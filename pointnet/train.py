"""Train PointNet on early-fused radar frames (per-point segmentation).

Data: folders written by pointnet/build_frames.py (one per capture). Each
sample is one CARLA frame with the returns of all radars pooled in world
coordinates. Classes: background, vehicle, pedestrian, cyclist.

Handling the density (about 4,000 fused points per frame):
  * every frame is randomly subsampled to --num-points (default 4096) for
    training, so batches have a fixed size and memory is bounded
  * evaluation uses up to --max-eval-points of every frame (default 16384,
    i.e. all of them) and scores real points only
  * --train-stride / --eval-stride skip near-duplicate 20 Hz frames
  * class-weighted cross entropy, because about 98% of points are background

Split (default --split scenario): whole scenarios are held out. All campaign
runs so far used seed 42, so the same scenario in two captures is near-identical
traffic and a random or per-capture split would leak. Defaults: validation
scenarios 4 and 10, test scenario 9, the rest train.

Usage (from the repo root):
  python pointnet/train.py --data dataset/Data/sensor_capture_*/pointnet_frames
  python pointnet/train.py --data ... --epochs 60 --batch-size 32 --num-points 4096
  python pointnet/train.py --data ... --aug strong        # site-independent: no absolute-position shortcut
  python pointnet/train.py --data ... --eval-only --checkpoint runs/pointnet_X/best.pt

Outputs: runs/pointnet_<timestamp>/ with best.pt, last.pt, log.csv, results.json, results.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.exit("PyTorch is required. GPU: pip install torch --index-url "
             "https://download.pytorch.org/whl/cu121   CPU: pip install torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PointNetSeg, recalibrate_bn, tnet_regularizer  # noqa: E402

FEATURES = ["x", "y", "z", "v_radial_mps", "rcs_dBsm", "los_x", "los_y"]
XYZ_SCALE = np.array([20.0, 20.0, 3.0], np.float32)
V_SCALE = 5.0
RCS_OFFSET, RCS_SCALE = 10.0, 15.0
AUG = {
    # rot (rad, +/-), translation (m, +/-), scale range, mirror probability, xyz jitter (m)
    "none":   (0.0, 0.0, (1.0, 1.0), 0.0, 0.0),
    "light":  (math.radians(5), 1.0, (0.95, 1.05), 0.0, 0.02),
    "strong": (math.pi, 10.0, (0.9, 1.1), 0.5, 0.02),
}


# Data ------------------------------------------------------------------------

def resolve_dirs(patterns):
    dirs = []
    for p in patterns:
        for d in sorted(glob.glob(p)) or [p]:
            d = Path(d)
            if (d / "pointnet_frames" / "meta.json").is_file():
                d = d / "pointnet_frames"
            if (d / "meta.json").is_file():
                dirs.append(d.resolve())
            else:
                print(f"[skip] {d}: no meta.json (run pointnet/build_frames.py)", flush=True)
    if not dirs:
        sys.exit("no frame folders found")
    return dirs


def load_index(dirs):
    """Per dir: frames (F,3) [carla frame, segment, scenario] and per-frame class counts (F,K)."""
    out = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        ptr = np.load(d / "frame_ptr.npy")
        frames = np.load(d / "frames.npy")
        label = np.load(d / "label.npy", mmap_mode="r")
        k = len(meta["classes"])
        fid = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
        counts = np.bincount(fid * k + np.asarray(label, np.int64),
                             minlength=(len(ptr) - 1) * k).reshape(-1, k)
        out.append({"dir": d, "meta": meta, "frames": frames, "counts": counts})
    classes = out[0]["meta"]["classes"]
    for o in out:
        if o["meta"]["classes"] != classes:
            sys.exit(f"class list differs in {o['dir']}")
    return out, classes


def split_items(index, args):
    """Return dict split -> array of (dir_idx, frame_idx)."""
    val_s = set(args.val_scenarios)
    test_s = set(args.test_scenarios)
    items = {"train": [], "val": [], "test": []}
    for di, o in enumerate(index):
        fr = o["frames"]
        name = o["dir"].parent.name
        seg_pos = np.zeros(len(fr))
        if args.split == "time":
            for sg in np.unique(fr[:, 1]):
                idx = np.where(fr[:, 1] == sg)[0]
                seg_pos[idx] = np.arange(len(idx)) / max(len(idx), 1)
        for fi in range(len(fr)):
            if args.split == "scenario":
                sc = int(fr[fi, 2])
                part = "test" if sc in test_s else "val" if sc in val_s else "train"
            elif args.split == "capture":
                part = ("test" if any(t in name for t in args.test_captures) else
                        "val" if any(v in name for v in args.val_captures) else "train")
            else:  # time: last val_frac of every segment
                part = "val" if seg_pos[fi] >= 1.0 - args.val_frac else "train"
            items[part].append((di, fi))
    out = {}
    for part, lst in items.items():
        arr = np.array(lst, dtype=np.int64).reshape(-1, 2)
        stride = args.train_stride if part == "train" else args.eval_stride
        out[part] = arr[::max(1, stride)]
    return out


class FusedFrames(Dataset):
    """One item = one fused frame. Arrays are memory-mapped lazily per worker."""

    def __init__(self, dirs, items, train, num_points, max_eval_points, center, aug):
        self.dirs = [str(d) for d in dirs]
        self.items = items
        self.train = train
        self.cap = num_points if train else max_eval_points
        self.center = np.asarray(center, np.float32)
        self.aug = AUG[aug] if train else AUG["none"]
        self._mm = None

    def __len__(self):
        return len(self.items)

    def _open(self):
        self._mm = []
        for d in self.dirs:
            d = Path(d)
            self._mm.append({
                "xyz": np.load(d / "xyz.npy", mmap_mode="r"),
                "feat": np.load(d / "feat.npy", mmap_mode="r"),
                "label": np.load(d / "label.npy", mmap_mode="r"),
                "ptr": np.load(d / "frame_ptr.npy"),
            })

    def __getitem__(self, i):
        if self._mm is None:
            self._open()
        di, fi = self.items[i]
        m = self._mm[di]
        s, e = int(m["ptr"][fi]), int(m["ptr"][fi + 1])
        rng = np.random.default_rng() if self.train else np.random.default_rng(int(i))
        n = e - s
        if n > self.cap:
            idx = np.sort(rng.choice(n, self.cap, replace=False)) + s
            xyz, feat, lab = m["xyz"][idx], m["feat"][idx], m["label"][idx]
        else:
            xyz, feat, lab = m["xyz"][s:e], m["feat"][s:e], m["label"][s:e]
        xyz = np.array(xyz, np.float32) - self.center
        feat = np.array(feat, np.float32)
        lab = np.array(lab, np.int64)

        rot, trans, (s_lo, s_hi), p_mirror, jit = self.aug
        if rot or trans or s_lo != 1.0 or p_mirror or jit:
            th = rng.uniform(-rot, rot)
            c, sn = math.cos(th), math.sin(th)
            r = np.array([[c, -sn], [sn, c]], np.float32)
            xyz[:, :2] = xyz[:, :2] @ r.T
            feat[:, 2:4] = feat[:, 2:4] @ r.T
            if rng.random() < p_mirror:
                xyz[:, 1] *= -1
                feat[:, 3] *= -1
            xyz *= rng.uniform(s_lo, s_hi)
            xyz[:, :2] += rng.uniform(-trans, trans, size=2).astype(np.float32)
            if jit:
                xyz += rng.normal(0, jit, size=xyz.shape).astype(np.float32)

        x = np.empty((len(lab), 7), np.float32)
        x[:, :3] = xyz / XYZ_SCALE
        x[:, 3] = feat[:, 0] / V_SCALE
        x[:, 4] = (feat[:, 1] + RCS_OFFSET) / RCS_SCALE
        x[:, 5:7] = feat[:, 2:4]
        return torch.from_numpy(x.T.copy()), torch.from_numpy(lab)


def make_collate(fixed_n):
    """Pad every frame to the same N by repeating its own points (max-pool safe)."""
    def collate(batch):
        n = fixed_n or max(lab.shape[0] for _, lab in batch)
        xs, ys, ms = [], [], []
        for x, lab in batch:
            k = lab.shape[0]
            if k < n:
                pad = torch.randint(0, k, (n - k,))
                x = torch.cat([x, x[:, pad]], 1)
                lab = torch.cat([lab, lab[pad]])
            m = torch.zeros(n, dtype=torch.bool)
            m[:k] = True
            xs.append(x)
            ys.append(lab)
            ms.append(m)
        return torch.stack(xs), torch.stack(ys), torch.stack(ms)
    return collate


# Metrics -----------------------------------------------------------------------

def scores(conf, classes):
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    gt = conf.sum(1)
    pr = conf.sum(0)
    iou = tp / np.maximum(gt + pr - tp, 1)
    rec = tp / np.maximum(gt, 1)
    prec = tp / np.maximum(pr, 1)
    present = gt > 0
    fg = present.copy()
    fg[0] = False
    return {
        "mIoU": float(iou[present].mean()) if present.any() else 0.0,
        "fg_mIoU": float(iou[fg].mean()) if fg.any() else 0.0,
        "accuracy": float(tp.sum() / max(conf.sum(), 1)),
        "per_class": {c: {"IoU": float(iou[k]), "precision": float(prec[k]), "recall": float(rec[k]),
                          "points": int(gt[k])} for k, c in enumerate(classes)},
        "confusion": conf.astype(np.int64).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device, k, amp):
    model.eval()
    conf = np.zeros((k, k), np.int64)
    for x, y, m in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(x)
        pred = logits.argmax(1).cpu()
        yt, pt = y[m].numpy(), pred[m].numpy()
        conf += np.bincount(yt * k + pt, minlength=k * k).reshape(k, k)
    return conf


# Main -----------------------------------------------------------------------------

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fmt_scores(s, classes):
    pc = "  ".join(f"{c[:4]} {s['per_class'][c]['IoU']:.3f}" for c in classes)
    return f"mIoU {s['mIoU']:.3f}  fg_mIoU {s['fg_mIoU']:.3f}  acc {s['accuracy']:.4f}  | {pc}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True,
                   help="pointnet_frames folders or capture folders (globs ok)")
    p.add_argument("--split", choices=("scenario", "capture", "time"), default="scenario")
    p.add_argument("--val-scenarios", type=int, nargs="*", default=[4, 10])
    p.add_argument("--test-scenarios", type=int, nargs="*", default=[9])
    p.add_argument("--val-captures", nargs="*", default=[])
    p.add_argument("--test-captures", nargs="*", default=[])
    p.add_argument("--val-frac", type=float, default=0.2, help="--split time only")
    p.add_argument("--train-stride", type=int, default=1, help="use every Nth training frame")
    p.add_argument("--eval-stride", type=int, default=2, help="use every Nth val/test frame")
    p.add_argument("--num-points", type=int, default=4096, help="points per training frame")
    p.add_argument("--max-eval-points", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--frames-per-epoch", type=int, default=0,
                   help="random training frames per epoch (0 = all)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=float, default=1.0, help="channel multiplier (0.5 = lighter)")
    p.add_argument("--feature-tnet", action="store_true", help="64x64 feature alignment T-Net")
    p.add_argument("--norm", choices=("bn", "gn"), default="bn",
                   help="bn = BatchNorm (re-estimated before eval), gn = GroupNorm")
    p.add_argument("--bn-batches", type=int, default=50,
                   help="training batches used to re-estimate BatchNorm statistics")
    p.add_argument("--aug", choices=tuple(AUG), default="light")
    p.add_argument("--class-weight-power", type=float, default=0.5,
                   help="weight = freq^-power (0 = unweighted)")
    p.add_argument("--max-class-weight", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="run folder (default runs/pointnet_<time>)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.no_amp

    dirs = resolve_dirs(args.data)
    index, classes = load_index(dirs)
    k = len(classes)
    items = split_items(index, args)

    def class_counts(arr):
        c = np.zeros(k, np.int64)
        for di, fi in arr:
            c += index[di]["counts"][fi]
        return c

    counts = {part: class_counts(arr) for part, arr in items.items()}
    print(f"[data] {len(dirs)} folders, device {device}{' (AMP)' if amp else ''}", flush=True)
    for part in ("train", "val", "test"):
        print(f"  {part:<5} {len(items[part]):>7,} frames  points per class "
              f"{dict(zip(classes, counts[part].tolist()))}", flush=True)
    if len(items["train"]) == 0 and not args.eval_only:
        sys.exit("no training frames - check --val-scenarios / --test-scenarios")

    ckpt = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        center = np.array(ckpt["config"]["center"], np.float32)
    else:
        center = np.median([o["meta"]["rig_center_xyz"] for o in index], 0).astype(np.float32)

    freq = counts["train"] / max(counts["train"].sum(), 1)
    if args.class_weight_power > 0 and counts["train"].sum():
        w = np.where(freq > 0, np.maximum(freq, 1e-6) ** -args.class_weight_power, 0.0)
        w = w / w[freq > 0].min()
        w = np.minimum(w, args.max_class_weight)
    else:
        w = np.ones(k)
    print(f"[data] class weights {dict(zip(classes, np.round(w, 2).tolist()))}", flush=True)
    weight = torch.tensor(w, dtype=torch.float32, device=device)

    def loader(part, train):
        ds = FusedFrames(dirs, items[part], train, args.num_points, args.max_eval_points,
                         center, args.aug)
        return DataLoader(ds, batch_size=args.batch_size if train else args.eval_batch_size,
                          shuffle=False, num_workers=args.workers,
                          collate_fn=make_collate(args.num_points if train else 0),
                          pin_memory=device.type == "cuda", drop_last=train,
                          persistent_workers=args.workers > 0)

    model = PointNetSeg(len(FEATURES), k, width=args.width,
                        feature_tnet=args.feature_tnet, norm=args.norm).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())

    run_dir = Path(args.out) if args.out else Path("runs") / time.strftime("pointnet_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"center": center.tolist(), "classes": classes, "features": FEATURES,
                           "class_weights": w.tolist(), "data_dirs": [str(d) for d in dirs],
                           "params": n_params}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[model] PointNetSeg width {args.width}  tnet {args.feature_tnet}  {n_params:,} params"
          f"  -> {run_dir}", flush=True)

    val_loader = loader("val", False) if len(items["val"]) else None
    test_loader = loader("test", False) if len(items["test"]) else None

    if not args.eval_only:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, (args.frames_per_epoch or len(items["train"])) // args.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        train_ds = FusedFrames(dirs, items["train"], True, args.num_points, args.max_eval_points,
                               center, args.aug)
        log_path = run_dir / "log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "lr", "frames_per_s", "val_mIoU",
                                    "val_fg_mIoU"] + [f"val_IoU_{c}" for c in classes])
        best = -1.0
        rng = np.random.default_rng(args.seed)
        for epoch in range(1, args.epochs + 1):
            n_ep = steps_per_epoch * args.batch_size
            order = rng.permutation(len(train_ds))
            if n_ep > len(order):
                order = np.concatenate([order, rng.integers(0, len(train_ds), n_ep - len(order))])
            sub = torch.utils.data.Subset(train_ds, order[:n_ep].tolist())
            tl = DataLoader(sub, batch_size=args.batch_size, shuffle=False, drop_last=True,
                            num_workers=args.workers, collate_fn=make_collate(args.num_points),
                            pin_memory=device.type == "cuda")
            model.train()
            t0 = time.time()
            tot = cnt = 0.0
            for x, y, m in tl:
                x, y, m = (t.to(device, non_blocking=True) for t in (x, y, m))
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits, trans = model(x)
                    nll = F.cross_entropy(logits.float(), y, reduction="none")
                    wy = weight[y] * m
                    loss = (nll * wy).sum() / wy.sum().clamp_min(1e-6)
                    if trans is not None:
                        loss = loss + 1e-3 * tnet_regularizer(trans)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * x.shape[0]
                cnt += x.shape[0]
            dt = time.time() - t0
            if args.norm == "bn":
                bn_idx = rng.choice(len(train_ds), min(len(train_ds), args.bn_batches * args.batch_size),
                                    replace=False)
                bn_dl = DataLoader(torch.utils.data.Subset(train_ds, bn_idx.tolist()),
                                   batch_size=args.batch_size, num_workers=args.workers,
                                   collate_fn=make_collate(args.num_points))
                recalibrate_bn(model, (b[0] for b in bn_dl), device)
            row = [epoch, tot / max(cnt, 1), sched.get_last_lr()[0], cnt / max(dt, 1e-6)]
            msg = f"[epoch {epoch:>3}/{args.epochs}] loss {tot / max(cnt, 1):.4f}  {cnt / max(dt, 1e-6):.1f} frames/s"
            if val_loader is not None:
                s = scores(evaluate(model, val_loader, device, k, amp), classes)
                row += [s["mIoU"], s["fg_mIoU"]] + [s["per_class"][c]["IoU"] for c in classes]
                msg += "  val " + fmt_scores(s, classes)
                metric = s["fg_mIoU"]
            else:
                metric = -tot
            print(msg, flush=True)
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
            state = {"model": model.state_dict(), "config": config, "epoch": epoch}
            torch.save(state, run_dir / "last.pt")
            if metric > best:
                best = metric
                torch.save(state, run_dir / "best.pt")
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[train] best val fg_mIoU {best:.3f} at epoch {ckpt['epoch']}", flush=True)

    results = {"run_dir": str(run_dir), "epoch": ckpt["epoch"] if ckpt else None,
               "params": n_params, "splits": {}}
    lines = ["# PointNet early fusion: results", "",
             f"Run: `{run_dir}`  best epoch: {results['epoch']}  params: {n_params:,}", "",
             "| split | frames | mIoU | fg mIoU | acc | " + " | ".join(f"IoU {c}" for c in classes) + " |",
             "|---|---|---|---|---|" + "---|" * k]
    for part, ld in (("val", val_loader), ("test", test_loader)):
        if ld is None:
            continue
        s = scores(evaluate(model, ld, device, k, amp), classes)
        results["splits"][part] = s | {"frames": int(len(items[part]))}
        print(f"[{part}] " + fmt_scores(s, classes), flush=True)
        lines.append(f"| {part} | {len(items[part]):,} | {s['mIoU']:.3f} | {s['fg_mIoU']:.3f} | "
                     f"{s['accuracy']:.4f} | " + " | ".join(f"{s['per_class'][c]['IoU']:.3f}"
                                                         for c in classes) + " |")
    lines += ["", "Per class precision / recall:", ""]
    for part, s in results["splits"].items():
        lines.append(f"- {part}: " + ", ".join(
            f"{c} P {s['per_class'][c]['precision']:.2f} R {s['per_class'][c]['recall']:.2f}"
            for c in classes))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {run_dir / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
