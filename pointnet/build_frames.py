"""Turn a labeled capture into early-fused radar frames for PointNet.

Early fusion = for every CARLA frame, pool the returns of ALL radars into one
point cloud in world coordinates (no sensor id). One fused cloud = one training
sample.

Reads  <capture>/radar_data_labeled.csv  (after LabelRadarCapture + PostProcessDataset)
Writes <capture>/pointnet_frames/  (or --out DIR):
    xyz.npy        float32 (P, 3)  world x, y, z of every point (m)
    feat.npy       float16 (P, 4)  radial velocity (m/s, + = receding), rcs (dBsm),
                                   unit line-of-sight from the radar, x and y
    label.npy      uint8   (P,)    0 background, 1 vehicle, 2 pedestrian, 3 cyclist
    instance.npy   int32   (P,)    matched actor id, -1 for background
    frame_ptr.npy  int64   (F+1,)  points of frame i are [frame_ptr[i], frame_ptr[i+1])
    frames.npy     int64   (F, 3)  carla frame number, segment id, scenario id (-1 if none)
    meta.json      classes, feature names, counts, settings

Modes
  measured (default)  what a real radar would report: only returns with
                      visible == 1 (FMCW detection model), positions rebuilt from
                      the NOISY depth / azimuth, noisy radial velocity.
  clean               every return, exact hit position, noise-free velocity.

Usage (from the repo root):
  python pointnet/build_frames.py dataset/Data/sensor_capture_20260929_210312
  python pointnet/build_frames.py dataset/Data/sensor_capture_* --frame-stride 2
  python pointnet/build_frames.py dataset/Data/sensor_capture_X --mode clean

Building in pieces (parallel, or under a per-command time limit):
  python pointnet/build_frames.py CAPTURE --parts 4 --part 0      (and 1, 2, 3)
  python pointnet/build_frames.py CAPTURE --parts 4 --merge
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")

CLASS_NAMES = ["background", "vehicle", "pedestrian", "cyclist"]
CLASS_OF = {
    "car": 1, "truck": 1, "bus": 1, "van": 1, "vehicle": 1,
    "pedestrian": 2, "walker": 2,
    "bicycle": 3, "motorcycle": 3, "bike": 3, "cyclist": 3,
}
FEATURE_NAMES = ["v_radial_mps", "rcs_dBsm", "los_x", "los_y"]

BASE_COLS = ["frame", "sensor_id", "altitude_rad",
             "sensor_world_x_m", "sensor_world_y_m", "sensor_world_z_m",
             "sensor_pitch_deg", "sensor_yaw_deg", "sensor_roll_deg",
             "matched_actor_id", "matched_actor_kind", "matched_actor_class",
             "rcs_dBsm", "segment_id", "scenario_id"]
CLEAN_COLS = ["depth_m", "azimuth_rad", "velocity_mps",
              "hit_world_x_m", "hit_world_y_m", "hit_world_z_m"]
MEASURED_COLS = ["visible", "depth_m_noisy", "azimuth_rad_noisy", "velocity_mps_noisy"]


def sensor_to_world(depth, az, alt, sx, sy, sz, pitch_deg, yaw_deg, roll_deg):
    """CARLA radar detection -> world point (carla.Transform.transform, UE axes)."""
    lx = depth * np.cos(alt) * np.cos(az)
    ly = depth * np.cos(alt) * np.sin(az)
    lz = depth * np.sin(alt)
    p, y, r = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)
    cp, sp, cy, sy_, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    wx = lx * (cp * cy) + ly * (cy * sp * sr - sy_ * cr) + lz * (-cy * sp * cr - sy_ * sr) + sx
    wy = lx * (cp * sy_) + ly * (sy_ * sp * sr + cy * cr) + lz * (-sy_ * sp * cr + cy * sr) + sy
    wz = lx * sp + ly * (-cp * sr) + lz * (cp * cr) + sz
    return wx, wy, wz


def _num(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(np.float64)


def _int_or(s, fill=-1):
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(fill).to_numpy(np.int64)


def labels_from(cls_series, kind_series):
    cls = cls_series.str.strip().str.lower()
    kind = kind_series.str.strip().str.lower()
    lab = cls.map(CLASS_OF)
    lab = lab.where(lab.notna(), kind.map({"vehicle": 1, "walker": 2, "pedestrian": 2}))
    return lab.fillna(0).to_numpy(np.uint8)


class _ByteRange(io.RawIOBase):
    """Read-only view of bytes [start, end) of a file."""

    def __init__(self, path, start, end):
        self.f = open(path, "rb")
        self.f.seek(start)
        self.left = end - start

    def readable(self):
        return True

    def readinto(self, b):
        if self.left <= 0:
            return 0
        n = self.f.readinto(memoryview(b)[: min(len(b), self.left)])
        self.left -= n
        return n

    def close(self):
        self.f.close()
        super().close()


def _part_bounds(csv_path, parts):
    """Byte offsets that cut the data lines (after the header) into `parts` pieces."""
    size = csv_path.stat().st_size
    with open(csv_path, "rb") as f:
        header = f.readline()
        first = f.tell()
        cuts = [first]
        for j in range(1, parts):
            f.seek(max(first, size * j // parts))
            f.readline()
            cuts.append(max(f.tell(), cuts[-1]))
    cuts.append(size)
    return header.decode("utf-8").strip().split(","), cuts


def _parts_dir(out, args):
    return Path(args.parts_dir) / out.parent.name if args.parts_dir else out / "_parts"


def build(capture: Path, args) -> None:
    csv_path = capture / "radar_data_labeled.csv"
    if not csv_path.is_file():
        print(f"[skip] {csv_path} not found (run LabelRadarCapture + PostProcessDataset)", flush=True)
        return
    out = Path(args.out) if args.out else capture / ("pointnet_frames" if args.mode == "measured"
                                                     else "pointnet_frames_clean")
    header = set(pd.read_csv(csv_path, nrows=0).columns)
    need = BASE_COLS + CLEAN_COLS + (MEASURED_COLS if args.mode == "measured" else [])
    missing = [c for c in need if c not in header and c not in ("segment_id", "scenario_id")]
    if missing:
        print(f"[skip] {csv_path.name} lacks {missing} - run PostProcessDataset first", flush=True)
        return
    cols = [c for c in need if c in header]

    parts = {k: [] for k in ("frame", "xyz", "feat", "label", "inst", "seg", "scen")}
    rows = kept = 0
    checked = False
    t0 = time.time()
    print(f"[frames] {csv_path}  mode={args.mode}  stride={args.frame_stride}", flush=True)
    if args.merge:
        _merge(capture, out, args)
        return
    src, extra = csv_path, {}
    if args.parts > 1:
        names, cuts = _part_bounds(csv_path, args.parts)
        src = io.BufferedReader(_ByteRange(csv_path, cuts[args.part], cuts[args.part + 1]),
                                buffer_size=1 << 20)
        extra = {"header": None, "names": names}
        print(f"  part {args.part + 1}/{args.parts}: bytes {cuts[args.part]:,}..{cuts[args.part + 1]:,}",
              flush=True)
    reader = pd.read_csv(src, usecols=cols, chunksize=args.chunksize, low_memory=False, **extra,
                         dtype={"matched_actor_kind": str, "matched_actor_class": str},
                         keep_default_na=False, na_values={c: [""] for c in cols
                                                           if c not in ("matched_actor_kind",
                                                                        "matched_actor_class")})
    for ch in reader:
        if args.max_rows and rows >= args.max_rows:
            break
        rows += len(ch)
        frame = _int_or(ch["frame"])
        keep = (frame % args.frame_stride) == 0
        if args.mode == "measured":
            keep &= _num(ch["visible"]) == 1
        if not keep.any():
            continue
        ch = ch[keep]
        frame = frame[keep]
        sx, sy, sz = (_num(ch[c]) for c in ("sensor_world_x_m", "sensor_world_y_m", "sensor_world_z_m"))
        sp, syaw, sr = (_num(ch[c]) for c in ("sensor_pitch_deg", "sensor_yaw_deg", "sensor_roll_deg"))
        alt = _num(ch["altitude_rad"])

        if not checked:
            # The spherical -> world formula must reproduce the labeler's exact hits.
            wx, wy, wz = sensor_to_world(_num(ch["depth_m"]), _num(ch["azimuth_rad"]), alt,
                                         sx, sy, sz, sp, syaw, sr)
            err = np.sqrt((wx - _num(ch["hit_world_x_m"])) ** 2 + (wy - _num(ch["hit_world_y_m"])) ** 2
                          + (wz - _num(ch["hit_world_z_m"])) ** 2)
            err = err[np.isfinite(err)]
            if err.size:
                print(f"  transform check vs hit_world: median {np.median(err):.4f} m, "
                      f"max {err.max():.4f} m", flush=True)
                if np.median(err) > 0.05:
                    sys.exit("ERROR: spherical->world transform disagrees with hit_world_*; aborting")
            checked = True

        if args.mode == "measured":
            x, y, z = sensor_to_world(_num(ch["depth_m_noisy"]), _num(ch["azimuth_rad_noisy"]), alt,
                                      sx, sy, sz, sp, syaw, sr)
            vr = _num(ch["velocity_mps_noisy"])
        else:
            x, y, z = (_num(ch[c]) for c in ("hit_world_x_m", "hit_world_y_m", "hit_world_z_m"))
            vr = _num(ch["velocity_mps"])
        rcs = _num(ch["rcs_dBsm"])
        rcs = np.where(np.isfinite(rcs), rcs, -10.0)
        dx, dy = x - sx, y - sy
        nrm = np.maximum(np.hypot(dx, dy), 1e-6)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(vr)

        lab = labels_from(ch["matched_actor_class"], ch["matched_actor_kind"])
        inst = _int_or(ch["matched_actor_id"]).astype(np.int32)
        inst = np.where(lab > 0, inst, -1)
        seg = _int_or(ch["segment_id"]) if "segment_id" in ch else np.full(len(ch), -1)
        scen = _int_or(ch["scenario_id"]) if "scenario_id" in ch else np.full(len(ch), -1)

        parts["frame"].append(frame[ok])
        parts["xyz"].append(np.stack([x, y, z], 1)[ok].astype(np.float32))
        parts["feat"].append(np.stack([vr, rcs, dx / nrm, dy / nrm], 1)[ok].astype(np.float16))
        parts["label"].append(lab[ok])
        parts["inst"].append(inst[ok])
        parts["seg"].append(seg[ok])
        parts["scen"].append(scen[ok])
        kept += int(ok.sum())
        print(f"  {rows:>12,} rows read  {kept:>12,} points kept  "
              f"{rows / max(time.time() - t0, 1e-6):>9,.0f} rows/s", flush=True)

    empty = {"frame": np.zeros(0, np.int64), "xyz": np.zeros((0, 3), np.float32),
             "feat": np.zeros((0, 4), np.float16), "label": np.zeros(0, np.uint8),
             "inst": np.zeros(0, np.int32), "seg": np.zeros(0, np.int64), "scen": np.zeros(0, np.int64)}
    arrays = {k: (np.concatenate(v) if v else empty[k]) for k, v in parts.items()}
    if args.parts > 1:
        pdir = _parts_dir(out, args)
        pdir.mkdir(parents=True, exist_ok=True)
        np.savez(pdir / f"part_{args.part}_of_{args.parts}.npz", rows=np.array(rows), **arrays)
        print(f"[frames] part {args.part + 1}/{args.parts}: {kept:,} points -> {pdir}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        return
    _write(out, csv_path, arrays, rows, args, t0)


def _merge(capture, out, args):
    t0 = time.time()
    pdir = _parts_dir(out, args)
    files = [pdir / f"part_{i}_of_{args.parts}.npz" for i in range(args.parts)]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        sys.exit(f"missing parts in {pdir}: {missing}")
    loaded = [np.load(f) for f in files]
    keys = ("frame", "xyz", "feat", "label", "inst", "seg", "scen")
    arrays = {k: np.concatenate([z[k] for z in loaded]) for k in keys}
    rows = int(sum(int(z["rows"]) for z in loaded))
    _write(out, capture / "radar_data_labeled.csv", arrays, rows, args, t0)
    if not args.keep_parts:
        for f in files:
            try:
                f.unlink()
            except OSError:
                pass


def _write(out, csv_path, arrays, rows, args, t0):
    if len(arrays["frame"]) == 0:
        print("[frames] nothing kept", flush=True)
        return
    frame = arrays.pop("frame")
    order = np.argsort(frame, kind="stable")
    frame = frame[order]
    arrays = {k: v[order] for k, v in arrays.items()}

    uniq, start = np.unique(frame, return_index=True)
    frame_ptr = np.append(start, len(frame)).astype(np.int64)
    min_pts = args.min_points
    if min_pts > 1:
        n = np.diff(frame_ptr)
        good = n >= min_pts
        if not good.all():
            sel = np.concatenate([np.arange(frame_ptr[i], frame_ptr[i + 1]) for i in np.where(good)[0]])
            frame = frame[sel]
            arrays = {k: v[sel] for k, v in arrays.items()}
            uniq, start = np.unique(frame, return_index=True)
            frame_ptr = np.append(start, len(frame)).astype(np.int64)
    seg_f = arrays["seg"][frame_ptr[:-1]]
    scen_f = arrays["scen"][frame_ptr[:-1]]

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "xyz.npy", arrays["xyz"])
    np.save(out / "feat.npy", arrays["feat"])
    np.save(out / "label.npy", arrays["label"])
    np.save(out / "instance.npy", arrays["inst"])
    np.save(out / "frame_ptr.npy", frame_ptr)
    np.save(out / "frames.npy", np.stack([uniq, seg_f, scen_f], 1).astype(np.int64))

    counts = np.bincount(arrays["label"], minlength=len(CLASS_NAMES))
    ppf = np.diff(frame_ptr)
    meta = {
        "format": 1,
        "source_csv": str(csv_path),
        "mode": args.mode,
        "frame_stride": args.frame_stride,
        "classes": CLASS_NAMES,
        "features": ["x", "y", "z"] + FEATURE_NAMES,
        "points": int(len(frame)),
        "frames": int(len(uniq)),
        "points_per_frame": {"mean": float(ppf.mean()), "median": float(np.median(ppf)),
                             "p95": float(np.percentile(ppf, 95)), "max": int(ppf.max())},
        "class_counts": {c: int(n) for c, n in zip(CLASS_NAMES, counts)},
        "rig_center_xyz": [float(v) for v in np.median(arrays["xyz"], 0)],
        "segments": sorted(int(s) for s in np.unique(seg_f)),
        "rows_read": int(rows),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[frames] wrote {out}: {meta['frames']:,} frames, {meta['points']:,} points, "
          f"{meta['points_per_frame']['mean']:.0f} pts/frame (p95 {meta['points_per_frame']['p95']:.0f}), "
          f"classes {meta['class_counts']}  ({time.time() - t0:.0f} s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("captures", nargs="+", help="capture folder(s); globs are expanded")
    p.add_argument("--mode", choices=("measured", "clean"), default="measured")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="keep every Nth CARLA frame (20 Hz frames are near-duplicates)")
    p.add_argument("--min-points", type=int, default=16, help="drop frames with fewer points")
    p.add_argument("--out", default=None, help="output folder (single capture only)")
    p.add_argument("--max-rows", type=int, default=0, help="stop after N CSV rows (quick test)")
    p.add_argument("--chunksize", type=int, default=1_000_000)
    p.add_argument("--parts", type=int, default=1, help="split the CSV into N byte ranges")
    p.add_argument("--part", type=int, default=0, help="which range to build (0..N-1)")
    p.add_argument("--merge", action="store_true", help="combine the N parts into the final folder")
    p.add_argument("--parts-dir", default="", help="where part files go (default <out>/_parts)")
    p.add_argument("--keep-parts", action="store_true")
    args = p.parse_args()
    caps = []
    for c in args.captures:
        caps += [Path(x) for x in sorted(glob.glob(c))] or [Path(c)]
    if args.out and len(caps) > 1:
        sys.exit("--out only works with a single capture")
    for c in caps:
        if c.is_dir():
            build(c, args)
        else:
            print(f"[skip] {c} is not a folder", flush=True)


if __name__ == "__main__":
    main()
