"""Covisibility: how many radars see each object, measured vs predicted.

For every object (vehicle / pedestrian / cyclist) on the monitored stretch in
every frame where ALL radars delivered a scan:

  measured k   = number of radars with >= 1 return matched to that object
                 (visible == 1 returns only, unless --all-returns)
  predicted k  = number of radars whose cone (range + horizontal / vertical FOV,
                 at the pose recorded in the capture) contains any part of the
                 object's bounding box. Pure geometry: no occlusion, no ray
                 sparsity, no FMCW filter.

The gap between the two is what occlusion (other actors, props), missed rays
and the FMCW visibility filter take away from the FOV-overlap prediction.

Outputs (in --out, default <capture>/covisibility):
  covis_summary.csv      per group x class x source: % of object-frames seen by
                         exactly k radars, k = 0..N, plus >= k and mean k
  covis_compare.csv      measured vs predicted side by side per class and k
  covis_confusion_<class>.csv   predicted k (rows) x measured k (columns)
  covis_per_radar.csv    per radar and class: realization = P(measured | predicted)
                         and the count measured outside the predicted cone
                         (should be ~0; a large value means the geometry is off)
  covis_report.md        the tables above in one page
  covis_<class>.png      bar chart, measured vs predicted, exactly k radars

Usage (from dataset/):
  python -m tools.covisibility Data/sensor_capture_20260929_210312
  python -m tools.covisibility "Data/sensor_capture_*" --out Data/covis_all --by scenario
  python -m tools.covisibility CAP --range 50          # capture taken at 50 m

This replaces the pedestrian-only, 8-radar rig_tests/covis_lib.py + emp.py/ab3.py
scripts; the cone test is the same (CARLA samples radar rays inside an ellipse
spanned by the two FOVs, see --shape).
"""

import argparse
import collections
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import capture_io as io  # noqa: E402


# ---------------------------------------------------------------- geometry
def bbox_samples(a):
    """15 world points on an actor's bounding box: centre + 4 corners at the
    bottom (+0.1 m), middle and top (-0.05 m)."""
    loc = a["location"]
    yaw = math.radians(a.get("rotation", {}).get("yaw", 0.0))
    bb = a.get("bbox") or {}
    bl = bb.get("location", {"x": 0.0, "y": 0.0, "z": 0.9})
    ex = bb.get("extent", {"x": 0.3, "y": 0.3, "z": 0.9})
    cx = loc["x"] + bl["x"] * math.cos(yaw) - bl["y"] * math.sin(yaw)
    cy = loc["y"] + bl["x"] * math.sin(yaw) + bl["y"] * math.cos(yaw)
    cz = loc["z"] + bl["z"]
    exx, eyy, ezz = max(ex["x"], 0.05), max(ex["y"], 0.05), max(ex["z"], 0.1)
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for z in (cz - ezz + 0.1, cz, cz + ezz - 0.05):
        pts.append((cx, cy, z))
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            pts.append((cx + sx * exx * c - sy * eyy * s, cy + sx * exx * s + sy * eyy * c, z))
    return np.array(pts)


class Cones:
    """Vectorized radar cone test for one capture's rig."""

    def __init__(self, poses, range_m, hfov_deg, vfov_deg, shape="ellipse"):
        self.labels = list(poses)
        P = [poses[k] for k in self.labels]
        self.pos = np.array([[p["x"], p["y"], p["z"]] for p in P])
        self.yaw = np.radians([p["yaw"] for p in P])
        self.pitch = np.radians([p["pitch"] for p in P])
        self.range = range_m
        self.th = math.tan(math.radians(hfov_deg / 2.0))
        self.tv = math.tan(math.radians(vfov_deg / 2.0))
        self.shape = shape

    def mask(self, pts):
        """Bitmask of radars whose cone contains any of the points (M x 3)."""
        d = pts[None, :, :] - self.pos[:, None, :]           # N x M x 3
        cy, sy = np.cos(self.yaw)[:, None], np.sin(self.yaw)[:, None]
        cp, sp = np.cos(self.pitch)[:, None], np.sin(self.pitch)[:, None]
        a = d[..., 0] * cy + d[..., 1] * sy                   # along yaw (horizontal)
        b = -d[..., 0] * sy + d[..., 1] * cy                  # lateral
        c = d[..., 2]
        xs = a * cp + c * sp                                  # along boresight
        zs = -a * sp + c * cp                                 # up in sensor frame
        with np.errstate(divide="ignore", invalid="ignore"):
            u = b / (xs * self.th)
            v = zs / (xs * self.tv)
        if self.shape == "ellipse":
            inside = u * u + v * v <= 1.0
        else:
            inside = (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
        ok = (xs > 0.3) & (xs <= self.range) & inside
        hit = ok.any(axis=1)
        return int(sum(1 << i for i in np.nonzero(hit)[0]))


def popcount(m):
    return bin(m).count("1")


MODES = set()   # how returns were counted, per capture (for the report)


# ---------------------------------------------------------------- per capture
def analyse_capture(cap, args, rows, per_radar):
    poses = io.radar_poses(cap)
    if not poses:
        print(f"[skip] {cap}: no radar poses in the CSV")
        return None, None
    labels = list(poses)
    idx = {k: i for i, k in enumerate(labels)}
    n = len(labels)
    full = (1 << n) - 1
    rng, hfov, vfov = io.radar_params(cap, args.range, args.hfov, args.vfov)
    cones = Cones(poses, rng, hfov, vfov, args.shape)
    zone = io.rig_zone(poses, args.zone_margin)
    segs = io.Segments(cap)

    def frame_ok(f):
        if args.frame_stride > 1 and f % args.frame_stride:
            return False
        return (not segs) or segs.lookup(f) is not None

    print(f"[{cap.name}] {n} radars, range {rng:.0f} m, hfov {hfov:.0f}, vfov {vfov:.0f}, "
          f"zone x {zone[0]:.1f}..{zone[1]:.1f} y {zone[2]:.1f}..{zone[3]:.1f}")

    # Pass 1: which radars reported each frame, and which hit each object.
    scans = collections.defaultdict(int)          # frame -> radar mask
    hits = collections.defaultdict(int)           # (frame, actor) -> radar mask
    cols = ["frame", "sensor_label", "matched_actor_id", "visible"]
    has_visible = "visible" in io.csv_columns(cap)
    use_visible = has_visible and not args.all_returns
    mode = ("visible == 1" if use_visible else
            ("all matched returns" if args.all_returns else "all matched returns (no visible column)"))
    if not has_visible and not args.all_returns:
        print("  note: no 'visible' column (not post-processed); counting all returns")
    for ch in io.iter_labeled(cap, cols, max_rows=args.max_rows):
        ch = ch[segs.mask(ch["frame"].to_numpy(np.int64), args.frame_stride)]
        if ch.empty:
            continue
        sc = ch[["frame", "sensor_label"]].drop_duplicates()
        for fr, lab in zip(sc["frame"].to_numpy(np.int64), sc["sensor_label"]):
            i = idx.get(lab)
            if i is not None:
                scans[int(fr)] |= 1 << i
        m = ch["matched_actor_id"].notna() & (ch["matched_actor_id"].astype(str).str.len() > 0)
        if use_visible:
            m &= ch["visible"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])
        h = ch.loc[m, ["frame", "matched_actor_id", "sensor_label"]].drop_duplicates()
        for fr, aid, lab in zip(h["frame"].to_numpy(np.int64), h["matched_actor_id"], h["sensor_label"]):
            i = idx.get(lab)
            if i is None:
                continue
            try:
                a = int(float(aid))
            except ValueError:
                continue
            hits[(int(fr), a)] |= 1 << i

    good = {f for f, m in scans.items() if m == full}
    print(f"  frames with all {n} radars: {len(good)} of {len(scans)}")

    # Pass 2: every object on the stretch in those frames.
    n_obj = 0
    for f, actors in io.iter_actor_frames(cap, frame_ok=lambda fr: fr in good):
        seg = segs.lookup(f) if segs else None
        group = io.scenario_key(seg, args.by) if seg else "all"
        for a in actors:
            cls = io.actor_class(a)
            if cls is None:
                continue
            x, y = a["location"]["x"], a["location"]["y"]
            if not io.in_zone(zone, x, y):
                continue
            meas = hits.get((f, int(a["id"])), 0)
            pred = cones.mask(bbox_samples(a))
            rows.append((cap.name, group, cls, n, popcount(meas), popcount(pred)))
            for i, lab in enumerate(labels):
                b = 1 << i
                per_radar[(lab, cls)][0] += bool(pred & b)
                per_radar[(lab, cls)][1] += bool(pred & b and meas & b)
                per_radar[(lab, cls)][2] += bool(meas & b and not pred & b)
            n_obj += 1
    print(f"  object-frames on the stretch: {n_obj}")
    return n, mode


# ---------------------------------------------------------------- reporting
def summarize(rows, nmax):
    """{(group, cls, source): counts array 0..nmax}"""
    out = collections.defaultdict(lambda: np.zeros(nmax + 1, np.int64))
    for _cap, group, cls, _n, km, kp in rows:
        for g in ("ALL", group):
            out[(g, cls, "measured")][km] += 1
            out[(g, cls, "predicted")][kp] += 1
    return out


def pct(v):
    t = v.sum()
    return (100.0 * v / t) if t else v.astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="capture folders (globs ok)")
    ap.add_argument("--out", default="", help="output folder")
    ap.add_argument("--by", choices=("scenario", "segment"), default="scenario")
    ap.add_argument("--range", type=float, default=None, help="radar range m (default: run_meta / 35)")
    ap.add_argument("--hfov", type=float, default=None, help="horizontal FOV deg (default: run_meta / 120)")
    ap.add_argument("--vfov", type=float, default=None, help="vertical FOV deg (default: run_meta / 30)")
    ap.add_argument("--shape", choices=("ellipse", "rect"), default="ellipse",
                    help="cone cross-section; CARLA samples rays in an ellipse")
    ap.add_argument("--all-returns", action="store_true",
                    help="count every matched return, not only visible == 1")
    ap.add_argument("--zone-margin", type=float, default=0.0,
                    help="grow the stretch box (radar-mount bounding box) by this many m")
    ap.add_argument("--frame-stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--max-rows", type=int, default=0, help="stop after N CSV rows (quick test)")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-read every capture (default: reuse <capture>/_analysis_cache)")
    ap.add_argument("--cache-only", action="store_true",
                    help="only build the per-capture cache, no report (one capture per call)")
    args = ap.parse_args()

    caps = io.expand_captures(args.captures)
    if not caps:
        sys.exit("no capture with radar_data_labeled.csv found")
    rows, per_radar = [], collections.defaultdict(lambda: [0, 0, 0])
    nmax = 0
    for cap in caps:
        cp = io.cache_path(cap, "covis", r=args.range, h=args.hfov, v=args.vfov, s=args.shape,
                           a=args.all_returns, z=args.zone_margin, f=args.frame_stride,
                           m=args.max_rows)
        got = None if args.no_cache else io.load_cache(cp, cap)
        if got is None:
            r_cap, pr_cap = [], collections.defaultdict(lambda: [0, 0, 0])
            n, mode = analyse_capture(cap, args, r_cap, pr_cap)
            if not n:
                continue
            got = {"rows": r_cap, "per_radar": dict(pr_cap), "n": n, "mode": mode}
            io.save_cache(cp, got)
        else:
            print(f"[{cap.name}] cached: {cp}")
        rows += got["rows"]
        for k, v in got["per_radar"].items():
            for i in range(3):
                per_radar[k][i] += v[i]
        MODES.add(got["mode"])
        nmax = max(nmax, got["n"])
    if args.cache_only:
        print("cache built")
        return
    if not rows:
        sys.exit("no object-frames found on the stretch")
    out = io.out_dir(args.out, caps, "covisibility")
    S = summarize(rows, nmax)
    ks = list(range(nmax + 1))

    # covis_summary.csv
    hdr = ["group", "class", "source", "object_frames", "mean_k"] + \
          [f"exactly_{k}_pct" for k in ks] + [f"atleast_{k}_pct" for k in ks[1:]]
    body = []
    for (g, cls, src), v in sorted(S.items()):
        p = pct(v)
        body.append([g, cls, src, int(v.sum()), round(float((v * np.arange(nmax + 1)).sum() / max(v.sum(), 1)), 2)]
                    + [round(float(x), 2) for x in p]
                    + [round(float(p[k:].sum()), 2) for k in ks[1:]])
    io.write_csv(out / "covis_summary.csv", body, hdr)

    # covis_compare.csv + confusion
    cmp_rows, md = [], []
    md.append(f"# Covisibility: measured vs FOV-predicted\n\nCaptures: {', '.join(c.name for c in caps)}  \n"
              f"Returns counted: {'; '.join(sorted(MODES))}; "
              f"cone: {args.shape}.\n")
    for cls in io.CLASS_NAMES:
        vm, vp = S.get(("ALL", cls, "measured")), S.get(("ALL", cls, "predicted"))
        if vm is None or vm.sum() == 0:
            continue
        pm, pp = pct(vm), pct(vp)
        md.append(f"\n## {cls}  ({int(vm.sum())} object-frames)\n")
        md.append("| radars k | measured exactly k | predicted exactly k | measured >= k | predicted >= k |")
        md.append("|---|---|---|---|---|")
        for k in ks:
            cmp_rows.append([cls, k, round(pm[k], 2), round(pp[k], 2),
                             round(pm[k:].sum(), 2), round(pp[k:].sum(), 2)])
            if k <= nmax and (vm[k] or vp[k] or k <= 1):
                md.append(f"| {k} | {pm[k]:.1f}% | {pp[k]:.1f}% | {pm[k:].sum():.1f}% | {pp[k:].sum():.1f}% |")
        mk = (vm * np.arange(nmax + 1)).sum() / vm.sum()
        pk = (vp * np.arange(nmax + 1)).sum() / vp.sum()
        md.append(f"\nMean radars per object: measured {mk:.2f}, predicted {pk:.2f} "
                  f"(realized {100 * mk / pk if pk else 0:.0f}% of the FOV overlap).\n")
        conf = np.zeros((nmax + 1, nmax + 1), np.int64)
        for _c, _g, c2, _n, km, kp in rows:
            if c2 == cls:
                conf[kp, km] += 1
        io.write_csv(out / f"covis_confusion_{cls}.csv",
                     [[f"pred_{k}"] + list(map(int, conf[k])) for k in ks],
                     ["predicted\\measured"] + [f"meas_{k}" for k in ks])
    io.write_csv(out / "covis_compare.csv", cmp_rows,
                 ["class", "k", "measured_exactly_pct", "predicted_exactly_pct",
                  "measured_atleast_pct", "predicted_atleast_pct"])

    # per radar realization
    pr_rows = []
    md.append("\n## Per radar: realization = P(returns on object | object in its cone)\n")
    md.append("| radar | class | in cone | realized | measured outside cone |")
    md.append("|---|---|---|---|---|")
    for (lab, cls), (incone, both, outside) in sorted(per_radar.items(), key=lambda kv: (io._natkey(kv[0][0]), kv[0][1])):
        r = 100.0 * both / incone if incone else float("nan")
        pr_rows.append([lab, cls, incone, both, round(r, 1), outside])
        md.append(f"| {lab} | {cls} | {incone} | {r:.0f}% | {outside} |")
    io.write_csv(out / "covis_per_radar.csv", pr_rows,
                 ["radar", "class", "object_frames_in_cone", "with_returns", "realization_pct",
                  "returns_outside_cone"])

    # by scenario, mean k
    groups = sorted({g for g, _c, _s in S if g != "ALL"})
    if len(groups) > 1:
        md.append("\n## Mean radars per object by scenario (measured / predicted)\n")
        md.append("| scenario | " + " | ".join(io.CLASS_NAMES) + " |")
        md.append("|---" * (len(io.CLASS_NAMES) + 1) + "|")
        for g in groups:
            cells = []
            for cls in io.CLASS_NAMES:
                vm, vp = S.get((g, cls, "measured")), S.get((g, cls, "predicted"))
                if vm is None or vm.sum() == 0:
                    cells.append("-")
                    continue
                ar = np.arange(nmax + 1)
                cells.append(f"{(vm * ar).sum() / vm.sum():.2f} / {(vp * ar).sum() / vp.sum():.2f}")
            md.append(f"| {g} | " + " | ".join(cells) + " |")
    (out / "covis_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    plt = io.try_pyplot()
    if plt is not None:
        for cls in io.CLASS_NAMES:
            vm, vp = S.get(("ALL", cls, "measured")), S.get(("ALL", cls, "predicted"))
            if vm is None or vm.sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 3.6))
            w = 0.4
            ax.bar(np.arange(nmax + 1) - w / 2, pct(vp), w, label="predicted (FOV overlap)", color="#9aa5b1")
            ax.bar(np.arange(nmax + 1) + w / 2, pct(vm), w, label="measured (returns)", color="#2f6fb0")
            ax.set_xticks(ks)
            ax.set_xlabel("radars seeing the object")
            ax.set_ylabel("% of object-frames")
            ax.set_title(f"{cls}: covisibility, {int(vm.sum())} object-frames")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(out / f"covis_{cls}.png", dpi=150)
            plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
