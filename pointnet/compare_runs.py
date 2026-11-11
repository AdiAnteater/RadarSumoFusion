"""Compare PointNet runs side by side, e.g. the point-budget sweep.

For each run folder (runs/pointnet_*), reads results.json + config.json and the
pointnet_frames/meta.json of the data it trained on, and prints one row per run:

    run | radar pps | pts/scan | fused pts/frame | split | fg mIoU | IoU per class | P/R

radar pps comes from the captures' run_meta.json (DATASET_RADAR_POINTS_PER_SECOND,
15000 when unset). pts/scan = fused points per frame / radar count.

Usage (from the repo root):
    python pointnet/compare_runs.py runs/pointnet_*            # all runs
    python pointnet/compare_runs.py runs/pps15000 runs/pps1300 runs/pps800 --split test
    python pointnet/compare_runs.py runs/pointnet_* --csv runs/sweep.csv
"""

import argparse
import csv
import glob
import json
from pathlib import Path

DEFAULT_PPS = 15000


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def data_info(data_dirs):
    """(pps set, radar counts, mean fused points per frame) over the run's data."""
    pps, radars, ppf, frames = set(), set(), 0.0, 0
    for d in data_dirs:
        d = Path(d)
        meta = load(d / "meta.json") or {}
        n = int(meta.get("frames", 0))
        if n:
            ppf += meta.get("points_per_frame", {}).get("mean", 0.0) * n
            frames += n
        rm = load(d.parent / "run_meta.json") or {}
        env = rm.get("dataset_env", {}) or {}
        pps.add(int(float(env.get("DATASET_RADAR_POINTS_PER_SECOND", DEFAULT_PPS) or DEFAULT_PPS)))
        if rm.get("radar_count"):
            radars.add(int(rm["radar_count"]))
    return pps, radars, (ppf / frames if frames else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (globs ok)")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--csv", default="", help="also write the table here")
    args = ap.parse_args()

    dirs = [Path(p) for g in args.runs for p in (sorted(glob.glob(g)) or [g])]
    rows = []
    for d in dirs:
        res, cfg = load(d / "results.json"), load(d / "config.json")
        if not res or not cfg:
            print(f"[skip] {d}: no results.json / config.json")
            continue
        s = res.get("splits", {}).get(args.split)
        if not s:
            print(f"[skip] {d}: no {args.split} split")
            continue
        pps, radars, ppf = data_info(cfg.get("data_dirs", []))
        nr = max(radars) if radars else 8
        classes = [c for c in cfg.get("classes", []) if c != "background"]
        row = {"run": d.name, "pps": "/".join(map(str, sorted(pps))) or "?",
               "pts_per_scan": round(ppf / nr, 1) if ppf == ppf else "",
               "fused_pts_per_frame": round(ppf, 1) if ppf == ppf else "",
               "num_points": cfg.get("num_points"), "frames": s.get("frames"),
               "mIoU": round(s["mIoU"], 3), "fg_mIoU": round(s["fg_mIoU"], 3)}
        for c in classes:
            pc = s["per_class"].get(c, {})
            row[f"IoU_{c}"] = round(pc.get("IoU", 0.0), 3)
            row[f"P_{c}"] = round(pc.get("precision", 0.0), 2)
            row[f"R_{c}"] = round(pc.get("recall", 0.0), 2)
        rows.append(row)
    if not rows:
        raise SystemExit("no runs to compare")
    rows.sort(key=lambda r: -float(str(r["pps"]).split("/")[0]) if str(r["pps"])[0].isdigit() else 0)
    cols = list(rows[0])
    for r in rows[1:]:
        cols += [c for c in r if c not in cols]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
