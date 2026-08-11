"""Realized cross-radar overlap from captured data (the data-driven analog of the
geometric S matrix in taca_topology_mask.py).

Geometric S asks "could radars i and j share a view" (from poses + FoV + range).
This asks "did they ACTUALLY co-observe the same targets" — i.e. for each
(frame, actor) [or (window, actor) when accumulating W frames per radar], which
radars produced a matched return, then a pairwise Jaccard:

    realized_overlap[i,j] = #(both i and j returned on the target) / #(i or j did)

Accumulating W frames per radar densifies sparse per-frame returns, so co-observation
(and thus realized overlap) rises with W — at the cost of ~v*W*dt motion smear.

Usage:
    python -m tools.realized_overlap <radar_data_labeled.csv> [--windows 1,5,10,20]
"""
import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

from tools.taca_topology_mask import (
    Radar, compute_windowed_taca_mask,
    mst_topology_ordering, select_implicit_K, build_windowed_attention_mask,
)


def geometric_overlap(extr_path, labels, R_max, fov_half, theta3db):
    """Geometric co-coverage Jaccard over road-band cells: |cells in both FoVs| /
    |cells in either|. Same Jaccard definition as the realized matrix, so the two are
    directly comparable (ratio = realization fraction). (TACA's S=Q*sqrt(C) shows the
    same structure but is BEV-area-normalized to a different scale.)"""
    from tools.taca_rig_sweep import corridor_points
    ext = {r["sensor_label"]: r for r in json.load(open(extr_path))}
    radars = [Radar(l, ext[l]["x"], ext[l]["y"], ext[l]["yaw"]) for l in labels if l in ext]
    pts = corridor_points(radars)
    rx = np.array([r.x for r in radars]); ry = np.array([r.y for r in radars])
    rphi = np.radians([r.phi_deg for r in radars]); M = len(radars)
    cover = np.zeros((len(pts), M), bool)
    for k in range(M):
        dx = pts[:, 0] - rx[k]; dy = pts[:, 1] - ry[k]
        off = np.abs((np.degrees(np.arctan2(dy, dx) - rphi[k]) + 180) % 360 - 180)
        cover[:, k] = (np.hypot(dx, dy) <= R_max) & (off <= fov_half)
    J = np.eye(M)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            either = (cover[:, i] | cover[:, j]).sum()
            J[i, j] = (cover[:, i] & cover[:, j]).sum() / either if either else 0.0
    return J


def _natkey(s):
    m = re.match(r"([A-Za-z]+)(\d+)", s)
    return (m.group(1), int(m.group(2))) if m else (s, 0)


def load_frame_actor_radars(path):
    """(frame, actor_id) -> set(sensor_label) for matched returns; + sorted labels."""
    fa = defaultdict(set)
    labels = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            a = row["matched_actor_id"]
            if not a:
                continue
            lab = row["sensor_label"]
            labels.add(lab)
            fa[(int(row["frame"]), a)] = fa[(int(row["frame"]), a)] | {lab}
    return fa, sorted(labels, key=_natkey)


def realized_overlap(fa, labels, W, minframe):
    idx = {l: i for i, l in enumerate(labels)}
    M = len(labels)
    # accumulate W frames per radar: (window, actor) -> union of radars seen across W frames
    wa = defaultdict(set)
    for (fr, a), rad in fa.items():
        wa[((fr - minframe) // W, a)] |= rad
    hist = defaultdict(int)
    co = np.zeros((M, M)); seen = np.zeros(M)
    for rad in wa.values():
        ridx = [idx[r] for r in rad]
        hist[len(ridx)] += 1
        for i in ridx:
            seen[i] += 1
            for j in ridx:
                co[i, j] += 1
    total = len(wa)
    R = np.eye(M)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            u = seen[i] + seen[j] - co[i, j]
            R[i, j] = co[i, j] / u if u > 0 else 0.0
    return R, hist, total, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--windows", default="1,5,10,20")
    ap.add_argument("--target-mass", type=float, default=0.9)
    ap.add_argument("--extrinsics", default=None,
                    help="radar_extrinsics.json for geometric side-by-side (default: capture dir)")
    ap.add_argument("--rmax", type=float, default=35.0, help="geometric S range (capture used 35)")
    ap.add_argument("--fov-half", type=float, default=60.0)
    ap.add_argument("--theta3db", type=float, default=15.0)
    args = ap.parse_args()
    windows = [int(x) for x in args.windows.split(",") if x.strip()]

    print(f"loading {args.path} ...", flush=True)
    fa, labels = load_frame_actor_radars(args.path)
    minframe = min(fr for fr, _ in fa.keys())
    M = len(labels)
    print(f"  {len(fa):,} (frame,actor) instances over {M} radars: {labels}\n")

    print("=== REALIZED co-visibility vs accumulation window W (frames/radar) ===")
    print(f"  {'W':>3} {'instances':>10} {'>=2 radars':>11} {'>=3':>7} {'>=4':>7} {'mean':>6}")
    for W in windows:
        _, hist, total, _ = realized_overlap(fa, labels, W, minframe)
        f2 = sum(v for n, v in hist.items() if n >= 2) / total
        f3 = sum(v for n, v in hist.items() if n >= 3) / total
        f4 = sum(v for n, v in hist.items() if n >= 4) / total
        mean = sum(n * v for n, v in hist.items()) / total
        print(f"  {W:>3} {total:>10,} {100*f2:>10.1f}% {100*f3:>6.1f}% {100*f4:>6.1f}% {mean:>6.2f}")

    for W in (windows[0], windows[-1]):
        R, hist, total, seen = realized_overlap(fa, labels, W, minframe)
        print(f"\n=== REALIZED overlap matrix (Jaccard), W={W} ===")
        print("      " + " ".join(f"{l:>5}" for l in labels))
        for i, l in enumerate(labels):
            print(f"  {l:>4} " + " ".join(f"{R[i, j]:5.2f}" for j in range(M)))
        order, oids = mst_topology_ordering(R, labels)
        K, _ = select_implicit_K(R, order, target_mass=args.target_mass, include_self=False)
        mask = build_windowed_attention_mask(R, order, K)
        print(f"  realized ordering: {'→'.join(oids)}   K*={K}   "
              f"mask density={mask.sum()/(M*M):.2f}")

    # ---- geometric vs realized side-by-side (same rig) ----
    extr = args.extrinsics or str(Path(args.path).resolve().parent / "radar_extrinsics.json")
    if Path(extr).is_file():
        Sgeo = geometric_overlap(extr, labels, args.rmax, args.fov_half, args.theta3db)
        R1, *_ = realized_overlap(fa, labels, 1, minframe)
        R10, *_ = realized_overlap(fa, labels, 10, minframe)

        def grid(name, Mx):
            print(f"\n  {name}")
            print("        " + " ".join(f"{l:>5}" for l in labels))
            for i, l in enumerate(labels):
                print(f"    {l:>4} " + " ".join(f"{Mx[i, j]:5.2f}" for j in range(M)))

        print("\n" + "=" * 72)
        print(f"GEOMETRIC (co-coverage Jaccard of road cells, R={args.rmax:.0f}) vs "
              f"REALIZED (Jaccard of co-observed targets)")
        print("=" * 72)
        grid("geometric Jaccard (FoVs overlap):", Sgeo)
        grid("realized (W=1):", R1)
        grid("realized (W=10):", R10)
        print("\n  pairwise — geometry predicts overlap; how much is actually realized?")
        print(f"  {'pair':>7} {'geom_S':>7} {'realW1':>7} {'realW10':>8} {'realized/geom(W10)':>19}")
        rows = []
        for i in range(M):
            for j in range(i + 1, M):
                if max(Sgeo[i, j], R1[i, j], R10[i, j]) > 0.01:
                    ratio = R10[i, j] / Sgeo[i, j] if Sgeo[i, j] > 1e-6 else float("nan")
                    rows.append((Sgeo[i, j], f"{labels[i]}-{labels[j]}", R1[i, j], R10[i, j], ratio))
        for s, pair, r1, r10, ratio in sorted(rows, reverse=True):
            print(f"  {pair:>7} {s:>7.2f} {r1:>7.2f} {r10:>8.2f} {ratio:>18.2f}x")
    else:
        print(f"\n(no extrinsics at {extr}; skipping geometric side-by-side)")


if __name__ == "__main__":
    main()
