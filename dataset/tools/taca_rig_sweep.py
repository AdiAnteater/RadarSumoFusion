"""Sweep the TACA overlap/banding analysis across the 4/8/12/14 radar rigs.

Drives tools/taca_topology_mask.compute_windowed_taca_mask from the saved
config/radar_layout_extrinsics_<map>.json layouts (exact map-computed yaws, no
CARLA needed). Reports, per rig:

  #1 Banding:   K*, K*/M, off-band S-mass discarded, mask density, and the
                global O(M^2) vs banded O(M*(2K*+1)) attention-entry compression.
  #2 Co-design: distance-vs-S agreement (Spearman, per-node top-neighbour Jaccard).
                If distance ~ S, a naive distance window ties TACA and the physics
                co-design is unprovable on that rig.

Usage:
    python -m tools.taca_rig_sweep [--map Town10HD_Opt] [--rmax 35] \
        [--fov-half 60] [--theta3db 15] [--target-mass 0.9]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.taca_topology_mask import Radar, compute_windowed_taca_mask


def _spearman(a, b):
    """Spearman rank correlation (scipy-free)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _wrap_rad(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def corridor_points(radars, res=2.0, margin=2.0):
    """Grid of candidate target locations over the road band between the radar rows."""
    xs = np.array([r.x for r in radars]); ys = np.array([r.y for r in radars])
    x = np.arange(xs.min(), xs.max() + res, res)
    y0, y1 = ys.min() + margin, ys.max() - margin
    y = np.arange(y0, y1 + res, res) if y1 > y0 else np.array([(ys.min() + ys.max()) / 2])
    X, Y = np.meshgrid(x, y)
    return np.column_stack([X.ravel(), Y.ravel()])


def reaim_convergent(radars, focal):
    """Re-aim every radar's boresight at a shared focal point (x, y)."""
    fx, fy = focal
    return [Radar(r.radar_id, r.x, r.y, float(np.degrees(np.arctan2(fy - r.y, fx - r.x))))
            for r in radars]


def coverage_and_ambiguity(radars, R_max, hfov_deg, pts, tau_lo=0.30, tau_hi=0.70):
    """Per candidate target point: how many radars cover it (range+FoV), and whether the
    radial-velocity signal is *ambiguous-but-reconcilable* — i.e. for a given motion
    direction at least one covering radar sees it ~tangentially (radial frac < tau_lo, looks
    static) AND another sees it ~radially (> tau_hi, clearly moving). That is exactly the
    cross-sensor case naive concatenation cannot resolve but a geometry-aware fuser can.
    Motion dirs: along-road (x) for vehicles, across-road (y) for crossers."""
    rx = np.array([r.x for r in radars]); ry = np.array([r.y for r in radars])
    rphi = np.radians([r.phi_deg for r in radars]); half = hfov_deg / 2.0
    covis = np.zeros(len(pts), int)
    info_along = np.zeros(len(pts), bool); info_across = np.zeros(len(pts), bool)
    for i, (px, py) in enumerate(pts):
        dx = px - rx; dy = py - ry; dist = np.hypot(dx, dy)
        off = np.abs(np.degrees(_wrap_rad(np.arctan2(dy, dx) - rphi)))
        cov = (dist <= R_max) & (off <= half)
        nc = int(cov.sum()); covis[i] = nc
        if nc >= 2:
            losx = dx[cov] / dist[cov]; losy = dy[cov] / dist[cov]
            for m, arr in (((1.0, 0.0), info_along), ((0.0, 1.0), info_across)):
                rf = np.abs(losx * m[0] + losy * m[1])
                if (rf < tau_lo).any() and (rf > tau_hi).any():
                    arr[i] = True
    return covis, info_along, info_across


def analyse_rig(radars, R_max, fov_half, theta3db, target_mass):
    xs = np.array([r.x for r in radars]); ys = np.array([r.y for r in radars])
    bev = (xs.min() - R_max, xs.max() + R_max, ys.min() - R_max, ys.max() + R_max)
    res = compute_windowed_taca_mask(
        radars=radars, bev_limits=bev, resolution=1.0,
        R_max=R_max, fov_half_angle_deg=fov_half, theta_3db_deg=theta3db,
        target_mass=target_mass, softmax_beta=1.0,
    )
    M = len(radars)
    K = res["selected_K"]
    S = res["soft_overlap_matrix"]
    order = res["ordered_radar_indices"]
    S_ord = S[np.ix_(order, order)]
    mask = res["windowed_attention_mask"]

    # ---- #1 banding ----
    offdiag = ~np.eye(M, dtype=bool)
    band = (np.abs(np.subtract.outer(np.arange(M), np.arange(M))) <= K) & offdiag
    tot_S = S_ord[offdiag].sum()
    inband_S = S_ord[band].sum()
    offband_frac = float((tot_S - inband_S) / tot_S) if tot_S > 0 else 0.0
    n_pairs_overlap = int((np.triu(S, 1) > 0).sum())
    mask_entries = int(mask.sum())
    global_entries = M * M
    compression = global_entries / mask_entries if mask_entries else float("nan")

    # ---- #2 distance vs S (over i<j pairs) ----
    dists, svals = [], []
    for i in range(M):
        for j in range(i + 1, M):
            d = float(np.hypot(radars[i].x - radars[j].x, radars[i].y - radars[j].y))
            dists.append(d); svals.append(float(S[i, j]))
    dists = np.array(dists); svals = np.array(svals)
    rho_all = _spearman(-dists, svals)
    ov = svals > 0
    rho_ov = _spearman(-dists[ov], svals[ov])

    # per-node: is the strongest-S neighbour also the nearest? + topK Jaccard
    jacc = []
    nearest_eq_strongest = 0
    for i in range(M):
        partners = [j for j in range(M) if j != i]
        by_S = sorted(partners, key=lambda j: S[i, j], reverse=True)
        by_d = sorted(partners, key=lambda j: np.hypot(radars[i].x - radars[j].x,
                                                        radars[i].y - radars[j].y))
        # only consider partners this radar actually overlaps
        ov_partners = [j for j in partners if S[i, j] > 0]
        if not ov_partners:
            continue
        if by_S[0] == by_d[0]:
            nearest_eq_strongest += 1
        kk = min(K if K > 0 else 1, len(ov_partners))
        topS = set(by_S[:kk]); topD = set(by_d[:kk])
        jacc.append(len(topS & topD) / len(topS | topD))
    mean_jacc = float(np.mean(jacc)) if jacc else float("nan")

    # most discordant pairs (close but weak / far but strong), among overlapping
    pair_recs = []
    for i in range(M):
        for j in range(i + 1, M):
            if S[i, j] > 0:
                d = float(np.hypot(radars[i].x - radars[j].x, radars[i].y - radars[j].y))
                pair_recs.append((radars[i].radar_id, radars[j].radar_id, d, float(S[i, j])))
    # rank by distance and by S, flag big rank gaps
    if pair_recs:
        dr = {p: r for r, p in enumerate(sorted(range(len(pair_recs)), key=lambda k: pair_recs[k][2]))}
        sr = {p: r for r, p in enumerate(sorted(range(len(pair_recs)), key=lambda k: -pair_recs[k][3]))}
        discord = sorted(range(len(pair_recs)), key=lambda k: -abs(dr[k] - sr[k]))[:3]
        discord_examples = [(pair_recs[k][0], pair_recs[k][1], round(pair_recs[k][2], 1),
                             round(pair_recs[k][3], 4)) for k in discord]
    else:
        discord_examples = []

    return {
        "M": M, "K": K, "K_over_M": K / M, "offband_frac": offband_frac,
        "n_overlap_pairs": n_pairs_overlap, "max_pairs": M * (M - 1) // 2,
        "mask_entries": mask_entries, "global_entries": global_entries,
        "compression": compression,
        "ordered_ids": res["ordered_radar_ids"],
        "rho_all": rho_all, "rho_overlap": rho_ov,
        "nearest_eq_strongest": nearest_eq_strongest,
        "mean_topK_jaccard": mean_jacc,
        "discord_examples": discord_examples,
    }


def _layout_radars(layouts, k):
    return [Radar(r["sensor_label"], r["x_m"], r["y_m"], r["yaw_deg"]) for r in layouts[k]]


def coverage_section(layouts, hfov, rmax_list, aim="toward-road", focal=None,
                     target_cov3=0.6, target_topo=0.5):
    print("\n" + "=" * 82)
    print("CO-VISIBILITY + RADIAL-VELOCITY AMBIGUITY (geometry only; road-band targets)")
    print("  cov>=k  = % of road points seen by >=k radars (the fusion material)")
    print("  topo-info = % where motion is tangential-to-one / radial-to-another radar")
    print("              -> single-radar Doppler is ambiguous; only geometry-aware cross-")
    print("                 sensor fusion resolves it. THIS is what naive concat can't do.")
    print(f"  aim={aim}   PASS = cov>=3 >= {target_cov3:.0%}  AND  max(topo) >= {target_topo:.0%}")
    print("=" * 82)
    print(f"  {'M':>2} {'R_max':>5}  {'cov>=1':>7} {'cov>=2':>7} {'cov>=3':>7}  "
          f"{'topo(along)':>12} {'topo(across)':>13}  {'PASS':>5}")
    for k in sorted(layouts, key=int):
        radars0 = _layout_radars(layouts, k)
        if aim == "convergent":
            fx = focal[0] if focal else float(np.mean([r.x for r in radars0]))
            fy = focal[1] if focal else float(np.mean([r.y for r in radars0]))
            radars = reaim_convergent(radars0, (fx, fy))
        else:
            radars = radars0
        pts = corridor_points(radars0)   # road band is fixed by the rig footprint, not the aim
        for R in rmax_list:
            covis, ia, ic = coverage_and_ambiguity(radars, R, hfov, pts)
            c3 = np.mean(covis >= 3); topo = max(np.mean(ia), np.mean(ic))
            ok = "PASS" if (c3 >= target_cov3 and topo >= target_topo) else "  -"
            print(f"  {len(radars):>2} {R:>5.0f}  {100*np.mean(covis>=1):>6.1f}% "
                  f"{100*np.mean(covis>=2):>6.1f}% {100*c3:>6.1f}%  "
                  f"{100*np.mean(ia):>11.1f}% {100*np.mean(ic):>12.1f}%  {ok:>5}")

    # Always-on aim sensitivity check (toward-road vs convergent) on the largest rig at R=50.
    big = max(layouts, key=int); radars0 = _layout_radars(layouts, big)
    cx = float(np.mean([r.x for r in radars0])); cy = float(np.mean([r.y for r in radars0]))
    conv = reaim_convergent(radars0, (cx, cy)); pts = corridor_points(radars0)
    print(f"\n  [aim sensitivity, M={len(radars0)} R=50]")
    for tag, rr in (("toward-road", radars0), ("convergent->centroid", conv)):
        covis, ia, ic = coverage_and_ambiguity(rr, 50.0, hfov, pts)
        print(f"    {tag:<22} cov>=3={100*np.mean(covis>=3):5.1f}%  "
              f"topo(across)={100*np.mean(ic):5.1f}%")


def main():
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--rmax", type=float, default=35.0)
    ap.add_argument("--fov-half", type=float, default=60.0)
    ap.add_argument("--theta3db", type=float, default=15.0)
    ap.add_argument("--target-mass", type=float, default=0.9)
    ap.add_argument("--cov-rmax", default="35,50,70",
                    help="comma-separated R_max (m) values for the co-visibility sweep")
    ap.add_argument("--aim", choices=["toward-road", "convergent"], default="toward-road",
                    help="boresight aim for the coverage analysis (convergent backfires; for comparison)")
    ap.add_argument("--focal", default=None, help="convergent focal point 'x,y' (default=rig centroid)")
    ap.add_argument("--target-cov3", type=float, default=0.6, help="PASS threshold on cov>=3 fraction")
    ap.add_argument("--target-topo", type=float, default=0.5, help="PASS threshold on max topo-info fraction")
    args = ap.parse_args()

    cfg = json.load(open(here / "config" / f"radar_layout_extrinsics_{args.map}.json"))
    layouts = cfg["layouts"]
    print(f"map={cfg['map']}  R_max={args.rmax}  fov_half={args.fov_half}  "
          f"theta_3db={args.theta3db}  target_mass={args.target_mass}\n")
    print(f"{'M':>3} {'K*':>3} {'K*/M':>6} {'offbandS':>9} {'overlap_pairs':>14} "
          f"{'maskE/M^2':>11} {'compress':>9} {'rho(-d,S)':>10} {'nn=strong':>10} {'topKjacc':>9}")
    rows = []
    for k in sorted(layouts, key=int):
        L = layouts[k]
        radars = [Radar(r["sensor_label"], r["x_m"], r["y_m"], r["yaw_deg"]) for r in L]
        a = analyse_rig(radars, args.rmax, args.fov_half, args.theta3db, args.target_mass)
        rows.append((k, a))
        print(f"{a['M']:>3} {a['K']:>3} {a['K_over_M']:>6.3f} {a['offband_frac']:>9.3f} "
              f"{a['n_overlap_pairs']:>5}/{a['max_pairs']:<8} "
              f"{a['mask_entries']:>4}/{a['global_entries']:<5} "
              f"{a['compression']:>9.2f} {a['rho_overlap']:>10.3f} "
              f"{a['nearest_eq_strongest']:>3}/{a['M']:<6} {a['mean_topK_jaccard']:>9.3f}")
    print("\nordering + most distance-vs-S discordant overlapping pairs (close-but-weak / far-but-strong):")
    for k, a in rows:
        print(f"  M={a['M']:>2}: order={'→'.join(a['ordered_ids'])}")
        print(f"        discord: {a['discord_examples']}")

    rmax_list = [float(x) for x in args.cov_rmax.split(",") if x.strip()]
    focal = tuple(float(v) for v in args.focal.split(",")) if args.focal else None
    coverage_section(layouts, hfov=args.fov_half * 2, rmax_list=rmax_list,
                     aim=args.aim, focal=focal,
                     target_cov3=args.target_cov3, target_topo=args.target_topo)


if __name__ == "__main__":
    main()
