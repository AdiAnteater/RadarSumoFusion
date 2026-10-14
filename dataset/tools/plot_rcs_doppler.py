"""RCS and Doppler distribution report for a post-processed radar capture.

Streams radar_data_labeled.csv in chunks (works on 5+ GB campaign captures with
flat memory use), accumulates EXACT histograms per return group and writes
plots + tables into <capture>/radar_distributions/.

Groups
  dynamic   one group per matched_actor_class (car, truck, bus, pedestrian, ...)
  static    every return with no matched actor; also split by return_class
            (road / structure / unassigned / ...)

What it reports
  RCS      per-group median / mean / std / p5 / p95 of rcs_dBsm, and the gap to
           the RadarScenes per-class median targets the post-processor
           calibrates to. Static RCS is overlaid with the RadarScenes static
           reference distribution.
  Doppler  per-group histogram of velocity_mps (synthesized radial velocity,
           positive = receding) with the raw CARLA value (velocity_raw_mps)
           overlaid, % of returns with |v| > 0.1 m/s, % beyond v_max.
  RCS vs range   median and IQR of rcs_dBsm per 2 m range bin (RCS should be
           range independent; the range dependence lives in SNR).
  Static vs dynamic separability   overlap coefficient (OVL) and AUC for RCS
           and |Doppler|. RadarScenes reference for RCS: OVL 0.86, AUC 0.55
           (RCS should NOT separate static from dynamic; Doppler should).
  Per segment (campaign captures)   per segment / scenario RCS median and
           Doppler stats.

Usage (from the dataset folder):
  python tools/plot_rcs_doppler.py Data/sensor_capture_20260929_210312
  python tools/plot_rcs_doppler.py Data/sensor_capture_A Data/sensor_capture_B
  python tools/plot_rcs_doppler.py Data/sensor_capture_X --visible-only
  python tools/plot_rcs_doppler.py Data/sensor_capture_X --max-rows 2000000   (quick look)

Options
  --visible-only        only returns that pass the FMCW detection model (visible == 1)
  --doppler-col COL     velocity column to analyse (default velocity_mps; or
                        velocity_mps_noisy for the measurement-noise version)
  --out DIR             output folder (default <capture>/radar_distributions)
  --max-rows N          stop after N rows (quick preview)
  --chunksize N         rows per chunk (default 1,000,000)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")

# Must match PostProcessDataset.py
TARGET_MEDIAN_DBSM = {
    "pedestrian": -7.9, "car": -4.9, "truck": 1.3, "bus": -2.5,
    "bicycle": -11.2, "motorcycle": -9.0,
}
RADARSCENES_STATIC_RCS_CDF = (
    (0.000, -30.60), (0.001, -29.50), (0.010, -26.53), (0.050, -23.33),
    (0.250, -15.18), (0.500, -7.38), (0.750, 1.57), (0.950, 16.16),
    (0.990, 26.55), (0.999, 37.40), (1.000, 58.71),
)
RADARSCENES_RCS_OVL = 0.86
RADARSCENES_RCS_AUC = 0.55
V_MAX_MPS = 50.0
MOVING_THRESH_MPS = 0.1

RCS_EDGES = np.arange(-40.0, 70.0 + 0.25, 0.5)
V_EDGES = np.arange(-55.0, 55.0 + 0.05, 0.1)
ABSV_EDGES = np.arange(0.0, 55.0 + 0.05, 0.05)
RANGE_EDGES = np.arange(0.0, 62.0 + 1e-9, 2.0)

USECOLS = ["matched_actor_class", "return_class", "rcs_dBsm", "velocity_raw_mps",
           "depth_m", "visible", "segment_id", "scenario_id"]


def _hist(values, edges):
    """Counts with out-of-range values clipped into the first / last bin."""
    if values.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.int64)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1).astype(np.int64)


class GroupStats:
    def __init__(self):
        self.n = 0
        self.n_rcs = 0
        self.rcs_sum = 0.0
        self.rcs_sq = 0.0
        self.rcs_h = np.zeros(len(RCS_EDGES) - 1, np.int64)
        self.v_h = np.zeros(len(V_EDGES) - 1, np.int64)
        self.vraw_h = np.zeros(len(V_EDGES) - 1, np.int64)
        self.absv_h = np.zeros(len(ABSV_EDGES) - 1, np.int64)
        self.rr_h = np.zeros((len(RANGE_EDGES) - 1, len(RCS_EDGES) - 1), np.int64)
        self.n_v = 0
        self.v_abs_sum = 0.0
        self.moving = 0
        self.raw_moving = 0
        self.n_raw = 0
        self.over_vmax = 0

    def add(self, rcs, v, vraw, rng):
        self.n += len(v)
        ok = np.isfinite(rcs)
        r = rcs[ok]
        self.n_rcs += r.size
        self.rcs_sum += float(r.sum())
        self.rcs_sq += float((r * r).sum())
        self.rcs_h += _hist(r, RCS_EDGES)
        rg = rng[ok]
        gok = np.isfinite(rg)
        if gok.any():
            ri = np.clip(np.searchsorted(RANGE_EDGES, rg[gok], side="right") - 1,
                         0, len(RANGE_EDGES) - 2)
            ci = np.clip(np.searchsorted(RCS_EDGES, r[gok], side="right") - 1,
                         0, len(RCS_EDGES) - 2)
            nc = len(RCS_EDGES) - 1
            flat = np.bincount(ri * nc + ci, minlength=self.rr_h.size)
            self.rr_h += flat.reshape(self.rr_h.shape)
        vv = v[np.isfinite(v)]
        self.n_v += vv.size
        av = np.abs(vv)
        self.v_abs_sum += float(av.sum())
        self.moving += int((av > MOVING_THRESH_MPS).sum())
        self.over_vmax += int((av > V_MAX_MPS).sum())
        self.v_h += _hist(vv, V_EDGES)
        self.absv_h += _hist(av, ABSV_EDGES)
        vr = vraw[np.isfinite(vraw)]
        self.n_raw += vr.size
        self.raw_moving += int((np.abs(vr) > MOVING_THRESH_MPS).sum())
        self.vraw_h += _hist(vr, V_EDGES)

    def merge(self, other: "GroupStats"):
        for k, val in other.__dict__.items():
            setattr(self, k, getattr(self, k) + val)


def _quantile(hist, edges, q):
    tot = hist.sum()
    if tot == 0:
        return float("nan")
    c = np.cumsum(hist)
    i = int(np.searchsorted(c, q * tot))
    i = min(i, len(hist) - 1)
    prev = c[i - 1] if i > 0 else 0
    frac = (q * tot - prev) / max(hist[i], 1)
    return float(edges[i] + frac * (edges[i + 1] - edges[i]))


def _ovl_auc(h_dyn, h_sta):
    """Overlap coefficient and AUC = P(dyn > static) from two histograms on the same bins."""
    if h_dyn.sum() == 0 or h_sta.sum() == 0:
        return float("nan"), float("nan")
    p = h_dyn / h_dyn.sum()
    q = h_sta / h_sta.sum()
    ovl = float(np.minimum(p, q).sum())
    cq = np.cumsum(q) - q          # P(static < bin)
    auc = float((p * (cq + 0.5 * q)).sum())
    return ovl, auc


def _static_ref_pdf():
    """RadarScenes static RCS pdf on RCS_EDGES (piecewise-uniform between knots)."""
    probs = np.array([k[0] for k in RADARSCENES_STATIC_RCS_CDF])
    vals = np.array([k[1] for k in RADARSCENES_STATIC_RCS_CDF])
    cdf_at_edges = np.interp(RCS_EDGES, vals, probs, left=0.0, right=1.0)
    mass = np.diff(cdf_at_edges)
    return mass / np.diff(RCS_EDGES)


def analyse(capture: Path, args) -> None:
    csv_path = capture / "radar_data_labeled.csv" if capture.is_dir() else capture
    if capture.is_file():
        capture = capture.parent
    if not csv_path.is_file():
        print(f"[skip] {csv_path} not found", flush=True)
        return
    out = Path(args.out) if args.out else capture / "radar_distributions"
    if args.visible_only and not args.out:
        out = capture / "radar_distributions_visible"
    out.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0).columns
    need = {"rcs_dBsm", args.doppler_col}
    missing = need - set(header)
    if missing:
        print(f"[skip] {csv_path.name} has no {sorted(missing)} - run PostProcessDataset first",
              flush=True)
        return
    cols = [c for c in dict.fromkeys(USECOLS + [args.doppler_col]) if c in header]
    has_seg = "segment_id" in header

    groups: dict[str, GroupStats] = {}
    segs: dict[tuple, dict] = {}
    t0 = time.time()
    rows = 0
    dtypes = {c: str for c in ("matched_actor_class", "return_class", "segment_id", "scenario_id")}
    reader = pd.read_csv(csv_path, usecols=cols, dtype=dtypes, chunksize=args.chunksize,
                         keep_default_na=False, na_values={"rcs_dBsm": [""],
                                                           args.doppler_col: [""]},
                         low_memory=False)
    print(f"[dist] {csv_path}", flush=True)
    for chunk in reader:
        if args.max_rows and rows >= args.max_rows:
            break
        if args.max_rows and rows + len(chunk) > args.max_rows:
            chunk = chunk.iloc[: args.max_rows - rows]
        rows += len(chunk)
        if args.visible_only and "visible" in chunk:
            chunk = chunk[pd.to_numeric(chunk["visible"], errors="coerce") == 1]
        klass = chunk["matched_actor_class"].str.strip()
        rc = chunk["return_class"].str.strip() if "return_class" in chunk else pd.Series(
            "static", index=chunk.index)
        rcs = pd.to_numeric(chunk["rcs_dBsm"], errors="coerce").to_numpy(float)
        v = pd.to_numeric(chunk[args.doppler_col], errors="coerce").to_numpy(float)
        vraw = (pd.to_numeric(chunk["velocity_raw_mps"], errors="coerce").to_numpy(float)
                if "velocity_raw_mps" in chunk else np.full(len(chunk), np.nan))
        rng = pd.to_numeric(chunk["depth_m"], errors="coerce").to_numpy(float)
        is_dyn = (klass != "").to_numpy()
        gkey = np.where(is_dyn, klass.to_numpy(object),
                        ("static:" + rc.replace("", "unlabeled")).to_numpy(object))
        for g in pd.unique(gkey):
            m = gkey == g
            groups.setdefault(g, GroupStats()).add(rcs[m], v[m], vraw[m], rng[m])
        if has_seg:
            sid = chunk["segment_id"].to_numpy(object)
            scen = chunk["scenario_id"].to_numpy(object) if "scenario_id" in chunk else sid
            df = pd.DataFrame({"sid": sid, "scen": scen, "dyn": is_dyn,
                               "absv": np.abs(v), "mv": np.abs(v) > MOVING_THRESH_MPS,
                               "rcs": rcs})
            for (s, sc, d), sub in df.groupby(["sid", "scen", "dyn"], sort=False):
                e = segs.setdefault((s, sc), {"dyn": [0, 0.0, 0, np.zeros(len(RCS_EDGES) - 1, np.int64)],
                                              "sta": [0, 0.0, 0, np.zeros(len(RCS_EDGES) - 1, np.int64)]})
                b = e["dyn" if d else "sta"]
                av = sub["absv"].to_numpy()
                av = av[np.isfinite(av)]
                b[0] += av.size
                b[1] += float(av.sum())
                b[2] += int(sub["mv"].sum())
                r = sub["rcs"].to_numpy()
                b[3] += _hist(r[np.isfinite(r)], RCS_EDGES)
        el = time.time() - t0
        print(f"  {rows:>12,} rows  {rows / max(el, 1e-6):>9,.0f} rows/s", flush=True)

    if not groups:
        print("[dist] no rows", flush=True)
        return

    dyn_names = sorted(g for g in groups if not g.startswith("static:"))
    sta_names = sorted(g for g in groups if g.startswith("static:"))
    static_all = GroupStats()
    for g in sta_names:
        static_all.merge(groups[g])
    dyn_all = GroupStats()
    for g in dyn_names:
        dyn_all.merge(groups[g])
    ordered = [(g, groups[g]) for g in dyn_names] + [("dynamic (all)", dyn_all)] + \
              [(g, groups[g]) for g in sta_names] + [("static (all)", static_all)]

    # Tables
    recs = []
    for name, st in ordered:
        tgt = TARGET_MEDIAN_DBSM.get(name)
        if name == "static (all)":
            tgt = RADARSCENES_STATIC_RCS_CDF[5][1]
        med = _quantile(st.rcs_h, RCS_EDGES, 0.5)
        mean = st.rcs_sum / st.n_rcs if st.n_rcs else float("nan")
        var = st.rcs_sq / st.n_rcs - mean * mean if st.n_rcs else float("nan")
        recs.append({
            "group": name,
            "returns": st.n,
            "rcs_n": st.n_rcs,
            "rcs_median_dBsm": med,
            "rcs_target_median_dBsm": tgt if tgt is not None else float("nan"),
            "rcs_median_minus_target_dB": (med - tgt) if tgt is not None else float("nan"),
            "rcs_mean_dBsm": mean,
            "rcs_std_dB": math.sqrt(max(var, 0.0)) if st.n_rcs else float("nan"),
            "rcs_p5_dBsm": _quantile(st.rcs_h, RCS_EDGES, 0.05),
            "rcs_p95_dBsm": _quantile(st.rcs_h, RCS_EDGES, 0.95),
            "doppler_mean_abs_mps": st.v_abs_sum / st.n_v if st.n_v else float("nan"),
            "doppler_p50_abs_mps": _quantile(st.absv_h, ABSV_EDGES, 0.5),
            "doppler_p95_abs_mps": _quantile(st.absv_h, ABSV_EDGES, 0.95),
            "doppler_moving_pct": 100.0 * st.moving / st.n_v if st.n_v else float("nan"),
            "raw_carla_moving_pct": 100.0 * st.raw_moving / st.n_raw if st.n_raw else float("nan"),
            "doppler_over_vmax_pct": 100.0 * st.over_vmax / st.n_v if st.n_v else float("nan"),
        })
    summary = pd.DataFrame(recs)
    summary.to_csv(out / "rcs_doppler_summary.csv", index=False, float_format="%.3f")

    rcs_ovl, rcs_auc = _ovl_auc(dyn_all.rcs_h, static_all.rcs_h)
    v_ovl, v_auc = _ovl_auc(dyn_all.absv_h, static_all.absv_h)

    seg_df = None
    if segs:
        srec = []
        for (s, sc), e in sorted(segs.items(), key=lambda kv: (str(kv[0][0]).zfill(6), kv[0][1])):
            d, a = e["dyn"], e["sta"]
            srec.append({
                "segment_id": s if s != "" else "(none)", "scenario_id": sc,
                "dynamic_returns": d[0], "static_returns": a[0],
                "dyn_rcs_median_dBsm": _quantile(d[3], RCS_EDGES, 0.5),
                "static_rcs_median_dBsm": _quantile(a[3], RCS_EDGES, 0.5),
                "dyn_mean_abs_doppler_mps": d[1] / d[0] if d[0] else float("nan"),
                "dyn_moving_pct": 100.0 * d[2] / d[0] if d[0] else float("nan"),
                "static_mean_abs_doppler_mps": a[1] / a[0] if a[0] else float("nan"),
            })
        seg_df = pd.DataFrame(srec)
        seg_df.to_csv(out / "rcs_doppler_by_segment.csv", index=False, float_format="%.3f")

    _write_md(out, csv_path, rows, args, summary, seg_df, rcs_ovl, rcs_auc, v_ovl, v_auc)
    _plots(out, ordered, dyn_names, groups, static_all, dyn_all, args, seg_df)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print(f"\n  static vs dynamic  RCS: OVL {rcs_ovl:.2f} AUC {rcs_auc:.2f}  "
          f"(RadarScenes {RADARSCENES_RCS_OVL:.2f} / {RADARSCENES_RCS_AUC:.2f})"
          f"   |Doppler|: OVL {v_ovl:.2f} AUC {v_auc:.2f}", flush=True)
    print(f"[dist] wrote {out}  ({time.time() - t0:.0f} s)", flush=True)


def _write_md(out, csv_path, rows, args, summary, seg_df, rcs_ovl, rcs_auc, v_ovl, v_auc):
    def fmt(x):
        return "" if isinstance(x, float) and math.isnan(x) else (
            f"{x:,.2f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, (int, np.integer)) else str(x))
    lines = [f"# RCS and Doppler distributions", "",
             f"Source: `{csv_path}`  ", f"Rows read: {rows:,}"
             + ("  (visible == 1 only)" if args.visible_only else "")
             + f"  Doppler column: `{args.doppler_col}` (positive = receding)", "",
             "## Per group", ""]
    cols = ["group", "returns", "rcs_median_dBsm", "rcs_target_median_dBsm",
            "rcs_median_minus_target_dB", "rcs_std_dB", "rcs_p5_dBsm", "rcs_p95_dBsm",
            "doppler_mean_abs_mps", "doppler_p95_abs_mps", "doppler_moving_pct",
            "raw_carla_moving_pct", "doppler_over_vmax_pct"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines += ["", "## Static vs dynamic separability", "",
              "| feature | OVL | AUC | RadarScenes reference |", "|---|---|---|---|",
              f"| RCS (dBsm) | {rcs_ovl:.2f} | {rcs_auc:.2f} | OVL {RADARSCENES_RCS_OVL:.2f}, AUC {RADARSCENES_RCS_AUC:.2f} |",
              f"| abs Doppler (m/s) | {v_ovl:.2f} | {v_auc:.2f} | separator (static near 0) |",
              "", "OVL = shared probability mass (1 = identical, 0 = disjoint). "
              "AUC = P(dynamic value > static value) (0.5 = no separation).", ""]
    if seg_df is not None:
        lines += ["## Per segment", "", "| " + " | ".join(seg_df.columns) + " |",
                  "|" + "---|" * len(seg_df.columns)]
        for _, r in seg_df.iterrows():
            lines.append("| " + " | ".join(fmt(r[c]) for c in seg_df.columns) + " |")
    (out / "rcs_doppler_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(out, ordered, dyn_names, groups, static_all, dyn_all, args, seg_df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[dist] matplotlib not installed - tables only", flush=True)
        return
    rc_mid = 0.5 * (RCS_EDGES[:-1] + RCS_EDGES[1:])
    v_mid = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])
    rc_w = np.diff(RCS_EDGES)
    panels = [(g, s) for g, s in ordered if s.n_rcs > 0 and g != "dynamic (all)"]

    def save(fig, name):
        fig.savefig(out / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out / name}", flush=True)

    # Fig 1: RCS per group
    ncol = min(4, len(panels))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    ref = _static_ref_pdf()
    for ax, (g, st) in zip(axes.flat, panels):
        dens = st.rcs_h / max(st.rcs_h.sum(), 1) / rc_w
        ax.bar(rc_mid, dens, width=rc_w, color="tab:blue" if not g.startswith("static")
               else "tab:gray", alpha=0.75)
        med = _quantile(st.rcs_h, RCS_EDGES, 0.5)
        ax.axvline(med, color="k", lw=1.2, label=f"median {med:.1f}")
        tgt = TARGET_MEDIAN_DBSM.get(g)
        if tgt is not None:
            ax.axvline(tgt, color="tab:red", ls="--", lw=1.2, label=f"target {tgt:.1f}")
        if g.startswith("static"):
            ax.plot(rc_mid, ref, color="tab:red", lw=1.2, label="RadarScenes static")
        ax.set_title(f"{g}  (n={st.n_rcs:,})", fontsize=9)
        ax.set_xlabel("RCS (dBsm)")
        ax.set_xlim(-40, 60)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("RCS distribution per return group", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_rcs_by_group.png")

    # Fig 2: RCS overlay dynamic classes vs static
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g in dyn_names:
        st = groups[g]
        if st.n_rcs:
            ax.step(rc_mid, st.rcs_h / st.rcs_h.sum() / rc_w, where="mid", label=g)
    if static_all.n_rcs:
        ax.step(rc_mid, static_all.rcs_h / static_all.rcs_h.sum() / rc_w, where="mid",
                color="k", lw=2, label="static (all)")
    ax.plot(rc_mid, ref, color="k", ls=":", label="RadarScenes static")
    ax.set_xlim(-40, 60)
    ax.set_xlabel("RCS (dBsm)")
    ax.set_ylabel("density")
    ax.set_title("RCS: dynamic classes vs static")
    ax.legend(fontsize=8)
    save(fig, "fig2_rcs_overlay.png")

    # Fig 3: Doppler per group (log y), raw CARLA overlay
    vpanels = [(g, s) for g, s in ordered if s.n_v > 0 and g != "dynamic (all)"]
    ncol = min(4, len(vpanels))
    nrow = math.ceil(len(vpanels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, (g, st) in zip(axes.flat, vpanels):
        ax.step(v_mid, np.maximum(st.v_h, 0.5), where="mid", color="tab:blue",
                label=args.doppler_col)
        if st.vraw_h.sum():
            ax.step(v_mid, np.maximum(st.vraw_h, 0.5), where="mid", color="tab:orange",
                    ls="--", lw=1, label="raw CARLA")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
        lim = max(2.0, min(55.0, 1.3 * _quantile(st.absv_h, ABSV_EDGES, 0.999)))
        ax.set_xlim(-lim, lim)
        mv = 100.0 * st.moving / st.n_v
        ax.set_title(f"{g}  |v|>{MOVING_THRESH_MPS:g}: {mv:.1f}%", fontsize=9)
        ax.set_xlabel("radial velocity (m/s, + = receding)")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(vpanels):]:
        ax.axis("off")
    fig.suptitle("Doppler distribution per return group (log counts)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_doppler_by_group.png")

    # Fig 4: RCS vs range (median and IQR per range bin)
    r_mid = 0.5 * (RANGE_EDGES[:-1] + RANGE_EDGES[1:])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g, st in [(g, groups[g]) for g in dyn_names] + [("static (all)", static_all)]:
        med, lo, hi = [], [], []
        for i in range(len(r_mid)):
            row = st.rr_h[i]
            if row.sum() < 50:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            med.append(_quantile(row, RCS_EDGES, 0.5))
            lo.append(_quantile(row, RCS_EDGES, 0.25))
            hi.append(_quantile(row, RCS_EDGES, 0.75))
        line, = ax.plot(r_mid, med, marker="o", ms=3, label=g,
                        color="k" if g.startswith("static") else None)
        ax.fill_between(r_mid, lo, hi, alpha=0.12, color=line.get_color())
    ax.set_xlabel("range (m)")
    ax.set_ylabel("RCS (dBsm), median and IQR")
    ax.set_title("RCS vs range (should be flat)")
    ax.legend(fontsize=8)
    save(fig, "fig4_rcs_vs_range.png")

    # Fig 5: |Doppler| static vs dynamic CDF
    fig, ax = plt.subplots(figsize=(7, 4))
    a_mid = ABSV_EDGES[1:]
    for name, st, col in [("dynamic (all)", dyn_all, "tab:blue"), ("static (all)", static_all, "k")]:
        if st.absv_h.sum():
            ax.plot(a_mid, np.cumsum(st.absv_h) / st.absv_h.sum(), color=col, label=name)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("|radial velocity| (m/s)")
    ax.set_ylabel("CDF")
    ax.set_title("|Doppler| CDF: static vs dynamic")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "fig5_doppler_cdf_static_vs_dynamic.png")

    # Fig 6: per segment
    if seg_df is not None and len(seg_df) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        lab = [f"{a}:{b}" for a, b in zip(seg_df["segment_id"], seg_df["scenario_id"])]
        x = np.arange(len(lab))
        axes[0].bar(x, seg_df["dyn_mean_abs_doppler_mps"], color="tab:blue")
        axes[0].set_ylabel("dynamic mean |v| (m/s)")
        axes[1].bar(x - 0.2, seg_df["dyn_rcs_median_dBsm"], 0.4, label="dynamic")
        axes[1].bar(x + 0.2, seg_df["static_rcs_median_dBsm"], 0.4, label="static", color="gray")
        axes[1].set_ylabel("RCS median (dBsm)")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(lab, rotation=60, fontsize=7)
            ax.set_xlabel("segment:scenario")
        fig.suptitle("Per segment", fontweight="bold")
        fig.tight_layout()
        save(fig, "fig6_by_segment.png")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("captures", nargs="+", type=Path,
                   help="capture folder(s) or radar_data_labeled.csv path(s)")
    p.add_argument("--visible-only", action="store_true")
    p.add_argument("--doppler-col", default="velocity_mps")
    p.add_argument("--out", default=None)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=1_000_000)
    args = p.parse_args()
    if args.out and len(args.captures) > 1:
        sys.exit("--out only works with a single capture")
    for c in args.captures:
        analyse(c, args)


if __name__ == "__main__":
    main()
