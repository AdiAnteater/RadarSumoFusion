"""
Visualize radar realism post-processing: before vs after.

Reads a real processed capture CSV  OR  generates synthetic demo data.

Usage:
    # Real data (after running PostProcessDataset):
    python tools/visualize_radar_realism.py --input Data/radar_data_processed.csv

    # Synthetic demo (no CARLA / no capture needed):
    python tools/visualize_radar_realism.py --synthetic
    python tools/visualize_radar_realism.py --synthetic --save plots/

    # Print data snippets to console only (no plots):
    python tools/visualize_radar_realism.py --synthetic --snippets-only
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# FMCW / RCS constants (must match PostProcessDataset defaults)
# ---------------------------------------------------------------------------
_C = 3e8

# RS35 FMCW preset
_F_C    = 76e9
_B      = 400e6
_T_C    = 19.74e-6
_N_S    = 256
_N_C    = 1024
_PT_DBM = 14.0
_G_DBI  = 15.0
_NF_DB  = 12.0
_L_DB   = 3.0
_SNR_THRESHOLD_DB = 10.0
_P_DETECT = 0.9

_LAM     = _C / _F_C
_T_FRAME = _N_C * _T_C
_DR      = _C / (2 * _B)
_R_MAX   = _N_S * _C / (4 * _B)
_V_MAX   = _LAM / (4 * _T_C)
_DV      = _LAM / (2 * _N_C * _T_C)
_PATH_BASE_DB = (
    (_PT_DBM - 30.0)
    + 2.0 * _G_DBI
    + 20.0 * math.log10(_LAM)
    + 174.0
    + 10.0 * math.log10(_T_FRAME)
    - 32.97
    - _NF_DB
    - _L_DB
)

# RCS calibration targets (dBsm)
_RCS_MEDIAN = {"car": 7.0, "truck": 15.0, "pedestrian": -11.0, "cyclist": -5.0}

# Swerling-1
_SWERLING_SIGMA = math.pi / math.sqrt(6.0) * 10.0 / math.log(10.0)
_RCS_NOISE = {
    "car":        (2.5, 3.5),
    "truck":      (2.0, 3.0),
    "pedestrian": (3.0, 4.0),
    "cyclist":    (2.5, 3.0),
}
_SPIKE_PROB = {"car": 0.05, "truck": 0.04, "pedestrian": 0.02, "cyclist": 0.03}
_ASPECT_AMP = {"car": 8.0, "truck": 10.0, "pedestrian": 4.0, "cyclist": 5.0}

# Azimuth noise
_AZ_SIGMA_0_RAD   = math.radians(6.0)
_AZ_SIGMA_FLOOR   = math.radians(0.3)


# ---------------------------------------------------------------------------
# Core math (mirrors PostProcessDataset helpers)
# ---------------------------------------------------------------------------
def _snr_db(rcs_dbsm: float, range_m: float) -> float:
    return _PATH_BASE_DB + rcs_dbsm - 40.0 * math.log10(max(range_m, 0.01))


def _swerling1(rng: random.Random, sigma: float) -> float:
    U = max(rng.random(), 1e-10)
    return (sigma / _SWERLING_SIGMA) * 10.0 * math.log10(-math.log(U) / math.log(2.0))


def _aspect_delta(heading_rad: float, los_rad: float, amp: float) -> float:
    alpha = heading_rad - los_rad
    sin2  = math.sin(alpha) ** 2
    return amp * (sin2 - 0.5)


def _noise_sigmas(snr_db_val: float) -> tuple:
    snr_lin = max(10.0 ** (snr_db_val / 10.0), 0.01)
    sqrt_2snr = math.sqrt(2.0 * snr_lin)
    sqrt_snr  = math.sqrt(snr_lin)
    return (
        _DR / sqrt_2snr,
        _DV / sqrt_2snr,
        max(_AZ_SIGMA_0_RAD / sqrt_snr, _AZ_SIGMA_FLOOR),
    )


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
CLASS_OBB_AREA_M2 = {          # typical projected OBB silhouette area
    "car":        8.5,
    "truck":      22.0,
    "pedestrian": 0.30,
    "cyclist":    0.55,
}
CLASS_COUNT = {"car": 600, "truck": 200, "pedestrian": 500, "cyclist": 150}
CLASS_COLORS = {"car": "#4C72B0", "truck": "#DD8452",
                "pedestrian": "#55A868", "cyclist": "#C44E52"}
CLASS_MARKERS = {"car": "o", "truck": "s", "pedestrian": "^", "cyclist": "D"}


def generate_synthetic(seed: int = 42) -> list[dict]:
    """Generate a synthetic radar_data_labeled-like dataset."""
    rng = random.Random(seed)
    rows = []
    actor_id = 1

    for klass, n in CLASS_COUNT.items():
        obb_area = CLASS_OBB_AREA_M2[klass]
        median_dbsm = _RCS_MEDIAN[klass]
        sa, sf = _RCS_NOISE[klass]
        spike_p = _SPIKE_PROB[klass]
        aspect_amp = _ASPECT_AMP[klass]

        # Per-actor calibration shift (proxy -> dBsm).
        # Synthetic proxy: lognormal around obb_area, so that median = obb_area.
        # Calibration offset: whatever shift brings median(proxy_dBsm) to target.
        # We compute it exactly for synthetic data.
        proxy_median_dbsm = 10.0 * math.log10(obb_area)
        calib_offset = median_dbsm - proxy_median_dbsm

        for i in range(n):
            aid = actor_id + i
            # Range: uniform in [3, R_MAX+5] to show detections both inside and outside R_max
            range_m   = rng.uniform(3.0, _R_MAX + 5.0)
            azimuth   = rng.uniform(-math.pi / 2, math.pi / 2)
            velocity  = rng.uniform(-_V_MAX * 1.1, _V_MAX * 1.1)

            # Actor heading (random; yaw gives orientation in world)
            actor_yaw = rng.uniform(-math.pi, math.pi)
            sensor_los = azimuth  # sensor at origin, actor direction ~ azimuth

            # rcs_proxy_m2: lognormal around obb_area
            log_sigma = 0.3
            proxy = obb_area * math.exp(rng.gauss(0.0, log_sigma))

            # --- BEFORE: calibrated median only (no noise) ---
            rcs_before = 10.0 * math.log10(proxy) + calib_offset

            # --- AFTER: + per-actor offset ---
            actor_offset = random.Random(f"42|{klass}|{aid}|actor").gauss(0.0, sa)
            rcs_after = rcs_before + actor_offset

            # --- AFTER: + Swerling-1 per-frame ---
            frame_rng = random.Random(f"42|{klass}|{aid}|0")
            swerling_noise = _swerling1(frame_rng, sf)
            rcs_after += swerling_noise

            # --- AFTER: + specular spike ---
            spike_rng = random.Random(f"42|{klass}|{aid}|0|spike")
            spiked = spike_rng.random() < spike_p
            if spiked:
                rcs_after += spike_rng.uniform(10.0, 25.0)

            # --- AFTER: + aspect-angle modulation ---
            aspect_delta = _aspect_delta(actor_yaw, sensor_los, aspect_amp)
            rcs_after += aspect_delta

            # --- SNR and visibility ---
            snr_before = _snr_db(rcs_before, range_m)
            snr_after  = _snr_db(rcs_after,  range_m)

            range_ok    = range_m  <= _R_MAX
            velocity_ok = abs(velocity) <= _V_MAX

            vis_rng = random.Random(f"42|vis|{aid}")
            snr_ok_before = (snr_before >= _SNR_THRESHOLD_DB
                             and vis_rng.random() < _P_DETECT)
            vis_rng2 = random.Random(f"42|vis2|{aid}")
            snr_ok_after  = (snr_after  >= _SNR_THRESHOLD_DB
                             and vis_rng2.random() < _P_DETECT)

            visible_before = 1 if (range_ok and velocity_ok and snr_ok_before) else 0
            visible_after  = 1 if (range_ok and velocity_ok and snr_ok_after)  else 0

            # --- Noisy measurements (after) ---
            sig_r, sig_v, sig_az = _noise_sigmas(snr_after)
            noise_rng = random.Random(f"42|noise|{aid}")
            range_noisy = range_m  + noise_rng.gauss(0.0, sig_r)
            vel_noisy   = velocity + noise_rng.gauss(0.0, sig_v)
            az_noisy    = azimuth  + noise_rng.gauss(0.0, sig_az)

            rows.append({
                "class":          klass,
                "actor_id":       aid,
                "depth_m":        range_m,
                "velocity_mps":   velocity,
                "azimuth_rad":    azimuth,
                "rcs_proxy_m2":   proxy,
                "rcs_before_dBsm": rcs_before,
                "rcs_after_dBsm":  rcs_after,
                "snr_before_dB":   snr_before,
                "snr_after_dB":    snr_after,
                "visible_before":  visible_before,
                "visible_after":   visible_after,
                "depth_m_noisy":   range_noisy,
                "velocity_mps_noisy": vel_noisy,
                "azimuth_rad_noisy":  az_noisy,
                "spiked":          spiked,
                "aspect_delta_dB": aspect_delta,
                "swerling_dB":     swerling_noise,
            })
        actor_id += n

    return rows


# ---------------------------------------------------------------------------
# Data snippets printer
# ---------------------------------------------------------------------------
def print_snippets(rows: list[dict], n_per_class: int = 4) -> None:
    classes = sorted({r["class"] for r in rows})

    print("\n" + "=" * 100)
    print("  DATA SNIPPETS — key columns for representative detections")
    print("=" * 100)

    hdr = (f"  {'Class':<12} {'Range':>7} {'RCS proxy':>10} {'RCS before':>11}"
           f" {'RCS after':>10} {'Swerling':>9} {'Aspect':>7} {'Spike':>6}"
           f" {'SNR_after':>9} {'Vis':>4}")
    print(hdr)
    print("  " + "-" * 98)
    print(f"  {'':12} {'(m)':>7} {'(m2)':>10} {'(dBsm)':>11}"
          f" {'(dBsm)':>10} {'(dB)':>9} {'(dB)':>7} {'':>6}"
          f" {'(dB)':>9} {'':>4}")
    print("  " + "-" * 98)

    for klass in classes:
        class_rows = [r for r in rows if r["class"] == klass]
        # pick a spread: near (< 20m), mid (20-35m), far (35-48m), OOR (> 48m)
        buckets = [
            ("near",   [r for r in class_rows if r["depth_m"] < 20]),
            ("mid",    [r for r in class_rows if 20 <= r["depth_m"] < 35]),
            ("far",    [r for r in class_rows if 35 <= r["depth_m"] <= _R_MAX]),
            ("OOR",    [r for r in class_rows if r["depth_m"] > _R_MAX]),
        ]
        printed = 0
        for label, bucket in buckets:
            if not bucket or printed >= n_per_class:
                break
            r = bucket[0]
            spike_flag = "*" if r["spiked"] else ""
            vis = r["visible_after"]
            print(f"  {klass:<12} {r['depth_m']:>7.1f} {r['rcs_proxy_m2']:>10.3f}"
                  f" {r['rcs_before_dBsm']:>+11.2f} {r['rcs_after_dBsm']:>+10.2f}"
                  f" {r['swerling_dB']:>+9.2f} {r['aspect_delta_dB']:>+7.2f}"
                  f" {spike_flag:>6} {r['snr_after_dB']:>+9.2f} {vis:>4}")
            printed += 1
        print()

    print(f"  * = specular spike event (+10-25 dB)")
    print(f"\n  FMCW limits:  R_max={_R_MAX:.0f} m  v_max={_V_MAX:.1f} m/s"
          f"  SNR_threshold={_SNR_THRESHOLD_DB:.0f} dB  P_detect={_P_DETECT:.1f}")
    dR_label  = f"dR={_DR:.3f} m"
    dV_label  = f"dv={_DV:.3f} m/s"
    az_label  = f"az_sigma_0={math.degrees(_AZ_SIGMA_0_RAD):.1f} deg"
    print(f"  Noise model:  {dR_label}  {dV_label}  {az_label}"
          f"  floor={math.degrees(_AZ_SIGMA_FLOOR):.1f} deg")

    # SNR horizon table
    print(f"\n  SNR at range (dB) — using calibrated median RCS, no noise:")
    print(f"  {'Class':<12} {'Median RCS':>11}  " +
          "  ".join(f"{r:>5}m" for r in [10, 20, 30, 40, 48]) +
          "   threshold")
    print("  " + "-" * 75)
    for klass, med in _RCS_MEDIAN.items():
        snrs = [_snr_db(med, r) for r in [10, 20, 30, 40, 48]]
        row_str = "  ".join(f"{s:>+6.1f}" for s in snrs)
        max_r = next((r for r in range(5, 60) if _snr_db(med, r) < _SNR_THRESHOLD_DB),
                     ">60")
        print(f"  {klass:<12} {med:>+11.1f}  {row_str}   det.range<{max_r}m")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(rows: list[dict], save_dir: Path | None) -> None:
    try:
        import matplotlib
        if save_dir:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        print("matplotlib not installed — cannot plot. Use --snippets-only.")
        return

    classes = sorted({r["class"] for r in rows})
    _fallback_colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                         "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    _fallback_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    rng_cmap    = {c: CLASS_COLORS.get(c, _fallback_colors[i % len(_fallback_colors)])
                   for i, c in enumerate(classes)}
    _class_markers = {c: CLASS_MARKERS.get(c, _fallback_markers[i % len(_fallback_markers)])
                      for i, c in enumerate(classes)}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def _save_or_show(fig, name):
        if save_dir:
            path = save_dir / name
            fig.savefig(path, dpi=130, bbox_inches="tight")
            print(f"  Saved: {path}")
            plt.close(fig)
        else:
            plt.show()

    # ================================================================
    # Figure 1: RCS distributions — before vs after, per class
    # ================================================================
    fig1, axes = plt.subplots(2, len(classes), figsize=(5 * len(classes), 7))
    fig1.suptitle("RCS Distributions: Before vs After Realism", fontsize=13,
                  fontweight="bold", y=1.01)

    for col, klass in enumerate(classes):
        cr = [r for r in rows if r["class"] == klass]
        before = [r["rcs_before_dBsm"] for r in cr]
        after  = [r["rcs_after_dBsm"]  for r in cr]
        color  = rng_cmap[klass]

        for ax_row, (data, label) in enumerate([(before, "Before\n(calib. median only)"),
                                                (after,  "After\n(+Swerling+spikes+aspect)")]):
            ax = axes[ax_row][col]
            arr = np.array(data)
            bins = np.linspace(arr.min() - 1, min(arr.max() + 1, arr.mean() + 30), 60)
            ax.hist(arr, bins=bins, color=color, alpha=0.75, edgecolor="none")
            med = float(np.median(arr))
            mn  = float(np.mean(arr))
            ax.axvline(med, color="black",  linestyle="--", linewidth=1.5,
                       label=f"median {med:.1f}")
            ax.axvline(mn,  color="gray",   linestyle=":",  linewidth=1.2,
                       label=f"mean {mn:.1f}")
            ax.axvline(_RCS_MEDIAN[klass], color="red", linestyle="-", linewidth=1,
                       alpha=0.7, label=f"target {_RCS_MEDIAN[klass]:.0f}")
            if ax_row == 0:
                ax.set_title(f"{klass.capitalize()}", fontsize=11, fontweight="bold")
            ax.set_xlabel("RCS (dBsm)")
            ax.set_ylabel("Count")
            if col == 0:
                ax.text(-0.22, 0.5, label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9,
                        fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig1, "fig1_rcs_distributions.png")

    # ================================================================
    # Figure 2: SNR analysis
    # ================================================================
    fig2 = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.35)
    fig2.suptitle("SNR Analysis (RS35 FMCW: R_max=48m, SNR_thr=10dB)",
                  fontsize=13, fontweight="bold")

    # 2a: SNR vs range scatter
    ax2a = fig2.add_subplot(gs[0, 0])
    for klass in classes:
        cr = [r for r in rows if r["class"] == klass]
        ax2a.scatter(
            [r["depth_m"] for r in cr],
            [r["snr_after_dB"] for r in cr],
            s=4, alpha=0.35, color=rng_cmap[klass], label=klass,
            marker=_class_markers[klass],
        )
    ax2a.axhline(_SNR_THRESHOLD_DB, color="red", linestyle="--", linewidth=1.5,
                 label=f"threshold {_SNR_THRESHOLD_DB} dB")
    ax2a.axvline(_R_MAX, color="orange", linestyle="--", linewidth=1.5,
                 label=f"R_max {_R_MAX:.0f} m")
    ax2a.set_xlabel("Range (m)")
    ax2a.set_ylabel("SNR after realism (dB)")
    ax2a.set_title("SNR vs Range")
    ax2a.legend(fontsize=7, markerscale=2)
    ax2a.grid(True, alpha=0.3)

    # 2b: SNR histogram before vs after
    ax2b = fig2.add_subplot(gs[0, 1])
    snr_before_all = [r["snr_before_dB"] for r in rows]
    snr_after_all  = [r["snr_after_dB"]  for r in rows]
    bins_snr = np.linspace(-20, 60, 80)
    ax2b.hist(snr_before_all, bins=bins_snr, alpha=0.5, color="steelblue",
              label="Before (calib. only)")
    ax2b.hist(snr_after_all,  bins=bins_snr, alpha=0.5, color="darkorange",
              label="After (all noise)")
    ax2b.axvline(_SNR_THRESHOLD_DB, color="red", linestyle="--", linewidth=1.5,
                 label=f"threshold {_SNR_THRESHOLD_DB} dB")
    ax2b.set_xlabel("SNR (dB)")
    ax2b.set_ylabel("Count")
    ax2b.set_title("SNR Distribution: Before vs After")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3)

    # 2c: Visibility breakdown stacked bar
    ax2c = fig2.add_subplot(gs[1, 0])
    vis_data = {}
    for klass in classes:
        cr = [r for r in rows if r["class"] == klass]
        total = len(cr)
        v1 = sum(r["visible_after"] for r in cr)
        # decompose invisible:
        inv_range = sum(1 for r in cr if r["depth_m"] > _R_MAX)
        inv_vel   = sum(1 for r in cr if abs(r["velocity_mps"]) > _V_MAX
                        and r["depth_m"] <= _R_MAX)
        inv_snr   = total - v1 - inv_range - inv_vel
        vis_data[klass] = {"visible": v1, "range": inv_range,
                           "velocity": inv_vel, "snr": max(inv_snr, 0)}

    x = np.arange(len(classes))
    w = 0.5
    bottoms = np.zeros(len(classes))
    for label, color in [("visible", "green"), ("snr", "gray"),
                         ("range", "tomato"), ("velocity", "gold")]:
        vals = np.array([vis_data[c][label] for c in classes])
        ax2c.bar(x, vals, w, bottom=bottoms, label=label, color=color, alpha=0.8)
        bottoms += vals
    ax2c.set_xticks(x)
    ax2c.set_xticklabels([c.capitalize() for c in classes])
    ax2c.set_ylabel("Detection count")
    ax2c.set_title("Visibility Breakdown After Realism")
    ax2c.legend(fontsize=8, title="Reason")
    ax2c.grid(True, alpha=0.3, axis="y")

    # 2d: P_detect curve vs SNR
    ax2d = fig2.add_subplot(gs[1, 1])
    snr_curve = np.linspace(-5, 40, 300)
    # Step P_d model (matches PostProcessDataset): 0 below threshold, P_DETECT above
    p_step = np.where(snr_curve >= _SNR_THRESHOLD_DB, _P_DETECT, 0.0)
    ax2d.plot(snr_curve, p_step, "b-", linewidth=2, label=f"P_detect (step at {_SNR_THRESHOLD_DB} dB)")
    # Mark class operating points (at median RCS, range = R_max/2)
    for klass, med in _RCS_MEDIAN.items():
        if klass not in rng_cmap:
            continue
        snr_pt = _snr_db(med, _R_MAX / 2)
        p_pt   = _P_DETECT if snr_pt >= _SNR_THRESHOLD_DB else 0.0
        ax2d.scatter([snr_pt], [p_pt], s=80, zorder=5, color=rng_cmap[klass],
                     marker=_class_markers[klass], label=f"{klass} @{_R_MAX/2:.0f}m")
    ax2d.axvline(_SNR_THRESHOLD_DB, color="red", linestyle="--",
                 linewidth=1.5, label=f"threshold {_SNR_THRESHOLD_DB} dB")
    ax2d.set_xlabel("SNR (dB)")
    ax2d.set_ylabel("P(detection)")
    ax2d.set_ylim(-0.05, 1.1)
    ax2d.set_title("Detection Probability Model")
    ax2d.legend(fontsize=7)
    ax2d.grid(True, alpha=0.3)

    _save_or_show(fig2, "fig2_snr_analysis.png")

    # ================================================================
    # Figure 3: Measurement noise vs SNR
    # ================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    fig3.suptitle("SNR-Scaled Measurement Noise (visible detections only)",
                  fontsize=12, fontweight="bold")

    vis_rows = [r for r in rows if r["visible_after"]]

    for klass in classes:
        cr = [r for r in vis_rows if r["class"] == klass]
        if not cr:
            continue
        snrs  = np.array([r["snr_after_dB"]            for r in cr])
        d_err = np.array([r["depth_m_noisy"] - r["depth_m"]         for r in cr])
        v_err = np.array([r["velocity_mps_noisy"] - r["velocity_mps"] for r in cr])
        a_err = np.array([math.degrees(r["azimuth_rad_noisy"] - r["azimuth_rad"]) for r in cr])

        for ax, err, label in zip(axes3, [d_err, v_err, a_err],
                                  ["Range error (m)", "Velocity error (m/s)",
                                   "Azimuth error (deg)"]):
            ax.scatter(snrs, err, s=3, alpha=0.3, color=rng_cmap[klass],
                       label=klass, marker=_class_markers[klass])

    # Overlay theoretical 1-sigma envelope (CRLB: sigma = resolution / sqrt(2*SNR))
    snr_env = np.linspace(_SNR_THRESHOLD_DB, 45, 200)
    snr_lin_env = 10.0 ** (snr_env / 10.0)
    sqrt_2snr_env = np.sqrt(2.0 * snr_lin_env)
    s_env         = np.sqrt(snr_lin_env)
    sigma_r_env  = _DR / sqrt_2snr_env
    sigma_v_env  = _DV / sqrt_2snr_env
    sigma_az_env = np.degrees(np.maximum(_AZ_SIGMA_0_RAD / s_env, _AZ_SIGMA_FLOOR))

    for ax, sigma_env, label in zip(axes3,
                                    [sigma_r_env, sigma_v_env, sigma_az_env],
                                    ["Range error (m)", "Velocity error (m/s)",
                                     "Azimuth error (deg)"]):
        ax.plot(snr_env,  sigma_env, "k--", linewidth=1.5, label="1-sigma (theory)")
        ax.plot(snr_env, -sigma_env, "k--", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(_SNR_THRESHOLD_DB, color="red", linestyle=":", linewidth=1,
                   label=f"threshold {_SNR_THRESHOLD_DB} dB")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(label)
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.3)

    axes3[0].set_title(f"Range noise  (dR={_DR:.3f} m)")
    axes3[1].set_title(f"Doppler noise  (dv={_DV:.3f} m/s)")
    axes3[2].set_title(f"Azimuth noise  (sigma_0=6 deg, floor=0.3 deg)")

    plt.tight_layout()
    _save_or_show(fig3, "fig3_measurement_noise.png")

    # ================================================================
    # Figure 4: BEV spatial — before vs after visibility
    # ================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle("Bird's-Eye View: Visible Detections Before vs After Realism",
                  fontsize=12, fontweight="bold")

    for ax, vis_key, title in [
        (ax4a, "visible_before", "Before (calib. SNR only)"),
        (ax4b, "visible_after",  "After (Swerling + spikes + aspect)")
    ]:
        for klass in classes:
            cr = [r for r in rows if r["class"] == klass]
            vis_r  = [r for r in cr if r[vis_key] == 1]
            invis_r = [r for r in cr if r[vis_key] == 0]

            def _xy(det_rows):
                xs = [r["depth_m"] * math.cos(r["azimuth_rad"]) for det in det_rows
                      for r in [det]]
                ys = [r["depth_m"] * math.sin(r["azimuth_rad"]) for det in det_rows
                      for r in [det]]
                return xs, ys

            xv, yv = _xy(vis_r)
            xi, yi = _xy(invis_r)
            ax.scatter(xv, yv, s=5, alpha=0.5, color=rng_cmap[klass],
                       marker=_class_markers[klass], label=f"{klass} visible")
            ax.scatter(xi, yi, s=3, alpha=0.12, color=rng_cmap[klass],
                       marker=_class_markers[klass])

        # R_max circle
        theta = [i * 2 * math.pi / 300 for i in range(301)]
        ax.plot([_R_MAX * math.cos(t) for t in theta],
                [_R_MAX * math.sin(t) for t in theta],
                "r--", linewidth=1.2, alpha=0.7, label=f"R_max={_R_MAX:.0f}m")
        ax.set_xlim(0, _R_MAX + 8)
        ax.set_ylim(-_R_MAX - 8, _R_MAX + 8)
        ax.set_xlabel("Forward (m)")
        ax.set_ylabel("Lateral (m)")
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=2, loc="upper right")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#f8f8f8")

        n_vis   = sum(1 for r in rows if r[vis_key] == 1)
        n_total = len(rows)
        ax.text(0.02, 0.02,
                f"visible {n_vis}/{n_total} ({100*n_vis/n_total:.0f}%)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    _save_or_show(fig4, "fig4_bev_visibility.png")

    # ================================================================
    # Figure 5: RCS component breakdown (violin)
    # ================================================================
    fig5, axes5 = plt.subplots(1, 4, figsize=(14, 5))
    fig5.suptitle("RCS Noise Component Breakdown by Class",
                  fontsize=12, fontweight="bold")

    component_keys = ["swerling_dB", "aspect_delta_dB"]
    component_labels = ["Swerling-1\nfluctuation (dB)", "Aspect-angle\ndelta (dB)"]

    for col, (key, label) in enumerate(zip(component_keys, component_labels)):
        ax = axes5[col]
        data_per_class = [[r[key] for r in rows if r["class"] == klass]
                          for klass in classes]
        parts = ax.violinplot(data_per_class, positions=range(len(classes)),
                              showmedians=True, showextrema=True)
        for pc, klass in zip(parts["bodies"], classes):
            pc.set_facecolor(rng_cmap[klass])
            pc.set_alpha(0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([c.capitalize() for c in classes], rotation=20)
        ax.set_ylabel(label)
        ax.set_title(label.replace("\n", " "))
        ax.grid(True, alpha=0.3, axis="y")

    # Spike fraction bar
    ax = axes5[2]
    spike_fracs = [sum(r["spiked"] for r in rows if r["class"] == klass)
                   / max(sum(1 for r in rows if r["class"] == klass), 1) * 100
                   for klass in classes]
    bars = ax.bar(range(len(classes)), spike_fracs, color=[rng_cmap[c] for c in classes],
                  alpha=0.8)
    for bar, v in zip(bars, spike_fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.capitalize() for c in classes], rotation=20)
    ax.set_ylabel("Spike rate (%)")
    ax.set_title("Specular Spike Rate")
    ax.grid(True, alpha=0.3, axis="y")

    # RCS spread comparison: sigma before vs sigma after
    ax = axes5[3]
    x = np.arange(len(classes))
    sigma_before = [float(np.std([r["rcs_before_dBsm"] for r in rows
                                  if r["class"] == klass])) for klass in classes]
    sigma_after  = [float(np.std([r["rcs_after_dBsm"]  for r in rows
                                  if r["class"] == klass])) for klass in classes]
    ax.bar(x - 0.18, sigma_before, 0.35, label="Before", color="steelblue", alpha=0.8)
    ax.bar(x + 0.18, sigma_after,  0.35, label="After",  color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], rotation=20)
    ax.set_ylabel("RCS std dev (dB)")
    ax.set_title("RCS Spread: Before vs After")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save_or_show(fig5, "fig5_rcs_components.png")

    print("\n  5 figures generated.")


# ---------------------------------------------------------------------------
# Real CSV loader
# ---------------------------------------------------------------------------
_TARGET_MEDIAN_DBSM = {
    "car": 7.0, "truck": 15.0, "pedestrian": -11.0,
    "cyclist": -5.0, "bus": 12.0, "motorcycle": 0.0,
}


def load_from_csv(csv_path: Path) -> list[dict]:
    """Load a processed radar_data_processed.csv.

    Derives the 'before' state from rcs_proxy_m2 + per-class calibration offset,
    so before/after comparisons work from a single file.
    """
    if not csv_path.is_file():
        sys.exit(f"Missing {csv_path}")

    # Pass 1: compute calibration offsets (same logic as PostProcessDataset step [3/4])
    print(f"  Pass 1/2: calibrating offsets from {csv_path.name} ...", flush=True)
    proxies_by_class: dict = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            klass = row.get("matched_actor_class", "").strip()
            proxy_str = row.get("rcs_proxy_m2", "").strip()
            if not klass or not proxy_str:
                continue
            try:
                v = float(proxy_str)
                if v > 0:
                    proxies_by_class[klass].append(v)
            except ValueError:
                pass

    calib_offsets: dict = {}
    for klass, xs in proxies_by_class.items():
        tgt = _TARGET_MEDIAN_DBSM.get(klass)
        if tgt is None:
            continue
        sorted_xs = sorted(xs)
        med = sorted_xs[len(sorted_xs) // 2]
        calib_offsets[klass] = tgt - 10.0 * math.log10(med)
        print(f"    {klass:<12} n={len(xs):>8,}  "
              f"proxy_median={med:.4f} m2  offset={calib_offsets[klass]:+.2f} dB",
              flush=True)

    # Pass 2: load rows with before/after derived
    print(f"  Pass 2/2: loading rows ...", flush=True)
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            klass = row.get("matched_actor_class", "").strip()
            if not klass:
                continue
            proxy_str   = row.get("rcs_proxy_m2", "").strip()
            rcs_str     = row.get("rcs_dBsm", "").strip()
            snr_str     = row.get("snr_dB", "").strip()
            vis_str     = row.get("visible", "").strip()
            if not proxy_str or not rcs_str or not snr_str:
                continue
            try:
                proxy       = float(proxy_str)
                rcs_after   = float(rcs_str)
                range_m     = float(row["depth_m"])
                vel         = float(row["velocity_mps"])
                az          = float(row["azimuth_rad"])
                snr_after   = float(snr_str)
                visible_after = int(vis_str) if vis_str else 0
            except (ValueError, KeyError):
                continue

            # Derive "before": calibrated median only (no Swerling / spikes)
            off = calib_offsets.get(klass)
            rcs_before = (10.0 * math.log10(max(proxy, 1e-12)) + off
                          if (proxy > 0 and off is not None)
                          else rcs_after)
            snr_before = _snr_db(rcs_before, range_m)
            range_ok   = range_m <= _R_MAX
            vel_ok     = abs(vel) <= _V_MAX
            # Use a deterministic "before" detection draw keyed on row identity
            det_rng = random.Random(
                f"before|{row.get('frame','')}|"
                f"{row.get('sensor_id','')}|{row.get('detection_index','')}"
            )
            visible_before = int(
                range_ok and vel_ok
                and snr_before >= _SNR_THRESHOLD_DB
                and det_rng.random() < _P_DETECT
            )

            rows.append({
                "class":              klass,
                "actor_id":           row.get("matched_actor_id", ""),
                "depth_m":            range_m,
                "velocity_mps":       vel,
                "azimuth_rad":        az,
                "rcs_proxy_m2":       proxy,
                "rcs_before_dBsm":    rcs_before,
                "rcs_after_dBsm":     rcs_after,
                "snr_before_dB":      snr_before,
                "snr_after_dB":       snr_after,
                "visible_before":     visible_before,
                "visible_after":      visible_after,
                "depth_m_noisy":      float(row.get("depth_m_noisy") or range_m),
                "velocity_mps_noisy": float(row.get("velocity_mps_noisy") or vel),
                "azimuth_rad_noisy":  float(row.get("azimuth_rad_noisy") or az),
                "spiked":             False,
                "aspect_delta_dB":    0.0,
                "swerling_dB":        rcs_after - rcs_before,  # residual ~ noise
            })

    print(f"  Loaded {len(rows):,} matched rows from {csv_path.name}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true",
                      help="Generate synthetic demo data.")
    mode.add_argument("--input", type=Path,
                      help="Path to a processed radar_data_processed.csv.")
    p.add_argument("--save", metavar="DIR", type=Path, default=None,
                   help="Save figures to this directory instead of displaying.")
    p.add_argument("--snippets-only", action="store_true",
                   help="Print data table only; skip matplotlib plots.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.synthetic:
        print("Generating synthetic demo data...", flush=True)
        rows = generate_synthetic(args.seed)
        print(f"  {len(rows):,} synthetic detections across "
              f"{len(CLASS_COUNT)} classes", flush=True)
    else:
        rows = load_from_csv(args.input)

    print_snippets(rows)

    if not args.snippets_only:
        print("\nGenerating plots...", flush=True)
        make_plots(rows, args.save)


if __name__ == "__main__":
    main()
