"""
FMCW Radar Configuration Calculator
=====================================
Computes key radar performance metrics from hardware parameters and compares
them against RadarScenes reference values.

Usage:
    python fmcw_radar_config.py                    # print all presets
    python fmcw_radar_config.py --preset rs50      # single preset
    python fmcw_radar_config.py --plot             # show tradeoff curves
    python fmcw_radar_config.py --plot --save fig.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Force UTF-8 stdout on Windows so Unicode labels print correctly
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C = 3e8          # speed of light (m/s)
K_B = 1.38e-23   # Boltzmann constant (J/K)
T_0 = 290.0      # reference temperature (K)

# ---------------------------------------------------------------------------
# RCS reference values (dBsm) for SNR calculation
# ---------------------------------------------------------------------------
RCS_TABLE_DBsm: Dict[str, float] = {
    "truck":       10.0,
    "car":          4.0,
    "cyclist":     -3.0,
    "pedestrian":  -8.0,
}


# ---------------------------------------------------------------------------
# Dataclass — hardware parameters
# ---------------------------------------------------------------------------
@dataclass
class FMCWRadarConfig:
    """Hardware / waveform parameters for one FMCW radar."""

    name: str = "unnamed"

    # RF
    center_freq_hz: float = 76e9        # carrier frequency (Hz)
    bandwidth_hz: float = 400e6         # sweep bandwidth B (Hz)

    # Waveform timing
    chirp_duration_s: float = 19.74e-6  # single chirp period T_c (s)
    n_chirps: int = 1024                # chirps per frame N_c
    n_adc_samples: int = 512            # ADC samples per chirp N_s

    # Link budget
    tx_power_dbm: float = 14.0          # transmit power (dBm)
    antenna_gain_dbi: float = 15.0      # one-way antenna gain (dBi)
    noise_figure_db: float = 12.0       # receiver noise figure (dB)
    system_loss_db: float = 3.0         # miscellaneous losses (dB)

    # Detection threshold
    snr_threshold_db: float = 10.0      # minimum detectable SNR (dB)


# ---------------------------------------------------------------------------
# Performance derived quantities
# ---------------------------------------------------------------------------
@dataclass
class FMCWPerformance:
    cfg: FMCWRadarConfig

    # Waveform
    wavelength_m: float = 0.0
    adc_sample_rate_hz: float = 0.0     # F_s = N_s / T_c
    chirp_slope_hz_per_s: float = 0.0   # S = B / T_c
    frame_duration_s: float = 0.0       # T_frame = N_c * T_c
    update_rate_hz: float = 0.0

    # Range
    range_resolution_m: float = 0.0     # delta_R = c / (2*B)
    range_max_m: float = 0.0            # R_max = N_s * c / (4*B)
    beat_freq_max_hz: float = 0.0       # f_beat_max = S * 2*R_max/c  (= F_s/2)
    if_bandwidth_hz: float = 0.0        # required IF filter BW = f_beat_max

    # Velocity
    velocity_resolution_ms: float = 0.0 # delta_v = lambda / (2 * N_c * T_c)
    velocity_max_ms: float = 0.0        # v_max = lambda / (4 * T_c)

    # Range-Doppler coupling: apparent range error for a moving target
    # ΔR = v * f_c / S  (before 2D-FFT correction)
    range_doppler_coupling_m_per_ms: float = 0.0  # ΔR per 1 m/s of target velocity
    coupling_at_vmax_m: float = 0.0               # ΔR at v_max

    # SNR per target class at selected ranges
    snr_table: Dict[str, List[float]] = field(default_factory=dict)
    snr_ranges_m: List[float] = field(default_factory=list)


def compute_performance(
    cfg: FMCWRadarConfig,
    snr_ranges_m: Optional[List[float]] = None,
) -> FMCWPerformance:
    """Derive all performance metrics from an FMCWRadarConfig."""

    if snr_ranges_m is None:
        snr_ranges_m = [10.0, 20.0, 35.0, 50.0, 75.0, 100.0]

    p = FMCWPerformance(cfg=cfg)
    p.snr_ranges_m = snr_ranges_m

    f_c = cfg.center_freq_hz
    B = cfg.bandwidth_hz
    T_c = cfg.chirp_duration_s
    N_s = cfg.n_adc_samples
    N_c = cfg.n_chirps

    lam = C / f_c
    p.wavelength_m = lam

    # ADC sample rate
    p.adc_sample_rate_hz = N_s / T_c

    # Chirp slope
    p.chirp_slope_hz_per_s = B / T_c

    # Frame timing
    p.frame_duration_s = N_c * T_c
    p.update_rate_hz = 1.0 / p.frame_duration_s

    S = B / T_c  # chirp slope (Hz/s)

    # Range metrics
    p.range_resolution_m = C / (2.0 * B)
    p.range_max_m = N_s * C / (4.0 * B)

    # Beat frequency at max range — must be <= F_s/2 (Nyquist)
    # f_beat = S * 2*R/c  =>  f_beat_max = S * 2*R_max/c = (B/T_c)*(N_s*c/(2*B))/c = N_s/(2*T_c) = F_s/2
    p.beat_freq_max_hz = S * 2.0 * p.range_max_m / C
    p.if_bandwidth_hz = p.beat_freq_max_hz  # IF filter must pass this

    # Velocity metrics
    p.velocity_max_ms = lam / (4.0 * T_c)
    p.velocity_resolution_ms = lam / (2.0 * N_c * T_c)

    # Range-Doppler coupling
    # A target moving at v m/s shifts the beat frequency by f_doppler = 2v/lambda.
    # The range FFT interprets this extra frequency as an apparent range offset:
    #   ΔR = (f_doppler / S) * c/2 = (2v/lambda) / (B/T_c) * c/2
    #      = v * f_c / S   (since lambda = c/f_c)
    # With 2D range-Doppler processing this is compensated, but it sets
    # the cross-coupling budget for single-chirp or limited-processing cases.
    p.range_doppler_coupling_m_per_ms = f_c / S  # m of range error per (m/s) of velocity
    p.coupling_at_vmax_m = p.velocity_max_ms * p.range_doppler_coupling_m_per_ms

    # SNR table
    # After 2-D range-Doppler FFT the processing gain is:
    #   G_proc = N_s * N_c  (coherent integration over full frame)
    # Noise bandwidth after FFT = 1/T_frame
    # SNR formula (linear):
    #   SNR = Pt * G_tx * G_rx * lambda^2 * sigma / ((4*pi)^3 * R^4)
    #         * T_frame / (k * T0 * NF * L)
    # In dB:
    #   SNR_dB = Pt_dBm - 30 + 2*G_dBi + 20*log10(lam)
    #            - 30*log10(4*pi)  [= ~-32.97 dB]
    #            + sigma_dBsm
    #            - 40*log10(R)
    #            + 174  [= -10*log10(k*T0) in dBm/Hz]
    #            + 10*log10(T_frame)
    #            - NF_dB - L_dB

    Pt_dBm = cfg.tx_power_dbm
    G_dBi = cfg.antenna_gain_dbi
    NF_dB = cfg.noise_figure_db
    L_dB = cfg.system_loss_db
    T_frame = p.frame_duration_s

    path_base_dB = (
        (Pt_dBm - 30.0)
        + 2.0 * G_dBi
        + 20.0 * math.log10(lam)
        - 20.0 * math.log10(4.0 * math.pi)   # NOTE: 20 not 30 — see below
        + 174.0
        + 10.0 * math.log10(T_frame)
        - NF_dB
        - L_dB
    )
    # Full free-space: ((4*pi)^3 * R^4) → 30*log10(4pi) + 40*log10(R)
    # But we split: Pt*Gt*Gr*lambda^2 uses (4pi*R)^2 in denominator twice
    # Standard form: SNR ∝ lambda^2 / (4pi)^3 / R^4
    # Rewrite:
    #   -20*log10(4pi) from lambda^2/(4pi)^2, then -20*log10(4pi) for the extra
    # Simpler to use the compact form directly:
    #   SNR_dB = Pt(dBm)-30 + 2*G + 20log(lambda) - 32.97(= 10log((4pi)^3/(1)^0)... )
    # Let's derive cleanly:
    #   SNR = Pt * Gt * Gr * lam^2 * sigma * T_frame
    #         / ( (4pi)^3 * R^4 * k * T0 * NF * L )
    # In dB:
    #   = [Pt_dBW] + [2*G] + [20*log10(lam)] + [sigma_dBsm]
    #     - [30*log10(4*pi)] + [10*log10(T_frame)]
    #     - [10*log10(k*T0)] - [NF_dB] - [L_dB]
    #     - 40*log10(R)
    # Note: 10*log10(k*T0) at T0=290K = 10*log10(1.38e-23*290) = -203.97 dBW/Hz
    #       => -173.97 dBm/Hz  ≈ -174 dBm/Hz
    # 30*log10(4*pi) = 30*log10(12.566) = 32.97 dB

    # So:
    #   path_base_dB = Pt_dBW + 2G + 20log(lam) + 174 + 10log(T_frame)
    #                  - 32.97 - NF - L
    # where Pt_dBW = Pt_dBm - 30

    path_base_dB = (
        (Pt_dBm - 30.0)
        + 2.0 * G_dBi
        + 20.0 * math.log10(lam)
        + 174.0
        + 10.0 * math.log10(T_frame)
        - 32.97
        - NF_dB
        - L_dB
    )

    snr_table: Dict[str, List[float]] = {}
    for target, sigma_dBsm in RCS_TABLE_DBsm.items():
        row: List[float] = []
        for R in snr_ranges_m:
            snr_dB = path_base_dB + sigma_dBsm - 40.0 * math.log10(R)
            row.append(snr_dB)
        snr_table[target] = row

    p.snr_table = snr_table
    return p


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

# Current CARLA default — ultra-fine range resolution, high ADC rate required
PRESET_5GHZ_35M = FMCWRadarConfig(
    name="5GHz_35m (current CARLA default)",
    center_freq_hz=76e9,
    bandwidth_hz=5e9,
    chirp_duration_s=19.74e-6,
    n_chirps=1024,
    n_adc_samples=4096,
    tx_power_dbm=14.0,
    antenna_gain_dbi=15.0,
    noise_figure_db=12.0,
    system_loss_db=3.0,
    snr_threshold_db=10.0,
)

# RadarScenes-matching resolution, 50 m design range, feasible ADC rate
PRESET_RS_MATCH_50M = FMCWRadarConfig(
    name="RS_match_50m (RadarScenes resolution, 50 m range)",
    center_freq_hz=76e9,
    bandwidth_hz=400e6,
    chirp_duration_s=19.74e-6,
    n_chirps=1024,
    n_adc_samples=512,
    tx_power_dbm=14.0,
    antenna_gain_dbi=15.0,
    noise_figure_db=12.0,
    system_loss_db=3.0,
    snr_threshold_db=10.0,
)

# RadarScenes-matching resolution, 35 m design range (fewer ADC samples)
PRESET_RS_MATCH_35M = FMCWRadarConfig(
    name="RS_match_35m (RadarScenes resolution, 35 m range)",
    center_freq_hz=76e9,
    bandwidth_hz=400e6,
    chirp_duration_s=19.74e-6,
    n_chirps=1024,
    n_adc_samples=256,
    tx_power_dbm=14.0,
    antenna_gain_dbi=15.0,
    noise_figure_db=12.0,
    system_loss_db=3.0,
    snr_threshold_db=10.0,
)

# 77 GHz variant with RadarScenes-matching resolution
PRESET_RS_MATCH_77GHZ = FMCWRadarConfig(
    name="RS_match_77GHz (77 GHz, RadarScenes resolution)",
    center_freq_hz=77e9,
    bandwidth_hz=400e6,
    chirp_duration_s=19.48e-6,   # lambda/(4*v_max) at 77GHz, v_max=50m/s
    n_chirps=1024,
    n_adc_samples=512,
    tx_power_dbm=14.0,
    antenna_gain_dbi=15.0,
    noise_figure_db=12.0,
    system_loss_db=3.0,
    snr_threshold_db=10.0,
)

ALL_PRESETS = [
    PRESET_5GHZ_35M,
    PRESET_RS_MATCH_50M,
    PRESET_RS_MATCH_35M,
    PRESET_RS_MATCH_77GHZ,
]

PRESET_MAP = {
    "5ghz": PRESET_5GHZ_35M,
    "rs50": PRESET_RS_MATCH_50M,
    "rs35": PRESET_RS_MATCH_35M,
    "rs77": PRESET_RS_MATCH_77GHZ,
}

# ---------------------------------------------------------------------------
# RadarScenes reference (Continental ARS430 approximate)
# ---------------------------------------------------------------------------
RADARSCENES_REF = {
    "range_resolution_m": 0.39,
    "velocity_resolution_ms": 0.12,
    "range_max_m": 200.0,
    "velocity_max_ms": 70.0,
    "center_freq_hz": 76.5e9,
    "bandwidth_hz_approx": 400e6,
}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
_SEP = "=" * 76
_SEP2 = "-" * 76


def _fmt(val: float, unit: str, decimals: int = 3) -> str:
    return f"{val:.{decimals}f} {unit}"


def print_performance(p: FMCWPerformance) -> None:
    cfg = p.cfg
    print(_SEP)
    print(f"  CONFIG: {cfg.name}")
    print(_SEP)

    # Waveform section
    print("\n  [Waveform]")
    print(f"    Carrier frequency  f_c : {cfg.center_freq_hz/1e9:.3f} GHz")
    print(f"    Bandwidth           B  : {cfg.bandwidth_hz/1e6:.1f} MHz")
    print(f"    Chirp duration      T_c: {cfg.chirp_duration_s*1e6:.3f} µs")
    print(f"    ADC samples        N_s : {cfg.n_adc_samples}")
    print(f"    Chirps per frame   N_c : {cfg.n_chirps}")
    print(f"    ADC sample rate    F_s : {p.adc_sample_rate_hz/1e6:.2f} MHz")
    print(f"    Chirp slope         S  : {p.chirp_slope_hz_per_s/1e12:.3f} THz/s")
    print(f"    Frame duration  T_frm  : {p.frame_duration_s*1e3:.2f} ms")
    print(f"    Update rate            : {p.update_rate_hz:.1f} Hz")
    print(f"    Wavelength        lam  : {p.wavelength_m*1e3:.3f} mm")

    # Performance section
    rs = RADARSCENES_REF
    dr_match = "OK" if abs(p.range_resolution_m - rs["range_resolution_m"]) < 0.05 else "NO"
    dv_match = "OK" if abs(p.velocity_resolution_ms - rs["velocity_resolution_ms"]) < 0.03 else "~"
    rm_ok    = "OK" if p.range_max_m >= 50.0 else "NO"
    vm_ok    = "OK" if p.velocity_max_ms >= 50.0 else "NO"

    print(f"\n  [Performance vs RadarScenes target]")
    print(f"    {'Metric':<30} {'This config':>14} {'RadarScenes ref':>16}  Match")
    print(f"    {'-'*30} {'-'*14} {'-'*16}  -----")
    print(f"    {'Range resolution dR (m)':<30} "
          f"{p.range_resolution_m:>14.3f} "
          f"{rs['range_resolution_m']:>16.3f}  {dr_match}")
    print(f"    {'Velocity resolution dv (m/s)':<30} "
          f"{p.velocity_resolution_ms:>14.3f} "
          f"{rs['velocity_resolution_ms']:>16.3f}  {dv_match}")
    print(f"    {'Max range R_max (m)':<30} "
          f"{p.range_max_m:>14.1f} "
          f"{rs['range_max_m']:>16.1f}  {rm_ok}")
    print(f"    {'Max velocity v_max (m/s)':<30} "
          f"{p.velocity_max_ms:>14.1f} "
          f"{rs['velocity_max_ms']:>16.1f}  {vm_ok}")

    # Slope, IF bandwidth, range-Doppler coupling
    S_mhz_us = p.cfg.bandwidth_hz / p.cfg.chirp_duration_s / 1e12  # MHz/µs
    if S_mhz_us <= 30:
        slope_note = "any automotive chip OK"
    elif S_mhz_us <= 70:
        slope_note = "TI AWR1843 class OK"
    elif S_mhz_us <= 100:
        slope_note = "TI AWR2944 class OK"
    else:
        slope_note = "exceeds standard automotive chips — specialized hardware needed"

    coupling_bins = p.coupling_at_vmax_m / p.range_resolution_m

    print(f"\n  [Chirp slope & IF bandwidth]")
    print(f"    Chirp slope  S = B/T_c  : {S_mhz_us:.2f} MHz/µs  ({slope_note})")
    print(f"    Beat freq at R_max       : {p.beat_freq_max_hz/1e6:.2f} MHz  (= F_s/2, Nyquist-exact)")
    print(f"    IF filter bandwidth req  : {p.if_bandwidth_hz/1e6:.2f} MHz")
    print(f"")
    print(f"    Range-Doppler coupling   : ΔR = v × f_c / S")
    print(f"      Coupling factor        : {p.range_doppler_coupling_m_per_ms*1000:.4f} m per (m/s)")
    print(f"      ΔR at v_max={p.velocity_max_ms:.0f} m/s   : {p.coupling_at_vmax_m:.3f} m  "
          f"({coupling_bins:.2f} range bins)")
    print(f"      Note: 2D range-Doppler FFT corrects this; single-chirp processing does not.")

    # Feasibility notes
    print(f"\n  [Feasibility]")
    fs_mhz = p.adc_sample_rate_hz / 1e6
    if fs_mhz <= 30:
        adc_note = "standard ADC (< 30 MHz)"
    elif fs_mhz <= 100:
        adc_note = "mid-range ADC (< 100 MHz)"
    elif fs_mhz <= 300:
        adc_note = "high-speed ADC (< 300 MHz)"
    else:
        adc_note = "very high-speed ADC (> 300 MHz)"
    print(f"    ADC sample rate {fs_mhz:.1f} MHz  -> {adc_note}")
    r_x_v = p.range_max_m * p.velocity_max_ms
    print(f"    R_max x v_max = {r_x_v:.0f} m^2/s  "
          f"(fundamental tradeoff: fixed for given F_s, B, f_c)")

    # SNR table
    print(f"\n  [SNR at threshold = {cfg.snr_threshold_db} dB]")
    ranges = p.snr_ranges_m
    header = f"    {'Target':<12}" + "".join(f"  {r:>5.0f}m" for r in ranges)
    print(header)
    print(f"    {'-'*12}" + "-" * (len(ranges) * 8))
    for target, snrs in p.snr_table.items():
        row = f"    {target:<12}"
        for snr in snrs:
            flag = ">" if snr >= cfg.snr_threshold_db else "<"
            row += f"  {snr:>5.1f}{flag}"
        print(row)
    print(f"    (> = detectable, < = below threshold)")


def print_comparison_table(perfs: List[FMCWPerformance]) -> None:
    print("\n" + _SEP)
    print("  SIDE-BY-SIDE COMPARISON")
    print(_SEP)
    rs = RADARSCENES_REF
    names = [p.cfg.name.split("(")[0].strip() for p in perfs] + ["RadarScenes"]
    col_w = max(len(n) for n in names) + 2

    def row(label: str, vals: List[str]) -> str:
        return f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in vals)

    headers = [f"{n}" for n in names]
    print(row("Metric", headers))
    print("  " + "-" * (28 + col_w * len(headers)))

    def fmt_vals(attr: str, decimals: int, rs_val: float) -> List[str]:
        out = []
        for p in perfs:
            out.append(f"{getattr(p, attr):.{decimals}f}")
        out.append(f"{rs_val:.{decimals}f}")
        return out

    print(row("f_c (GHz)",
              [f"{p.cfg.center_freq_hz/1e9:.2f}" for p in perfs] + [f"{rs['center_freq_hz']/1e9:.2f}"]))
    print(row("B (MHz)",
              [f"{p.cfg.bandwidth_hz/1e6:.0f}" for p in perfs] + [f"{rs['bandwidth_hz_approx']/1e6:.0f}"]))
    print(row("dR (m)",    fmt_vals("range_resolution_m",     3, rs["range_resolution_m"])))
    print(row("R_max (m)", fmt_vals("range_max_m",            1, rs["range_max_m"])))
    print(row("v_max (m/s)", fmt_vals("velocity_max_ms",      1, rs["velocity_max_ms"])))
    print(row("dv (m/s)",  fmt_vals("velocity_resolution_ms", 3, rs["velocity_resolution_ms"])))
    print(row("F_s (MHz)", [f"{p.adc_sample_rate_hz/1e6:.1f}" for p in perfs] + ["—"]))
    print(row("S (MHz/us)",
              [f"{p.chirp_slope_hz_per_s/1e12:.1f}" for p in perfs] + ["—"]))
    print(row("IF BW (MHz)",
              [f"{p.if_bandwidth_hz/1e6:.2f}" for p in perfs] + ["—"]))
    print(row("DR coupling (m/ms)",
              [f"{p.range_doppler_coupling_m_per_ms*1000:.4f}" for p in perfs] + ["—"]))
    print(row("DR coupling @ v_max (m)",
              [f"{p.coupling_at_vmax_m:.3f}" for p in perfs] + ["—"]))
    print(row("T_frame (ms)", [f"{p.frame_duration_s*1e3:.2f}" for p in perfs] + ["—"]))
    print(row("N_s",       [str(p.cfg.n_adc_samples) for p in perfs] + ["—"]))
    print(row("N_c",       [str(p.cfg.n_chirps) for p in perfs] + ["—"]))


# ---------------------------------------------------------------------------
# Range-velocity tradeoff plot
# ---------------------------------------------------------------------------
def plot_tradeoff(
    perfs: List[FMCWPerformance],
    save_path: Optional[str] = None,
) -> None:
    try:
        import matplotlib
        if save_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FMCW Radar Tradeoff Analysis", fontsize=13, fontweight="bold")

    # ---- Panel 1: R_max vs v_max tradeoff for varying B (at fixed f_c=76GHz) ----
    ax = axes[0]
    ax.set_title("R_max vs v_max tradeoff (vary B)")
    ax.set_xlabel("Max Velocity v_max (m/s)")
    ax.set_ylabel("Max Range R_max (m)")

    B_values_mhz = [200, 400, 1000, 5000]
    T_c_vals = np.linspace(5e-6, 100e-6, 300)
    N_s = 512
    f_c = 76e9
    lam = C / f_c
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(B_values_mhz)))

    for B_mhz, color in zip(B_values_mhz, colors):
        B = B_mhz * 1e6
        v_max_arr = lam / (4 * T_c_vals)
        R_max_arr = N_s * C / (4 * B) * np.ones_like(T_c_vals)
        ax.plot(v_max_arr, R_max_arr, color=color, linewidth=1.5,
                label=f"B={B_mhz} MHz")

    for p in perfs:
        ax.scatter(p.velocity_max_ms, p.range_max_m, s=80, zorder=5,
                   label=p.cfg.name.split("(")[0].strip())

    ax.axvline(50, color="red", linestyle="--", linewidth=1, label="50 m/s target")
    ax.axhline(50, color="orange", linestyle="--", linewidth=1, label="50 m target")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 250)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Range resolution vs bandwidth ----
    ax = axes[1]
    ax.set_title("Range Resolution vs Bandwidth")
    ax.set_xlabel("Bandwidth B (MHz)")
    ax.set_ylabel("Range Resolution δR (m)")

    B_sweep = np.linspace(50e6, 6000e6, 500)
    dR_sweep = C / (2 * B_sweep)
    ax.plot(B_sweep / 1e6, dR_sweep, "b-", linewidth=2)

    ax.axhline(RADARSCENES_REF["range_resolution_m"], color="red", linestyle="--",
               linewidth=1.5, label=f"RadarScenes {RADARSCENES_REF['range_resolution_m']} m")
    for p in perfs:
        ax.scatter(p.cfg.bandwidth_hz / 1e6, p.range_resolution_m,
                   s=80, zorder=5, label=p.cfg.name.split("(")[0].strip())

    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: SNR vs range for all presets (car target) ----
    ax = axes[2]
    ax.set_title("SNR vs Range (car, σ=4 dBsm)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("SNR (dB)")

    R_arr = np.linspace(5, 120, 300)
    sigma_car_dBsm = RCS_TABLE_DBsm["car"]

    line_styles = ["-", "--", "-.", ":"]
    for i, p in enumerate(perfs):
        cfg = p.cfg
        lam_i = C / cfg.center_freq_hz
        T_frame_i = cfg.n_chirps * cfg.chirp_duration_s
        base = (
            (cfg.tx_power_dbm - 30.0)
            + 2.0 * cfg.antenna_gain_dbi
            + 20.0 * math.log10(lam_i)
            + 174.0
            + 10.0 * math.log10(T_frame_i)
            - 32.97
            - cfg.noise_figure_db
            - cfg.system_loss_db
            + sigma_car_dBsm
        )
        snr_arr = base - 40.0 * np.log10(R_arr)
        ax.plot(R_arr, snr_arr, line_styles[i % len(line_styles)],
                label=p.cfg.name.split("(")[0].strip(), linewidth=1.8)

    ax.axhline(PRESET_RS_MATCH_50M.snr_threshold_db, color="red", linestyle="--",
               linewidth=1.5, label=f"SNR threshold {PRESET_RS_MATCH_50M.snr_threshold_db} dB")
    ax.axvline(50, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="50 m")
    ax.set_xlim(5, 120)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FMCW radar configuration calculator"
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESET_MAP.keys()) + ["all"],
        default="all",
        help="Which preset to display (default: all)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show range-velocity tradeoff plots",
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save plot to file instead of displaying",
    )
    args = parser.parse_args()

    if args.preset == "all":
        selected = ALL_PRESETS
    else:
        selected = [PRESET_MAP[args.preset]]

    perfs = [compute_performance(cfg) for cfg in selected]

    for p in perfs:
        print_performance(p)
        print()

    if len(perfs) > 1:
        print_comparison_table(perfs)

    print("\n" + _SEP)
    print("  FEASIBILITY SUMMARY")
    print(_SEP)
    print("""
  Goal: match RadarScenes (δR≈0.39m, δv≈0.12m/s) with R_max≥50m, v_max≥50m/s
  at f_c = 76 GHz.

  Key insight - Fundamental range-velocity tradeoff:
    R_max x v_max = (N_s / T_c) x c x lam / (16 x B)
                  = F_s x c x lam / (16 x B)

  For fixed B=400MHz, f_c=76GHz:
    dR = c/(2B) = 0.375 m  [OK]  (matches RadarScenes 0.39m)
    To hit v_max>=50m/s: T_c <= lam/(4x50) = 19.74 us
    To hit R_max>=50m:   N_s >= 4xBx50/c = 267 -> use N_s=512 -> R_max=96m [OK]
    Resulting F_s = N_s/T_c = 25.9 MHz  -> STANDARD ADC [OK]
    dv = lam/(2xN_cxT_c) at N_c=1024 -> 0.097 m/s  approx 0.12m/s [OK]

  CONCLUSION: RadarScenes resolution at 50m range + 50m/s velocity IS FEASIBLE
  with a standard ADC at 76 GHz.  -> Use preset 'rs50'.

  B=5GHz config gives dR=0.03m (x13 finer) but requires F_s=207MHz ADC
  (high-end, available in TI AWR2944 class devices).
""")

    if args.plot:
        plot_tradeoff(perfs if len(perfs) > 1 else ALL_PRESETS,
                      save_path=args.save)


if __name__ == "__main__":
    main()
