"""Post-process a labeled radar capture to align with RadarScenes-style features.

Transforms applied in-order to radar_data_labeled.csv:

1. Pedestrian Doppler fix.
   CARLA's walker controller returns zero velocity; fix by central-differencing
   actor_frames.jsonl positions and projecting onto sensor line-of-sight.

2. RCS dBsm calibration + realism.
   rcs_proxy_m2 (geometric OBB area) -> rcs_dBsm with:
     a) Per-class median calibration shift.
     b) Per-actor Gaussian offset (inter-individual variation, persistent).
     c) Per-frame Swerling-1 fluctuation (right-skewed, replaces Gaussian jitter).
     d) Specular spikes: occasional +10-25 dB events (class-specific probability).
     e) Aspect-angle modulation: broadside view gives higher RCS than end-on,
        derived from actor yaw (rotation) stored in actor_frames.jsonl.

3. FMCW realism (RS35 preset defaults: B=400 MHz, N_s=256, f_c=76 GHz).
   Adds five new columns without removing any existing rows:
     snr_dB            - radar range-equation SNR (dB); blank if RCS unknown
     visible           - 1 if detectable, 0 if filtered (range / velocity / SNR)
     depth_m_noisy     - range with SNR-scaled Gaussian noise (Cramer-Rao)
     velocity_mps_noisy- Doppler with SNR-scaled Gaussian noise
     azimuth_rad_noisy - azimuth with SNR-scaled Gaussian noise

Importable entry point:
    from capture.PostProcessDataset import post_process_capture_dir
    post_process_capture_dir(run_dir, seed=42)

CLI:
    python -m capture.PostProcessDataset --capture-dir <path>
    python -m capture.PostProcessDataset --input <csv-path>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc)
_dc.bootstrap(__file__)

LABELED_CSV = "radar_data_labeled.csv"
ACTOR_FRAMES_JSONL = "actor_frames.jsonl"

# ---------------------------------------------------------------------------
# RCS calibration defaults
# ---------------------------------------------------------------------------
# Targets are the EMPIRICAL per-class median dBsm measured in RadarScenes (relative/
# uncalibrated dBsm — comparable within-dataset only). We match these so our dynamic RCS
# sits on the SAME scale as the static distribution ([[static_rcs_inverse_cdf]]), i.e.
# static-vs-dynamic RCS overlaps near-totally (RadarScenes OVL 0.86, AUC 0.55) and RCS is
# NOT an artificial static/dynamic shortcut — Doppler (static |v|≈0) is the real separator.
# (Earlier values were ABSOLUTE physical RCS from literature — car +7, truck +15 — which put
# our dynamics ~13-14 dB above the static bulk and made RCS a fake separator: OVL 0.67/AUC 0.68.)
DEFAULT_TARGET_MEDIAN_DBSM = {
    "pedestrian": -7.9,   # RadarScenes pedestrian
    "car":        -4.9,   # RadarScenes car
    "truck":       1.3,   # RadarScenes truck
    "bus":        -2.5,   # RadarScenes bus
    "bicycle":   -11.2,   # RadarScenes bike
    "motorcycle": -9.0,   # not in RadarScenes; between bike and ped (moot: 4-wheel filter spawns none)
}

# Two-scale noise sigmas: (per-actor Gaussian offset, per-frame Swerling-1 scale).
# Per-actor: inter-individual variation (e.g. different car sizes/models), persistent
#   across all frames — GNN sees a stable per-target signature.
# Per-frame: rapid RCS fluctuation due to scintillation / target motion.
#   Modeled with Swerling-1 (exponential power distribution -> right-skewed dB).
#   sigma_swerling gives the desired approx. spread; internally scaled to the
#   theoretical Swerling-1 standard deviation (~5.57 dB at unit scale).
DEFAULT_RCS_DB_NOISE = {
    # class:       (sigma_actor_dB, sigma_swerling_dB)
    "pedestrian":  (3.0, 4.0),
    "car":         (2.5, 3.5),
    "truck":       (2.0, 3.0),
    "bus":         (2.0, 3.0),
    "bicycle":     (2.5, 3.0),
    "motorcycle":  (2.0, 2.5),
}

# Swerling-1 theoretical standard deviation in dB (unit scale = 1.0):
# sigma = pi/sqrt(6) * 10/ln(10)  ~= 5.57 dB
_SWERLING_SIGMA_DB = math.pi / math.sqrt(6.0) * 10.0 / math.log(10.0)

# Specular spike model: occasional large RCS jumps due to coherent reflections
# (metallic corners, flat surfaces at normal incidence, etc.)
DEFAULT_SPIKE_PROB = {
    "pedestrian":  0.02,
    "car":         0.05,
    "truck":       0.04,
    "bus":         0.04,
    "bicycle":     0.03,
    "motorcycle":  0.03,
}
DEFAULT_SPIKE_DB_LO = 10.0
DEFAULT_SPIKE_DB_HI = 25.0

# Aspect-angle modulation: RCS is higher when the radar sees a large projected area
# (broadside view) and lower for end-on views (front/rear).
# Model:  delta_dB = amplitude * (sin^2(alpha) - 0.5)
#   alpha = angle between actor heading and sensor line-of-sight
#   delta_dB = 0 when averaged over uniform aspect angles (unbiased)
#   Range: [-amplitude/2, +amplitude/2] dB
DEFAULT_ASPECT_AMPLITUDE_DB = {
    "pedestrian":  4.0,
    "car":         8.0,
    "truck":      10.0,
    "bus":         9.0,
    "bicycle":     5.0,
    "motorcycle":  5.0,
}

DEFAULT_MICRO_DOPPLER_SIGMA = 1.0
DEFAULT_FD_STRIDE = 10
DEFAULT_MAX_WALKER_SPEED_MPS = 3.0

# Static/clutter RCS — sampled from the RadarScenes static class (label 11) distribution
# (115M detections). Static returns otherwise carry NO rcs_dBsm, which lets RCS
# presence-vs-absence trivially separate static from dynamic. Knots = (cumulative prob,
# dBsm percentile); piecewise-linear inverse CDF with a HARD floor at -30.6 (the sensor
# CFAR/detection cutoff — a censored edge, not physics, so never extrapolate below it),
# right-skewed bright tail to +58.7 (specular metal). Sampled once per world voxel so a
# given reflector keeps a consistent RCS across frames (no flicker).
RADARSCENES_STATIC_RCS_CDF = (
    (0.000, -30.60), (0.001, -29.50), (0.010, -26.53), (0.050, -23.33),
    (0.250, -15.18), (0.500, -7.38), (0.750, 1.57), (0.950, 16.16),
    (0.990, 26.55), (0.999, 37.40), (1.000, 58.71),
)
DEFAULT_STATIC_RCS_VOXEL_M = 1.0


def static_rcs_inverse_cdf(u: float) -> float:
    """Map u in [0,1] to a dBsm value via the RadarScenes static percentile knots."""
    knots = RADARSCENES_STATIC_RCS_CDF
    if u <= knots[0][0]:
        return knots[0][1]
    for (p0, v0), (p1, v1) in zip(knots, knots[1:]):
        if u <= p1:
            t = (u - p0) / (p1 - p0) if p1 > p0 else 0.0
            return v0 + t * (v1 - v0)
    return knots[-1][1]

# ---------------------------------------------------------------------------
# FMCW realism defaults  (RS35 preset: B=400 MHz, N_s=256, f_c=76 GHz)
# See DatasetCreation/tools/fmcw_radar_config.py for derivation.
# Derived performance:
#   range_resolution = c/(2B)       = 0.375 m
#   range_max        = N_s*c/(4B)   = 48.0 m
#   velocity_max     = lam/(4*T_c)  = 50.0 m/s
#   doppler_res      = lam/(2*N_c*T_c) = 0.097 m/s
# ---------------------------------------------------------------------------
_C_LIGHT = 3e8

DEFAULT_FMCW_CENTER_FREQ_HZ     = 76e9
DEFAULT_FMCW_BANDWIDTH_HZ       = 400e6
DEFAULT_FMCW_CHIRP_DURATION_S   = 19.74e-6
DEFAULT_FMCW_N_CHIRPS           = 1024
DEFAULT_FMCW_N_ADC_SAMPLES      = 256
DEFAULT_FMCW_TX_POWER_DBM       = 14.0
DEFAULT_FMCW_ANTENNA_GAIN_DBI   = 15.0
DEFAULT_FMCW_NOISE_FIGURE_DB    = 12.0
DEFAULT_FMCW_SYSTEM_LOSS_DB     = 3.0
DEFAULT_FMCW_SNR_THRESHOLD_DB   = 10.0
# Static class: NO SNR re-gating. Static rcs_dBsm is sampled from the RadarScenes static
# distribution, which is ALREADY post-CFAR (its -30.6 dBsm floor IS the detector cutoff).
# Re-applying our SNR gate would double-censor and bias the detected-static RCS high
# (median +0.3 instead of -7.4). -inf threshold = pass all sampled static returns; range/
# velocity/P_d gates still apply. Dynamic returns keep the physical 10 dB threshold.
DEFAULT_FMCW_SNR_THRESHOLD_STATIC_DB = float("-inf")
DEFAULT_FMCW_P_DETECT           = 0.9
DEFAULT_FMCW_AZ_SIGMA_0_DEG     = 6.0
DEFAULT_FMCW_AZ_SIGMA_FLOOR_DEG = 0.3


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_walker_positions(frames_path: Path) -> dict:
    """Return {(frame_id, actor_id): (x, y, z)} for pedestrians only."""
    walker_pos: dict = {}
    with frames_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            frame = int(rec["frame"])
            for a in rec.get("actors", []):
                if a.get("kind") != "pedestrian":
                    continue
                loc = a["location"]
                walker_pos[(frame, int(a["id"]))] = (
                    float(loc["x"]),
                    float(loc["y"]),
                    float(loc["z"]),
                )
    return walker_pos


def load_actor_transforms(frames_path: Path) -> dict:
    """Return {(frame_id, actor_id): (x, y, z, yaw_deg)} for ALL actors.

    Used for aspect-angle RCS modulation. yaw_deg is the actor's world heading
    in CARLA's left-hand coordinate system (0 = +X direction, clockwise positive).
    """
    transforms: dict = {}
    with frames_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            frame = int(rec["frame"])
            for a in rec.get("actors", []):
                loc = a.get("location") or {}
                rot = a.get("rotation") or {}
                transforms[(frame, int(a["id"]))] = (
                    float(loc.get("x", 0.0)),
                    float(loc.get("y", 0.0)),
                    float(loc.get("z", 0.0)),
                    float(rot.get("yaw", 0.0)),
                )
    return transforms


def estimate_frame_dt(labeled_path: Path) -> float:
    """Mean simulation tick interval (s) from CSV frame/timestamp pairs."""
    seen: dict = {}
    with labeled_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = int(row["frame"])
            if fr not in seen:
                seen[fr] = float(row["timestamp"])
    if len(seen) < 2:
        return 1.0 / 30.0
    frames = sorted(seen.keys())
    f0, fN = frames[0], frames[-1]
    return (seen[fN] - seen[f0]) / (fN - f0)


def class_offsets_from_data(labeled_path: Path, target_median_dbsm: dict) -> tuple:
    """Per-class dB offsets so median(rcs_proxy_m2)->dBsm hits target."""
    rcs_by_class: dict = defaultdict(list)
    with labeled_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = row["rcs_proxy_m2"].strip()
            klass = row["matched_actor_class"].strip()
            if not r or not klass:
                continue
            try:
                v = float(r)
            except ValueError:
                continue
            if v > 0:
                rcs_by_class[klass].append(v)

    offsets: dict = {}
    info: dict = {}
    for klass, xs in rcs_by_class.items():
        med = _median(xs)
        if med is None or med <= 0:
            continue
        cur_dbsm = 10.0 * math.log10(med)
        tgt = target_median_dbsm.get(klass)
        if tgt is None:
            continue
        offsets[klass] = tgt - cur_dbsm
        info[klass] = (med, cur_dbsm, tgt, offsets[klass], len(xs))
    return offsets, info


# ---------------------------------------------------------------------------
# RCS noise helpers
# ---------------------------------------------------------------------------
def _median(xs: list) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _swerling1_dB(rng_local: random.Random, sigma_dB: float) -> float:
    """Sample a zero-median Swerling-1 fluctuation scaled to approx sigma_dB.

    Swerling-1 models power ~ Exp(mean): each independent look draws a fresh
    RCS sample from an exponential distribution.  In dB space this is:
        X = 10 * log10(-ln(U) / ln(2))   (U ~ Uniform(0,1))
    giving median=0 and std~5.57 dB (right-skewed: occasional high values,
    concentrated near/below median).

    We scale X by sigma_dB / 5.57 to match the desired per-class spread while
    preserving the right-skewed shape characteristic of Swerling targets.
    """
    U = max(rng_local.random(), 1e-10)
    unit_sample = 10.0 * math.log10(-math.log(U) / math.log(2.0))
    return (sigma_dB / _SWERLING_SIGMA_DB) * unit_sample


def _aspect_delta_dB(
    sensor_x: float,
    sensor_y: float,
    actor_x: float,
    actor_y: float,
    actor_yaw_deg: float,
    amplitude_dB: float,
) -> float:
    """RCS change (dB) from aspect-angle geometry.

    Model:  delta_dB = amplitude * (sin^2(alpha) - 0.5)
      alpha  = angle between actor heading and sensor line-of-sight
      Result is zero on average over uniform heading angles (unbiased).
      Range: [-amplitude/2, +amplitude/2] dB

    Broadside view (alpha = 90 deg): delta = +amplitude/2  (large projected area)
    End-on view   (alpha = 0/180 deg): delta = -amplitude/2 (narrow cross-section)
    """
    los_x = actor_x - sensor_x
    los_y = actor_y - sensor_y
    los_len = math.sqrt(los_x * los_x + los_y * los_y)
    if los_len < 0.1:
        return 0.0
    los_x /= los_len
    los_y /= los_len

    yaw_rad = math.radians(actor_yaw_deg)
    heading_x = math.cos(yaw_rad)
    heading_y = math.sin(yaw_rad)

    cos_alpha = heading_x * los_x + heading_y * los_y
    sin2_alpha = 1.0 - cos_alpha * cos_alpha

    return amplitude_dB * (sin2_alpha - 0.5)


def _fmcw_performance(
    center_freq_hz: float,
    bandwidth_hz: float,
    chirp_duration_s: float,
    n_adc_samples: int,
    n_chirps: int,
    tx_power_dbm: float,
    antenna_gain_dbi: float,
    noise_figure_db: float,
    system_loss_db: float,
    az_sigma_0_deg: float,
    az_sigma_floor_deg: float,
) -> dict:
    lam     = _C_LIGHT / center_freq_hz
    T_frame = n_chirps * chirp_duration_s
    path_base = (
        (tx_power_dbm - 30.0)
        + 2.0 * antenna_gain_dbi
        + 20.0 * math.log10(lam)
        + 174.0
        + 10.0 * math.log10(T_frame)
        - 32.97
        - noise_figure_db
        - system_loss_db
    )
    return {
        "lam": lam,
        "T_frame": T_frame,
        "range_resolution_m":     _C_LIGHT / (2.0 * bandwidth_hz),
        "range_max_m":            n_adc_samples * _C_LIGHT / (4.0 * bandwidth_hz),
        "velocity_max_ms":        lam / (4.0 * chirp_duration_s),
        "velocity_resolution_ms": lam / (2.0 * n_chirps * chirp_duration_s),
        "path_base_dB":           path_base,
        "az_sigma_0_rad":         math.radians(az_sigma_0_deg),
        "az_sigma_floor_rad":     math.radians(az_sigma_floor_deg),
    }


# ---------------------------------------------------------------------------
# Core post-processing (importable entry point)
# ---------------------------------------------------------------------------
def post_process_capture_dir(
    capture_dir=None,
    *,
    input_path=None,
    micro_doppler_sigma: float = DEFAULT_MICRO_DOPPLER_SIGMA,
    fd_stride: int = DEFAULT_FD_STRIDE,
    max_walker_speed: float = DEFAULT_MAX_WALKER_SPEED_MPS,
    seed: int = 0,
    rcs_db_noise_enabled: bool = True,
    rcs_spikes_enabled: bool = True,
    rcs_aspect_enabled: bool = True,
    no_fmcw_realism: bool = False,
    fmcw_center_freq_hz: float = DEFAULT_FMCW_CENTER_FREQ_HZ,
    fmcw_bandwidth_hz: float = DEFAULT_FMCW_BANDWIDTH_HZ,
    fmcw_chirp_duration_s: float = DEFAULT_FMCW_CHIRP_DURATION_S,
    fmcw_n_chirps: int = DEFAULT_FMCW_N_CHIRPS,
    fmcw_n_adc_samples: int = DEFAULT_FMCW_N_ADC_SAMPLES,
    fmcw_tx_power_dbm: float = DEFAULT_FMCW_TX_POWER_DBM,
    fmcw_antenna_gain_dbi: float = DEFAULT_FMCW_ANTENNA_GAIN_DBI,
    fmcw_noise_figure_db: float = DEFAULT_FMCW_NOISE_FIGURE_DB,
    fmcw_system_loss_db: float = DEFAULT_FMCW_SYSTEM_LOSS_DB,
    fmcw_snr_threshold_db: float = DEFAULT_FMCW_SNR_THRESHOLD_DB,
    fmcw_snr_threshold_static_db: float = DEFAULT_FMCW_SNR_THRESHOLD_STATIC_DB,
    fmcw_p_detect: float = DEFAULT_FMCW_P_DETECT,
    fmcw_az_sigma_0_deg: float = DEFAULT_FMCW_AZ_SIGMA_0_DEG,
    fmcw_az_sigma_floor_deg: float = DEFAULT_FMCW_AZ_SIGMA_FLOOR_DEG,
    out=None,
) -> Path:
    """Pedestrian-Doppler fix + RCS realism + FMCW post-process.

    Importable entry point; also invoked by CaptureRadarCameraData after
    capture when DATASET_POSTPROCESS_AFTER_CAPTURE=1.
    Returns the output Path; raises FileNotFoundError / RuntimeError on
    bad input.

    Either capture_dir (directory containing radar_data_labeled.csv and
    actor_frames.jsonl) or input_path (direct CSV path) must be supplied.
    When only input_path is given, pedestrian Doppler fix and aspect-angle
    modulation are skipped (no actor_frames.jsonl available).
    """
    if input_path is not None:
        labeled = Path(input_path)
        frames_path = None
    elif capture_dir is not None:
        capture_dir = Path(capture_dir)
        labeled     = capture_dir / LABELED_CSV
        frames_path = capture_dir / ACTOR_FRAMES_JSONL
    else:
        raise ValueError("Provide capture_dir or input_path.")

    out = Path(out) if out is not None else labeled

    if not labeled.is_file():
        raise FileNotFoundError(f"Missing {labeled}")

    has_frames = frames_path is not None and frames_path.is_file()
    if not has_frames:
        if frames_path is not None:
            print(f"  WARNING: {frames_path} not found — "
                  "skipping pedestrian Doppler fix and aspect-angle modulation.",
                  flush=True)
        else:
            print("  No capture-dir given — "
                  "skipping pedestrian Doppler fix and aspect-angle modulation.",
                  flush=True)
        rcs_aspect_enabled = False

    # ------------------------------------------------------------------ [1/4]
    if has_frames:
        print(f"[1/4] Loading actor data from {frames_path.name} ...", flush=True)  # type: ignore[union-attr]
        walker_pos = load_walker_positions(frames_path)  # type: ignore[arg-type]
        walker_actors = {aid for (_, aid) in walker_pos}
        print(f"      {len(walker_pos):,} (frame, walker) entries; "
              f"{len(walker_actors)} distinct walkers", flush=True)

        actor_transforms: dict = {}
        if rcs_aspect_enabled:
            actor_transforms = load_actor_transforms(frames_path)  # type: ignore[arg-type]
            print(f"      {len(actor_transforms):,} (frame, actor) transforms loaded "
                  f"for aspect-angle modulation", flush=True)
    else:
        print("[1/4] Skipping actor data load (no actor_frames.jsonl).", flush=True)
        walker_pos = {}
        walker_actors: set = set()
        actor_transforms = {}

    # ------------------------------------------------------------------ [2/4]
    print(f"[2/4] Estimating mean dt from {labeled.name} ...", flush=True)
    dt_mean = estimate_frame_dt(labeled)
    print(f"      dt ~ {dt_mean * 1000:.2f} ms  (~{1.0/max(dt_mean, 1e-9):.0f} Hz)",
          flush=True)

    # ------------------------------------------------------------------ [3/4]
    print("[3/4] Calibrating RCS dBsm offsets ...", flush=True)
    offsets, info = class_offsets_from_data(labeled, DEFAULT_TARGET_MEDIAN_DBSM)
    if not offsets:
        raise RuntimeError("No matched rows with rcs_proxy_m2 — nothing to calibrate.")
    for klass, (med_m2, cur_db, tgt_db, off, n) in info.items():
        print(f"      {klass:<12} n={n:>7,}  median(m2)={med_m2:.4f}  "
              f"cur_dBsm={cur_db:+.2f}  tgt_dBsm={tgt_db:+.2f}  "
              f"offset={off:+.2f} dB", flush=True)

    noise_desc = []
    if rcs_db_noise_enabled:
        noise_desc.append("Swerling-1 per-frame + Gaussian per-actor")
    if rcs_spikes_enabled:
        noise_desc.append(f"specular spikes [{DEFAULT_SPIKE_DB_LO:.0f}"
                          f"-{DEFAULT_SPIKE_DB_HI:.0f} dB]")
    if rcs_aspect_enabled:
        noise_desc.append("aspect-angle modulation")
    print(f"      RCS realism: {' | '.join(noise_desc) if noise_desc else 'disabled'}",
          flush=True)

    # FMCW performance (derived once, used per-row)
    fmcw: dict | None = None
    if not no_fmcw_realism:
        fmcw = _fmcw_performance(
            fmcw_center_freq_hz, fmcw_bandwidth_hz, fmcw_chirp_duration_s,
            fmcw_n_adc_samples, fmcw_n_chirps,
            fmcw_tx_power_dbm, fmcw_antenna_gain_dbi,
            fmcw_noise_figure_db, fmcw_system_loss_db,
            fmcw_az_sigma_0_deg, fmcw_az_sigma_floor_deg,
        )
        print(f"\n  FMCW realism (RS35-like preset):", flush=True)
        print(f"    f_c={fmcw_center_freq_hz/1e9:.3f} GHz  "
              f"B={fmcw_bandwidth_hz/1e6:.0f} MHz  "
              f"N_s={fmcw_n_adc_samples}  N_c={fmcw_n_chirps}", flush=True)
        print(f"    dR={fmcw['range_resolution_m']:.3f} m  "
              f"R_max={fmcw['range_max_m']:.1f} m  "
              f"v_max={fmcw['velocity_max_ms']:.1f} m/s  "
              f"dv={fmcw['velocity_resolution_ms']:.3f} m/s", flush=True)
        print(f"    SNR_thr={fmcw_snr_threshold_db:.1f} dB  "
              f"P_det={fmcw_p_detect:.2f}  "
              f"az_sigma_0={fmcw_az_sigma_0_deg:.1f} deg  "
              f"az_floor={fmcw_az_sigma_floor_deg:.1f} deg", flush=True)

    # ------------------------------------------------------------------ [4/4]
    print(f"\n[4/4] Rewriting {out.name} ...", flush=True)

    rng = random.Random(seed)
    fd  = fd_stride

    actor_offset_cache: dict = {}
    frame_offset_cache: dict = {}

    def _actor_offset(klass: str, actor_id: int) -> float:
        """Persistent Gaussian offset representing inter-individual RCS variation."""
        sigmas = DEFAULT_RCS_DB_NOISE.get(klass)
        if sigmas is None:
            return 0.0
        sa = sigmas[0]
        key = (klass, actor_id)
        if key not in actor_offset_cache:
            actor_offset_cache[key] = random.Random(
                f"{seed}|{klass}|{actor_id}|actor"
            ).gauss(0.0, sa)
        return actor_offset_cache[key]

    def _frame_fluctuation(klass: str, actor_id: int, frame_id: int) -> float:
        """Swerling-1 per-frame RCS fluctuation (independent each frame)."""
        sigmas = DEFAULT_RCS_DB_NOISE.get(klass)
        if sigmas is None:
            return 0.0
        sf = sigmas[1]
        key = (klass, actor_id, frame_id)
        if key not in frame_offset_cache:
            local_rng = random.Random(f"{seed}|{klass}|{actor_id}|{frame_id}")
            frame_offset_cache[key] = _swerling1_dB(local_rng, sf)
        return frame_offset_cache[key]

    # Static/clutter RCS, sampled once per world voxel (per-reflector correlation).
    static_rcs_cache: dict = {}

    def _static_rcs_dbsm(wx: float, wy: float, wz: float) -> float:
        vx = int(math.floor(wx / DEFAULT_STATIC_RCS_VOXEL_M))
        vy = int(math.floor(wy / DEFAULT_STATIC_RCS_VOXEL_M))
        vz = int(math.floor(wz / DEFAULT_STATIC_RCS_VOXEL_M))
        key = (vx, vy, vz)
        if key not in static_rcs_cache:
            u = random.Random(f"{seed}|static|{vx}|{vy}|{vz}").random()
            static_rcs_cache[key] = static_rcs_inverse_cdf(u)
        return static_rcs_cache[key]

    stats = {
        "walker_fixed": 0, "walker_skipped": 0, "walker_clamped": 0,
        "rcs_written": 0, "static_rcs": 0, "spikes": 0,
        "snr_written": 0, "visible_1": 0,
        "invis_range": 0, "invis_vel": 0, "invis_snr": 0,
    }
    max_walker_speed = max(0.1, float(max_walker_speed))

    FMCW_COLS = ["snr_dB", "visible",
                 "depth_m_noisy", "velocity_mps_noisy", "azimuth_rad_noisy"]

    tmp = out.with_suffix(out.suffix + ".tmp")
    with labeled.open(newline="", encoding="utf-8") as fin:
        reader    = csv.DictReader(fin)
        in_fields = list(reader.fieldnames or [])
        out_fields = list(in_fields)
        if "rcs_dBsm" not in out_fields:
            out_fields.append("rcs_dBsm")
        if fmcw is not None:
            for col in FMCW_COLS:
                if col not in out_fields:
                    out_fields.append(col)

        with tmp.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                aid_raw  = row["matched_actor_id"].strip()
                kind     = row["matched_actor_kind"].strip()
                frame_id = int(row["frame"])

                # --------------------------------------------------------
                # [A] Pedestrian velocity fix (bulk + micro-Doppler)
                # --------------------------------------------------------
                if aid_raw and kind == "pedestrian":
                    try:
                        aid = int(aid_raw)
                    except ValueError:
                        aid = None
                    if aid is not None:
                        p_prev = walker_pos.get((frame_id - fd, aid))
                        p_next = walker_pos.get((frame_id + fd, aid))
                        p_now  = walker_pos.get((frame_id, aid))
                        if p_prev and p_next and p_now:
                            dt  = 2.0 * fd * dt_mean
                            vx  = (p_next[0] - p_prev[0]) / dt
                            vy  = (p_next[1] - p_prev[1]) / dt
                            vz  = (p_next[2] - p_prev[2]) / dt
                            spd = math.sqrt(vx*vx + vy*vy + vz*vz)
                            if spd > max_walker_speed:
                                scale = max_walker_speed / spd
                                vx *= scale
                                vy *= scale
                                vz *= scale
                                stats["walker_clamped"] += 1
                            sx = float(row["sensor_world_x_m"])
                            sy = float(row["sensor_world_y_m"])
                            sz = float(row["sensor_world_z_m"])
                            ux = p_now[0] - sx
                            uy = p_now[1] - sy
                            uz = p_now[2] - sz
                            n  = math.sqrt(ux*ux + uy*uy + uz*uz)
                            if n > 1e-6:
                                ux /= n
                                uy /= n
                                uz /= n
                                v_rad = vx*ux + vy*uy + vz*uz
                                if micro_doppler_sigma > 0:
                                    v_rad += rng.gauss(0.0, micro_doppler_sigma)
                                row["velocity_mps"] = f"{v_rad:.6f}"
                                stats["walker_fixed"] += 1
                            else:
                                stats["walker_skipped"] += 1
                        else:
                            stats["walker_skipped"] += 1

                # --------------------------------------------------------
                # [B] RCS dBsm calibration + realism
                # --------------------------------------------------------
                r_raw = row["rcs_proxy_m2"].strip()
                klass = row["matched_actor_class"].strip()
                rcs_dBsm_val: float | None = None

                if r_raw and klass in offsets:
                    try:
                        v = float(r_raw)
                        if v > 0:
                            dbsm = 10.0 * math.log10(v) + offsets[klass]

                            if rcs_db_noise_enabled and aid_raw:
                                try:
                                    actor_id_int = int(aid_raw)
                                    dbsm += _actor_offset(klass, actor_id_int)
                                    dbsm += _frame_fluctuation(
                                        klass, actor_id_int, frame_id
                                    )
                                except ValueError:
                                    pass

                            if rcs_spikes_enabled and aid_raw:
                                spike_p = DEFAULT_SPIKE_PROB.get(klass, 0.0)
                                spike_rng = random.Random(
                                    f"{seed}|{klass}|{aid_raw}|{frame_id}|spike"
                                )
                                if spike_rng.random() < spike_p:
                                    dbsm += spike_rng.uniform(
                                        DEFAULT_SPIKE_DB_LO, DEFAULT_SPIKE_DB_HI
                                    )
                                    stats["spikes"] += 1

                            if rcs_aspect_enabled and aid_raw:
                                try:
                                    tf = actor_transforms.get(
                                        (frame_id, int(aid_raw))
                                    )
                                    if tf is not None:
                                        ax, ay, _az, actor_yaw = tf
                                        sx = float(row["sensor_world_x_m"])
                                        sy = float(row["sensor_world_y_m"])
                                        amp = DEFAULT_ASPECT_AMPLITUDE_DB.get(klass, 0.0)
                                        dbsm += _aspect_delta_dB(
                                            sx, sy, ax, ay, actor_yaw, amp
                                        )
                                except (ValueError, TypeError):
                                    pass

                            row["rcs_dBsm"] = f"{dbsm:.4f}"
                            rcs_dBsm_val = dbsm
                            stats["rcs_written"] += 1
                        else:
                            row["rcs_dBsm"] = ""
                    except ValueError:
                        row["rcs_dBsm"] = ""
                elif kind in ("vehicle", "pedestrian"):
                    # Dynamic actor without a usable rcs_proxy_m2 — leave blank.
                    row["rcs_dBsm"] = ""
                else:
                    # Static/clutter: sample the RadarScenes static distribution,
                    # correlated per reflector (world-space voxel). Set rcs_dBsm_val so
                    # the FMCW stage below gates static returns by SNR like dynamic ones.
                    row["rcs_dBsm"] = ""
                    try:
                        depth = float(row["depth_m"])
                        az    = float(row["azimuth_rad"])
                        alt   = float(row["altitude_rad"])
                        sx    = float(row["sensor_world_x_m"])
                        sy    = float(row["sensor_world_y_m"])
                        sz    = float(row["sensor_world_z_m"])
                        yaw   = math.radians(float(row["sensor_yaw_deg"]))
                        ca = math.cos(alt)
                        lx = depth * ca * math.cos(az)
                        ly = depth * ca * math.sin(az)
                        lz = depth * math.sin(alt)
                        wx = sx + lx * math.cos(yaw) - ly * math.sin(yaw)
                        wy = sy + lx * math.sin(yaw) + ly * math.cos(yaw)
                        wz = sz + lz
                        sdbsm = _static_rcs_dbsm(wx, wy, wz)
                        row["rcs_dBsm"] = f"{sdbsm:.4f}"
                        rcs_dBsm_val = sdbsm
                        stats["static_rcs"] += 1
                    except (ValueError, KeyError):
                        pass

                # --------------------------------------------------------
                # [C] FMCW realism: SNR, visibility flag, noisy measurements
                # --------------------------------------------------------
                if fmcw is not None:
                    depth_m = float(row["depth_m"])
                    vel_mps = float(row["velocity_mps"])
                    az_rad  = float(row["azimuth_rad"])

                    range_ok    = depth_m <= fmcw["range_max_m"]
                    velocity_ok = abs(vel_mps) <= fmcw["velocity_max_ms"]

                    snr_dB_out    = ""
                    snr_ok        = True
                    snr_lin_noise = 1.0

                    if rcs_dBsm_val is not None and depth_m > 0.01:
                        snr_dB_val = (
                            fmcw["path_base_dB"]
                            + rcs_dBsm_val
                            - 40.0 * math.log10(depth_m)
                        )
                        snr_dB_out    = f"{snr_dB_val:.4f}"
                        snr_lin_noise = max(10.0 ** (snr_dB_val / 10.0), 0.01)
                        # Static returns bypass SNR gating (their RCS is already a
                        # post-CFAR sample); dynamics keep the physical threshold.
                        thr = (fmcw_snr_threshold_db
                               if kind in ("vehicle", "pedestrian")
                               else fmcw_snr_threshold_static_db)
                        if snr_dB_val >= thr:
                            snr_ok = rng.random() < fmcw_p_detect
                        else:
                            snr_ok = False
                        stats["snr_written"] += 1

                    visible = 1 if (range_ok and velocity_ok and snr_ok) else 0
                    if visible:
                        stats["visible_1"] += 1
                    else:
                        if not range_ok:    stats["invis_range"] += 1
                        if not velocity_ok: stats["invis_vel"]   += 1
                        if not snr_ok:      stats["invis_snr"]   += 1

                    sqrt_2snr = math.sqrt(2.0 * snr_lin_noise)
                    sqrt_snr  = math.sqrt(snr_lin_noise)
                    sig_r  = fmcw["range_resolution_m"]     / sqrt_2snr
                    sig_v  = fmcw["velocity_resolution_ms"] / sqrt_2snr
                    sig_az = max(
                        fmcw["az_sigma_0_rad"] / sqrt_snr,
                        fmcw["az_sigma_floor_rad"],
                    )

                    row["snr_dB"]             = snr_dB_out
                    row["visible"]            = str(visible)
                    row["depth_m_noisy"]      = f"{depth_m + rng.gauss(0.0, sig_r):.6f}"
                    row["velocity_mps_noisy"] = f"{vel_mps + rng.gauss(0.0, sig_v):.6f}"
                    row["azimuth_rad_noisy"]  = f"{az_rad  + rng.gauss(0.0, sig_az):.6f}"

                writer.writerow(row)

    tmp.replace(out)

    print(flush=True)
    print(f"  Walker velocities fixed     : {stats['walker_fixed']:,}", flush=True)
    print(f"  Walker rows skipped         : {stats['walker_skipped']:,}", flush=True)
    if stats["walker_clamped"]:
        pct = 100.0 * stats["walker_clamped"] / max(stats["walker_fixed"], 1)
        print(f"  Walker speeds clamped       : {stats['walker_clamped']:,} "
              f"({pct:.2f}% above {max_walker_speed:.1f} m/s)", flush=True)
    print(f"  rcs_dBsm cells written      : {stats['rcs_written']:,}", flush=True)
    print(f"  static rcs_dBsm (RadarScenes): {stats['static_rcs']:,} "
          f"({len(static_rcs_cache):,} reflector voxels)", flush=True)
    print(f"  Specular spikes added       : {stats['spikes']:,} "
          f"({100.0*stats['spikes']/max(stats['rcs_written'],1):.2f}% of matched rows)",
          flush=True)

    if fmcw is not None:
        total = stats["visible_1"] + stats["invis_range"] + stats["invis_vel"] + stats["invis_snr"]
        pct_v = 100.0 * stats["visible_1"] / max(total, 1)
        print(f"  snr_dB cells written        : {stats['snr_written']:,}", flush=True)
        print(f"  visible=1                   : {stats['visible_1']:,} ({pct_v:.1f}%)",
              flush=True)
        print(f"  invisible (range > R_max)   : {stats['invis_range']:,}", flush=True)
        print(f"  invisible (|v| > v_max)     : {stats['invis_vel']:,}", flush=True)
        print(f"  invisible (SNR / P_d)       : {stats['invis_snr']:,}", flush=True)

    print(f"Done -> {out}", flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--capture-dir", type=Path, default=None,
                   help="Directory containing radar_data_labeled.csv and "
                        "actor_frames.jsonl.")
    p.add_argument("--input", type=Path, default=None,
                   help="Direct path to radar_data_labeled.csv "
                        "(actor frames unavailable => pedestrian Doppler fix "
                        "and aspect-angle modulation are skipped automatically).")

    # -- pedestrian Doppler fix --
    p.add_argument("--micro-doppler-sigma", type=float,
                   default=DEFAULT_MICRO_DOPPLER_SIGMA,
                   help="Gaussian stddev (m/s) added to bulk pedestrian Doppler. "
                        "0 disables.")
    p.add_argument("--fd-stride", type=int, default=DEFAULT_FD_STRIDE,
                   help="Central-difference stride in world ticks for walker velocity.")
    p.add_argument("--max-walker-speed", type=float,
                   default=DEFAULT_MAX_WALKER_SPEED_MPS,
                   help="Hard ceiling on bulk walker speed (m/s).")

    # -- RCS calibration --
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rcs-db-noise", action="store_true", default=True,
                   help="Add Swerling-1 per-frame + per-actor offset noise to rcs_dBsm.")
    p.add_argument("--no-rcs-db-noise", dest="rcs_db_noise", action="store_false",
                   help="Disable all RCS noise (median-only calibration).")
    p.add_argument("--rcs-spikes", action="store_true", default=True,
                   help="Add specular spike events to rcs_dBsm.")
    p.add_argument("--no-rcs-spikes", dest="rcs_spikes", action="store_false",
                   help="Disable specular spikes.")
    p.add_argument("--rcs-aspect", action="store_true", default=True,
                   help="Apply aspect-angle RCS modulation.")
    p.add_argument("--no-rcs-aspect", dest="rcs_aspect", action="store_false",
                   help="Disable aspect-angle modulation.")

    # -- FMCW realism --
    p.add_argument("--fmcw-center-freq-hz",    type=float, default=DEFAULT_FMCW_CENTER_FREQ_HZ)
    p.add_argument("--fmcw-bandwidth-hz",       type=float, default=DEFAULT_FMCW_BANDWIDTH_HZ)
    p.add_argument("--fmcw-chirp-duration-s",   type=float, default=DEFAULT_FMCW_CHIRP_DURATION_S)
    p.add_argument("--fmcw-n-chirps",           type=int,   default=DEFAULT_FMCW_N_CHIRPS)
    p.add_argument("--fmcw-n-adc-samples",      type=int,   default=DEFAULT_FMCW_N_ADC_SAMPLES)
    p.add_argument("--fmcw-tx-power-dbm",       type=float, default=DEFAULT_FMCW_TX_POWER_DBM)
    p.add_argument("--fmcw-antenna-gain-dbi",   type=float, default=DEFAULT_FMCW_ANTENNA_GAIN_DBI)
    p.add_argument("--fmcw-noise-figure-db",    type=float, default=DEFAULT_FMCW_NOISE_FIGURE_DB)
    p.add_argument("--fmcw-system-loss-db",     type=float, default=DEFAULT_FMCW_SYSTEM_LOSS_DB)
    p.add_argument("--fmcw-snr-threshold-db",   type=float, default=DEFAULT_FMCW_SNR_THRESHOLD_DB)
    p.add_argument("--fmcw-p-detect",           type=float, default=DEFAULT_FMCW_P_DETECT)
    p.add_argument("--fmcw-az-sigma-0-deg",     type=float, default=DEFAULT_FMCW_AZ_SIGMA_0_DEG)
    p.add_argument("--fmcw-az-sigma-floor-deg", type=float, default=DEFAULT_FMCW_AZ_SIGMA_FLOOR_DEG)
    p.add_argument("--no-fmcw-realism", action="store_true", default=False,
                   help="Skip FMCW realism step.")

    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path (defaults to overwriting the input).")
    args = p.parse_args()

    if args.input is None and args.capture_dir is None:
        p.error("Provide --input <csv-path>  or  --capture-dir <dir>")

    try:
        post_process_capture_dir(
            args.capture_dir,
            input_path=args.input,
            micro_doppler_sigma=args.micro_doppler_sigma,
            fd_stride=args.fd_stride,
            max_walker_speed=args.max_walker_speed,
            seed=args.seed,
            rcs_db_noise_enabled=args.rcs_db_noise,
            rcs_spikes_enabled=args.rcs_spikes,
            rcs_aspect_enabled=args.rcs_aspect,
            no_fmcw_realism=args.no_fmcw_realism,
            fmcw_center_freq_hz=args.fmcw_center_freq_hz,
            fmcw_bandwidth_hz=args.fmcw_bandwidth_hz,
            fmcw_chirp_duration_s=args.fmcw_chirp_duration_s,
            fmcw_n_chirps=args.fmcw_n_chirps,
            fmcw_n_adc_samples=args.fmcw_n_adc_samples,
            fmcw_tx_power_dbm=args.fmcw_tx_power_dbm,
            fmcw_antenna_gain_dbi=args.fmcw_antenna_gain_dbi,
            fmcw_noise_figure_db=args.fmcw_noise_figure_db,
            fmcw_system_loss_db=args.fmcw_system_loss_db,
            fmcw_snr_threshold_db=args.fmcw_snr_threshold_db,
            fmcw_p_detect=args.fmcw_p_detect,
            fmcw_az_sigma_0_deg=args.fmcw_az_sigma_0_deg,
            fmcw_az_sigma_floor_deg=args.fmcw_az_sigma_floor_deg,
            out=args.out,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
