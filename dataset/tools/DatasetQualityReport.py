"""Offline dataset quality / sanity report for a capture folder.

Replays labeled radar CSV into the existing labeling QA collector (accuracy +
density), then adds Doppler-vs-GT residuals per radar/actor, RCS/SNR/visible
checks, and capture completeness. Writes into radar_labeling_qa/ (same folder
as the labeling report). No CARLA required.

Usage:
    python dataset/tools/DatasetQualityReport.py --capture-dir Data/sensor_capture_*
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc)
_dc.bootstrap(__file__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capture.PostProcessDataset import (
    ACTOR_FRAMES_JSONL,
    DEFAULT_FD_STRIDE,
    DEFAULT_FMCW_ANTENNA_GAIN_DBI,
    DEFAULT_FMCW_AZ_SIGMA_0_DEG,
    DEFAULT_FMCW_AZ_SIGMA_FLOOR_DEG,
    DEFAULT_FMCW_BANDWIDTH_HZ,
    DEFAULT_FMCW_CENTER_FREQ_HZ,
    DEFAULT_FMCW_CHIRP_DURATION_S,
    DEFAULT_FMCW_N_ADC_SAMPLES,
    DEFAULT_FMCW_N_CHIRPS,
    DEFAULT_FMCW_NOISE_FIGURE_DB,
    DEFAULT_FMCW_SNR_THRESHOLD_DB,
    DEFAULT_FMCW_SNR_THRESHOLD_STATIC_DB,
    DEFAULT_FMCW_SYSTEM_LOSS_DB,
    DEFAULT_FMCW_TX_POWER_DBM,
    DEFAULT_MAX_WALKER_SPEED_MPS,
    DEFAULT_TARGET_MEDIAN_DBSM,
    LABELED_CSV,
    _detection_ray_world_unit,
    _fmcw_performance,
    estimate_frame_dt,
    estimate_walker_world_velocities,
    load_actor_positions,
)
from testing.RadarLabelingTestReport import (
    SCATTER_RESERVOIR_MAX,
    DetectionRecord,
    LabelingStatsCollector,
    write_report,
)

RAW_CSV = "radar_data.csv"
RUN_META = "run_meta.json"
QA_DIRNAME = "radar_labeling_qa"
DEFAULT_MAX_VEHICLE_SPEED_MPS = 40.0
DEFAULT_MIN_MATCH_RATE = 0.05
MOVING_SPEED_MPS = 0.5
PED_UNFIXED_MEAS_MPS = 0.05
PED_UNFIXED_GT_MPS = 0.5
VEHICLE_MAE_WARN_MPS = 2.0
PED_UNFIXED_FRAC_WARN = 0.10
MISSING_FRAMES_FRAC_WARN = 0.01
RCS_DELTA_WARN_DB = 3.0
QUALITY_TXT_MARKER = "\nDataset quality / sanity\n"

# Match CaptureRadarCameraData write_capture_labeling_report defaults without
# importing that module (it requires carla).
_DEFAULT_REPORT_KWARGS = {
    "min_match_rate": DEFAULT_MIN_MATCH_RATE,
    "proximity_m": 40.0,
    "hit_match_m": 0.5,
    "hit_match_max_margin_m": 0.5,
    "bbox_extent_inflation_m": 0.75,
    "labelable_min_speed_mps": 0.0,
    "candidate_max_range_m": 35.0,
    "candidate_horizontal_fov_deg": 120.0,
    "candidate_depth_margin_m": 3.0,
    "candidate_azimuth_margin_deg": 8.0,
    "candidate_hit_max_bbox_margin_m": 7.0,
    "single_candidate_max_margin_m": 1.0,
}

DOPPLER_FIELDS = (
    "count",
    "bias_mps",
    "mae_mps",
    "rmse_mps",
    "p50_abs_mps",
    "p95_abs_mps",
)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _sensor_sort_key(label: str) -> tuple:
    if len(label) >= 2 and label[0] in "Rr" and label[1:].isdigit():
        return (0, int(label[1:]))
    return (1, label)


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _residual_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "bias_mps": None,
            "mae_mps": None,
            "rmse_mps": None,
            "p50_abs_mps": None,
            "p95_abs_mps": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    abs_arr = np.abs(arr)
    return {
        "count": int(arr.size),
        "bias_mps": _round(float(arr.mean())),
        "mae_mps": _round(float(abs_arr.mean())),
        "rmse_mps": _round(float(math.sqrt(float(np.mean(arr * arr))))),
        "p50_abs_mps": _round(float(np.percentile(abs_arr, 50))),
        "p95_abs_mps": _round(float(np.percentile(abs_arr, 95))),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_csv(capture_dir: Path, csv_path: Path | None = None) -> Path:
    if csv_path is not None:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        return path
    labeled = capture_dir / LABELED_CSV
    if labeled.is_file():
        return labeled
    raw = capture_dir / RAW_CSV
    if raw.is_file():
        return raw
    raise FileNotFoundError(
        f"Missing {labeled.name} (and fallback {raw.name}) in {capture_dir}"
    )


def _load_run_meta(capture_dir: Path) -> dict[str, Any]:
    path = capture_dir / RUN_META
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_prior_summary(qa_dir: Path) -> dict[str, Any]:
    path = qa_dir / "summary.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _report_kwargs(qa_dir: Path, sensor_labels: set[str], run_meta: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(_DEFAULT_REPORT_KWARGS)
    prior = _load_prior_summary(qa_dir)
    params = prior.get("parameters") or {}
    if isinstance(params, dict):
        if params.get("proximity_candidate_radius_m_legacy") is not None:
            kwargs["proximity_m"] = float(params["proximity_candidate_radius_m_legacy"])
        if params.get("hit_match_max_margin_m") is not None:
            kwargs["hit_match_max_margin_m"] = float(params["hit_match_max_margin_m"])
            kwargs["hit_match_m"] = float(params["hit_match_max_margin_m"])
        elif params.get("hit_match_m_legacy_alias") is not None:
            kwargs["hit_match_m"] = float(params["hit_match_m_legacy_alias"])
        if params.get("bbox_extent_inflation_m") is not None:
            kwargs["bbox_extent_inflation_m"] = float(params["bbox_extent_inflation_m"])
        if params.get("labelable_min_speed_mps") is not None:
            kwargs["labelable_min_speed_mps"] = float(params["labelable_min_speed_mps"])
        if params.get("min_pass_match_rate") is not None:
            kwargs["min_match_rate"] = float(params["min_pass_match_rate"])
        if params.get("candidate_max_range_m") is not None:
            kwargs["candidate_max_range_m"] = float(params["candidate_max_range_m"])
        if params.get("candidate_horizontal_fov_deg") is not None:
            kwargs["candidate_horizontal_fov_deg"] = float(
                params["candidate_horizontal_fov_deg"]
            )
        if params.get("candidate_depth_margin_m") is not None:
            kwargs["candidate_depth_margin_m"] = float(params["candidate_depth_margin_m"])
        if params.get("candidate_azimuth_margin_deg") is not None:
            kwargs["candidate_azimuth_margin_deg"] = float(
                params["candidate_azimuth_margin_deg"]
            )
        if params.get("single_candidate_max_margin_m") is not None:
            kwargs["single_candidate_max_margin_m"] = float(
                params["single_candidate_max_margin_m"]
            )
    labels = set(sensor_labels)
    prior_labels = prior.get("expected_radars") or []
    if isinstance(prior_labels, list):
        labels.update(str(x) for x in prior_labels)
    radar_count = run_meta.get("radar_count")
    try:
        n_radar = int(radar_count) if radar_count not in (None, "") else 0
    except (TypeError, ValueError):
        n_radar = 0
    if not n_radar:
        nums = [int(s[1:]) for s in labels if len(s) >= 2 and s[1:].isdigit()]
        n_radar = max(nums) if nums else 0
    if n_radar:
        labels |= {f"R{i}" for i in range(1, n_radar + 1)}
    kwargs["expected_radar_labels"] = labels
    return kwargs


def _fmcw_defaults() -> dict[str, Any]:
    return _fmcw_performance(
        DEFAULT_FMCW_CENTER_FREQ_HZ,
        DEFAULT_FMCW_BANDWIDTH_HZ,
        DEFAULT_FMCW_CHIRP_DURATION_S,
        DEFAULT_FMCW_N_ADC_SAMPLES,
        DEFAULT_FMCW_N_CHIRPS,
        DEFAULT_FMCW_TX_POWER_DBM,
        DEFAULT_FMCW_ANTENNA_GAIN_DBI,
        DEFAULT_FMCW_NOISE_FIGURE_DB,
        DEFAULT_FMCW_SYSTEM_LOSS_DB,
        DEFAULT_FMCW_AZ_SIGMA_0_DEG,
        DEFAULT_FMCW_AZ_SIGMA_FLOOR_DEG,
    )


def _estimate_actor_velocities(
    actor_data: dict[str, Any],
    dt_mean: float,
    fd_stride: int,
    max_walker_speed: float,
    max_vehicle_speed: float,
) -> tuple[dict, dict[str, Any]]:
    positions = actor_data["positions"]
    kind = actor_data["kind"]
    walker_pos = {}
    other_pos = {}
    for key, xyz in positions.items():
        aid = key[1]
        if kind.get(aid) == "pedestrian":
            walker_pos[key] = xyz
        else:
            other_pos[key] = xyz
    walker_vel, walker_info = (
        estimate_walker_world_velocities(walker_pos, dt_mean, fd_stride, max_walker_speed)
        if walker_pos
        else ({}, {"n_central": 0, "n_onesided": 0, "n_clamped": 0, "n_total": 0})
    )
    other_vel, other_info = (
        estimate_walker_world_velocities(other_pos, dt_mean, fd_stride, max_vehicle_speed)
        if other_pos
        else ({}, {"n_central": 0, "n_onesided": 0, "n_clamped": 0, "n_total": 0})
    )
    velocities = dict(other_vel)
    velocities.update(walker_vel)
    return velocities, {"pedestrian": walker_info, "vehicle": other_info}


def _stats_rows(
    grouped: dict[Any, list[float]],
    *,
    key_name: str,
    extra: dict[Any, dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [key_name, *DOPPLER_FIELDS]
    extra_keys: list[str] = []
    if extra:
        for payload in extra.values():
            for k in payload:
                if k not in extra_keys:
                    extra_keys.append(k)
        fieldnames = [key_name, *extra_keys, *DOPPLER_FIELDS]
    rows = []
    for key in grouped:
        stats = _residual_stats(grouped[key])
        row: dict[str, Any] = {key_name: key, **stats}
        if extra and key in extra:
            row.update(extra[key])
        rows.append(row)
    return fieldnames, rows


def _plot_doppler(
    out_path: Path,
    residuals: list[float],
    residuals_by_kind: dict[str, list[float]],
    residuals_by_sensor: dict[str, list[float]],
    scatter: list[tuple[float, float, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    if residuals:
        ax.hist(residuals, bins=80, color="steelblue", edgecolor="none")
        ax.axvline(0.0, color="k", linewidth=0.8)
        ax.set_xlabel("residual velocity_mps − v_gt_radial (m/s)")
        ax.set_ylabel("count")
        ax.set_title("Doppler residual")
    else:
        ax.set_title("Doppler residual (no samples)")
        ax.axis("off")

    ax = axes[0, 1]
    plotted = False
    for kind, color in (("vehicle", "tab:blue"), ("pedestrian", "tab:orange")):
        vals = residuals_by_kind.get(kind) or []
        if not vals:
            continue
        ax.hist(vals, bins=60, alpha=0.55, label=kind, color=color, edgecolor="none")
        plotted = True
    if plotted:
        ax.axvline(0.0, color="k", linewidth=0.8)
        ax.legend(frameon=False)
        ax.set_xlabel("residual (m/s)")
        ax.set_ylabel("count")
        ax.set_title("Residual by actor kind")
    else:
        ax.set_title("Residual by actor kind (no samples)")
        ax.axis("off")

    ax = axes[1, 0]
    labels = sorted(residuals_by_sensor, key=_sensor_sort_key)
    maes = []
    for sl in labels:
        stats = _residual_stats(residuals_by_sensor[sl])
        maes.append(stats["mae_mps"] if stats["mae_mps"] is not None else 0.0)
    if labels:
        ax.bar(labels, maes, color="tab:green")
        ax.set_ylabel("MAE (m/s)")
        ax.set_title("Per-radar Doppler MAE")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.set_title("Per-radar Doppler MAE (no samples)")
        ax.axis("off")

    ax = axes[1, 1]
    if scatter:
        classes = sorted({c for _, _, c in scatter if c})
        cmap = plt.get_cmap("tab10")
        color_of = {c: cmap(i % 10) for i, c in enumerate(classes)}
        if not classes:
            xs = [g for g, _, _ in scatter]
            ys = [m for _, m, _ in scatter]
            ax.scatter(xs, ys, s=6, alpha=0.35, color="steelblue")
        else:
            for cls in classes:
                pts = [(g, m) for g, m, c in scatter if c == cls]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=6, alpha=0.4, color=color_of[cls], label=cls)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        lims = []
        for g, m, _ in scatter:
            lims.append(g)
            lims.append(m)
        lo, hi = float(min(lims)), float(max(lims))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8)
        ax.set_xlabel("v_gt_radial (m/s)")
        ax.set_ylabel("velocity_mps (m/s)")
        ax.set_title("Measured vs GT radial")
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_title("Measured vs GT radial (no samples)")
        ax.axis("off")

    fig.suptitle("Dataset Doppler sanity", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _append_summary_txt(qa_dir: Path, block: str) -> None:
    path = qa_dir / "summary.txt"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if QUALITY_TXT_MARKER in existing:
        existing = existing.split(QUALITY_TXT_MARKER)[0].rstrip() + "\n"
    path.write_text(existing + QUALITY_TXT_MARKER + block.rstrip() + "\n", encoding="utf-8")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr))


def write_capture_quality_report(
    capture_dir,
    *,
    csv_path: Path | None = None,
    refresh_labeling_qa: bool = True,
    fd_stride: int = DEFAULT_FD_STRIDE,
    max_walker_speed: float = DEFAULT_MAX_WALKER_SPEED_MPS,
    max_vehicle_speed: float = DEFAULT_MAX_VEHICLE_SPEED_MPS,
    min_match_rate: float = DEFAULT_MIN_MATCH_RATE,
) -> Path:
    """Build / extend radar_labeling_qa/ for a capture directory.

    Returns the QA output directory.
    """
    capture_dir = Path(capture_dir)
    csv_file = _resolve_csv(capture_dir, csv_path)
    qa_dir = capture_dir / QA_DIRNAME
    qa_dir.mkdir(parents=True, exist_ok=True)
    run_meta = _load_run_meta(capture_dir)
    frames_path = capture_dir / ACTOR_FRAMES_JSONL

    print(f"[quality] Reading {csv_file.name} ...", flush=True)
    dt_mean = estimate_frame_dt(csv_file)

    actor_data: dict[str, Any] | None = None
    velocities: dict = {}
    vel_info = {
        "pedestrian": {"n_central": 0, "n_onesided": 0, "n_clamped": 0, "n_total": 0},
        "vehicle": {"n_central": 0, "n_onesided": 0, "n_clamped": 0, "n_total": 0},
    }
    if frames_path.is_file():
        print(f"[quality] Loading {frames_path.name} ...", flush=True)
        actor_data = load_actor_positions(frames_path)
        velocities, vel_info = _estimate_actor_velocities(
            actor_data, dt_mean, fd_stride, max_walker_speed, max_vehicle_speed,
        )
        print(
            f"      actors: {len(actor_data['kind']):,}  frames: {len(actor_data['frames']):,}  "
            f"GT v: ped={vel_info['pedestrian']['n_total']:,}  "
            f"other={vel_info['vehicle']['n_total']:,}",
            flush=True,
        )
    else:
        print(f"[quality] WARNING: {frames_path.name} missing — skipping Doppler GT.", flush=True)

    collector = LabelingStatsCollector(labelable_min_speed_mps=_DEFAULT_REPORT_KWARGS["labelable_min_speed_mps"])
    residuals_all: list[float] = []
    residuals_by_sensor: dict[str, list[float]] = defaultdict(list)
    residuals_by_class: dict[str, list[float]] = defaultdict(list)
    residuals_by_sensor_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    residuals_by_actor: dict[int, list[float]] = defaultdict(list)
    residuals_by_kind: dict[str, list[float]] = defaultdict(list)
    actor_meta: dict[int, dict[str, Any]] = {}
    scatter: list[tuple[float, float, str]] = []
    scatter_seen = 0
    rng = random.Random(0)
    n_matched_no_gt = 0
    n_ped_unfixed = 0
    n_ped_doppler = 0
    noisy_residuals: list[float] = []
    rcs_by_class: dict[str, list[float]] = defaultdict(list)
    visible_n = 0
    visible_1 = 0
    invis_range = 0
    invis_vel = 0
    invis_snr = 0
    visible_by_sensor: Counter[str] = Counter()
    moving_by_key: Counter[tuple[str, str]] = Counter()
    static_by_key: Counter[tuple[str, str]] = Counter()
    matched_ids: set[int] = set()
    sensor_labels: set[str] = set()
    radar_frames_seen: set[int] = set()
    msg_counts: Counter[tuple[str, int]] = Counter()
    t_min: float | None = None
    t_max: float | None = None
    n_noisy = 0
    fmcw = _fmcw_defaults()
    postprocessed = False

    with csv_file.open(newline="", encoding="utf-8") as rf:
        reader = csv.DictReader(rf)
        fieldnames = set(reader.fieldnames or [])
        postprocessed = "rcs_dBsm" in fieldnames or "visible" in fieldnames
        has_visible = "visible" in fieldnames
        has_rcs = "rcs_dBsm" in fieldnames
        has_noisy = "velocity_mps_noisy" in fieldnames
        has_snr = "snr_dB" in fieldnames
        n_rows = 0
        for row in reader:
            n_rows += 1
            sensor_label = (row.get("sensor_label") or "").strip() or "?"
            frame_id = int(row["frame"])
            sensor_labels.add(sensor_label)
            radar_frames_seen.add(frame_id)
            msg_counts[(sensor_label, frame_id)] += 1
            ts = _opt_float(row.get("timestamp"))
            if ts is not None:
                if t_min is None or ts < t_min:
                    t_min = ts
                if t_max is None or ts > t_max:
                    t_max = ts

            velocity_mps = float(row["velocity_mps"])
            depth_m = float(row["depth_m"])
            azimuth_rad = float(row.get("azimuth_rad") or 0.0)
            altitude_rad = float(row.get("altitude_rad") or 0.0)
            actor_id = _opt_int(row.get("matched_actor_id"))
            actor_kind = (row.get("matched_actor_kind") or "").strip()
            actor_class = (row.get("matched_actor_class") or "").strip()
            had_candidates = (row.get("had_actor_candidates") or "").strip() == "1"
            label_scored = (row.get("label_scored") or "1").strip() not in ("0", "false", "False")
            nearest_margin = _opt_float(row.get("nearest_actor_bbox_margin_m"))
            match_margin = _opt_float(row.get("matched_actor_bbox_margin_m"))

            if not label_scored:
                collector.record_static_skipped(sensor_label)
            else:
                collector.record_detection(
                    DetectionRecord(
                        sensor_label=sensor_label,
                        frame=frame_id,
                        had_candidates=had_candidates,
                        matched=actor_id is not None,
                        depth_m=depth_m,
                        velocity_mps=velocity_mps,
                        azimuth_rad=azimuth_rad,
                        actor_id=actor_id,
                        actor_kind=actor_kind,
                        actor_class=actor_class,
                        match_bbox_margin_m=match_margin,
                        nearest_bbox_margin_m=nearest_margin,
                        uncensored_nearest_bbox_margin_m=nearest_margin,
                    )
                )

            class_key = actor_class or ("unmatched" if actor_id is None else "unknown")
            if abs(velocity_mps) > MOVING_SPEED_MPS:
                moving_by_key[(sensor_label, class_key)] += 1
            else:
                static_by_key[(sensor_label, class_key)] += 1

            if has_rcs:
                rcs_val = _opt_float(row.get("rcs_dBsm"))
                if rcs_val is not None and actor_class:
                    rcs_by_class[actor_class].append(rcs_val)

            if has_visible:
                vis = (row.get("visible") or "").strip()
                if vis != "":
                    visible_n += 1
                    if vis == "1":
                        visible_1 += 1
                        visible_by_sensor[sensor_label] += 1
                    else:
                        if depth_m > fmcw["range_max_m"]:
                            invis_range += 1
                        if abs(velocity_mps) > fmcw["velocity_max_ms"]:
                            invis_vel += 1
                        snr_val = _opt_float(row.get("snr_dB")) if has_snr else None
                        if snr_val is not None:
                            thr = (
                                DEFAULT_FMCW_SNR_THRESHOLD_DB
                                if actor_kind in ("vehicle", "pedestrian")
                                else DEFAULT_FMCW_SNR_THRESHOLD_STATIC_DB
                            )
                            if snr_val < thr:
                                invis_snr += 1

            if has_noisy:
                noisy = _opt_float(row.get("velocity_mps_noisy"))
                if noisy is not None:
                    n_noisy += 1
                    delta = noisy - velocity_mps
                    if len(noisy_residuals) < SCATTER_RESERVOIR_MAX:
                        noisy_residuals.append(delta)
                    else:
                        j = rng.randint(0, n_noisy - 1)
                        if j < SCATTER_RESERVOIR_MAX:
                            noisy_residuals[j] = delta

            if actor_id is None:
                continue
            matched_ids.add(actor_id)
            pitch = _opt_float(row.get("sensor_pitch_deg")) or 0.0
            yaw = _opt_float(row.get("sensor_yaw_deg")) or 0.0
            v_world = velocities.get((frame_id, actor_id))
            if v_world is None:
                n_matched_no_gt += 1
                continue
            ray = _detection_ray_world_unit(azimuth_rad, altitude_rad, pitch, yaw)
            if ray is None:
                n_matched_no_gt += 1
                continue
            v_gt = v_world[0] * ray[0] + v_world[1] * ray[1] + v_world[2] * ray[2]
            residual = velocity_mps - v_gt
            residuals_all.append(residual)
            residuals_by_sensor[sensor_label].append(residual)
            residuals_by_class[actor_class or "unknown"].append(residual)
            residuals_by_sensor_class[(sensor_label, actor_class or "unknown")].append(residual)
            residuals_by_actor[actor_id].append(residual)
            kind_key = actor_kind or "unknown"
            residuals_by_kind[kind_key].append(residual)
            if actor_id not in actor_meta:
                actor_meta[actor_id] = {"kind": actor_kind, "class": actor_class}
            if kind_key == "pedestrian":
                n_ped_doppler += 1
                if abs(velocity_mps) < PED_UNFIXED_MEAS_MPS and abs(v_gt) > PED_UNFIXED_GT_MPS:
                    n_ped_unfixed += 1
            scatter_seen += 1
            item = (v_gt, velocity_mps, actor_class or kind_key)
            if len(scatter) < SCATTER_RESERVOIR_MAX:
                scatter.append(item)
            else:
                j = rng.randint(0, scatter_seen - 1)
                if j < SCATTER_RESERVOIR_MAX:
                    scatter[j] = item

    for (_sl, _fr), n in msg_counts.items():
        collector.record_message(raw_returns=n)

    collector.finalize_co_visibility()
    snap = collector.snapshot()
    report_kwargs = _report_kwargs(qa_dir, sensor_labels, run_meta)
    if min_match_rate != DEFAULT_MIN_MATCH_RATE:
        report_kwargs["min_match_rate"] = min_match_rate
    collector.labelable_min_speed_mps = report_kwargs["labelable_min_speed_mps"]

    if refresh_labeling_qa and snap.get("total_detections", 0) > 0:
        print("[quality] Refreshing labeling QA plots from CSV ...", flush=True)
        write_report(collector, qa_dir, **report_kwargs)

    density = dict(snap.get("density_per_radar_per_frame") or {})
    radar_frames = int(density.get("radar_frames") or 0)
    visible_density = None
    visible_by_sensor_density: dict[str, float] = {}
    if visible_n and radar_frames:
        visible_density = _round(visible_1 / radar_frames)
        by_sensor_frames = (density.get("by_sensor") or {})
        for sl, n_vis in visible_by_sensor.items():
            nf = int((by_sensor_frames.get(sl) or {}).get("frames") or 0)
            visible_by_sensor_density[sl] = round(n_vis / nf, 4) if nf else 0.0

    n_unique_frames = len(radar_frames_seen)
    n_radars = len(sensor_labels)
    n_msgs = len(msg_counts)
    expected_msgs = n_radars * n_unique_frames
    t_span = (t_max - t_min) if t_min is not None and t_max is not None else 0.0
    actor_frames = actor_data["frames"] if actor_data else set()
    missing_actor_frames = sorted(radar_frames_seen - actor_frames) if actor_data else []
    extra_actor_frames = (
        len(actor_frames - radar_frames_seen) if actor_data else 0
    )

    never_matched = 0
    never_matched_by_kind: dict[str, int] = {}
    if actor_data:
        present_ids = set(actor_data["kind"])
        never_ids = present_ids - matched_ids
        never_matched = len(never_ids)
        never_matched_by_kind = dict(
            Counter(actor_data["kind"].get(aid, "") or "unknown" for aid in never_ids)
        )

    zero_match_radars = sorted(
        (
            sl
            for sl, bucket in (snap.get("by_sensor") or {}).items()
            if int(bucket.get("matched") or 0) == 0
        ),
        key=_sensor_sort_key,
    )
    expected_missing = sorted(
        report_kwargs["expected_radar_labels"] - sensor_labels,
        key=_sensor_sort_key,
    )

    doppler_overall = _residual_stats(residuals_all)
    doppler_by_sensor = {
        sl: _residual_stats(vals)
        for sl, vals in sorted(residuals_by_sensor.items(), key=lambda kv: _sensor_sort_key(kv[0]))
    }
    doppler_by_class = {
        cls: _residual_stats(vals) for cls, vals in sorted(residuals_by_class.items())
    }
    doppler_by_sensor_class = {
        f"{sl}|{cls}": _residual_stats(vals)
        for (sl, cls), vals in sorted(residuals_by_sensor_class.items())
    }
    ped_stats = _residual_stats(residuals_by_kind.get("pedestrian") or [])
    veh_kind_residuals: list[float] = []
    for kind, vals in residuals_by_kind.items():
        if kind != "pedestrian":
            veh_kind_residuals.extend(vals)
    veh_stats = _residual_stats(veh_kind_residuals)
    ped_unfixed_frac = (n_ped_unfixed / n_ped_doppler) if n_ped_doppler else 0.0

    rcs_summary: dict[str, Any] = {}
    for cls, vals in sorted(rcs_by_class.items()):
        med = _median(vals)
        target = DEFAULT_TARGET_MEDIAN_DBSM.get(cls)
        delta = (med - target) if med is not None and target is not None else None
        rcs_summary[cls] = {
            "n": len(vals),
            "median_dbsm": _round(med, 3) if med is not None else None,
            "target_dbsm": target,
            "delta_db": _round(delta, 3) if delta is not None else None,
        }

    moving_static: dict[str, Any] = {}
    all_class_keys = {k[1] for k in moving_by_key} | {k[1] for k in static_by_key}
    for cls in sorted(all_class_keys):
        n_move = sum(cnt for (_sl, cname), cnt in moving_by_key.items() if cname == cls)
        n_stat = sum(cnt for (_sl, cname), cnt in static_by_key.items() if cname == cls)
        moving_static[cls] = {
            "moving": n_move,
            "static": n_stat,
            "moving_frac": _round(n_move / (n_move + n_stat)) if (n_move + n_stat) else None,
        }

    pass_ok = (
        int(snap.get("with_candidates") or 0) >= 50
        and float(snap.get("match_rate_given_candidates") or 0.0) >= report_kwargs["min_match_rate"]
    )
    warnings: list[str] = []
    if not pass_ok:
        warnings.append(
            f"labeling match_rate_given_candidates="
            f"{100 * float(snap.get('match_rate_given_candidates') or 0):.1f}% "
            f"(PASS needs ≥ {100 * report_kwargs['min_match_rate']:.0f}% and ≥50 candidates)"
        )
    if zero_match_radars:
        warnings.append(f"radars with zero matches: {', '.join(zero_match_radars)}")
    if expected_missing:
        warnings.append(f"expected radars missing from CSV: {', '.join(expected_missing)}")
    if veh_stats["mae_mps"] is not None and veh_stats["mae_mps"] > VEHICLE_MAE_WARN_MPS:
        warnings.append(f"vehicle Doppler MAE {veh_stats['mae_mps']:.2f} m/s > {VEHICLE_MAE_WARN_MPS:.1f}")
    if n_ped_doppler and ped_unfixed_frac > PED_UNFIXED_FRAC_WARN:
        warnings.append(
            f"pedestrian Doppler looks unfixed: {100 * ped_unfixed_frac:.1f}% of ped "
            f"returns have |v|≈0 while GT radial > {PED_UNFIXED_GT_MPS} m/s"
        )
    if radar_frames_seen and missing_actor_frames:
        miss_frac = len(missing_actor_frames) / max(n_unique_frames, 1)
        if miss_frac >= MISSING_FRAMES_FRAC_WARN:
            warnings.append(
                f"missing actor_frames for {len(missing_actor_frames):,} / "
                f"{n_unique_frames:,} radar frames ({100 * miss_frac:.1f}%)"
            )
    for cls, info in rcs_summary.items():
        delta = info.get("delta_db")
        if delta is not None and abs(delta) > RCS_DELTA_WARN_DB:
            warnings.append(
                f"RCS median for {cls} is {info['median_dbsm']:+.1f} dBsm "
                f"(target {info['target_dbsm']:+.1f}, Δ {delta:+.1f} dB)"
            )

    worst_radar = None
    worst_radar_mae = None
    for sl, stats in doppler_by_sensor.items():
        mae = stats.get("mae_mps")
        if mae is None:
            continue
        if worst_radar_mae is None or mae > worst_radar_mae:
            worst_radar, worst_radar_mae = sl, mae
    worst_class = None
    worst_class_mae = None
    for cls, stats in doppler_by_class.items():
        mae = stats.get("mae_mps")
        if mae is None:
            continue
        if worst_class_mae is None or mae > worst_class_mae:
            worst_class, worst_class_mae = cls, mae

    top_actors = []
    for aid, vals in residuals_by_actor.items():
        stats = _residual_stats(vals)
        meta = actor_meta.get(aid) or {}
        top_actors.append(
            {
                "actor_id": aid,
                "kind": meta.get("kind", ""),
                "class": meta.get("class", ""),
                **stats,
            }
        )
    top_actors.sort(key=lambda r: (-(r["mae_mps"] or -1.0), -(r["count"] or 0)))

    quality = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "csv": csv_file.name,
        "postprocessed": postprocessed,
        "pass": pass_ok,
        "warnings": warnings,
        "accuracy": {
            "total_detections": snap.get("total_detections", 0),
            "matched_detections": snap.get("matched_detections", 0),
            "with_candidates": snap.get("with_candidates", 0),
            "match_rate": snap.get("match_rate", 0.0),
            "match_rate_given_candidates": snap.get("match_rate_given_candidates", 0.0),
            "failed_match_rate_given_candidates": snap.get(
                "failed_match_rate_given_candidates", 0.0
            ),
            "clutter_rate": snap.get("clutter_rate", 0.0),
            "by_sensor": snap.get("by_sensor", {}),
        },
        "density": {
            **{k: v for k, v in density.items() if k != "note"},
            "visible": visible_density,
            "visible_by_sensor": visible_by_sensor_density,
        },
        "doppler": {
            "overall": doppler_overall,
            "vehicle": veh_stats,
            "pedestrian": ped_stats,
            "by_sensor": doppler_by_sensor,
            "by_class": doppler_by_class,
            "by_sensor_class": doppler_by_sensor_class,
            "n_matched_no_gt": n_matched_no_gt,
            "gt_velocity_info": vel_info,
            "pedestrian_fix": {
                "n_compared": n_ped_doppler,
                "n_near_zero_meas_moving_gt": n_ped_unfixed,
                "frac_unfixed": _round(ped_unfixed_frac),
            },
            "fmcw_noise": _residual_stats(noisy_residuals) if noisy_residuals else None,
            "worst_radar": {"sensor_label": worst_radar, "mae_mps": worst_radar_mae},
            "worst_class": {"class": worst_class, "mae_mps": worst_class_mae},
            "top_actor_mae": top_actors[:10],
        },
        "completeness": {
            "n_rows": n_rows,
            "n_radars": n_radars,
            "n_unique_frames": n_unique_frames,
            "n_messages": n_msgs,
            "expected_messages": expected_msgs,
            "message_coverage": _round(n_msgs / expected_msgs) if expected_msgs else None,
            "timestamp_span_s": _round(t_span, 3) if t_min is not None else None,
            "dt_mean_s": _round(dt_mean, 6),
            "approx_hz": _round(1.0 / dt_mean, 3) if dt_mean else None,
            "actor_frames": len(actor_frames),
            "missing_actor_frames": len(missing_actor_frames),
            "extra_actor_frames": extra_actor_frames,
            "expected_radars_missing": expected_missing,
        },
        "rcs": rcs_summary,
        "visibility": {
            "n_scored": visible_n,
            "visible_1": visible_1,
            "visible_rate": _round(visible_1 / visible_n) if visible_n else None,
            "invisible_range": invis_range,
            "invisible_velocity": invis_vel,
            "invisible_snr": invis_snr,
        }
        if visible_n
        else None,
        "moving_static": moving_static,
        "flags": {
            "zero_match_radars": zero_match_radars,
            "never_matched_actors": {
                "count": never_matched,
                "by_kind": never_matched_by_kind,
            },
        },
    }

    sensor_fields, sensor_rows = _stats_rows(residuals_by_sensor, key_name="sensor_label")
    sensor_rows.sort(key=lambda r: _sensor_sort_key(str(r["sensor_label"])))
    _write_csv(qa_dir / "doppler_by_sensor.csv", list(sensor_fields), sensor_rows)

    class_fields, class_rows = _stats_rows(residuals_by_class, key_name="matched_actor_class")
    class_rows.sort(key=lambda r: str(r["matched_actor_class"]))
    _write_csv(qa_dir / "doppler_by_class.csv", list(class_fields), class_rows)

    actor_rows = []
    for aid, vals in residuals_by_actor.items():
        meta = actor_meta.get(aid) or {}
        actor_rows.append(
            {
                "actor_id": aid,
                "kind": meta.get("kind", ""),
                "class": meta.get("class", ""),
                **_residual_stats(vals),
            }
        )
    actor_rows.sort(key=lambda r: (-int(r["count"]), int(r["actor_id"])))
    _write_csv(
        qa_dir / "doppler_by_actor.csv",
        ["actor_id", "kind", "class", *DOPPLER_FIELDS],
        actor_rows,
    )

    plot_path = qa_dir / "radar_quality_doppler.png"
    _plot_doppler(
        plot_path,
        residuals_all,
        residuals_by_kind,
        residuals_by_sensor,
        scatter,
    )

    summary = _load_prior_summary(qa_dir)
    if not summary:
        summary = {
            "generated_at": quality["generated_at"],
            "expected_radars": sorted(report_kwargs["expected_radar_labels"], key=_sensor_sort_key),
            "pass": pass_ok,
            "summary": snap,
        }
    summary["quality"] = quality
    summary["pass"] = bool(summary.get("pass", True)) and pass_ok
    files = summary.setdefault("files", {})
    files.update(
        {
            "doppler_by_sensor": "doppler_by_sensor.csv",
            "doppler_by_class": "doppler_by_class.csv",
            "doppler_by_actor": "doppler_by_actor.csv",
            "doppler_plot": plot_path.name,
        }
    )
    (qa_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dn = density
    checklist_lines = [
        "=" * 40,
        f"CSV: {csv_file.name}  postprocessed={postprocessed}  rows={n_rows:,}",
        f"Match (w/ candidates): {100 * float(snap.get('match_rate_given_candidates') or 0):.1f}%  "
        f"({snap.get('matched_detections', 0):,}/{snap.get('with_candidates', 0):,})  "
        f"{'PASS' if pass_ok else 'FAIL'} "
        f"(need ≥ {100 * report_kwargs['min_match_rate']:.0f}% and ≥50 candidates)",
        f"Match (all returns):   {100 * float(snap.get('match_rate') or 0):.2f}%   "
        f"clutter {100 * float(snap.get('clutter_rate') or 0):.1f}%",
        f"Density / radar / frame: all={dn.get('all', 0):.2f}  matched={dn.get('matched', 0):.3f}  "
        f"vehicle={dn.get('vehicle', 0):.3f}  ped={dn.get('pedestrian', 0):.3f}",
    ]
    if visible_density is not None:
        checklist_lines.append(f"Visible density / radar / frame: {visible_density:.3f}")
    if doppler_overall["count"]:
        checklist_lines.append(
            f"Doppler MAE: overall={doppler_overall['mae_mps']:.3f} m/s  "
            f"vehicle={veh_stats['mae_mps'] if veh_stats['mae_mps'] is not None else float('nan'):.3f}  "
            f"ped={ped_stats['mae_mps'] if ped_stats['mae_mps'] is not None else float('nan'):.3f}  "
            f"n={doppler_overall['count']:,}"
        )
        if worst_radar is not None:
            checklist_lines.append(
                f"Worst radar Doppler MAE: {worst_radar} {worst_radar_mae:.3f} m/s"
            )
        if worst_class is not None:
            checklist_lines.append(
                f"Worst class Doppler MAE: {worst_class} {worst_class_mae:.3f} m/s"
            )
        checklist_lines.append(
            f"Ped Doppler fix: {n_ped_unfixed:,}/{n_ped_doppler:,} still ~0 while GT moving "
            f"({100 * ped_unfixed_frac:.1f}%)"
        )
    else:
        checklist_lines.append("Doppler: no matched returns with GT velocity")
    checklist_lines.append(
        f"Completeness: {n_msgs:,}/{expected_msgs:,} (sensor,frame) messages  "
        f"missing actor frames={len(missing_actor_frames):,}"
    )
    if rcs_summary:
        bits = [
            f"{cls} {info['median_dbsm']:+.1f}"
            for cls, info in rcs_summary.items()
            if info.get("median_dbsm") is not None
        ]
        checklist_lines.append("RCS median dBsm: " + ", ".join(bits))
    if visible_n:
        checklist_lines.append(
            f"Visible: {100 * visible_1 / visible_n:.1f}%  "
            f"(invis range={invis_range:,} vel={invis_vel:,} snr={invis_snr:,})"
        )
    if zero_match_radars:
        checklist_lines.append("Zero-match radars: " + ", ".join(zero_match_radars))
    if warnings:
        checklist_lines.append("Warnings:")
        checklist_lines.extend(f"  - {w}" for w in warnings)
    else:
        checklist_lines.append("Warnings: none")
    checklist_lines.append(f"Outputs in: {qa_dir.resolve()}")
    checklist = "\n".join(checklist_lines)
    _append_summary_txt(qa_dir, checklist)

    print(flush=True)
    print("Dataset quality / sanity", flush=True)
    print(checklist, flush=True)
    return qa_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="sensor_capture_* folder with radar_data_labeled.csv (or radar_data.csv)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Override radar CSV path (defaults to radar_data_labeled.csv)",
    )
    parser.add_argument(
        "--no-refresh-labeling-qa",
        action="store_true",
        help="Do not regenerate radar_labeling_summary.png / collector CSVs",
    )
    parser.add_argument("--fd-stride", type=int, default=DEFAULT_FD_STRIDE)
    parser.add_argument(
        "--max-walker-speed", type=float, default=DEFAULT_MAX_WALKER_SPEED_MPS,
    )
    parser.add_argument(
        "--max-vehicle-speed", type=float, default=DEFAULT_MAX_VEHICLE_SPEED_MPS,
    )
    parser.add_argument(
        "--min-match-rate", type=float, default=DEFAULT_MIN_MATCH_RATE,
    )
    args = parser.parse_args()
    try:
        write_capture_quality_report(
            args.capture_dir,
            csv_path=args.csv,
            refresh_labeling_qa=not args.no_refresh_labeling_qa,
            fd_stride=args.fd_stride,
            max_walker_speed=args.max_walker_speed,
            max_vehicle_speed=args.max_vehicle_speed,
            min_match_rate=args.min_match_rate,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
