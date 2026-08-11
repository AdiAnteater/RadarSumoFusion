"""
Label a fast-capture radar_data.csv offline using actor_frames.jsonl.

Run after simulation (no CARLA required if actor_frames.jsonl exists):

  python capture/LabelRadarCapture.py --capture-dir Data/sensor_capture_YYYYMMDD_HHMMSS

Or set DATASET_LABEL_RADAR_AFTER_CAPTURE=1 (default) to run automatically when
CaptureRadarCameraData.py stops.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc)
_dc.bootstrap(__file__)

import carla

from capture.actor_frame_log import ActorFrameLogger
from capture.CaptureRadarCameraData import (
    RADAR_HORIZONTAL_FOV_DEG,
    RADAR_MAX_RANGE_M,
    actor_rcs_proxy_projected_area_m2,
    evaluate_radar_detection_label,
    labelable_min_speed_from_env,
    radar_hit_match_max_margin_m_from_env,
    radar_single_candidate_max_margin_m_from_env,
    write_capture_labeling_report,
)
from testing.RadarLabelingTestReport import (
    MARGIN_N_BINS,
    DetectionRecord,
    LabelingStatsCollector,
    derive_margin_threshold,
    margin_bin_index,
)

RADAR_CSV = "radar_data.csv"
LABELED_CSV = "radar_data_labeled.csv"


def label_after_capture_from_env() -> bool:
    raw = os.environ.get("DATASET_LABEL_RADAR_AFTER_CAPTURE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _detection_from_row(row: dict) -> SimpleNamespace:
    return SimpleNamespace(
        depth=float(row["depth_m"]),
        azimuth=float(row["azimuth_rad"]),
        altitude=float(row["altitude_rad"]),
        velocity=float(row["velocity_mps"]),
    )


def _transform_from_row(row: dict) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            float(row["sensor_world_x_m"]),
            float(row["sensor_world_y_m"]),
            float(row["sensor_world_z_m"]),
        ),
        carla.Rotation(
            pitch=float(row["sensor_pitch_deg"]),
            yaw=float(row["sensor_yaw_deg"]),
            roll=float(row["sensor_roll_deg"]),
        ),
    )


def auto_margin_from_env() -> bool:
    """DATASET_RADAR_AUTO_MARGIN=1 derives the match margin from this capture's own
    nearest-margin distribution instead of using the constant/env. Default OFF so
    dataset runs stay reproducible with a fixed, documented threshold."""
    raw = os.environ.get("DATASET_RADAR_AUTO_MARGIN", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def auto_margin_stride_from_env() -> int:
    raw = os.environ.get("DATASET_RADAR_AUTO_MARGIN_STRIDE", "20").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 20


def derive_capture_margins(
    radar_path: Path,
    actors_by_frame: dict,
    labelable_min_speed: float,
    *,
    stride: int,
    base_primary: float,
    base_single: float,
) -> tuple[float, float, dict]:
    """Sampled first pass: build the uncensored nearest-margin histogram and derive
    a primary + single-candidate margin from its trough. The margins passed in only
    drive matching (which we ignore here) — the histogram is threshold-independent.
    Falls back to base_* when the sample has too few candidate returns.
    """
    import numpy as np

    hist = np.zeros(MARGIN_N_BINS, dtype=np.int64)
    with radar_path.open(newline="", encoding="utf-8") as rf:
        for row in csv.DictReader(rf):
            if int(row["frame"]) % stride:
                continue
            actors = actors_by_frame.get(int(row["frame"]))
            if not actors:
                continue
            label = evaluate_radar_detection_label(
                None,
                _transform_from_row(row),
                _detection_from_row(row),
                actors,
                range_m=RADAR_MAX_RANGE_M,
                hfov_deg=RADAR_HORIZONTAL_FOV_DEG,
                labelable_min_speed_mps=labelable_min_speed,
                hit_match_max_margin_m=base_primary,
                single_candidate_max_margin_m=base_single,
            )
            bi = margin_bin_index(label["uncensored_nearest_bbox_margin_m"])
            if bi is not None:
                hist[bi] += 1
    diag = derive_margin_threshold(hist)
    if diag["enough_data"] and diag["suggested_primary_m"] is not None:
        return diag["suggested_primary_m"], diag["suggested_single_m"], diag
    return base_primary, base_single, diag


def label_radar_capture_dir(
    capture_dir: str | Path,
    *,
    in_place: bool = False,
    progress_every: int = 50_000,
) -> Path:
    capture_dir = Path(capture_dir)
    radar_path = capture_dir / RADAR_CSV
    frames_path = capture_dir / ActorFrameLogger.ACTOR_FRAMES_FILENAME
    out_path = capture_dir / RADAR_CSV if in_place else capture_dir / LABELED_CSV

    if not radar_path.is_file():
        raise FileNotFoundError(f"Missing {radar_path}")
    if not frames_path.is_file():
        raise FileNotFoundError(
            f"Missing {frames_path}. Re-run capture with fast mode (default) so actor "
            "frames are logged alongside radar_data.csv."
        )

    banner = "=" * 64
    print(flush=True)
    print(banner, flush=True)
    print(">>> OFFLINE RADAR LABELING — DO NOT CLOSE THIS WINDOW <<<", flush=True)
    print(banner, flush=True)
    print(f"  Capture folder: {capture_dir}", flush=True)
    print(f"  Input:          {radar_path.name}", flush=True)
    print(f"  Output:         {out_path.name}", flush=True)
    print(banner, flush=True)

    print(f"[1/3] Loading actor frames from {frames_path.name}...", flush=True)
    t0 = time.time()
    actors_by_frame = ActorFrameLogger.load_by_frame(frames_path)
    print(f"      {len(actors_by_frame):,} frames loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"[2/3] Counting radar returns in {radar_path.name}...", flush=True)
    t_count = time.time()
    with radar_path.open(encoding="utf-8") as cf:
        total_rows = max(sum(1 for _ in cf) - 1, 0)
    print(f"      {total_rows:,} rows in {time.time() - t_count:.1f}s", flush=True)

    labelable_min_speed = labelable_min_speed_from_env()
    hit_match_margin = radar_hit_match_max_margin_m_from_env()
    single_cand_margin = radar_single_candidate_max_margin_m_from_env()
    print(
        f"      hit_match_max_margin_m={hit_match_margin:.2f}  "
        f"single_candidate_max_margin_m={single_cand_margin:.2f}  "
        f"labelable_min_speed_mps={labelable_min_speed:.2f}",
        flush=True,
    )

    if auto_margin_from_env():
        stride = auto_margin_stride_from_env()
        print(
            f"[auto-margin] DATASET_RADAR_AUTO_MARGIN=1: deriving margin from this "
            f"capture (every {stride}th frame)...",
            flush=True,
        )
        t_am = time.time()
        auto_primary, auto_single, diag = derive_capture_margins(
            radar_path,
            actors_by_frame,
            labelable_min_speed,
            stride=stride,
            base_primary=hit_match_margin,
            base_single=single_cand_margin,
        )
        if diag["enough_data"] and diag["suggested_primary_m"] is not None:
            print(
                f"[auto-margin] samples={diag['samples']:,} "
                f"spike={100 * diag['spike_frac']:.1f}% trough={diag['trough_m']} m "
                f"-> primary {auto_primary} m, single {auto_single} m "
                f"(was {hit_match_margin:.2f}/{single_cand_margin:.2f}) "
                f"in {time.time() - t_am:.1f}s",
                flush=True,
            )
            hit_match_margin, single_cand_margin = auto_primary, auto_single
        else:
            print(
                f"[auto-margin] only {diag['samples']:,} candidate samples (<2000) — "
                f"keeping {hit_match_margin:.2f}/{single_cand_margin:.2f} m",
                flush=True,
            )

    collector = LabelingStatsCollector(labelable_min_speed_mps=labelable_min_speed)

    rows_written = 0
    missing_frames = 0
    progress_interval_s = 2.0

    print(f"[3/3] Labeling -> {out_path.name}...", flush=True)
    t1 = time.time()
    last_progress = t1
    with radar_path.open(newline="", encoding="utf-8") as rf, out_path.open(
        "w", newline="", encoding="utf-8"
    ) as wf:
        reader = csv.DictReader(rf)
        fieldnames = list(reader.fieldnames or [])
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            frame_id = int(row["frame"])
            sensor_label = row["sensor_label"]
            actors = actors_by_frame.get(frame_id)
            if actors is None:
                missing_frames += 1
                actors = []

            sensor_transform = _transform_from_row(row)
            label = evaluate_radar_detection_label(
                None,
                sensor_transform,
                _detection_from_row(row),
                actors,
                range_m=RADAR_MAX_RANGE_M,
                hfov_deg=RADAR_HORIZONTAL_FOV_DEG,
                labelable_min_speed_mps=labelable_min_speed,
                hit_match_max_margin_m=hit_match_margin,
                single_candidate_max_margin_m=single_cand_margin,
            )

            row["matched_actor_id"] = ""
            row["matched_actor_kind"] = ""
            row["matched_actor_type_id"] = ""
            row["matched_actor_class"] = ""
            row["matched_actor_bbox_margin_m"] = ""
            row["matched_vehicle_id"] = ""
            row["matched_vehicle_type_id"] = ""
            row["matched_vehicle_class"] = ""
            row["matched_vehicle_distance_m"] = ""
            row["rcs_proxy_m2"] = ""
            row["had_actor_candidates"] = "1" if label["had_candidates"] else "0"
            row["label_scored"] = "1" if label["scored"] else "0"
            row["nearest_actor_bbox_margin_m"] = (
                f"{label['nearest_bbox_margin_m']:.6f}"
                if label["nearest_bbox_margin_m"] is not None
                else ""
            )

            if label["matched"] and label["actor_id"] is not None:
                row["matched_actor_id"] = str(label["actor_id"])
                row["matched_actor_kind"] = label["actor_kind"]
                row["matched_actor_type_id"] = label["actor_type_id"]
                row["matched_actor_class"] = label["actor_class"]
                if label["match_bbox_margin_m"] is not None:
                    row["matched_actor_bbox_margin_m"] = f"{label['match_bbox_margin_m']:.6f}"
                if label["actor_kind"] == "vehicle":
                    row["matched_vehicle_id"] = row["matched_actor_id"]
                    row["matched_vehicle_type_id"] = label["actor_type_id"]
                    row["matched_vehicle_class"] = label["actor_class"]
                    row["matched_vehicle_distance_m"] = row["matched_actor_bbox_margin_m"]
                row["rcs_proxy_m2"] = actor_rcs_proxy_projected_area_m2(
                    label.get("actor_snapshot"), sensor_transform.location
                )

            writer.writerow(row)
            rows_written += 1

            if label["scored"]:
                collector.record_detection(
                    DetectionRecord(
                        sensor_label=sensor_label,
                        frame=frame_id,
                        had_candidates=label["had_candidates"],
                        matched=label["matched"],
                        depth_m=float(row["depth_m"]),
                        velocity_mps=label["velocity_mps"],
                        azimuth_rad=float(row["azimuth_rad"]),
                        actor_id=label["actor_id"],
                        actor_kind=label["actor_kind"],
                        actor_class=label["actor_class"],
                        match_bbox_margin_m=label["match_bbox_margin_m"],
                        nearest_bbox_margin_m=label["nearest_bbox_margin_m"],
                        uncensored_nearest_bbox_margin_m=label[
                            "uncensored_nearest_bbox_margin_m"
                        ],
                    )
                )

            now = time.time()
            tick_by_row = progress_every > 0 and rows_written % progress_every == 0
            tick_by_time = now - last_progress >= progress_interval_s
            if tick_by_row or tick_by_time:
                snap = collector.snapshot()
                elapsed = now - t1
                rate = rows_written / max(elapsed, 1e-6)
                if total_rows > 0:
                    pct = 100.0 * rows_written / total_rows
                    bar_width = 28
                    filled = max(0, min(bar_width, int(bar_width * rows_written / total_rows)))
                    bar = "#" * filled + "." * (bar_width - filled)
                    remaining = max(0, total_rows - rows_written)
                    eta_s = remaining / max(rate, 1.0)
                    print(
                        f"      [{bar}] {pct:5.1f}% | {rows_written:,}/{total_rows:,} rows "
                        f"| matched={snap['matched_detections']:,} w/cand={snap['with_candidates']:,} "
                        f"| {rate:,.0f} rows/s | ETA {eta_s:5.0f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"      {rows_written:,} rows | matched={snap['matched_detections']:,} "
                        f"w/cand={snap['with_candidates']:,} | {elapsed:.0f}s",
                        flush=True,
                    )
                last_progress = now

    elapsed = time.time() - t1
    snap = collector.snapshot()
    print(banner, flush=True)
    print(
        f">>> OFFLINE LABELING COMPLETE in {elapsed:.1f}s "
        f"({rows_written / max(elapsed, 1e-6):,.0f} rows/s) <<<",
        flush=True,
    )
    print(
        f"  Rows:    {rows_written:,}  (missing actor frames: {missing_frames:,})",
        flush=True,
    )
    print(
        f"  Matched: {snap['matched_detections']:,} "
        f"({100 * snap.get('match_rate', 0):.2f}% of all, "
        f"{100 * snap.get('match_rate_given_candidates', 0):.1f}% of candidates)",
        flush=True,
    )
    print(f"  Output:  {out_path}", flush=True)
    print(banner, flush=True)
    print(flush=True)

    write_capture_labeling_report(
        collector,
        str(capture_dir),
        labelable_min_speed_mps=labelable_min_speed,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline radar point labeling for a capture folder.")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="sensor_capture_* folder with radar_data.csv and actor_frames.jsonl",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=f"Overwrite {RADAR_CSV} instead of writing {LABELED_CSV}",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Print progress every N rows (0=disable)",
    )
    parser.add_argument(
        "--auto-margin",
        action="store_true",
        help="Derive the hit-match margin from this capture's own margin distribution "
        "(sets DATASET_RADAR_AUTO_MARGIN=1) instead of the fixed 0.5/1.0 m defaults.",
    )
    parser.add_argument(
        "--auto-margin-stride",
        type=int,
        default=None,
        help="Sample every Nth frame when deriving --auto-margin (default 20).",
    )
    args = parser.parse_args()
    if args.auto_margin:
        os.environ["DATASET_RADAR_AUTO_MARGIN"] = "1"
    if args.auto_margin_stride is not None:
        os.environ["DATASET_RADAR_AUTO_MARGIN_STRIDE"] = str(args.auto_margin_stride)
    try:
        out = label_radar_capture_dir(
            args.capture_dir,
            in_place=args.in_place,
            progress_every=args.progress_every,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
