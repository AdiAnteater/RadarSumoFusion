"""
Build plots + tabular summaries from TestRadarLabeling.py aggregated stats.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEPTH_BIN_EDGES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 100)
SCATTER_RESERVOIR_MAX = 8000
FAILURE_SAMPLE_MAX = 5000
BUSIEST_FRAME_POINT_CAP = 25000

# Uncensored nearest-OBB-margin histogram (hit -> nearest candidate actor surface,
# 0.75 m inflation already applied). Bounded by the 7 m candidate pre-filter.
MARGIN_BIN_M = 0.05
MARGIN_MAX_M = 7.0
MARGIN_N_BINS = int(round(MARGIN_MAX_M / MARGIN_BIN_M))
# Env clamp shared with DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M.
MARGIN_FLOOR_M = 0.5
MARGIN_CEIL_M = 25.0


def margin_bin_centers() -> np.ndarray:
    edges = np.arange(0.0, MARGIN_MAX_M + MARGIN_BIN_M, MARGIN_BIN_M)
    return edges[:-1] + MARGIN_BIN_M / 2.0


def margin_bin_index(margin_m: float | None) -> int | None:
    if margin_m is None or margin_m < 0:
        return None
    return min(int(margin_m / MARGIN_BIN_M), MARGIN_N_BINS - 1)


def derive_margin_threshold(
    hist,
    *,
    spike_edge_m: float = 0.15,
    trough_lo_m: float = 0.10,
    trough_hi_m: float = 1.0,
    min_samples: int = 2000,
    floor_m: float = MARGIN_FLOOR_M,
    ceil_m: float = MARGIN_CEIL_M,
) -> dict[str, Any]:
    """Derive a grounded match margin from an uncensored nearest-margin histogram.

    Genuine ray hits land ON the actor OBB -> a spike at margin ~0; clutter that
    merely shares the beam forms a monotonically rising ramp with no far lobe. The
    only natural cut is the trough just past the spike. Returns spike/trough stats,
    a precision (= spike / accepted) table, and suggested primary + single-candidate
    margins (trough, floored at floor_m, so the value only moves UP for captures
    whose clutter encroaches closer — e.g. sparse pps / high speed).
    """
    hist = np.asarray(hist, dtype=np.float64)
    centers = margin_bin_centers()
    total = float(hist.sum())
    out: dict[str, Any] = {
        "samples": int(total),
        "enough_data": total >= min_samples,
        "spike_edge_m": spike_edge_m,
        "spike_count": 0,
        "spike_frac": 0.0,
        "trough_m": None,
        "trough_count": None,
        "suggested_primary_m": None,
        "suggested_single_m": None,
        "precision_table": [],
    }
    if total <= 0:
        return out

    spike = float(hist[centers < spike_edge_m].sum())
    out["spike_count"] = int(spike)
    out["spike_frac"] = spike / total

    cum = np.cumsum(hist)

    def accepted_le(thresh: float) -> float:
        k = int(np.searchsorted(centers, thresh, side="right"))
        return float(cum[k - 1]) if k > 0 else 0.0

    out["precision_table"] = [
        [float(t), int(accepted_le(t)), (spike / accepted_le(t)) if accepted_le(t) > 0 else 0.0]
        for t in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    ]

    win = (centers >= trough_lo_m) & (centers < trough_hi_m)
    idxs = np.where(win)[0]
    if idxs.size and out["enough_data"]:
        smooth = np.convolve(hist, np.ones(3) / 3.0, mode="same")
        ti = int(idxs[int(np.argmin(smooth[idxs]))])
        trough = float(centers[ti])
        out["trough_m"] = round(trough, 3)
        out["trough_count"] = int(hist[ti])
        primary = min(max(trough, floor_m), ceil_m)
        out["suggested_primary_m"] = round(primary, 2)
        out["suggested_single_m"] = round(min(max(primary * 2.0, floor_m), ceil_m), 2)
    return out


@dataclass
class BusiestFramePoint:
    sensor_label: str
    x_m: float
    y_m: float
    depth_m: float
    velocity_mps: float
    category: str
    actor_id: int | None = None
    actor_kind: str = ""


@dataclass
class DetectionRecord:
    sensor_label: str
    frame: int
    had_candidates: bool
    matched: bool
    depth_m: float
    velocity_mps: float
    azimuth_rad: float
    actor_id: int | None = None
    actor_kind: str = ""
    actor_class: str = ""
    match_bbox_margin_m: float | None = None
    nearest_bbox_margin_m: float | None = None
    # Threshold-independent nearest margin (pre-match), for the margin-distribution panel.
    uncensored_nearest_bbox_margin_m: float | None = None


class LabelingStatsCollector:
    """Thread-safe aggregates + reservoir sample for plots (bounded memory)."""

    def __init__(self, *, labelable_min_speed_mps: float | None = None) -> None:
        import threading

        self.lock = threading.Lock()
        self.labelable_min_speed_mps = labelable_min_speed_mps
        self.total_detections = 0
        self.static_skipped = 0
        self.matched_detections = 0
        self.no_actor_candidates = 0
        self.failed_match = 0
        self.radar_messages = 0
        self.raw_radar_returns = 0
        self.legacy_matched = 0
        self.legacy_failed = 0

        self.by_sensor: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "detections": 0,
                "matched": 0,
                "no_candidates": 0,
                "failed_match": 0,
            }
        )
        self.points_by_vehicle: Counter[int] = Counter()
        self.points_by_pedestrian: Counter[int] = Counter()
        self.points_by_actor_class: Counter[str] = Counter()
        self.points_by_vehicle_class: Counter[str] = Counter()
        self.points_by_sensor: Counter[str] = Counter()
        self.matched_vehicle_by_sensor: Counter[str] = Counter()
        self.matched_pedestrian_by_sensor: Counter[str] = Counter()
        self.vehicle_sensor_matrix: Counter[tuple[int, str]] = Counter()
        self.depth_hist_matched = np.zeros(len(DEPTH_BIN_EDGES) - 1, dtype=np.int64)
        self.depth_hist_unmatched = np.zeros(len(DEPTH_BIN_EDGES) - 1, dtype=np.int64)
        # Uncensored nearest-OBB margin over all returns that had a candidate.
        self.margin_hist = np.zeros(MARGIN_N_BINS, dtype=np.int64)
        self.margin_samples = 0
        self.velocity_samples_matched: list[float] = []
        self.velocity_samples_unmatched: list[float] = []
        self._velocity_cap = 12000

        # (frame, vehicle_id) -> sensors that matched that car in that frame
        self._frame_vehicle_sensors: dict[tuple[int, int], set[str]] = defaultdict(set)
        self.co_visibility_hist: Counter[int] = Counter()

        self._scatter_seen = 0
        self.scatter_xy: list[tuple[float, float, str, str]] = []

        self._frame_totals: Counter[int] = Counter()
        self._frame_matched: Counter[int] = Counter()
        self._frame_no_candidates: Counter[int] = Counter()
        self._frame_failed: Counter[int] = Counter()
        self._frame_sensors: dict[int, set[str]] = defaultdict(set)
        self._frame_actors: dict[int, set[int]] = defaultdict(set)
        self._frame_vehicle_ids: dict[int, set[int]] = defaultdict(set)
        self._frame_pedestrian_ids: dict[int, set[int]] = defaultdict(set)
        self._frame_points: dict[int, list[BusiestFramePoint]] = defaultdict(list)

        self._failure_samples_seen = 0
        self.failure_samples: list[dict[str, Any]] = []

        # Per-run telemetry set by the test loop (queue drops, per-radar arrival timing,
        # world frame range). Mirrored into summary.json under "runtime".
        self.runtime_telemetry: dict[str, Any] = {}

    def set_runtime_telemetry(self, telemetry: dict[str, Any]) -> None:
        with self.lock:
            self.runtime_telemetry = dict(telemetry)

    def _depth_bin_index(self, depth_m: float) -> int | None:
        if depth_m < 0:
            return None
        for i in range(len(DEPTH_BIN_EDGES) - 1):
            if DEPTH_BIN_EDGES[i] <= depth_m < DEPTH_BIN_EDGES[i + 1]:
                return i
        return len(DEPTH_BIN_EDGES) - 2

    def _maybe_add_scatter(
        self,
        depth_m: float,
        azimuth_rad: float,
        sensor_label: str,
        category: str,
    ) -> None:
        x_m = depth_m * math.cos(azimuth_rad)
        y_m = depth_m * math.sin(azimuth_rad)
        self._scatter_seen += 1
        if len(self.scatter_xy) < SCATTER_RESERVOIR_MAX:
            self.scatter_xy.append((x_m, y_m, sensor_label, category))
        else:
            j = np.random.randint(0, self._scatter_seen)
            if j < SCATTER_RESERVOIR_MAX:
                self.scatter_xy[j] = (x_m, y_m, sensor_label, category)

    def _maybe_add_velocity(self, velocity_mps: float, matched: bool) -> None:
        bucket = self.velocity_samples_matched if matched else self.velocity_samples_unmatched
        if len(bucket) < self._velocity_cap:
            bucket.append(velocity_mps)

    def _maybe_add_failure_sample(self, rec: DetectionRecord) -> None:
        if rec.matched or not rec.had_candidates:
            return
        row = {
            "frame": rec.frame,
            "sensor_label": rec.sensor_label,
            "depth_m": round(rec.depth_m, 4),
            "velocity_mps": round(rec.velocity_mps, 4),
            "nearest_bbox_margin_m": rec.nearest_bbox_margin_m,
        }
        self._failure_samples_seen += 1
        if len(self.failure_samples) < FAILURE_SAMPLE_MAX:
            self.failure_samples.append(row)
        else:
            j = np.random.randint(0, self._failure_samples_seen)
            if j < FAILURE_SAMPLE_MAX:
                self.failure_samples[j] = row

    def _append_frame_point(self, frame: int, point: BusiestFramePoint) -> None:
        buf = self._frame_points[frame]
        if len(buf) < BUSIEST_FRAME_POINT_CAP:
            buf.append(point)

    def record_message(self, *, raw_returns: int = 0) -> None:
        with self.lock:
            self.radar_messages += 1
            self.raw_radar_returns += max(0, raw_returns)

    def record_static_skipped(self, sensor_label: str) -> None:
        """Radar return with |velocity| at or below labelable threshold (static clutter)."""
        with self.lock:
            self.static_skipped += 1
            self.by_sensor[sensor_label]["static_skipped"] = (
                self.by_sensor[sensor_label].get("static_skipped", 0) + 1
            )

    def record_detection(self, rec: DetectionRecord, *, legacy_matched: bool | None = None) -> None:
        with self.lock:
            self.total_detections += 1
            sl = rec.sensor_label
            bucket = self.by_sensor[sl]
            bucket["detections"] += 1
            self.points_by_sensor[sl] += 1

            category = "no_candidates"
            hist = None
            if not rec.had_candidates:
                self.no_actor_candidates += 1
                bucket["no_candidates"] += 1
            elif rec.matched:
                self.matched_detections += 1
                bucket["matched"] += 1
                category = "matched"
                hist = self.depth_hist_matched
                if rec.actor_kind == "pedestrian":
                    self.matched_pedestrian_by_sensor[sl] += 1
                else:
                    self.matched_vehicle_by_sensor[sl] += 1
                aid = rec.actor_id
                if aid is not None:
                    if rec.actor_kind == "pedestrian":
                        self.points_by_pedestrian[aid] += 1
                    else:
                        self.points_by_vehicle[aid] += 1
                        self.vehicle_sensor_matrix[(aid, sl)] += 1
                        self._frame_vehicle_sensors[(rec.frame, aid)].add(sl)
                    if rec.actor_class:
                        self.points_by_actor_class[rec.actor_class] += 1
                        if rec.actor_kind == "vehicle":
                            self.points_by_vehicle_class[rec.actor_class] += 1
            else:
                self.failed_match += 1
                bucket["failed_match"] += 1
                category = "failed_match"
                hist = self.depth_hist_unmatched
                self._maybe_add_failure_sample(rec)

            if legacy_matched is True:
                self.legacy_matched += 1
            elif legacy_matched is False:
                self.legacy_failed += 1

            if hist is not None:
                idx = self._depth_bin_index(rec.depth_m)
                if idx is not None:
                    hist[idx] += 1

            if rec.had_candidates:
                mbi = margin_bin_index(rec.uncensored_nearest_bbox_margin_m)
                if mbi is not None:
                    self.margin_hist[mbi] += 1
                    self.margin_samples += 1

            if category != "no_candidates":
                self._maybe_add_scatter(rec.depth_m, rec.azimuth_rad, sl, category)
                self._maybe_add_velocity(rec.velocity_mps, rec.matched)

            fr = rec.frame
            self._frame_totals[fr] += 1
            self._frame_sensors[fr].add(sl)
            if not rec.had_candidates:
                self._frame_no_candidates[fr] += 1
            elif rec.matched:
                self._frame_matched[fr] += 1
                if rec.actor_id is not None:
                    self._frame_actors[fr].add(rec.actor_id)
                    if rec.actor_kind == "pedestrian":
                        self._frame_pedestrian_ids[fr].add(rec.actor_id)
                    else:
                        self._frame_vehicle_ids[fr].add(rec.actor_id)
            else:
                self._frame_failed[fr] += 1

            x_m = rec.depth_m * math.cos(rec.azimuth_rad)
            y_m = rec.depth_m * math.sin(rec.azimuth_rad)
            self._append_frame_point(
                fr,
                BusiestFramePoint(
                    sensor_label=sl,
                    x_m=x_m,
                    y_m=y_m,
                    depth_m=rec.depth_m,
                    velocity_mps=rec.velocity_mps,
                    category=category,
                    actor_id=rec.actor_id,
                    actor_kind=rec.actor_kind,
                ),
            )

    def _resolve_busiest_frame_unlocked(self) -> int | None:
        """Frame with most matched points (tie: more total, then higher frame id). Caller holds lock."""
        if not self._frame_totals:
            return None

        def sort_key(frame_id: int) -> tuple:
            return (
                self._frame_matched[frame_id],
                self._frame_totals[frame_id],
                frame_id,
            )

        return max(self._frame_totals.keys(), key=sort_key)

    def resolve_busiest_frame(self) -> int | None:
        with self.lock:
            return self._resolve_busiest_frame_unlocked()

    def busiest_frame_snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            frame_id = self._resolve_busiest_frame_unlocked()
            if frame_id is None:
                return None
            total = self._frame_totals[frame_id]
            matched = self._frame_matched[frame_id]
            return {
                "frame": frame_id,
                "total_points": total,
                "matched_points": matched,
                "match_rate": (matched / total) if total else 0.0,
                "no_candidates": self._frame_no_candidates[frame_id],
                "failed_match": self._frame_failed[frame_id],
                "distinct_radars": len(self._frame_sensors[frame_id]),
                "distinct_actors": len(self._frame_actors[frame_id]),
                "distinct_vehicles": len(self._frame_vehicle_ids[frame_id]),
                "distinct_pedestrians": len(self._frame_pedestrian_ids[frame_id]),
                "radar_labels": sorted(self._frame_sensors[frame_id]),
                "actor_ids": sorted(self._frame_actors[frame_id]),
                "vehicle_ids": sorted(self._frame_vehicle_ids[frame_id]),
                "stored_points_for_plot": len(self._frame_points.get(frame_id, [])),
                "with_candidates": total
                - self._frame_no_candidates[frame_id],
            }

    @staticmethod
    def _derived_rates(
        total: int,
        matched: int,
        no_candidates: int,
        failed_match: int,
    ) -> dict[str, float | int]:
        with_candidates = total - no_candidates
        return {
            "with_candidates": with_candidates,
            "match_rate_given_candidates": (matched / with_candidates)
            if with_candidates
            else 0.0,
            "clutter_or_out_of_fov": no_candidates,
            "clutter_rate": (no_candidates / total) if total else 0.0,
            "failed_match_rate_given_candidates": (failed_match / with_candidates)
            if with_candidates
            else 0.0,
        }

    def finalize_co_visibility(self) -> None:
        with self.lock:
            self.co_visibility_hist.clear()
            for sensors in self._frame_vehicle_sensors.values():
                if sensors:
                    self.co_visibility_hist[len(sensors)] += 1

    def margin_threshold_diagnostics(self) -> dict[str, Any]:
        """Spike/trough/precision stats + suggested margins from the uncensored hist."""
        with self.lock:
            hist = self.margin_hist.copy()
        return derive_margin_threshold(hist)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            total = self.total_detections
            matched = self.matched_detections
            unique_vehicles = len(self.points_by_vehicle)
            unique_pedestrians = len(self.points_by_pedestrian)
            sensors_per_vehicle = [
                len({s for (vid, s), _ in self.vehicle_sensor_matrix.items() if vid == v})
                for v in self.points_by_vehicle
            ]
            derived = self._derived_rates(
                total, matched, self.no_actor_candidates, self.failed_match
            )
            by_sensor_out = {}
            for label, b in self.by_sensor.items():
                det = b["detections"]
                nc = b["no_candidates"]
                m = b["matched"]
                wc = det - nc
                by_sensor_out[label] = {
                    **b,
                    "with_candidates": wc,
                    "match_rate_given_candidates": (m / wc) if wc else 0.0,
                }

            # Point density per radar per frame. radar_frames = distinct (sensor,frame)
            # sweeps; each density = point count / radar_frames.
            radar_frames = sum(len(s) for s in self._frame_sensors.values())
            per_sensor_frames: Counter[str] = Counter()
            for sensors in self._frame_sensors.values():
                for s in sensors:
                    per_sensor_frames[s] += 1
            veh_pts = int(sum(self.points_by_vehicle.values()))
            ped_pts = int(sum(self.points_by_pedestrian.values()))

            def _pd(x: float) -> float:
                return round(x / radar_frames, 4) if radar_frames else 0.0

            density_by_sensor = {}
            for label in sorted(self.by_sensor):
                nf = per_sensor_frames.get(label, 0)
                density_by_sensor[label] = {
                    "frames": nf,
                    "all": round(self.points_by_sensor.get(label, 0) / nf, 3) if nf else 0.0,
                    "matched": round(self.by_sensor[label].get("matched", 0) / nf, 4) if nf else 0.0,
                    "vehicle": round(self.matched_vehicle_by_sensor.get(label, 0) / nf, 4) if nf else 0.0,
                    "pedestrian": round(self.matched_pedestrian_by_sensor.get(label, 0) / nf, 4) if nf else 0.0,
                }
            density_per_radar_per_frame = {
                "radar_frames": radar_frames,
                "all": _pd(total),
                "matched": _pd(matched),
                "vehicle": _pd(veh_pts),
                "pedestrian": _pd(ped_pts),
                "clutter_unmatched": _pd(total - matched),
                "by_vehicle_class": {
                    c: _pd(n) for c, n in sorted(self.points_by_vehicle_class.items())
                },
                "by_sensor": density_by_sensor,
                "note": "points per radar per frame; radar_frames = distinct (sensor,frame) sweeps",
            }
            return {
                "total_detections": total,
                "labelable_detections": total,
                "static_skipped": self.static_skipped,
                "total_radar_points": total + self.static_skipped,
                "labelable_min_speed_mps": self.labelable_min_speed_mps,
                "matched_detections": matched,
                "match_rate": (matched / total) if total else 0.0,
                **derived,
                "no_actor_candidates": self.no_actor_candidates,
                "no_vehicle_candidates": self.no_actor_candidates,
                "failed_match": self.failed_match,
                "radar_messages": self.radar_messages,
                "raw_radar_returns": self.raw_radar_returns,
                "avg_raw_returns_per_message": (
                    self.raw_radar_returns / self.radar_messages
                    if self.radar_messages
                    else 0.0
                ),
                "by_sensor": by_sensor_out,
                "density_per_radar_per_frame": density_per_radar_per_frame,
                "legacy_matched": self.legacy_matched,
                "legacy_failed": self.legacy_failed,
                "unique_actors_matched": unique_vehicles + unique_pedestrians,
                "unique_vehicles_matched": unique_vehicles,
                "unique_pedestrians_matched": unique_pedestrians,
                "median_radars_per_vehicle": float(np.median(sensors_per_vehicle))
                if sensors_per_vehicle
                else 0.0,
                "max_radars_per_vehicle": max(sensors_per_vehicle) if sensors_per_vehicle else 0,
                # derive_margin_threshold reads the array only (no lock) — safe under self.lock.
                "margin_samples": int(self.margin_samples),
                "margin_threshold": derive_margin_threshold(self.margin_hist),
            }


def _bin_labels() -> list[str]:
    labels = []
    for i in range(len(DEPTH_BIN_EDGES) - 1):
        a, b = DEPTH_BIN_EDGES[i], DEPTH_BIN_EDGES[i + 1]
        labels.append(f"{a}-{b}m")
    return labels


def write_csv_tables(
    out_dir: Path,
    collector: LabelingStatsCollector,
    *,
    hit_match_max_margin_m: float = 2.0,
) -> None:
    with collector.lock:
        vehicle_rows = []
        for vid, count in collector.points_by_vehicle.most_common():
            sensors = sorted(
                s for (v, s), c in collector.vehicle_sensor_matrix.items() if v == vid and c > 0
            )
            vehicle_rows.append(
                {
                    "vehicle_id": vid,
                    "matched_point_count": count,
                    "distinct_radar_count": len(sensors),
                    "radar_labels": ";".join(sensors),
                }
            )

        pedestrian_rows = []
        for pid, count in collector.points_by_pedestrian.most_common():
            pedestrian_rows.append(
                {
                    "pedestrian_id": pid,
                    "matched_point_count": count,
                }
            )

        sensor_rows = []
        for label in sorted(collector.by_sensor):
            b = collector.by_sensor[label]
            det = b["detections"]
            wc = det - b["no_candidates"]
            sensor_rows.append(
                {
                    "sensor_label": label,
                    "total_points": det,
                    "with_candidates": wc,
                    "matched_points": b["matched"],
                    "match_rate_pct_all": round(100.0 * b["matched"] / det, 4) if det else 0.0,
                    "match_rate_pct_given_candidates": round(
                        100.0 * b["matched"] / wc, 4
                    )
                    if wc
                    else 0.0,
                    "no_candidates": b["no_candidates"],
                    "failed_match": b["failed_match"],
                    "static_skipped": b.get("static_skipped", 0),
                }
            )

        matrix_path = out_dir / "vehicle_radar_matrix.csv"
        with matrix_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["vehicle_id", "sensor_label", "matched_point_count"])
            for (vid, sl), cnt in sorted(collector.vehicle_sensor_matrix.items()):
                w.writerow([vid, sl, cnt])

        frame_rows = []
        for fid in collector._frame_totals:
            tot = collector._frame_totals[fid]
            m = collector._frame_matched[fid]
            nc = collector._frame_no_candidates[fid]
            wc = tot - nc
            frame_rows.append(
                {
                    "frame": fid,
                    "total_points": tot,
                    "with_candidates": wc,
                    "matched_points": m,
                    "match_rate_pct_all": round(100.0 * m / tot, 4) if tot else 0.0,
                    "match_rate_pct_given_candidates": round(100.0 * m / wc, 4) if wc else 0.0,
                    "no_candidates": nc,
                    "failed_match": collector._frame_failed[fid],
                    "distinct_radars": len(collector._frame_sensors[fid]),
                    "distinct_vehicles": len(collector._frame_vehicle_ids[fid]),
                    "distinct_pedestrians": len(collector._frame_pedestrian_ids[fid]),
                }
            )
        frame_rows.sort(key=lambda r: (-r["matched_points"], -r["with_candidates"], -r["frame"]))

    vpath = out_dir / "per_vehicle_summary.csv"
    with vpath.open("w", newline="", encoding="utf-8") as f:
        if vehicle_rows:
            w = csv.DictWriter(f, fieldnames=list(vehicle_rows[0].keys()))
            w.writeheader()
            w.writerows(vehicle_rows)

    ppath = out_dir / "per_pedestrian_summary.csv"
    with ppath.open("w", newline="", encoding="utf-8") as f:
        if pedestrian_rows:
            w = csv.DictWriter(f, fieldnames=list(pedestrian_rows[0].keys()))
            w.writeheader()
            w.writerows(pedestrian_rows)

    spath = out_dir / "per_sensor_summary.csv"
    with spath.open("w", newline="", encoding="utf-8") as f:
        if sensor_rows:
            w = csv.DictWriter(f, fieldnames=list(sensor_rows[0].keys()))
            w.writeheader()
            w.writerows(sensor_rows)

    fpath = out_dir / "per_frame_summary.csv"
    with fpath.open("w", newline="", encoding="utf-8") as f:
        if frame_rows:
            w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            w.writeheader()
            w.writerows(frame_rows)

    fail_path = out_dir / "labeling_failure_samples.csv"
    with fail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "sensor_label",
                "depth_m",
                "velocity_mps",
                "nearest_bbox_margin_m",
                "over_match_threshold_m",
            ],
        )
        w.writeheader()
        for row in collector.failure_samples:
            margin = row.get("nearest_bbox_margin_m")
            over = ""
            if margin is not None:
                over = round(max(0.0, float(margin) - hit_match_max_margin_m), 4)
            w.writerow({**row, "over_match_threshold_m": over})


def _busiest_frame_points_for_plot(collector: LabelingStatsCollector) -> list[BusiestFramePoint]:
    with collector.lock:
        frame_id = collector._resolve_busiest_frame_unlocked()
        if frame_id is None:
            return []
        return list(collector._frame_points.get(frame_id, []))


def write_busiest_frame_report(collector: LabelingStatsCollector, out_dir: Path) -> Path | None:
    bf = collector.busiest_frame_snapshot()
    if bf is None:
        return None

    points = _busiest_frame_points_for_plot(collector)

    # CSV of busiest-frame points (capped buffer)
    bcsv = out_dir / "busiest_frame_points.csv"
    with bcsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sensor_label",
                "x_m",
                "y_m",
                "depth_m",
                "velocity_mps",
                "category",
                "actor_id",
                "actor_kind",
            ],
        )
        w.writeheader()
        for p in points:
            w.writerow(
                {
                    "sensor_label": p.sensor_label,
                    "x_m": round(p.x_m, 4),
                    "y_m": round(p.y_m, 4),
                    "depth_m": round(p.depth_m, 4),
                    "velocity_mps": round(p.velocity_mps, 4),
                    "category": p.category,
                    "actor_id": p.actor_id if p.actor_id is not None else "",
                    "actor_kind": p.actor_kind,
                }
            )

    # Per-sensor / per-actor counts within busiest frame
    by_sensor = Counter(p.sensor_label for p in points)
    by_actor = Counter(p.actor_id for p in points if p.actor_id is not None)

    colors = {"matched": "#27ae60", "failed_match": "#e74c3c", "no_candidates": "#bdc3c7"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    if points:
        for cat in colors:
            xs = [p.x_m for p in points if p.category == cat]
            ys = [p.y_m for p in points if p.category == cat]
            if xs:
                ax.scatter(xs, ys, s=8, alpha=0.5, c=colors[cat], label=cat)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(markerscale=2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No point buffer", ha="center", va="center")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y lateral (m)")
    ax.set_title(f"Busiest frame {bf['frame']} — top-down cloud")

    ax = axes[0, 1]
    if by_sensor:
        labels = sorted(by_sensor.keys())
        matched_c = [
            sum(1 for p in points if p.sensor_label == l and p.category == "matched")
            for l in labels
        ]
        other_c = [by_sensor[l] - matched_c[i] for i, l in enumerate(labels)]
        x = np.arange(len(labels))
        ax.bar(x, other_c, label="Unlabeled", color="#bdc3c7")
        ax.bar(x, matched_c, bottom=other_c, label="Matched", color="#27ae60")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Points")
        ax.set_title("Points per radar (this frame)")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    ax = axes[0, 2]
    if by_actor:
        items = by_actor.most_common(15)
        ids = [str(v) for v, _ in items]
        counts = [c for _, c in items]
        ax.barh(ids[::-1], counts[::-1], color="#3498db")
        ax.set_xlabel("Matched points")
        ax.set_title("Points per actor (this frame)")
    else:
        ax.text(0.5, 0.5, "No matched actors\nin this frame", ha="center", va="center")

    ax = axes[1, 0]
    sizes = [bf["matched_points"], bf["no_candidates"], bf["failed_match"]]
    labels_p = ["Matched", "No actor nearby", "Failed match"]
    if sum(sizes):
        ax.pie(sizes, labels=labels_p, autopct="%1.1f%%", startangle=90)
    ax.set_title("Outcome mix (busiest frame)")

    ax = axes[1, 1]
    if points:
        depths = [p.depth_m for p in points]
        ax.hist(depths, bins=30, color="#34495e", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Count")
        ax.set_title("Depth histogram (busiest frame)")
    else:
        ax.axis("off")

    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"Busiest frame: {bf['frame']}",
        f"  Total points:     {bf['total_points']}",
        f"  Matched:          {bf['matched_points']} ({100 * bf['match_rate']:.1f}%)",
        f"  Distinct radars:  {bf['distinct_radars']} {bf['radar_labels']}",
        f"  Distinct actors:  {bf['distinct_actors']} (veh={bf['distinct_vehicles']}, ped={bf.get('distinct_pedestrians', 0)})",
        f"  Actor ids:        {bf['actor_ids'][:12]}{'...' if len(bf['actor_ids']) > 12 else ''}",
        "",
        "Per-sensor totals:",
    ]
    for sl, cnt in sorted(by_sensor.items(), key=lambda x: -x[1]):
        m = sum(1 for p in points if p.sensor_label == sl and p.category == "matched")
        lines.append(f"  {sl}: {cnt} pts ({m} matched)")
    if bf.get("plot_note"):
        lines.extend(["", bf["plot_note"]])
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=9, family="monospace")

    fig.suptitle(
        f"Busiest frame analysis — frame {bf['frame']} ({bf['total_points']} points)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = out_dir / "busiest_frame_summary.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    json_path = out_dir / "busiest_frame_summary.json"
    payload = {
        **bf,
        "by_sensor_point_count": dict(by_sensor),
        "by_actor_matched_point_count": {str(k): v for k, v in by_actor.items()},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return plot_path


def _plot_outcome_pie(ax, snap: dict) -> None:
    """Matched vs unmatched among returns that had an actor candidate (excludes static/clutter)."""
    matched = snap["matched_detections"]
    unmatched = snap["failed_match"]
    pool = snap.get("with_candidates", matched + unmatched)
    excluded = snap.get("no_actor_candidates", 0)
    static = snap.get("static_skipped", 0)

    sizes = [matched, unmatched]
    labels = ["Matched", "Unmatched"]
    colors = ["#27ae60", "#e74c3c"]

    if pool == 0:
        ax.text(
            0.5,
            0.5,
            "No returns with\nactor candidates",
            ha="center",
            va="center",
        )
        ax.set_title("Labeling outcomes (target pool)")
        return

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    title = f"Labeling outcomes (n={pool:,} w/ candidates)"
    note_parts = [f"excluded clutter/no actor: {excluded:,}"]
    if static:
        note_parts.append(f"static skipped: {static:,}")
    title += "\n" + ", ".join(note_parts)
    ax.set_title(title, fontsize=10)


def _plot_depth_hist(ax, collector: LabelingStatsCollector) -> None:
    labels = _bin_labels()
    x = np.arange(len(labels))
    w = 0.38
    with collector.lock:
        m = collector.depth_hist_matched
        u = collector.depth_hist_unmatched
    ax.bar(x - w / 2, m, width=w, label="Matched", color="#2ecc71")
    ax.bar(x + w / 2, u, width=w, label="Unmatched (should label)", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Point count")
    ax.set_title("Depth (w/ actor candidates only)")
    ax.legend()


def _plot_points_per_vehicle(ax, collector: LabelingStatsCollector) -> None:
    with collector.lock:
        items = collector.points_by_vehicle.most_common(25)
    if not items:
        ax.text(0.5, 0.5, "No matched vehicles", ha="center", va="center")
        return
    ids = [str(v) for v, _ in items]
    counts = [c for _, c in items]
    ax.barh(ids[::-1], counts[::-1], color="#3498db")
    ax.set_xlabel("Matched radar points")
    ax.set_title("Top vehicles by point count")


def _plot_points_per_sensor(ax, collector: LabelingStatsCollector) -> None:
    with collector.lock:
        labels = sorted(collector.by_sensor)
        pool = []
        matched = []
        for l in labels:
            b = collector.by_sensor[l]
            wc = b["detections"] - b["no_candidates"]
            pool.append(wc)
            matched.append(b["matched"])
    x = np.arange(len(labels))
    unmatched = [max(0, pool[i] - matched[i]) for i in range(len(labels))]
    ax.bar(x, unmatched, label="Unmatched (should label)", color="#e74c3c", alpha=0.85)
    ax.bar(x, matched, bottom=unmatched, label="Matched", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Point count")
    ax.set_title("Points per radar (w/ candidates)")
    ax.legend(fontsize=8)


def _plot_sensor_match_rate(ax, collector: LabelingStatsCollector, min_match_rate: float) -> None:
    with collector.lock:
        labels = sorted(collector.by_sensor)
        rates = []
        for l in labels:
            b = collector.by_sensor[l]
            wc = b["detections"] - b["no_candidates"]
            m = b["matched"]
            rates.append(100.0 * m / wc if wc else 0.0)
    ax.bar(labels, rates, color="#9b59b6")
    ax.set_ylabel("Match rate (%)")
    ax.set_title("Label success per radar (given candidates)")
    ax.axhline(
        100.0 * min_match_rate,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"{100 * min_match_rate:.0f}% PASS",
    )
    ax.legend()


def _plot_co_visibility(ax, collector: LabelingStatsCollector) -> None:
    with collector.lock:
        items = sorted(collector.co_visibility_hist.items())
    if not items:
        ax.text(0.5, 0.5, "No co-visible (frame, vehicle) pairs", ha="center", va="center")
        return
    counts = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.bar([str(c) for c in counts], vals, color="#e67e22")
    ax.set_xlabel("Radars seeing same car in same frame")
    ax.set_ylabel("Frame–vehicle pairs")
    ax.set_title("Multi-radar coverage (same frame)")


def _plot_scatter(ax, collector: LabelingStatsCollector) -> None:
    with collector.lock:
        pts = list(collector.scatter_xy)
    if not pts:
        ax.text(0.5, 0.5, "No scatter sample", ha="center", va="center")
        return
    colors = {"matched": "#27ae60", "failed_match": "#e74c3c"}
    labels = {"matched": "Matched", "failed_match": "Unmatched"}
    for cat in colors:
        xs = [p[0] for p in pts if p[3] == cat]
        ys = [p[1] for p in pts if p[3] == cat]
        if xs:
            ax.scatter(xs, ys, s=4, alpha=0.35, c=colors[cat], label=labels[cat])
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y lateral (m)")
    ax.set_title(f"Top-down sample (w/ candidates, n≤{SCATTER_RESERVOIR_MAX})")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(markerscale=3, loc="upper right")


def _plot_velocity(ax, collector: LabelingStatsCollector) -> None:
    with collector.lock:
        vm = collector.velocity_samples_matched
        vu = collector.velocity_samples_unmatched
    if not vm and not vu:
        ax.text(0.5, 0.5, "No velocity samples", ha="center", va="center")
        return
    if vm:
        ax.hist(vm, bins=40, alpha=0.6, label="Matched", color="#2ecc71", density=True)
    if vu:
        ax.hist(
            vu,
            bins=40,
            alpha=0.5,
            label="Unmatched (should label)",
            color="#e74c3c",
            density=True,
        )
    ax.set_xlabel("Radial velocity (m/s)")
    ax.set_ylabel("Density")
    ax.set_title("Velocity (w/ candidates only)")
    ax.legend(fontsize=8)


def _precision_at(hist: np.ndarray, centers: np.ndarray, spike: float, thresh: float) -> float:
    k = int(np.searchsorted(centers, thresh, side="right"))
    accepted = float(np.cumsum(hist)[k - 1]) if k > 0 else 0.0
    return (spike / accepted) if accepted > 0 else 0.0


def _plot_margin_hist(
    ax,
    collector: LabelingStatsCollector,
    diag: dict[str, Any],
    *,
    current_primary: float | None,
    current_single: float | None,
    x_max: float = 4.0,
) -> None:
    with collector.lock:
        hist = collector.margin_hist.copy()
    centers = margin_bin_centers()
    if hist.sum() <= 0:
        ax.text(0.5, 0.5, "No candidate returns", ha="center", va="center")
        ax.set_title("Hit→OBB margin distribution")
        return
    keep = centers <= x_max
    ax.bar(centers[keep], hist[keep], width=MARGIN_BIN_M, color="#34495e", align="center")
    ax.set_yscale("log")
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Nearest-OBB margin (m)  [0.75 m inflation applied]")
    ax.set_ylabel("Returns (log)")
    ax.set_title(f"Hit→OBB margin (uncensored, n={int(hist.sum()):,}, 7 m cap)")

    def vline(x, color, label):
        if x is not None and 0 <= x <= x_max:
            ax.axvline(x, color=color, ls="--", lw=1.4, label=label)

    vline(diag.get("spike_edge_m"), "#7f8c8d", f"spike <{diag.get('spike_edge_m')} m")
    vline(diag.get("trough_m"), "#27ae60", f"trough {diag.get('trough_m')} m")
    vline(current_primary, "#e74c3c", f"current primary {current_primary} m")
    sug = diag.get("suggested_primary_m")
    if sug is not None and sug != current_primary:
        vline(sug, "#2980b9", f"suggested {sug} m")
    ax.legend(fontsize=7, loc="upper right")


def _plot_margin_precision(
    ax,
    collector: LabelingStatsCollector,
    diag: dict[str, Any],
    *,
    current_primary: float | None,
    x_max: float = 3.0,
) -> None:
    with collector.lock:
        hist = collector.margin_hist.copy()
    centers = margin_bin_centers()
    total = float(hist.sum())
    if total <= 0:
        ax.text(0.5, 0.5, "No candidate returns", ha="center", va="center")
        ax.set_title("On-body precision vs threshold")
        return
    spike = float(diag.get("spike_count") or 0.0)
    cum = np.cumsum(hist)
    keep = centers <= x_max
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(cum > 0, spike / cum, 0.0)
    ax.plot(centers[keep], 100.0 * precision[keep], color="#2980b9", lw=2, label="precision (on-body)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Match threshold T (m)")
    ax.set_ylabel("On-body precision = spike / accepted (%)")
    ax.set_title("Precision vs match threshold")
    if current_primary is not None and 0 <= current_primary <= x_max:
        p = 100.0 * _precision_at(hist, centers, spike, current_primary)
        ax.axvline(current_primary, color="#e74c3c", ls="--", lw=1.4)
        ax.annotate(f"current {current_primary} m\n{p:.0f}%", (current_primary, p),
                    textcoords="offset points", xytext=(6, -4), fontsize=8, color="#e74c3c")
    sug = diag.get("suggested_primary_m")
    if sug is not None and 0 <= sug <= x_max and sug != current_primary:
        p = 100.0 * _precision_at(hist, centers, spike, sug)
        ax.axvline(sug, color="#27ae60", ls="--", lw=1.4)
        ax.annotate(f"suggested {sug} m\n{p:.0f}%", (sug, p),
                    textcoords="offset points", xytext=(6, 8), fontsize=8, color="#27ae60")
    ax.grid(True, alpha=0.3)


def _plot_margin_text(
    ax,
    collector: LabelingStatsCollector,
    diag: dict[str, Any],
    *,
    current_primary: float | None,
    current_single: float | None,
) -> None:
    ax.axis("off")
    with collector.lock:
        hist = collector.margin_hist.copy()
    centers = margin_bin_centers()
    spike = float(diag.get("spike_count") or 0.0)
    lines = ["Match-margin calibration", "(uncensored hit→nearest-OBB margin)", ""]
    if not diag.get("enough_data"):
        lines.append(f"samples: {diag.get('samples', 0):,}  (need >=2000")
        lines.append("for a stable trough — using config as-is)")
    else:
        lines.append(f"samples:  {diag['samples']:,}")
        lines.append(
            f"spike:    {diag['spike_count']:,} "
            f"({100 * diag['spike_frac']:.1f}%) < {diag['spike_edge_m']} m"
        )
        lines.append(f"trough:   {diag['trough_m']} m ({diag['trough_count']}/bin)")
        lines.append("")
        cp = current_primary
        if cp is not None:
            lines.append(
                f"current:  primary {cp} m  -> {100 * _precision_at(hist, centers, spike, cp):.0f}% on-body"
            )
        if current_single is not None:
            lines.append(f"          single {current_single} m")
        lines.append(
            f"suggest:  primary {diag['suggested_primary_m']} m  -> "
            f"{100 * _precision_at(hist, centers, spike, diag['suggested_primary_m']):.0f}% on-body"
        )
        lines.append(f"          single  {diag['suggested_single_m']} m")
        lines.append("")
        lines.append("precision (on-body / accepted) by T:")
        for t, acc, prec in diag["precision_table"]:
            lines.append(f"   T={t:>4} m  acc={acc:>8,}  {100 * prec:5.1f}%")
        if diag["suggested_primary_m"] != current_primary:
            lines.append("")
            lines.append("apply: DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M=")
            lines.append(f"       {diag['suggested_primary_m']}  (+ ..._SINGLE_...={diag['suggested_single_m']})")
            lines.append("or set DATASET_RADAR_AUTO_MARGIN=1")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=9, family="monospace")


def write_live_snapshot(out_dir: Path, snap: dict[str, Any]) -> None:
    """Lightweight progress file while the test is still running."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "live_stats.json"
    seq = 1
    if dest.is_file():
        try:
            prev = json.loads(dest.read_text(encoding="utf-8"))
            seq = int(prev.get("seq", 0)) + 1
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            seq = 1
    payload = {
        "seq": seq,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **snap,
    }
    body = json.dumps(payload, indent=2)
    tmp = out_dir / ".live_stats.json.tmp"
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, dest)
    meta_path = out_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("status") == "running":
            meta["last_stats"] = {
                "total_detections": snap.get("total_detections", 0),
                "matched_detections": snap.get("matched_detections", 0),
                "with_candidates": snap.get("with_candidates", 0),
                "match_rate": snap.get("match_rate", 0.0),
                "match_rate_given_candidates": snap.get(
                    "match_rate_given_candidates", 0.0
                ),
                "unique_vehicles_matched": snap.get("unique_vehicles_matched", 0),
            }
            meta["last_updated_at"] = payload["updated_at"]
            meta["live_stats_seq"] = seq
            meta_tmp = out_dir / ".run_meta.json.tmp"
            meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            os.replace(meta_tmp, meta_path)


def write_report(
    collector: LabelingStatsCollector,
    out_dir: Path,
    *,
    min_match_rate: float,
    expected_radar_labels: set[str],
    proximity_m: float,
    hit_match_m: float,
    labelable_min_speed_mps: float,
    candidate_max_range_m: float | None = None,
    candidate_horizontal_fov_deg: float | None = None,
    candidate_depth_margin_m: float | None = None,
    candidate_hit_max_bbox_margin_m: float | None = None,
    candidate_azimuth_margin_deg: float | None = None,
    hit_match_max_margin_m: float | None = None,
    single_candidate_max_margin_m: float | None = None,
    bbox_extent_inflation_m: float | None = None,
) -> Path:
    collector.finalize_co_visibility()
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = collector.snapshot()

    write_csv_tables(
        out_dir,
        collector,
        hit_match_max_margin_m=hit_match_max_margin_m
        if hit_match_max_margin_m is not None
        else hit_match_m,
    )
    busiest_plot = write_busiest_frame_report(collector, out_dir)

    eff_primary = hit_match_max_margin_m if hit_match_max_margin_m is not None else hit_match_m
    diag = collector.margin_threshold_diagnostics()

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    _plot_outcome_pie(axes[0, 0], snap)
    _plot_depth_hist(axes[0, 1], collector)
    _plot_velocity(axes[0, 2], collector)
    _plot_points_per_sensor(axes[1, 0], collector)
    _plot_sensor_match_rate(axes[1, 1], collector, min_match_rate)
    _plot_points_per_vehicle(axes[1, 2], collector)
    _plot_co_visibility(axes[2, 0], collector)
    _plot_scatter(axes[2, 1], collector)
    axes[2, 2].axis("off")
    _plot_margin_hist(
        axes[3, 0], collector, diag,
        current_primary=eff_primary, current_single=single_candidate_max_margin_m,
    )
    _plot_margin_precision(axes[3, 1], collector, diag, current_primary=eff_primary)
    _plot_margin_text(
        axes[3, 2], collector, diag,
        current_primary=eff_primary, current_single=single_candidate_max_margin_m,
    )
    with collector.lock:
        top_v = collector.points_by_vehicle.most_common(8)
        wc = snap.get("with_candidates", 0)
        rate_cand = snap.get("match_rate_given_candidates", 0.0)
        lines = [
            "Quality checklist",
            f"  Match (all returns): {100 * snap['match_rate']:.2f}%",
            f"  Match (w/ candidates): {100 * rate_cand:.1f}% "
            f"({snap['matched_detections']}/{wc}, PASS if ≥ {100 * min_match_rate:.0f}%)",
            f"  Clutter / no actor: {snap.get('clutter_or_out_of_fov', 0):,} "
            f"({100 * snap.get('clutter_rate', 0):.1f}%)",
            f"  Unique vehicles labeled: {snap['unique_vehicles_matched']}",
            f"  Median radars / vehicle: {snap['median_radars_per_vehicle']:.1f}",
            f"  Max radars / vehicle: {snap['max_radars_per_vehicle']}",
            "",
            "Top vehicles (points):",
        ]
        for vid, cnt in top_v:
            n_rad = len({s for (v, s) in collector.vehicle_sensor_matrix if v == vid})
            lines.append(f"  id {vid}: {cnt} pts, {n_rad} radars")
        if snap["match_rate"] < 0.01 and snap["no_vehicle_candidates"] > 0.9 * snap["total_detections"]:
            lines.append("")
            lines.append("WARNING: Most points have no actor in radar FOV/range near the hit.")
            lines.append("  Spawn traffic and wait before stopping.")
    axes[2, 2].text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10, family="monospace")

    fig.suptitle(
        "Radar labeling test summary (charts exclude clutter / no-actor returns)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = out_dir / "radar_labeling_summary.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    pass_ok = (
        snap.get("with_candidates", 0) >= 50
        and snap.get("match_rate_given_candidates", 0.0) >= min_match_rate
    )
    with collector.lock:
        runtime_snapshot = dict(collector.runtime_telemetry)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expected_radars": sorted(expected_radar_labels),
        "pass": pass_ok,
        "parameters": {
            "proximity_candidate_radius_m_legacy": proximity_m,
            "hit_match_max_margin_m": hit_match_max_margin_m
            if hit_match_max_margin_m is not None
            else hit_match_m,
            "bbox_extent_inflation_m": bbox_extent_inflation_m,
            "hit_match_m_legacy_alias": hit_match_m,
            "labelable_min_speed_mps": labelable_min_speed_mps,
            "min_pass_match_rate": min_match_rate,
            "candidate_max_range_m": candidate_max_range_m,
            "candidate_horizontal_fov_deg": candidate_horizontal_fov_deg,
            "candidate_depth_margin_m": candidate_depth_margin_m,
            "candidate_azimuth_margin_deg": candidate_azimuth_margin_deg,
            "single_candidate_max_margin_m": single_candidate_max_margin_m,
        },
        "summary": snap,
        "runtime": runtime_snapshot,
        "co_visibility": dict(collector.co_visibility_hist),
        "busiest_frame": collector.busiest_frame_snapshot(),
        "files": {
            "plot": plot_path.name,
            "per_vehicle_summary": "per_vehicle_summary.csv",
            "per_sensor_summary": "per_sensor_summary.csv",
            "per_frame_summary": "per_frame_summary.csv",
            "vehicle_radar_matrix": "vehicle_radar_matrix.csv",
            "busiest_frame_plot": busiest_plot.name if busiest_plot else None,
            "busiest_frame_summary": "busiest_frame_summary.json",
            "busiest_frame_points": "busiest_frame_points.csv",
        },
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    wc = snap.get("with_candidates", 0)
    rate_cand = snap.get("match_rate_given_candidates", 0.0)
    txt_lines = [
        "Radar labeling test summary",
        "=" * 40,
        f"Scored returns:      {snap['total_detections']:,} (|velocity| >= {labelable_min_speed_mps} m/s)",
        f"Static skipped:      {snap.get('static_skipped', 0):,}",
        f"With candidates:     {wc:,}",
        f"Matched (labeled):   {snap['matched_detections']:,} "
        f"({100 * rate_cand:.1f}% of w/ candidates, {100 * snap['match_rate']:.2f}% of all)",
        f"No candidates:       {snap['no_vehicle_candidates']:,} (clutter / out of FOV)",
        f"Failed match:        {snap['failed_match']:,}",
        f"Vehicles labeled:    {snap['unique_vehicles_matched']}",
        f"Median radars/vehicle:{snap['median_radars_per_vehicle']:.1f}",
        "",
        f"Outputs in: {out_dir.resolve()}",
        f"  - {plot_path.name}",
        "  - per_vehicle_summary.csv",
        "  - per_sensor_summary.csv",
        "  - vehicle_radar_matrix.csv",
        "  - per_frame_summary.csv",
        "  - busiest_frame_summary.png",
        "  - busiest_frame_summary.json",
        "  - busiest_frame_points.csv",
        "  - summary.json",
    ]
    dn = snap.get("density_per_radar_per_frame")
    if dn:
        bvc = dn.get("by_vehicle_class", {})
        txt_lines.extend(
            [
                "",
                f"Point density per radar per frame ({dn.get('radar_frames', 0):,} sweeps):",
                f"  all returns: {dn.get('all', 0):.2f}   clutter: {dn.get('clutter_unmatched', 0):.2f}",
                f"  matched:     {dn.get('matched', 0):.3f}  = vehicle {dn.get('vehicle', 0):.3f} "
                f"(car {bvc.get('car', 0):.3f} + truck {bvc.get('truck', 0):.3f}) "
                f"+ pedestrian {dn.get('pedestrian', 0):.3f}",
            ]
        )
    if diag.get("enough_data"):
        with collector.lock:
            mhist = collector.margin_hist.copy()
        mcenters = margin_bin_centers()
        spike = float(diag.get("spike_count") or 0.0)
        cur_p = 100 * _precision_at(mhist, mcenters, spike, eff_primary)
        sug_p = 100 * _precision_at(mhist, mcenters, spike, diag["suggested_primary_m"])
        txt_lines.extend(
            [
                "",
                "Match-margin calibration (uncensored hit->OBB margin):",
                f"  on-body spike:    {diag['spike_count']:,} "
                f"({100 * diag['spike_frac']:.1f}%) within {diag['spike_edge_m']} m",
                f"  trough:           {diag['trough_m']} m",
                f"  current primary:  {eff_primary} m  -> {cur_p:.0f}% on-body",
                f"  suggested:        primary {diag['suggested_primary_m']} m "
                f"(-> {sug_p:.0f}% on-body), single {diag['suggested_single_m']} m",
            ]
        )

    bf = summary.get("busiest_frame")
    if bf:
        txt_lines.extend(
            [
                "",
                f"Busiest frame: {bf['frame']} ({bf['total_points']} points, "
                f"{bf['matched_points']} matched, {bf['distinct_radars']} radars, "
                f"{bf['distinct_vehicles']} vehicles)",
            ]
        )
    (out_dir / "summary.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return out_dir
