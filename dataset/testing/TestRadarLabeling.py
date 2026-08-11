"""
Validate radar -> actor labeling (vehicles + pedestrians via OBB) until you stop.

Scores returns with an actor in the detection beam (incl. parked) or |velocity| above threshold.
Static clutter with no actor in-beam is not scored. Matching uses beam/depth candidates,
primary + legacy hit, and single-target fallbacks. Use --debug-draws to visualize hits in CARLA.

Run standalone (after setup/RadarCameraSetup* and traffic are up):
    python testing/TestRadarLabeling.py

Or via Start.py test mode:
    python Start.py --test-labeling

Press Enter to stop. Writes a summary folder with plots + CSV tables (no full radar CSV).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import argparse
import datetime
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import carla

from _kbhit_compat import enter_pressed

from carla_connect import carla_timeout_s
from capture.CaptureRadarCameraData import (
    BBOX_MATCH_EXTENT_INFLATION_M,
    DATASET_RADAR_ROLE_PREFIX,
    EXPECTED_RADAR_LABELS,
    RADAR_ACTOR_PROXIMITY_M,
    RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
    RADAR_CANDIDATE_DEPTH_MARGIN_M,
    RADAR_HIT_MATCH_MAX_DISTANCE_M,
    RADAR_HIT_MATCH_MAX_MARGIN_M,
    RADAR_HORIZONTAL_FOV_DEG,
    RADAR_LABELABLE_MIN_SPEED_MPS,
    RADAR_MAX_RANGE_M,
    RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M,
    RADAR_VEHICLE_PROXIMITY_M,
    RadarActorSnapshotCache,
    actor_snapshots_for_radar_detection,
    evaluate_radar_detection_label,
    filter_tagged_sensors,
    list_radar_actors,
    precompute_actor_frame_cache,
    radar_detection_world_location,
    radar_candidate_hit_max_bbox_margin_m,
    radar_sensor_limits,
    select_one_sensor_per_label,
    sensor_label_from_role_name,
    wait_for_sensors,
)
from capture.radar_stream import PerRadarLatestBuffer, make_radar_measurement_buffer
from dataset_paths import data_output_dir, testing_dir
from testing.RadarLabelingTestReport import (
    DetectionRecord,
    LabelingStatsCollector,
    write_live_snapshot,
    write_report,
)

DEFAULT_MIN_MATCH_RATE = 0.05
DEFAULT_MIN_DETECTIONS = 50
DEFAULT_MIN_WITH_CANDIDATES = 50
SENSOR_WAIT_TIMEOUT_S = 60.0
STATUS_INTERVAL_S = 2.0
AUTOSAVE_REPORT_INTERVAL_S = 90.0
REQUEST_STOP_FILENAME = ".request_stop"
REPORT_COMPLETE_FILENAME = ".report_complete"
RADAR_TEST_LOG_PREFIX = "[RadarTest]"


def maybe_debug_draw_match(
    world,
    sensor_transform,
    detection,
    *,
    candidates,
    matched_actor_id,
    debug_draws: bool,
    debug_counter: list[int],
    debug_sample_every: int,
) -> None:
    if not debug_draws or world is None:
        return
    debug_counter[0] += 1
    if debug_counter[0] % debug_sample_every != 0:
        return
    hit_loc = radar_detection_world_location(sensor_transform, detection)
    world.debug.draw_point(hit_loc, size=0.12, color=carla.Color(255, 40, 40), life_time=0.15)
    if matched_actor_id is not None:
        try:
            actor = world.get_actor(int(matched_actor_id))
            if actor.is_alive:
                bbox = actor.bounding_box
                tf = actor.get_transform()
                center = tf.transform(bbox.location)
                world.debug.draw_box(
                    carla.BoundingBox(center, bbox.extent),
                    tf.rotation,
                    thickness=0.06,
                    color=carla.Color(40, 220, 80),
                    life_time=0.15,
                )
        except RuntimeError:
            pass
    elif candidates:
        try:
            actor = world.get_actor(candidates[0]["id"])
            if actor.is_alive:
                bbox = actor.bounding_box
                tf = actor.get_transform()
                center = tf.transform(bbox.location)
                world.debug.draw_box(
                    carla.BoundingBox(center, bbox.extent),
                    tf.rotation,
                    thickness=0.04,
                    color=carla.Color(220, 180, 40),
                    life_time=0.15,
                )
        except RuntimeError:
            pass


def format_status_line(snap: dict) -> str:
    wc = snap.get("with_candidates", 0)
    rate_cand = snap.get("match_rate_given_candidates", 0.0)
    raw = snap.get("raw_radar_returns", 0)
    avg_raw = snap.get("avg_raw_returns_per_message", 0.0)
    msgs = snap.get("radar_messages", 0)
    pending = snap.get("queue_pending", 0)
    dropped = snap.get("queue_dropped", 0)
    return (
        f"{RADAR_TEST_LOG_PREFIX} msgs={msgs:,} raw={raw:,} ({avg_raw:.1f}/msg) "
        f"q={pending} drop={dropped} | "
        f"scored={snap['total_detections']:,} | "
        f"matched={snap['matched_detections']:,} ({100 * snap['match_rate']:.2f}% all) | "
        f"w/ candidates={wc:,} → {100 * rate_cand:.1f}% labeled | "
        f"fail={snap['failed_match']:,} | actors={snap['unique_actors_matched']}"
    )


def print_report(collector: LabelingStatsCollector, min_match_rate: float, out_dir: Path | None) -> bool:
    snap = collector.snapshot()
    total = snap["total_detections"]
    matched = snap["matched_detections"]
    wc = snap.get("with_candidates", 0)
    rate_cand = snap.get("match_rate_given_candidates", 0.0)
    rate_all = snap["match_rate"]

    print(f"\n{RADAR_TEST_LOG_PREFIX} === Final report ===")
    print(f"  Radars: {len(EXPECTED_RADAR_LABELS)}  |  messages: {snap['radar_messages']:,}")
    print(
        f"  Raw CARLA returns: {snap.get('raw_radar_returns', 0):,} "
        f"({snap.get('avg_raw_returns_per_message', 0.0):.1f} per callback, actor hits only)"
    )
    static = snap.get("static_skipped", 0)
    min_speed = snap.get("labelable_min_speed_mps", RADAR_LABELABLE_MIN_SPEED_MPS)
    print(f"  Scored returns: {total:,} (|v| >= {min_speed} m/s)", end="")
    if static:
        print(f", static skipped: {static:,}")
    else:
        print()
    print(
        f"  Matched: {matched:,}  |  "
        f"{100 * rate_cand:.1f}% of {wc:,} w/ candidates  |  "
        f"{100 * rate_all:.2f}% of all returns"
    )
    print(
        f"  No candidate: {snap['no_actor_candidates']:,} (clutter / no actor near hit)  |  "
        f"Failed: {snap['failed_match']:,} (hit > {RADAR_HIT_MATCH_MAX_MARGIN_M} m from OBB; "
        f"see labeling_failure_samples.csv)"
    )
    print(
        f"  Actors: {snap['unique_actors_matched']} "
        f"(veh={snap['unique_vehicles_matched']}, ped={snap['unique_pedestrians_matched']})  |  "
        f"radars/veh median={snap['median_radars_per_vehicle']:.1f} max={snap['max_radars_per_vehicle']}"
    )
    if snap["legacy_matched"] or snap["legacy_failed"]:
        legacy_total = snap["legacy_matched"] + snap["legacy_failed"]
        legacy_rate = snap["legacy_matched"] / legacy_total if legacy_total else 0.0
        print(
            f"  Legacy spherical: {snap['legacy_matched']}/{legacy_total} "
            f"({100.0 * legacy_rate:.1f}% w/ candidates)"
        )

    print("  Per radar (match % = among returns with candidates):")
    for label in sorted(snap["by_sensor"]):
        b = snap["by_sensor"][label]
        det = b["detections"]
        m = b["matched"]
        sensor_wc = b.get("with_candidates", det - b["no_candidates"])
        pct = 100.0 * b.get("match_rate_given_candidates", m / sensor_wc if sensor_wc else 0.0)
        print(
            f"    {label}: {m:,}/{sensor_wc:,} matched ({pct:.1f}%), "
            f"no_cand={b['no_candidates']:,}, fail={b['failed_match']:,}"
        )

    bf = collector.busiest_frame_snapshot()
    if bf:
        bf_wc = bf.get("with_candidates", bf["total_points"] - bf["no_candidates"])
        bf_rate = (bf["matched_points"] / bf_wc * 100) if bf_wc else 0.0
        print(
            f"  Best labeling frame: {bf['frame']} — {bf['matched_points']} matched / "
            f"{bf_wc} w/ candidates ({bf_rate:.1f}%), {bf['distinct_radars']} radars, "
            f"{bf['distinct_actors']} actors"
        )

    if out_dir is not None:
        print(f"  Outputs: {out_dir.resolve()}")

    passed = (
        total >= DEFAULT_MIN_DETECTIONS
        and wc >= DEFAULT_MIN_WITH_CANDIDATES
        and rate_cand >= min_match_rate
    )
    print(f"\n{RADAR_TEST_LOG_PREFIX} Result:", "PASS" if passed else "FAIL")
    if total < DEFAULT_MIN_DETECTIONS:
        print(f"    Need >= {DEFAULT_MIN_DETECTIONS} scored returns.")
    if wc < DEFAULT_MIN_WITH_CANDIDATES:
        print(
            f"    Need >= {DEFAULT_MIN_WITH_CANDIDATES} returns with actor candidates "
            "(spawn traffic near radars)."
        )
    if rate_cand < min_match_rate:
        print(
            f"    Match rate among candidates {rate_cand:.2%} < {min_match_rate:.2%} threshold."
        )
    return passed


def _stop_requested(out_dir: Path | None) -> bool:
    if out_dir is None:
        return False
    return (out_dir / REQUEST_STOP_FILENAME).is_file()


def run_test(
    world,
    radar_sensors,
    compare_legacy: bool,
    duration_s: float | None = None,
    *,
    labelable_min_speed_mps: float = RADAR_LABELABLE_MIN_SPEED_MPS,
    out_dir: Path | None = None,
    report_kwargs: dict | None = None,
    debug_draws: bool = False,
) -> LabelingStatsCollector:
    collector = LabelingStatsCollector(labelable_min_speed_mps=labelable_min_speed_mps)
    stop_at = (time.time() + duration_s) if duration_s is not None else None
    report_kwargs = report_kwargs or {}
    sigint_stop = False

    def _on_sigint(_signum, _frame) -> None:
        nonlocal sigint_stop
        sigint_stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    debug_counter = [0]
    actor_cache = RadarActorSnapshotCache(world)
    radar_queue = make_radar_measurement_buffer()
    # Per-radar arrival instrumentation: when did each sensor's first/last msg arrive
    # at our callback, and how many did each sensor try to deliver.
    per_radar_stats: dict[str, dict] = {}
    per_radar_lock = threading.Lock()
    # Worker pool for the per-message labeling work. Env: DATASET_RADAR_LABEL_WORKERS
    # (default 4). 1 disables the pool (single-threaded drain).
    try:
        _worker_count = max(1, int(os.environ.get("DATASET_RADAR_LABEL_WORKERS", "4")))
    except ValueError:
        _worker_count = 4
    label_executor = None
    if _worker_count > 1:
        from concurrent.futures import ThreadPoolExecutor
        label_executor = ThreadPoolExecutor(
            max_workers=_worker_count, thread_name_prefix="radar-label"
        )
        print(
            f"{RADAR_TEST_LOG_PREFIX} drain pool: {_worker_count} workers",
            flush=True,
        )
    else:
        print(f"{RADAR_TEST_LOG_PREFIX} drain pool: disabled (single-threaded)", flush=True)

    def process_radar_measurement(sensor_label: str, radar_actor, measurement, actors=None) -> None:
        collector.record_message(raw_returns=len(measurement))
        sensor_transform = measurement.transform
        if actors is None:
            actors = actor_cache.get(int(measurement.frame))
        range_m, hfov_deg = radar_sensor_limits(radar_actor)

        for detection in measurement:
            label = evaluate_radar_detection_label(
                world,
                sensor_transform,
                detection,
                actors,
                range_m=range_m,
                hfov_deg=hfov_deg,
                labelable_min_speed_mps=labelable_min_speed_mps,
                compare_legacy=compare_legacy,
            )
            if not label["scored"]:
                collector.record_static_skipped(sensor_label)
                continue

            if label["had_candidates"] and debug_draws:
                match_candidates = actor_snapshots_for_radar_detection(
                    sensor_transform,
                    detection,
                    actors,
                    world,
                    max_range_m=range_m,
                    horizontal_fov_deg=hfov_deg,
                    hit_max_bbox_margin_m=radar_candidate_hit_max_bbox_margin_m(),
                )
                maybe_debug_draw_match(
                    world,
                    sensor_transform,
                    detection,
                    candidates=match_candidates,
                    matched_actor_id=label["actor_id"],
                    debug_draws=True,
                    debug_counter=debug_counter,
                    debug_sample_every=200,
                )

            collector.record_detection(
                DetectionRecord(
                    sensor_label=sensor_label,
                    frame=int(measurement.frame),
                    had_candidates=label["had_candidates"],
                    matched=label["matched"],
                    depth_m=float(detection.depth),
                    velocity_mps=label["velocity_mps"],
                    azimuth_rad=float(detection.azimuth),
                    actor_id=label["actor_id"],
                    actor_kind=label["actor_kind"],
                    actor_class=label["actor_class"],
                    match_bbox_margin_m=label["match_bbox_margin_m"],
                    nearest_bbox_margin_m=label["nearest_bbox_margin_m"],
                    uncensored_nearest_bbox_margin_m=label[
                        "uncensored_nearest_bbox_margin_m"
                    ],
                ),
                legacy_matched=label["legacy_matched"],
            )

    def drain_radar_queue(*, max_items: int = 0, budget_s: float = 0.25) -> None:
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            batch = (
                radar_queue.drain_all()
                if isinstance(radar_queue, PerRadarLatestBuffer)
                else radar_queue.drain(max_items if max_items > 0 else 64)
            )
            if not batch:
                break
            # Resolve actor snapshots per unique frame on the main thread (cache
            # mutation must not race with workers), and precompute world-space
            # bbox center / inverse rotations once per frame so all messages
            # in this batch share the work.
            frame_actors_cache: dict[int, list] = {}
            for sensor_label, radar_actor, measurement in batch:
                fid = int(measurement.frame)
                if fid not in frame_actors_cache:
                    actors = actor_cache.get(fid)
                    precompute_actor_frame_cache(actors, world)
                    frame_actors_cache[fid] = actors
            if label_executor is None:
                for sensor_label, radar_actor, measurement in batch:
                    process_radar_measurement(
                        sensor_label, radar_actor, measurement,
                        frame_actors_cache[int(measurement.frame)],
                    )
            else:
                futures = [
                    label_executor.submit(
                        process_radar_measurement,
                        sensor_label, radar_actor, measurement,
                        frame_actors_cache[int(measurement.frame)],
                    )
                    for sensor_label, radar_actor, measurement in batch
                ]
                for f in futures:
                    try:
                        f.result()
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"{RADAR_TEST_LOG_PREFIX} worker error: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
            if isinstance(radar_queue, PerRadarLatestBuffer):
                break

    def make_callback(sensor_label: str, radar_actor):
        def radar_callback(measurement):
            now = time.time()
            with per_radar_lock:
                st = per_radar_stats.setdefault(sensor_label, {
                    "first_msg_at": now,
                    "last_msg_at": now,
                    "first_msg_frame": int(measurement.frame),
                    "last_msg_frame": int(measurement.frame),
                    "arrivals": 0,
                })
                st["last_msg_at"] = now
                st["last_msg_frame"] = int(measurement.frame)
                st["arrivals"] += 1
            item = (sensor_label, radar_actor, measurement)
            if isinstance(radar_queue, PerRadarLatestBuffer):
                radar_queue.enqueue(sensor_label, item)
            else:
                radar_queue.enqueue(item)

        return radar_callback

    for radar in radar_sensors:
        label = sensor_label_from_role_name(
            radar.attributes.get("role_name", ""), DATASET_RADAR_ROLE_PREFIX
        )
        radar.listen(make_callback(label, radar))

    if stop_at is not None:
        print(
            f"{RADAR_TEST_LOG_PREFIX} Listening ({len(radar_sensors)} radars, {duration_s:.0f}s)...",
            flush=True,
        )
    else:
        print(
            f"{RADAR_TEST_LOG_PREFIX} Listening on {len(radar_sensors)} radars — "
            "press Enter here to stop and save.",
            flush=True,
        )
        if out_dir is not None:
            print(
                f"{RADAR_TEST_LOG_PREFIX} Output: {out_dir.resolve()} "
                f"(live_stats every {STATUS_INTERVAL_S:.0f}s, autosave every {AUTOSAVE_REPORT_INTERVAL_S:.0f}s)",
                flush=True,
            )
    last_status = time.time()
    last_autosave = time.time()
    last_status_line = ""
    autosave_busy = False

    listen_attached_at = time.time()
    first_world_frame = int(world.get_snapshot().frame)

    def publish_status(force: bool = False) -> None:
        nonlocal last_status_line
        snap = collector.snapshot()
        snap["queue_pending"] = radar_queue.pending()
        snap["queue_dropped"] = radar_queue.dropped
        try:
            ws = world.get_snapshot()
            snap["world_frame"] = int(ws.frame)
            snap["world_frames_since_attach"] = int(ws.frame) - first_world_frame
            snap["wall_seconds_since_attach"] = round(time.time() - listen_attached_at, 3)
        except Exception:  # noqa: BLE001
            pass
        with per_radar_lock:
            snap["per_radar_arrivals"] = {
                k: {"arrivals": v["arrivals"],
                    "first_at": round(v["first_msg_at"] - listen_attached_at, 3),
                    "last_at": round(v["last_msg_at"] - listen_attached_at, 3),
                    "first_frame": v["first_msg_frame"],
                    "last_frame": v["last_msg_frame"]}
                for k, v in per_radar_stats.items()
            }
        # Mirror runtime telemetry into the collector so autosaves + final
        # summary.json carry these fields too (not just live_stats.json).
        telemetry = {
            "queue_pending": snap.get("queue_pending"),
            "queue_dropped": snap.get("queue_dropped"),
            "queue_maxsize": getattr(radar_queue, "_queue", None).maxsize
                if hasattr(radar_queue, "_queue") else None,
            "queue_type": type(radar_queue).__name__,
            "world_frame": snap.get("world_frame"),
            "world_frames_since_attach": snap.get("world_frames_since_attach"),
            "wall_seconds_since_attach": snap.get("wall_seconds_since_attach"),
            "per_radar_arrivals": snap.get("per_radar_arrivals", {}),
        }
        total_enq = (telemetry["queue_dropped"] or 0) + (snap.get("radar_messages") or 0)
        wall = telemetry["wall_seconds_since_attach"] or 0
        wf = telemetry["world_frames_since_attach"] or 0
        telemetry["enqueue_attempts_total"] = total_enq
        telemetry["enqueue_retention_pct"] = (
            round(100 * (snap.get("radar_messages") or 0) / total_enq, 3)
            if total_enq > 0 else None
        )
        telemetry["world_fps_hz"] = round(wf / wall, 3) if wall > 0 else None
        telemetry["arrival_hz_per_radar"] = (
            round(total_enq / wall / max(1, len(per_radar_stats)), 3)
            if wall > 0 and per_radar_stats else None
        )
        telemetry["worker_count"] = _worker_count
        telemetry["drain_hz_total"] = (
            round((snap.get("radar_messages") or 0) / wall, 3) if wall > 0 else None
        )
        collector.set_runtime_telemetry(telemetry)
        line = format_status_line(snap)
        if force or line != last_status_line:
            print(line, flush=True)
            last_status_line = line
        if out_dir is not None:
            write_live_snapshot(out_dir, snap)

    publish_status(force=True)

    def _run_autosave() -> None:
        nonlocal autosave_busy
        try:
            write_report(collector, out_dir, **report_kwargs)
            print(
                f"{RADAR_TEST_LOG_PREFIX} Autosaved reports → {out_dir.name}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{RADAR_TEST_LOG_PREFIX} Autosave failed: {exc}", flush=True)
        finally:
            autosave_busy = False

    try:
        while True:
            drain_radar_queue()
            now = time.time()
            if stop_at is not None and now >= stop_at:
                break
            if sigint_stop or _stop_requested(out_dir):
                print(
                    f"{RADAR_TEST_LOG_PREFIX} Stopping — final summary will print next...",
                    flush=True,
                )
                break
            if enter_pressed():
                break

            now = time.time()
            if now - last_status >= STATUS_INTERVAL_S:
                publish_status(force=True)
                last_status = now

            if (
                out_dir is not None
                and report_kwargs
                and not autosave_busy
                and now - last_autosave >= AUTOSAVE_REPORT_INTERVAL_S
            ):
                autosave_busy = True
                threading.Thread(target=_run_autosave, daemon=True).start()
                last_autosave = now

            if radar_queue.pending():
                continue
            time.sleep(0.002)
    finally:
        drain_radar_queue(budget_s=30.0)
        for radar in radar_sensors:
            try:
                radar.stop()
            except RuntimeError:
                pass
        if label_executor is not None:
            try:
                label_executor.shutdown(wait=True, cancel_futures=False)
            except Exception:  # noqa: BLE001
                pass
        try:
            publish_status(force=True)  # final snapshot with all instrumentation
        except Exception:  # noqa: BLE001
            pass

    return collector


def resolve_output_dir(script_dir: Path, explicit: str = "") -> Path:
    """
    New folder per run: radar_labeling_test_YYYYMMDD_HHMMSS under Data/
    or DATASET_CAPTURE_BASE_DIR when set.
    """
    if explicit.strip():
        path = Path(explicit.strip())
        if not path.is_absolute():
            path = (script_dir / path).resolve()
        else:
            path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.environ.get("DATASET_CAPTURE_BASE_DIR", "").strip()
    parent = Path(base).resolve() if base else data_output_dir()
    out = parent / f"radar_labeling_test_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_run_started_marker(out_dir: Path, *, expected_radars: set[str]) -> None:
    meta = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "expected_radars": sorted(expected_radars),
        "folder": str(out_dir.resolve()),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_run_error(out_dir: Path, message: str) -> None:
    (out_dir / "error.txt").write_text(message.rstrip() + "\n", encoding="utf-8")
    meta_path = out_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        meta["status"] = "failed"
        meta["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        meta["error"] = message.strip()
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def publish_output_pointer(script_dir: Path, out_dir: Path) -> None:
    pointer = testing_dir() / ".last_radar_labeling_test_dir"
    pointer.write_text(str(out_dir.resolve()) + "\n", encoding="utf-8")


def print_output_artifacts(out_dir: Path) -> None:
    """List report files written under the test output folder."""
    names = [
        "summary.txt",
        "summary.json",
        "radar_labeling_summary.png",
        "busiest_frame_summary.png",
        "busiest_frame_summary.json",
        "busiest_frame_points.csv",
        "per_vehicle_summary.csv",
        "per_pedestrian_summary.csv",
        "per_sensor_summary.csv",
        "per_frame_summary.csv",
        "labeling_failure_samples.csv",
        "vehicle_radar_matrix.csv",
        "live_stats.json",
        "run_meta.json",
    ]
    print(f"\n{RADAR_TEST_LOG_PREFIX} Output artifacts ({out_dir.resolve()}):")
    for name in names:
        path = out_dir / name
        mark = "ok" if path.is_file() else "—"
        print(f"  [{mark}] {name}")


def finalize_labeling_test_run(
    collector: LabelingStatsCollector | None,
    out_dir: Path | None,
    script_dir: Path,
    *,
    report_kwargs: dict,
    min_match_rate: float,
    no_plots: bool = False,
) -> bool:
    """
    Write plots/CSVs/JSON and print the full terminal summary once at end of run.
    """
    if collector is None or out_dir is None:
        return False

    complete_marker = out_dir / REPORT_COMPLETE_FILENAME
    if complete_marker.is_file():
        complete_marker.unlink()

    passed = False
    try:
        print(f"\n{RADAR_TEST_LOG_PREFIX} Writing final reports...", flush=True)
        if not no_plots:
            write_report(collector, out_dir, **report_kwargs)
        print(f"{RADAR_TEST_LOG_PREFIX} ========== Final summary ==========", flush=True)
        passed = print_report(collector, min_match_rate, out_dir)
        if not no_plots:
            print_output_artifacts(out_dir)

        summary_txt = out_dir / "summary.txt"
        if summary_txt.is_file():
            print(f"\n{RADAR_TEST_LOG_PREFIX} --- summary.txt ---", flush=True)
            print(summary_txt.read_text(encoding="utf-8"), end="", flush=True)

        meta_path = out_dir / "run_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "completed"
            meta["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            meta["pass"] = passed
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        publish_output_pointer(script_dir, out_dir)
        complete_marker.write_text(
            datetime.datetime.now().isoformat(timespec="seconds") + "\n",
            encoding="utf-8",
        )
        print(f"{RADAR_TEST_LOG_PREFIX} Reports complete.", flush=True)
    except ImportError as exc:
        print(f"WARNING: Could not write plots (matplotlib missing?): {exc}", file=sys.stderr)
        write_run_error(out_dir, f"Report export failed: {exc}")
    except Exception as exc:
        print(f"WARNING: Report export failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        write_run_error(out_dir, f"Report export failed: {exc}\n{traceback.format_exc()}")

    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate radar vehicle labeling in CARLA.")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional fixed run time in seconds (default: run until Enter).",
    )
    parser.add_argument(
        "--min-match-rate",
        type=float,
        default=DEFAULT_MIN_MATCH_RATE,
        help=(
            f"PASS threshold: match rate among returns WITH candidates (default {DEFAULT_MIN_MATCH_RATE}). "
            "|velocity| >= labelable-min-speed (0 includes parked)."
        ),
    )
    parser.add_argument(
        "--labelable-min-speed",
        type=float,
        default=RADAR_LABELABLE_MIN_SPEED_MPS,
        help=(
            "Min |radial velocity| (m/s) to score a return (default 0 = parked/stalled OK). "
            "Use 0.5 to ignore near-static returns."
        ),
    )
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Also score the old spherical hit conversion for comparison.",
    )
    parser.add_argument(
        "--debug-draws",
        action="store_true",
        help=(
            "Draw sample hit points / OBBs in CARLA (every 200th scored return w/ candidates). "
            "Red point = hit; green box = matched; yellow = nearest candidate on miss."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Folder for plots/CSVs (default: radar_labeling_test_<timestamp> in script dir).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing plots and CSV tables.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA host (default 127.0.0.1; use this on Windows instead of localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA port (default 2000).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir: Path | None = None
    collector: LabelingStatsCollector | None = None
    exit_code = 1
    passed = False

    if not args.no_plots:
        out_dir = resolve_output_dir(script_dir, args.output_dir)
        write_run_started_marker(out_dir, expected_radars=EXPECTED_RADAR_LABELS)
        publish_output_pointer(script_dir, out_dir)
        print(f"{RADAR_TEST_LOG_PREFIX} Output: {out_dir.resolve()}", flush=True)

    cand_hit_m = radar_candidate_hit_max_bbox_margin_m()
    cand_hit_label = (
        f"hit≤{cand_hit_m} m" if cand_hit_m is not None else "beam-only (no hit gate)"
    )
    print(
        f"{RADAR_TEST_LOG_PREFIX} Config: "
        f"beam FOV {RADAR_HORIZONTAL_FOV_DEG}°+{RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG}° | "
        f"depth±{RADAR_CANDIDATE_DEPTH_MARGIN_M} m | "
        f"candidates: {cand_hit_label} | "
        f"match≤{RADAR_HIT_MATCH_MAX_MARGIN_M} m (single≤{RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M} m) | "
        f"score if in-beam or |v|≥{args.labelable_min_speed} m/s | "
        f"PASS ≥{100 * args.min_match_rate:.0f}% among w/ candidates"
    )
    if args.debug_draws:
        print(f"{RADAR_TEST_LOG_PREFIX} Debug draws enabled (CARLA viewport).")

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(carla_timeout_s())
        world = client.get_world()

        print(f"{RADAR_TEST_LOG_PREFIX} Waiting for radars (up to {SENSOR_WAIT_TIMEOUT_S:.0f}s)...")
        time.sleep(1.0)
        radar_sensors, _ = wait_for_sensors(world, SENSOR_WAIT_TIMEOUT_S, log_progress=False)
        expected = len(EXPECTED_RADAR_LABELS)
        all_tagged = filter_tagged_sensors(
            world, "sensor.other.radar", DATASET_RADAR_ROLE_PREFIX, EXPECTED_RADAR_LABELS
        )
        if len(all_tagged) > expected:
            print(
                f"{RADAR_TEST_LOG_PREFIX} Using newest of {len(all_tagged)} radars ({expected} labels).",
                flush=True,
            )
        radar_sensors = select_one_sensor_per_label(
            all_tagged, DATASET_RADAR_ROLE_PREFIX, EXPECTED_RADAR_LABELS
        )
        if len(radar_sensors) != expected:
            all_radars = list_radar_actors(world)
            print(
                f"ERROR: Found {len(radar_sensors)} unique radar labels, expected {expected}.",
                file=sys.stderr,
            )
            print(f"  Radars in world (any role): {len(all_radars)}", file=sys.stderr)
            for actor in all_radars[:24]:
                role = actor.attributes.get("role_name", "")
                print(
                    f"    id={actor.id} type_id={actor.type_id} role_name={role!r}",
                    file=sys.stderr,
                )
            if len(all_radars) >= expected and len(radar_sensors) == 0:
                print(
                    "  Radars exist but role_name is missing/wrong. "
                    "Keep RadarCameraSetup*.py running; it must set role_name "
                    f"to {DATASET_RADAR_ROLE_PREFIX}R1 etc.",
                    file=sys.stderr,
                )
            elif len(all_radars) == 0:
                print(
                    "  No radars in the world. Start RadarCameraSetup*.py first and wait until "
                    "you see 'Spawned R1'..'Spawned R8'. Do not press Enter in that window.",
                    file=sys.stderr,
                )
            else:
                print(
                    "  Missing labels: "
                    + ", ".join(
                        sorted(
                            EXPECTED_RADAR_LABELS
                            - {
                                sensor_label_from_role_name(
                                    a.attributes.get("role_name", ""),
                                    DATASET_RADAR_ROLE_PREFIX,
                                )
                                for a in radar_sensors
                            }
                        )
                    ),
                    file=sys.stderr,
                )
            raise RuntimeError(
                f"Found {len(radar_sensors)} unique radar labels, expected {expected}."
            )

        labels_ready = ", ".join(
            f"{sensor_label_from_role_name(a.attributes.get('role_name', ''), DATASET_RADAR_ROLE_PREFIX)}:{a.id}"
            for a in radar_sensors
        )
        vehicle_count = len(world.get_actors().filter("vehicle.*"))
        pedestrian_count = len(world.get_actors().filter("walker.pedestrian.*"))
        print(
            f"{RADAR_TEST_LOG_PREFIX} Ready: {labels_ready} | "
            f"{vehicle_count} vehicles, {pedestrian_count} pedestrians",
            flush=True,
        )
        if vehicle_count == 0 and pedestrian_count == 0:
            print(
                "WARNING: No vehicles or pedestrians in world — match rate will be 0. "
                "Run SpawnCarsAtPosition14.py / SpawnPedestriansAcrossMap.py or Start.py test mode.",
                file=sys.stderr,
            )

        report_kwargs = {
            "min_match_rate": args.min_match_rate,
            "expected_radar_labels": EXPECTED_RADAR_LABELS,
            "proximity_m": RADAR_VEHICLE_PROXIMITY_M,
            "hit_match_m": RADAR_HIT_MATCH_MAX_DISTANCE_M,
            "hit_match_max_margin_m": RADAR_HIT_MATCH_MAX_MARGIN_M,
            "bbox_extent_inflation_m": BBOX_MATCH_EXTENT_INFLATION_M,
            "labelable_min_speed_mps": args.labelable_min_speed,
            "candidate_max_range_m": RADAR_MAX_RANGE_M,
            "candidate_horizontal_fov_deg": RADAR_HORIZONTAL_FOV_DEG,
            "candidate_depth_margin_m": RADAR_CANDIDATE_DEPTH_MARGIN_M,
            "candidate_azimuth_margin_deg": RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
            "candidate_hit_max_bbox_margin_m": radar_candidate_hit_max_bbox_margin_m(),
            "single_candidate_max_margin_m": RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M,
        }
        collector = run_test(
            world,
            radar_sensors,
            compare_legacy=args.compare_legacy,
            duration_s=args.duration,
            labelable_min_speed_mps=args.labelable_min_speed,
            out_dir=out_dir,
            report_kwargs=report_kwargs if out_dir is not None else None,
            debug_draws=args.debug_draws,
        )
        exit_code = 0

    except Exception as exc:
        msg = str(exc)
        print(f"ERROR: {msg}", file=sys.stderr)
        if out_dir is not None:
            write_run_error(out_dir, msg)
            print(f"Wrote error log: {out_dir / 'error.txt'}", file=sys.stderr)
        return 1

    finally:
        if collector is not None:
            passed = finalize_labeling_test_run(
                collector,
                out_dir,
                script_dir,
                report_kwargs=report_kwargs,
                min_match_rate=args.min_match_rate,
                no_plots=args.no_plots,
            )

    if collector is None:
        return 1

    return 0 if passed and exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
