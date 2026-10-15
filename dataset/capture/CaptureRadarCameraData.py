import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import csv
import datetime
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import carla
from _kbhit_compat import enter_pressed

from carla_connect import get_world
from capture.ExportCameraExtrinsics import write_camera_extrinsics_to_dataset_dir
from capture.ExportRadarExtrinsics import write_radar_extrinsics_live_to_dataset_dir
from capture.radar_layout import RADAR_PITCH_DEG, apply_radar_pitch
from capture.radar_stream import is_per_radar_buffer, make_radar_capture_buffer
from capture.campaign_control import CampaignGate, campaign_control_dir_from_env
from capture.actor_frame_log import (
    ActorFrameLogger,
    TickActorSnapshotter,
    snapshot_location,
)
from dataset_paths import capture_dir, data_output_dir
from testing.RadarLabelingTestReport import (
    DetectionRecord,
    LabelingStatsCollector,
    write_report,
)

# Camera "nearby actor" bookkeeping radius (metadata only; every camera frame is
# saved regardless). 90 m covers the whole stretch from the set-back camera.
NEARBY_DISTANCE_M = 90.0
# Default radar sensor limits (overridden per actor when attributes are present).
# NOTE on CARLA's radar "range": the sensor casts rays to a RECTANGLE at x=range
# (y half-size range*tan(hfov/2), z half-size range*tan(vfov/2)), so oblique rays
# are LONGER than range: up to range*sqrt(1+tan^2(hfov/2)+tan^2(vfov/2)). With
# 35 m / 120 deg / 60 deg that is 72.9 m, and capture 20260917_125123 indeed
# reports depths up to 66 m (p90 = 42 m). radar_effective_max_depth_m() gives
# the true reach; the candidate gate must use it, not RADAR_MAX_RANGE_M.
RADAR_MAX_RANGE_M = 35.0
RADAR_HORIZONTAL_FOV_DEG = 120.0
# 30 deg (was 60): a 60 deg cone from a 3 m pole spends most rays on the sky,
# facades and the road within 5 m. 30 deg (+/-15) with the -6 deg down-tilt
# covers the road from ~8 m to beyond the far lane. Typical roadside 4D radars
# have 20-30 deg elevation FOV.
RADAR_VERTICAL_FOV_DEG = 30.0
# CARLA default points_per_second is 1500; raise for denser returns (CPU cost scales up).
# Very high values block the client callback thread and stall the sensor stream.
# 15000 is the tuned corridor default: with VFOV=60° it gives ~55% per-tick vehicle
# hit-rate and median ~14 detections/vehicle/frame (vs ~12% / 0.7 at 3000). CPU cost
# is ~7 cores on the EPYC host — well within budget. Drop to 3000 for low-CPU runs;
# CARLA hard-clamps the env override at 20000.
RADAR_POINTS_PER_SECOND_DEFAULT = 15000
# 0.0 = emit every simulation step (floods multi-radar setups); 0.05 ≈ 20 Hz per radar.
RADAR_SENSOR_TICK_S = 0.05
# Extra range beyond reported depth when building per-detection candidates.
RADAR_CANDIDATE_DEPTH_MARGIN_M = 3.0
# Extra horizontal tolerance (deg) for beam vs actor bearing / OBB angular width.
RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG = 8.0
# Pre-filter: actor must be within this OBB margin (m) of the hit to count as a candidate.
# This ONLY defines the denominator of the QA "match rate given candidates" (it never
# changes which returns get labeled; labeling is decided by the 0.5 m margin below).
# The old 7 m bubble made that denominator meaningless: in capture 20260917_125123
# 65% of "unmatched with candidates" were road-surface returns (z ~ 0) up to 7 m
# from a car, so the report showed "35% matched" while recall on true on-body
# returns was 99%. 2.0 m keeps the metric about the actor, not the road around it.
RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M = 2.0
# Legacy wide bubble (reports only).
RADAR_ACTOR_PROXIMITY_M = 40.0
RADAR_VEHICLE_PROXIMITY_M = RADAR_ACTOR_PROXIMITY_M
# Inflate each actor OBB extent when computing margin (m per axis).
# 0.2 (was 0.75): with the exact hit reconstruction below, true on-body returns
# fall INSIDE the OBB; 0.75 mostly admitted road returns beside the car (99% of
# the returns that lived in the 0..0.75 m shell were at z = 0). 0.2 covers mesh
# parts that poke out of the CARLA bounding box (mirrors, bumpers).
BBOX_MATCH_EXTENT_INFLATION_M = 0.2
# Ground rejection: a hit that lies at/below the actor's own ground plane
# (OBB bottom + this clearance) and is OUTSIDE the un-inflated OBB is a road
# return next to / under the car, never a body hit. Tyre hits inside the OBB
# are unaffected.
GROUND_REJECT_CLEARANCE_M = 0.12
# Height above which a return cannot be a vehicle/pedestrian (trucks/buses < 4.5 m).
STRUCTURE_MIN_Z_M = 4.5
# Below this world z a return is on the road surface (stretch is flat at z=0).
ROAD_SURFACE_MAX_Z_M = 0.25
# Max distance from hit to OBB surface for a primary match (m).
# Default 0.5 m: derived from the uncensored nearest-margin distribution of a real
# capture (tools/derive_match_threshold_uncensored.py). Genuine ray hits land ON
# the actor OBB, so they pile into a spike at margin ~0 (here ~12% of candidate
# returns, all within the 0.75 m inflation). Past a trough at ~0.2 m the histogram
# is a monotonically RISING road/structure-clutter ramp with no second lobe and no
# valley — i.e. raising the threshold buys ~zero extra on-body returns and only
# admits clutter. Precision (on-body / accepted) on that capture: 0.5 m -> 87%,
# 1.5 m -> 53%, 2.0 m -> 42%. 0.5 m keeps a little slack for bbox-underfit / pose
# jitter; drop toward 0.25 m for max precision (~96%). The trough shifts with pps,
# so re-derive per capture: the QA report (radar_labeling_summary.png) now plots the
# spike/trough/precision curve, or set DATASET_RADAR_AUTO_MARGIN=1 to derive+apply it
# automatically. Override the constant via DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M
# (clamped 0.5–25 m).
RADAR_HIT_MATCH_MAX_MARGIN_M = 0.5
# Margin when exactly one actor is in the depth/azimuth gate. Was 1.0 m: that
# loosening (plus the approximate hit reconstruction) accepted ~13% wrong labels
# in capture 20260917_125123 (road returns 0.67-0.98 m from the car, z = 0).
# Same as the primary margin now; raise deliberately via env if ever needed.
RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M = 0.5
# Pedestrian tolerances (distance from the TRUE, un-inflated walker box). A walker
# is ~0.4 m wide, so the vehicle tolerances (0.2 m inflation + 0.5 m margin =
# 0.7 m) let sidewalk and facade returns next to a walker be labeled pedestrian.
# Measured on capture 20260929_012142: of returns that far from a walker, the
# share also hit in frames with NO actor nearby (i.e. static scene) is 4% inside
# the box, 6-14% within 0.15 m, 25-36% at 0.15-0.3 m, 53-57% at 0.3-0.7 m.
# 0.15 m keeps swinging arms/legs (mesh slightly outside the CARLA bbox) and
# rejects the static scene around them. A/B on the same capture: share of
# pedestrian labels that are static scene 24% -> 6% (the floor inside the box
# itself is ~4-6%), pedestrian recall 100% -> 100%, vehicles unchanged.
# Env: DATASET_PED_MATCH_MAX_DIST_M.
PED_MATCH_MAX_DIST_M = 0.15
# QA candidate bubble for pedestrians (vehicles keep RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M).
# With the 2 m vehicle bubble, a walker on the boulevard sidewalk pulled facade /
# shopfront returns into "with candidates", so "match rate given candidates" fell
# to 22% for pedestrians although every true walker hit was labeled.
PED_CANDIDATE_MAX_DIST_M = 0.5   # env: DATASET_PED_CANDIDATE_MAX_DIST_M


def _env_float_clamped(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(lo, min(float(raw), hi))
        except ValueError:
            pass
    return default


def ped_match_max_dist_m() -> float:
    return _env_float_clamped("DATASET_PED_MATCH_MAX_DIST_M", PED_MATCH_MAX_DIST_M, 0.0, 2.0)


def ped_candidate_max_dist_m() -> float:
    return _env_float_clamped("DATASET_PED_CANDIDATE_MAX_DIST_M", PED_CANDIDATE_MAX_DIST_M, 0.0, 5.0)


def _is_pedestrian(actor_snapshot) -> bool:
    return actor_snapshot.get("kind") == "pedestrian"
# Backward-compatible alias for reports / CLI (near-surface threshold, not extent inflation).
RADAR_HIT_MATCH_MAX_DISTANCE_M = RADAR_HIT_MATCH_MAX_MARGIN_M
# Min |radial velocity| (m/s) to score a return. Default 0 includes parked/stalled actors.
# Set > 0 (e.g. 0.5) to exclude near-static clutter from match stats.
RADAR_LABELABLE_MIN_SPEED_MPS = 0.0
# CARLA does not simulate electromagnetic RCS; `rcs_proxy_m2` is a geometric OBB silhouette.
SENSOR_WAIT_TIMEOUT_S = 30.0
SENSOR_WAIT_POLL_S = 0.5
DATASET_RADAR_ROLE_PREFIX = "dataset_radar_"
DATASET_CAMERA_ROLE_PREFIX = "dataset_camera_"


def _expected_radar_count_from_env() -> int:
    raw = os.environ.get("DATASET_EXPECTED_RADAR_COUNT", "8")
    try:
        n = int(raw)
    except ValueError:
        return 8
    return max(1, min(n, 64))


EXPECTED_RADAR_LABELS = {f"R{i}" for i in range(1, _expected_radar_count_from_env() + 1)}


def vehicle_class_from_type_id(type_id):
    type_lower = type_id.lower()
    if any(token in type_lower for token in ("firetruck", "ambulance", "truck")):
        return "truck"
    if "bus" in type_lower:
        return "bus"
    if any(token in type_lower for token in ("motorcycle", "vespa", "yamaha", "kawasaki", "harley")):
        return "motorcycle"
    if any(token in type_lower for token in ("bicycle", "bike", "crossbike")):
        return "bicycle"
    if "van" in type_lower:
        return "van"
    return "car"


def _capture_name_suffix_from_env() -> str:
    """Optional DATASET_CAPTURE_NAME -> sensor_capture_<ts>_<name> (campaigns)."""
    raw = os.environ.get("DATASET_CAPTURE_NAME", "").strip()
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)[:48]
    return f"_{safe}" if safe else ""


def make_output_paths(base_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"sensor_capture_{timestamp}{_capture_name_suffix_from_env()}")
    camera_dir = os.path.join(run_dir, "camera_frames")
    os.makedirs(camera_dir, exist_ok=True)
    radar_csv = os.path.join(run_dir, "radar_data.csv")
    camera_csv = os.path.join(run_dir, "camera_data.csv")
    return run_dir, camera_dir, radar_csv, camera_csv


def setup_radar_writer(path):
    """RadarDetection fields + pose; actor match via OBB margin; rcs_proxy_m2 from OBB geometry."""
    file_handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(
        [
            "sensor_id",
            "sensor_label",
            "frame",
            "timestamp",
            "detection_index",
            "depth_m",
            "azimuth_rad",
            "altitude_rad",
            "velocity_mps",
            "sensor_world_x_m",
            "sensor_world_y_m",
            "sensor_world_z_m",
            "sensor_pitch_deg",
            "sensor_yaw_deg",
            "sensor_roll_deg",
            "matched_actor_id",
            "matched_actor_kind",
            "matched_actor_type_id",
            "matched_actor_class",
            "matched_actor_bbox_margin_m",
            "matched_vehicle_id",
            "matched_vehicle_type_id",
            "matched_vehicle_class",
            "matched_vehicle_distance_m",
            "rcs_proxy_m2",
            "had_actor_candidates",
            "label_scored",
            "nearest_actor_bbox_margin_m",
        ]
    )
    return file_handle, writer


def setup_camera_writer(path):
    file_handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(
        [
            "sensor_id",
            "sensor_label",
            "frame",
            "timestamp",
            "width",
            "height",
            "image_path",
            "nearest_actor_id",
            "nearest_actor_kind",
            "nearest_actor_type_id",
            "nearest_actor_class",
            "nearest_actor_distance_m",
            "nearby_actor_ids",
            "nearby_actor_kinds",
            "nearby_actor_classes",
            "nearest_vehicle_id",
            "nearest_vehicle_type_id",
            "nearest_vehicle_class",
            "nearest_vehicle_distance_m",
            "nearby_vehicle_ids",
            "nearby_vehicle_classes",
            "nearest_pedestrian_id",
            "nearest_pedestrian_type_id",
            "nearest_pedestrian_class",
            "nearest_pedestrian_distance_m",
            "nearby_pedestrian_ids",
            "nearby_pedestrian_classes",
            "segment_id",
        ]
    )
    return file_handle, writer


def sensor_label_from_role_name(role_name, prefix):
    if role_name.startswith(prefix):
        return role_name[len(prefix) :]
    return ""


def list_radar_actors(world):
    """All radar sensors in the world (robust type_id match across CARLA builds)."""
    return [a for a in world.get_actors() if "sensor.other.radar" in a.type_id]


def filter_tagged_sensors(world, actor_pattern, role_prefix, allowed_labels=None):
    filtered = []
    if actor_pattern == "sensor.other.radar":
        actors = list_radar_actors(world)
    else:
        actors = world.get_actors().filter(actor_pattern)
    for actor in actors:
        role_name = actor.attributes.get("role_name", "")
        if not role_name.startswith(role_prefix):
            continue
        if allowed_labels is not None:
            label = sensor_label_from_role_name(role_name, role_prefix)
            if label not in allowed_labels:
                continue
        filtered.append(actor)
    return filtered


def select_one_sensor_per_label(sensors, role_prefix, allowed_labels):
    """
    When multiple actors share the same role label (e.g. leftover radars from a prior run),
    keep the newest actor id per label.
    """
    best_by_label = {}
    for actor in sensors:
        label = sensor_label_from_role_name(actor.attributes.get("role_name", ""), role_prefix)
        if not label or label not in allowed_labels:
            continue
        prev = best_by_label.get(label)
        if prev is None or actor.id > prev.id:
            best_by_label[label] = actor
    return [best_by_label[label] for label in sorted(allowed_labels) if label in best_by_label]


def destroy_dataset_radars(world):
    """Remove stale dataset radars before a fresh RadarCameraSetup spawn."""
    removed = 0
    for actor in list_radar_actors(world):
        role_name = actor.attributes.get("role_name", "")
        if not role_name.startswith(DATASET_RADAR_ROLE_PREFIX):
            continue
        try:
            actor.destroy()
            removed += 1
        except RuntimeError:
            pass
    return removed


def wait_for_sensors(world, timeout_s, log_progress=False):
    deadline = time.time() + timeout_s
    last_log = 0.0
    last_radar_sensors: list = []
    last_camera_sensors: list = []
    expected = len(EXPECTED_RADAR_LABELS)

    while time.time() < deadline:
        all_radars = list_radar_actors(world)
        radar_sensors = filter_tagged_sensors(
            world,
            "sensor.other.radar",
            DATASET_RADAR_ROLE_PREFIX,
            EXPECTED_RADAR_LABELS,
        )
        camera_sensors = filter_tagged_sensors(
            world,
            "sensor.camera.rgb",
            DATASET_CAMERA_ROLE_PREFIX,
        )
        radar_sensors = select_one_sensor_per_label(
            radar_sensors, DATASET_RADAR_ROLE_PREFIX, EXPECTED_RADAR_LABELS
        )
        last_radar_sensors = radar_sensors
        last_camera_sensors = camera_sensors

        if len(radar_sensors) == expected:
            return radar_sensors, camera_sensors

        if log_progress and time.time() - last_log >= 5.0:
            unique_labels = len(radar_sensors)
            print(
                f"  Waiting for radars: {len(all_radars)} in world, "
                f"{unique_labels}/{expected} unique labels ready",
                flush=True,
            )
            last_log = time.time()

        time.sleep(SENSOR_WAIT_POLL_S)

    last_radar_sensors = select_one_sensor_per_label(
        last_radar_sensors, DATASET_RADAR_ROLE_PREFIX, EXPECTED_RADAR_LABELS
    )
    return last_radar_sensors, last_camera_sensors


def pedestrian_class_from_type_id(type_id):
    return "pedestrian"


def get_vehicle_snapshots(world):
    return [s for s in get_radar_target_snapshots(world) if s["kind"] == "vehicle"]


def _snapshot_from_actor(actor, kind: str, class_label: str) -> dict | None:
    """One RPC (get_transform) + local bbox attribute access — no enrich pass needed."""
    try:
        actor_tf = actor.get_transform()
    except RuntimeError:
        return None
    bbox = actor.bounding_box
    bbox_rotation = None
    if bbox.rotation is not None:
        bbox_rotation = {
            "pitch": float(bbox.rotation.pitch),
            "yaw": float(bbox.rotation.yaw),
            "roll": float(bbox.rotation.roll),
        }
    return {
        "id": actor.id,
        "kind": kind,
        "type_id": actor.type_id,
        "class_label": class_label,
        "location": actor_tf.location,
        "rotation": {
            "pitch": float(actor_tf.rotation.pitch),
            "yaw": float(actor_tf.rotation.yaw),
            "roll": float(actor_tf.rotation.roll),
        },
        "bbox": {
            "location": {
                "x": float(bbox.location.x),
                "y": float(bbox.location.y),
                "z": float(bbox.location.z),
            },
            "extent": {
                "x": float(bbox.extent.x),
                "y": float(bbox.extent.y),
                "z": float(bbox.extent.z),
            },
            "rotation": bbox_rotation,
        },
    }


def get_radar_target_snapshots(world):
    """Vehicles and pedestrians (walkers) eligible for radar point labeling.

    Each snapshot already contains rotation + bbox so offline labeling can use it
    directly without re-querying CARLA per actor (hot-path RPC saver).
    """
    snapshots = []
    for vehicle in world.get_actors().filter("vehicle.*"):
        snap = _snapshot_from_actor(
            vehicle, "vehicle", vehicle_class_from_type_id(vehicle.type_id)
        )
        if snap is not None:
            snapshots.append(snap)
    for walker in world.get_actors().filter("walker.pedestrian.*"):
        snap = _snapshot_from_actor(
            walker, "pedestrian", pedestrian_class_from_type_id(walker.type_id)
        )
        if snap is not None:
            snapshots.append(snap)
    return snapshots


def make_fast_tick_snapshot_fn(world):
    """Returns a ``(world, world_snapshot) -> list[dict]`` for ``TickActorSnapshotter``.

    The naive ``get_radar_target_snapshots(world)`` issues one CARLA RPC per
    actor (~60 RPCs per tick at full traffic). When this runs inside CARLA's
    ``on_tick`` callback, the server's tick budget is ~33 ms at 30 Hz and the
    server WILL silently stop dispatching the callback once a previous one
    overruns. Capture 231410 hit this: ``_on_tick`` fired for ~12 s while the
    spawner ramped up actors, then went silent for 7.4 min once the actor
    population reached ~60.

    This builder avoids the RPC storm by:

    * Iterating ``world_snapshot`` directly — it already contains the current
      pose of every actor in the world at this exact frame, at zero RPC cost.
    * Caching static per-actor metadata (kind, type_id, class_label, bbox) the
      first time each ``actor_id`` is seen. After the cache warms up (first few
      ticks), the steady-state per-tick cost is **zero CARLA RPCs**.

    The returned dicts are shape-compatible with ``get_radar_target_snapshots``
    output, so downstream code (offline labeler, ``ActorFrameLogger``, etc.)
    needs no changes.
    """
    actor_meta_cache: dict[int, dict | None] = {}

    def _build_meta(actor) -> dict | None:
        type_id = actor.type_id
        if type_id.startswith("vehicle."):
            kind = "vehicle"
            class_label = vehicle_class_from_type_id(type_id)
        elif type_id.startswith("walker.pedestrian."):
            kind = "pedestrian"
            class_label = pedestrian_class_from_type_id(type_id)
        else:
            return None
        bbox = actor.bounding_box
        bbox_rotation = None
        if bbox.rotation is not None:
            bbox_rotation = {
                "pitch": float(bbox.rotation.pitch),
                "yaw": float(bbox.rotation.yaw),
                "roll": float(bbox.rotation.roll),
            }
        return {
            "id": int(actor.id),
            "kind": kind,
            "type_id": type_id,
            "class_label": class_label,
            "bbox": {
                "location": {
                    "x": float(bbox.location.x),
                    "y": float(bbox.location.y),
                    "z": float(bbox.location.z),
                },
                "extent": {
                    "x": float(bbox.extent.x),
                    "y": float(bbox.extent.y),
                    "z": float(bbox.extent.z),
                },
                "rotation": bbox_rotation,
            },
        }

    def fast_tick_snapshot(world, world_snapshot):
        out = []
        for actor_snap in world_snapshot:
            aid = int(actor_snap.id)
            if aid in actor_meta_cache:
                meta = actor_meta_cache[aid]
                if meta is None:
                    # Known non-target (props, sensors, spectator, etc.) — skip.
                    continue
            else:
                # First time we've seen this actor — one RPC to populate cache.
                try:
                    actor = world.get_actor(aid)
                except RuntimeError:
                    actor_meta_cache[aid] = None
                    continue
                if actor is None:
                    actor_meta_cache[aid] = None
                    continue
                meta = _build_meta(actor)
                actor_meta_cache[aid] = meta
                if meta is None:
                    continue

            tf = actor_snap.get_transform()
            out.append(
                {
                    "id": meta["id"],
                    "kind": meta["kind"],
                    "type_id": meta["type_id"],
                    "class_label": meta["class_label"],
                    "location": tf.location,
                    "rotation": {
                        "pitch": float(tf.rotation.pitch),
                        "yaw": float(tf.rotation.yaw),
                        "roll": float(tf.rotation.roll),
                    },
                    "bbox": meta["bbox"],
                }
            )
        return out

    return fast_tick_snapshot


class RadarActorSnapshotCache:
    """Lazy single-frame fallback cache used when no TickActorSnapshotter is wired up.

    Prefer ``TickActorSnapshotter`` (see ``actor_frame_log.py``): it captures
    actor state synchronously with the simulation tick via ``world.on_tick``,
    so radar messages can be matched against the exact frame they belong to —
    even when processing falls behind or runs after Ctrl+C. This class remains
    for legacy code paths and is no longer used by the main capture loop.

    Per-actor static metadata (bbox, type_id, class_label) is cached across
    frames — ``actor.bounding_box`` is a ~4 ms CARLA RPC and never changes for
    a spawned actor, so refetching it every frame dominates the test-loop
    drain time. Only ``get_transform()`` (location + rotation) is refreshed
    per frame.
    """

    def __init__(self, world) -> None:
        self._world = world
        self._frame: int | None = None
        self._snapshots: list = []
        # actor_id -> static meta dict: id, kind, type_id, class_label, bbox
        self._meta_cache: dict[int, dict] = {}

    def _build_meta(self, actor, kind: str, class_label: str) -> dict:
        bbox = actor.bounding_box  # 1 RPC, cached for the lifetime of the actor
        bbox_rotation = None
        if bbox.rotation is not None:
            bbox_rotation = {
                "pitch": float(bbox.rotation.pitch),
                "yaw": float(bbox.rotation.yaw),
                "roll": float(bbox.rotation.roll),
            }
        return {
            "id": int(actor.id),
            "kind": kind,
            "type_id": actor.type_id,
            "class_label": class_label,
            "bbox": {
                "location": {
                    "x": float(bbox.location.x),
                    "y": float(bbox.location.y),
                    "z": float(bbox.location.z),
                },
                "extent": {
                    "x": float(bbox.extent.x),
                    "y": float(bbox.extent.y),
                    "z": float(bbox.extent.z),
                },
                "rotation": bbox_rotation,
            },
        }

    def _snapshot_with_cache(self, actor, kind: str, class_label: str) -> dict | None:
        try:
            actor_tf = actor.get_transform()
        except RuntimeError:
            return None
        aid = int(actor.id)
        meta = self._meta_cache.get(aid)
        if meta is None:
            try:
                meta = self._build_meta(actor, kind, class_label)
            except RuntimeError:
                return None
            self._meta_cache[aid] = meta
        # Fresh-per-frame fields (cheap): location + rotation.
        return {
            **meta,
            "location": actor_tf.location,
            "rotation": {
                "pitch": float(actor_tf.rotation.pitch),
                "yaw": float(actor_tf.rotation.yaw),
                "roll": float(actor_tf.rotation.roll),
            },
        }

    def _build_snapshots(self) -> list:
        snapshots: list = []
        seen_ids: set[int] = set()
        for vehicle in self._world.get_actors().filter("vehicle.*"):
            snap = self._snapshot_with_cache(
                vehicle, "vehicle", vehicle_class_from_type_id(vehicle.type_id)
            )
            if snap is not None:
                snapshots.append(snap)
                seen_ids.add(int(vehicle.id))
        for walker in self._world.get_actors().filter("walker.pedestrian.*"):
            snap = self._snapshot_with_cache(
                walker, "pedestrian", pedestrian_class_from_type_id(walker.type_id)
            )
            if snap is not None:
                snapshots.append(snap)
                seen_ids.add(int(walker.id))
        # Drop stale meta entries so destroyed actors don't leak memory.
        stale = [aid for aid in self._meta_cache if aid not in seen_ids]
        for aid in stale:
            self._meta_cache.pop(aid, None)
        return snapshots

    def get(self, frame_id: int):
        if self._frame != frame_id:
            self._frame = frame_id
            self._snapshots = self._build_snapshots()
        return self._snapshots


def normalize_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def radar_detection_is_labelable(
    velocity_mps: float,
    *,
    min_speed_mps: float = RADAR_LABELABLE_MIN_SPEED_MPS,
    had_candidates: bool = False,
) -> bool:
    """
    True when this return should be scored.

    With had_candidates=True, parked actors in the beam are included even if |v| is low.
    Static clutter with no actor in the beam is excluded unless |v| exceeds min_speed_mps.
    """
    if had_candidates:
        return True
    return abs(velocity_mps) >= min_speed_mps


def should_score_radar_return(
    velocity_mps: float,
    had_candidates: bool,
    *,
    min_speed_mps: float = RADAR_LABELABLE_MIN_SPEED_MPS,
) -> bool:
    return radar_detection_is_labelable(
        velocity_mps, min_speed_mps=min_speed_mps, had_candidates=had_candidates
    )


def radar_candidate_hit_max_bbox_margin_m() -> float | None:
    """
    Candidate hit proximity (m). Override via DATASET_RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M;
    set to 'none'/'off'/'disable' for beam-only candidacy (legacy behavior).
    """
    raw = os.environ.get("DATASET_RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M", "").strip().lower()
    if raw in ("none", "off", "disable"):
        return None
    if raw:
        try:
            value = float(raw)
            return None if value <= 0 else min(max(value, 1.0), 25.0)
        except ValueError:
            pass
    return RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M


def radar_hit_match_max_margin_m_from_env() -> float:
    """Override the primary hit-to-OBB acceptance margin via
    ``DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M``. Clamped to [0.5, 25.0] m."""
    raw = os.environ.get("DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M", "").strip()
    if raw:
        try:
            return max(0.5, min(float(raw), 25.0))
        except ValueError:
            pass
    return RADAR_HIT_MATCH_MAX_MARGIN_M


def radar_single_candidate_max_margin_m_from_env() -> float:
    """Override the looser single-candidate fallback margin via
    ``DATASET_RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M``. Clamped to [0.5, 25.0] m."""
    raw = os.environ.get("DATASET_RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M", "").strip()
    if raw:
        try:
            return max(0.5, min(float(raw), 25.0))
        except ValueError:
            pass
    return RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M


def radar_points_per_second_from_env() -> int:
    """Override via DATASET_RADAR_POINTS_PER_SECOND (e.g. 6000). Clamped 500–20000."""
    raw = os.environ.get("DATASET_RADAR_POINTS_PER_SECOND", "").strip()
    if raw:
        try:
            return max(500, min(int(raw), 20000))
        except ValueError:
            pass
    return RADAR_POINTS_PER_SECOND_DEFAULT


def radar_horizontal_fov_deg_from_env() -> float:
    """Override via DATASET_RADAR_HORIZONTAL_FOV_DEG. Narrower FOV = denser actor hits."""
    raw = os.environ.get("DATASET_RADAR_HORIZONTAL_FOV_DEG", "").strip()
    if raw:
        try:
            return max(10.0, min(float(raw), 120.0))
        except ValueError:
            pass
    return RADAR_HORIZONTAL_FOV_DEG


def radar_vertical_fov_deg_from_env() -> float:
    raw = os.environ.get("DATASET_RADAR_VERTICAL_FOV_DEG", "").strip()
    if raw:
        try:
            return max(10.0, min(float(raw), 90.0))
        except ValueError:
            pass
    return RADAR_VERTICAL_FOV_DEG


def radar_sensor_tick_s_from_env() -> float:
    """Override via DATASET_RADAR_SENSOR_TICK_S (seconds between measurements)."""
    raw = os.environ.get("DATASET_RADAR_SENSOR_TICK_S", "").strip()
    if raw:
        try:
            return max(0.0, min(float(raw), 1.0))
        except ValueError:
            pass
    return RADAR_SENSOR_TICK_S


def radar_capture_fast_from_env() -> bool:
    """When true, write every CARLA return without per-detection actor matching (much higher throughput)."""
    raw = os.environ.get("DATASET_RADAR_CAPTURE_FAST", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def sync_mode_from_env() -> bool:
    """When True, the capture script becomes the world tick driver: it enables CARLA
    synchronous mode and calls ``world.tick()`` on its own cadence so every
    listening radar fires on the exact same world frame. Default off (preserves
    async behavior)."""
    raw = os.environ.get("DATASET_SYNC_MODE", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def sync_fixed_delta_s_from_env() -> float:
    """``fixed_delta_seconds`` used while sync mode is active. When unset, defaults
    to the configured radar sensor_tick so each tick produces exactly one radar
    measurement per sensor — i.e. all 8 sensors share every frame_id."""
    raw = os.environ.get("DATASET_SYNC_FIXED_DELTA_S", "").strip()
    if raw:
        try:
            return max(0.005, min(float(raw), 0.5))
        except ValueError:
            pass
    tick = radar_sensor_tick_s_from_env()
    return tick if tick > 0 else RADAR_SENSOR_TICK_S


def sync_min_period_s_from_env() -> float:
    """Minimum wall-clock gap between ``world.tick()`` calls.

    Simulation time still advances by ``fixed_delta_seconds`` every tick. This
    only stops a fast machine from ticking faster than the radar CSV writer
    can drain, which otherwise overflows the per-radar deque and drops
    measurements. Unset or 0 keeps the previous behavior (tick as soon as
    CARLA returns). ``DATASET_SYNC_MIN_PERIOD_S``.
    """
    raw = os.environ.get("DATASET_SYNC_MIN_PERIOD_S", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, min(float(raw), 1.0))
    except ValueError:
        return 0.0


def radar_watchdog_stale_ticks_from_env() -> int:
    """Number of world ticks a single radar can fall behind its peers before the
    watchdog re-attaches its ``listen()`` callback. Capture 231410 lost R7 after
    ~51 s because CARLA's per-sensor listen callback silently stopped firing
    while every other radar kept going. Default 60 ticks (~2 s at 30 Hz). Set
    to 0 to disable the watchdog. Override via ``DATASET_RADAR_WATCHDOG_STALE_TICKS``."""
    raw = os.environ.get("DATASET_RADAR_WATCHDOG_STALE_TICKS", "").strip()
    if not raw:
        return 60
    try:
        return max(0, int(raw))
    except ValueError:
        return 60


def camera_watchdog_stale_ticks_from_env() -> int:
    """Same listen() watchdog for RGB cameras. Defaults to the radar knob
    (``DATASET_RADAR_WATCHDOG_STALE_TICKS``) so one env var covers both.
    Override cameras alone with ``DATASET_CAMERA_WATCHDOG_STALE_TICKS``."""
    raw = os.environ.get("DATASET_CAMERA_WATCHDOG_STALE_TICKS", "").strip()
    if not raw:
        return radar_watchdog_stale_ticks_from_env()
    try:
        return max(0, int(raw))
    except ValueError:
        return radar_watchdog_stale_ticks_from_env()


def traffic_manager_port_from_env() -> int:
    """TM port to align with the world's sync mode. Defaults to CARLA's standard
    port; override via ``DATASET_TRAFFIC_MANAGER_PORT`` if your spawner uses
    something else (e.g. 8000)."""
    raw = os.environ.get("DATASET_TRAFFIC_MANAGER_PORT", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 8000


def configure_dataset_radar_blueprint(radar_bp) -> int:
    """
    Apply shared dataset radar settings to a CARLA sensor.other.radar blueprint.
    Returns the points_per_second value applied (for logging).
    """
    if radar_bp.has_attribute("range"):
        radar_bp.set_attribute("range", str(int(RADAR_MAX_RANGE_M)))
    hfov = int(radar_horizontal_fov_deg_from_env())
    vfov = int(radar_vertical_fov_deg_from_env())
    if radar_bp.has_attribute("horizontal_fov"):
        radar_bp.set_attribute("horizontal_fov", str(hfov))
    if radar_bp.has_attribute("vertical_fov"):
        radar_bp.set_attribute("vertical_fov", str(vfov))
    pps = radar_points_per_second_from_env()
    if radar_bp.has_attribute("points_per_second"):
        radar_bp.set_attribute("points_per_second", str(pps))
    if radar_bp.has_attribute("sensor_tick"):
        radar_bp.set_attribute("sensor_tick", str(radar_sensor_tick_s_from_env()))
    return pps


def radar_effective_max_depth_m(
    range_m=RADAR_MAX_RANGE_M, hfov_deg=None, vfov_deg=None
) -> float:
    """Longest ray CARLA's radar can return for the given blueprint values.

    CARLA aims each ray at a point on the rectangle x=range, |y|<=range*tan(hfov/2),
    |z|<=range*tan(vfov/2), so corner rays are range*sqrt(1+tan^2+tan^2) long.
    """
    if hfov_deg is None:
        hfov_deg = radar_horizontal_fov_deg_from_env()
    if vfov_deg is None:
        vfov_deg = radar_vertical_fov_deg_from_env()
    ty = math.tan(math.radians(min(hfov_deg, 179.0) / 2.0))
    tz = math.tan(math.radians(min(vfov_deg, 179.0) / 2.0))
    return float(range_m) * math.sqrt(1.0 + ty * ty + tz * tz)


def radar_sensor_limits(radar_actor):
    """Read range (m) and horizontal FOV (deg) from a spawned radar actor."""
    attrs = radar_actor.attributes
    range_m = RADAR_MAX_RANGE_M
    hfov_deg = RADAR_HORIZONTAL_FOV_DEG
    if attrs.get("range"):
        try:
            range_m = float(attrs["range"])
        except ValueError:
            pass
    if attrs.get("horizontal_fov"):
        try:
            hfov_deg = float(attrs["horizontal_fov"])
        except ValueError:
            pass
    return range_m, hfov_deg


def _planar_range_bearing_deg(sensor_location, target_location):
    dx = target_location.x - sensor_location.x
    dy = target_location.y - sensor_location.y
    return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))


def _detection_beam_yaw_deg(sensor_transform, detection):
    return normalize_angle_deg(
        sensor_transform.rotation.yaw + math.degrees(float(detection.azimuth))
    )


def _actor_bbox_world_center_and_extent(world, actor_snapshot):
    # Fast path: per-frame precompute populates these once per actor (see
    # precompute_actor_frame_cache). Saves ~270 carla.Transform.transform() calls
    # per radar message in the test loop.
    cached_center = actor_snapshot.get("_world_center")
    cached_extent = actor_snapshot.get("_extent")
    if cached_center is not None and cached_extent is not None:
        return cached_center, cached_extent

    bbox = actor_snapshot.get("bbox")
    if bbox and actor_snapshot.get("location") and actor_snapshot.get("rotation"):
        rot = actor_snapshot["rotation"]
        actor_tf = carla.Transform(
            snapshot_location(actor_snapshot),
            carla.Rotation(float(rot["pitch"]), float(rot["yaw"]), float(rot["roll"])),
        )
        bl = bbox["location"]
        ex = bbox["extent"]
        center = actor_tf.transform(
            carla.Location(float(bl["x"]), float(bl["y"]), float(bl["z"]))
        )
        return center, carla.Vector3D(float(ex["x"]), float(ex["y"]), float(ex["z"]))

    if world is None:
        return None, None
    try:
        actor = world.get_actor(actor_snapshot["id"])
    except RuntimeError:
        return None, None
    bbox = actor.bounding_box
    actor_tf = actor.get_transform()
    center = actor_tf.transform(bbox.location)
    return center, bbox.extent


def precompute_actor_frame_cache(actors, world=None):
    """Populate per-actor cached quantities used by the radar labeling hot path.

    For each actor snapshot with a logged bbox + location + rotation, computes:
      _world_center      : carla.Location, bbox center in world coords
      _extent            : carla.Vector3D, bbox extent
      _max_extent_xy_m   : float, max planar extent (for beam angular gate)
      _inv_actor_tf      : carla.Transform(loc=0, rot=-actor_rot) — pre-built
                           rotation-only inverse, reused by actor_bbox_margin_m
      _inv_bbox_tf       : same for bbox local rotation, when bbox has rotation

    All keys are sensor-independent, so a single call per frame covers every
    detection from every radar firing at that frame. Idempotent (skips actors
    already cached).
    """
    for actor in actors:
        if "_world_center" in actor:
            continue
        center, extent = _actor_bbox_world_center_and_extent(world, actor)
        if center is None or extent is None:
            continue
        actor["_world_center"] = center
        actor["_extent"] = extent
        actor["_max_extent_xy_m"] = math.hypot(extent.x, extent.y)
        rot = actor.get("rotation") or {}
        try:
            actor["_inv_actor_tf"] = _inverse_rotation_transform(carla.Rotation(
                pitch=float(rot.get("pitch", 0.0)),
                yaw=float(rot.get("yaw", 0.0)),
                roll=float(rot.get("roll", 0.0)),
            ))
        except Exception:  # noqa: BLE001
            pass
        bbox = actor.get("bbox") or {}
        bbox_rot = bbox.get("rotation")
        if bbox_rot:
            try:
                actor["_inv_bbox_tf"] = _inverse_rotation_transform(carla.Rotation(
                    pitch=float(bbox_rot.get("pitch", 0.0)),
                    yaw=float(bbox_rot.get("yaw", 0.0)),
                    roll=float(bbox_rot.get("roll", 0.0)),
                ))
            except Exception:  # noqa: BLE001
                pass


def actor_snapshot_in_sensor_fov(
    sensor_transform, actor_location, max_distance_m, horizontal_fov_deg
):
    """True when actor center is within horizontal FOV and planar range of the sensor."""
    sensor_location = sensor_transform.location
    distance, bearing_deg = _planar_range_bearing_deg(sensor_location, actor_location)
    if distance > max_distance_m:
        return False
    sensor_yaw = sensor_transform.rotation.yaw
    yaw_delta = abs(normalize_angle_deg(bearing_deg - sensor_yaw))
    return yaw_delta <= horizontal_fov_deg * 0.5


def actor_visible_in_detection_beam(
    world,
    sensor_transform,
    detection,
    actor_snapshot,
    *,
    max_range_m=RADAR_MAX_RANGE_M,
    horizontal_fov_deg=RADAR_HORIZONTAL_FOV_DEG,
    depth_margin_m=RADAR_CANDIDATE_DEPTH_MARGIN_M,
    azimuth_margin_deg=RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
):
    """
    True when the actor OBB is plausibly illuminated by this detection (depth + bearing gate).
    Uses bbox center range and angular extent, not only the actor origin.
    Works offline when actor_snapshot includes logged bbox (actor_frames.jsonl).
    """
    center, extent = _actor_bbox_world_center_and_extent(world, actor_snapshot)
    if center is None or extent is None:
        if world is not None:
            return False
        return actor_snapshot_in_sensor_fov(
            sensor_transform,
            actor_snapshot["location"],
            min(max_range_m, float(detection.depth) + depth_margin_m),
            horizontal_fov_deg,
        )

    sensor_loc = sensor_transform.location
    range_m, bearing_deg = _planar_range_bearing_deg(sensor_loc, center)
    max_extent_m = actor_snapshot.get("_max_extent_xy_m")
    if max_extent_m is None:
        max_extent_m = math.hypot(extent.x, extent.y)
    depth = float(detection.depth)
    depth_min = max(0.0, depth - depth_margin_m - max_extent_m)
    depth_max = min(max_range_m, depth + depth_margin_m + max_extent_m)
    if range_m < depth_min or range_m > depth_max:
        return False

    beam_yaw = _detection_beam_yaw_deg(sensor_transform, detection)
    angular_half_deg = math.degrees(math.atan2(max_extent_m, max(range_m, 0.5)))
    yaw_delta = abs(normalize_angle_deg(bearing_deg - beam_yaw))
    half_fov = horizontal_fov_deg * 0.5
    return yaw_delta <= half_fov + azimuth_margin_deg or yaw_delta <= angular_half_deg + azimuth_margin_deg


def actor_snapshots_near_sensor(sensor_location, actor_snapshots, max_distance_m):
    """Actors whose transform location is within max_distance_m (3D) of the sensor."""
    out = []
    for actor in actor_snapshots:
        if sensor_location.distance(actor["location"]) <= max_distance_m:
            out.append(actor)
    return out


_HIT_MARGIN_DEFAULT = object()


def actor_snapshots_for_radar_detection(
    sensor_transform,
    detection,
    actor_snapshots,
    world=None,
    *,
    max_range_m=RADAR_MAX_RANGE_M,
    horizontal_fov_deg=RADAR_HORIZONTAL_FOV_DEG,
    depth_margin_m=RADAR_CANDIDATE_DEPTH_MARGIN_M,
    azimuth_margin_deg=RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
    hit_max_bbox_margin_m=_HIT_MARGIN_DEFAULT,
):
    """
    Per-detection candidates: actor OBB in the detection beam, optionally near the hit.

    When hit_max_bbox_margin_m is set, actors only qualify if the reconstructed hit is within
    that distance of their OBB (reduces beam-only false candidates).
    """
    if hit_max_bbox_margin_m is _HIT_MARGIN_DEFAULT:
        hit_max_bbox_margin_m = radar_candidate_hit_max_bbox_margin_m()
    candidates = []
    for actor in actor_snapshots:
        if not actor_visible_in_detection_beam(
            world,
            sensor_transform,
            detection,
            actor,
            max_range_m=max_range_m,
            horizontal_fov_deg=horizontal_fov_deg,
            depth_margin_m=depth_margin_m,
            azimuth_margin_deg=azimuth_margin_deg,
        ):
            continue
        candidates.append(actor)

    if not candidates or hit_max_bbox_margin_m is None:
        return candidates

    hit_loc = radar_detection_world_location(sensor_transform, detection)
    near_hit = []
    ped_limit = min(hit_max_bbox_margin_m, ped_candidate_max_dist_m())
    for actor in candidates:
        if _is_pedestrian(actor):
            margin = actor_bbox_margin_m(world, hit_loc, actor, inflation_m=0.0)
            limit = ped_limit
        else:
            margin = actor_bbox_margin_m(world, hit_loc, actor)
            limit = hit_max_bbox_margin_m
        if margin is not None and margin <= limit:
            near_hit.append(actor)
    return near_hit


def _actors_in_depth_azimuth_gate(
    world,
    sensor_transform,
    detection,
    candidate_actors,
    *,
    max_range_m=RADAR_MAX_RANGE_M,
    depth_margin_m=RADAR_CANDIDATE_DEPTH_MARGIN_M,
    azimuth_margin_deg=RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
):
    if world is None:
        return list(candidate_actors)
    max_range_m = max(
        float(max_range_m),
        radar_effective_max_depth_m(max_range_m, None, max(radar_vertical_fov_deg_from_env(), 60.0)),
    )
    gated = []
    for actor in candidate_actors:
        if actor_visible_in_detection_beam(
            world,
            sensor_transform,
            detection,
            actor,
            max_range_m=max_range_m,
            horizontal_fov_deg=180.0,
            depth_margin_m=depth_margin_m,
            azimuth_margin_deg=azimuth_margin_deg,
        ):
            gated.append(actor)
    return gated


def vehicle_snapshots_near_sensor(sensor_location, vehicle_snapshots, max_distance_m):
    return actor_snapshots_near_sensor(sensor_location, vehicle_snapshots, max_distance_m)


def radar_detection_world_location_legacy(sensor_transform, detection):
    """DEPRECATED approximation, kept only for TestRadarLabeling's comparison mode.

    Composes the beam as Euler offsets (sensor pitch + altitude, sensor yaw +
    azimuth), which is what manual_control.py does for drawing. That is only
    exact when the sensor pitch is 0: with an 8 deg mount pitch it is off by
    0.8 m on average and up to 4.6 m at the cone edges (measured on capture
    20260917_125123). Do not use for labeling.
    """
    rot = sensor_transform.rotation
    beam_rot = carla.Rotation(
        pitch=rot.pitch + math.degrees(detection.altitude),
        yaw=rot.yaw + math.degrees(detection.azimuth),
        roll=rot.roll,
    )
    offset = carla.Transform(carla.Location(), beam_rot).transform(
        carla.Vector3D(x=detection.depth)
    )
    loc = sensor_transform.location
    return carla.Location(loc.x + offset.x, loc.y + offset.y, loc.z + offset.z)


def radar_detection_world_location(sensor_transform, detection):
    """Exact world-space hit point.

    CARLA computes each detection's azimuth/altitude with
    FMath::GetAzimuthAndElevation() against the sensor's own X/Y/Z axes, so the
    ray direction in the SENSOR frame is the spherical unit vector
        (cos(alt) cos(az), cos(alt) sin(az), sin(alt))
    and the hit is that vector scaled by depth, pushed through the sensor's
    full transform (carla.Transform.transform applies the same UE rotation
    matrix the server used). No Euler-angle addition, valid for any mount
    pitch/roll.
    """
    d = float(detection.depth)
    ca = math.cos(detection.altitude)
    local = carla.Location(
        x=d * ca * math.cos(detection.azimuth),
        y=d * ca * math.sin(detection.azimuth),
        z=d * math.sin(detection.altitude),
    )
    return sensor_transform.transform(local)


_HAS_INVERSE_TRANSFORM = hasattr(carla.Transform, "inverse_transform")


def _inverse_rotation_transform(rotation):
    """Transform that maps a WORLD offset into the frame rotated by ``rotation``.

    Uses carla.Transform.inverse_transform on a rotation-only transform when the
    API has it (exact R^T). Otherwise falls back to Rotation(-p, -y, -r), which
    is only exact for yaw-only rotations (fine for road vehicles on the flat).
    """
    if _HAS_INVERSE_TRANSFORM:
        return carla.Transform(carla.Location(), carla.Rotation(
            pitch=float(rotation.pitch), yaw=float(rotation.yaw), roll=float(rotation.roll)))
    return carla.Transform(carla.Location(), carla.Rotation(
        pitch=-float(rotation.pitch), yaw=-float(rotation.yaw), roll=-float(rotation.roll)))


def _apply_inverse_rotation(inv_tf, offset):
    if _HAS_INVERSE_TRANSFORM:
        return inv_tf.inverse_transform(carla.Location(offset.x, offset.y, offset.z))
    return inv_tf.transform(carla.Location(offset.x, offset.y, offset.z))


def _world_offset_in_actor_frame(world_offset, actor_rotation):
    """Rotate a world-space offset into the actor's local frame."""
    return _apply_inverse_rotation(_inverse_rotation_transform(actor_rotation), world_offset)


GROUND_REJECT_MARGIN_M = 9.0


def _obb_margin_from_local(lx, ly, lz, ex, ey, ez, inflation):
    """Margin (m) of a hit expressed in the actor OBB frame.

    0 when inside the true box; GROUND_REJECT_MARGIN_M when the hit is below
    the actor's ground plane (a road return beside/under the actor); otherwise
    the distance to the box inflated by ``inflation`` on every axis.
    """
    ax, ay, az = abs(lx), abs(ly), abs(lz)
    if ax <= ex and ay <= ey and az <= ez:
        return 0.0
    if lz < -ez + GROUND_REJECT_CLEARANCE_M:
        return GROUND_REJECT_MARGIN_M
    dx = max(0.0, ax - (ex + inflation))
    dy = max(0.0, ay - (ey + inflation))
    dz = max(0.0, az - (ez + inflation))
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return 0.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def actor_bbox_margin_m(world, hit_location, actor_snapshot, inflation_m=BBOX_MATCH_EXTENT_INFLATION_M):
    """
    Signed margin to the actor OBB in meters: 0 if inside (with optional inflation),
    otherwise the shortest distance from the hit to the box surface. Hits below the
    actor's ground plane are returned as GROUND_REJECT_MARGIN_M (see
    _obb_margin_from_local).

    Fast path: when precompute_actor_frame_cache has populated _world_center,
    _extent, _inv_actor_tf (and optionally _inv_bbox_tf), skips reconstructing
    Transform objects per call.
    """
    logged_bbox = actor_snapshot.get("bbox")
    cached_center = actor_snapshot.get("_world_center")
    cached_extent = actor_snapshot.get("_extent")
    cached_inv_actor_tf = actor_snapshot.get("_inv_actor_tf")
    cached_inv_bbox_tf = actor_snapshot.get("_inv_bbox_tf")  # may be None
    if (
        cached_center is not None
        and cached_extent is not None
        and cached_inv_actor_tf is not None
    ):
        delta = carla.Location(
            hit_location.x - cached_center.x,
            hit_location.y - cached_center.y,
            hit_location.z - cached_center.z,
        )
        local = _apply_inverse_rotation(cached_inv_actor_tf, delta)
        if cached_inv_bbox_tf is not None:
            local = _apply_inverse_rotation(cached_inv_bbox_tf, local)
        return _obb_margin_from_local(
            local.x, local.y, local.z,
            cached_extent.x, cached_extent.y, cached_extent.z, inflation_m,
        )

    if logged_bbox and actor_snapshot.get("location") and actor_snapshot.get("rotation"):
        rot = actor_snapshot["rotation"]
        actor_tf = carla.Transform(
            snapshot_location(actor_snapshot),
            carla.Rotation(float(rot["pitch"]), float(rot["yaw"]), float(rot["roll"])),
        )
        bl = logged_bbox["location"]
        ex = logged_bbox["extent"]
        bbox_loc = carla.Location(float(bl["x"]), float(bl["y"]), float(bl["z"]))
        bbox_rot = logged_bbox.get("rotation")
        center_world = actor_tf.transform(bbox_loc)
        delta = carla.Location(
            hit_location.x - center_world.x,
            hit_location.y - center_world.y,
            hit_location.z - center_world.z,
        )
        local = _world_offset_in_actor_frame(delta, actor_tf.rotation)
        if bbox_rot:
            local = _world_offset_in_actor_frame(
                local,
                carla.Rotation(
                    float(bbox_rot["pitch"]),
                    float(bbox_rot["yaw"]),
                    float(bbox_rot["roll"]),
                ),
            )
        return _obb_margin_from_local(
            local.x, local.y, local.z,
            float(ex["x"]), float(ex["y"]), float(ex["z"]), inflation_m,
        )

    if world is None:
        return None
    try:
        actor = world.get_actor(actor_snapshot["id"])
    except RuntimeError:
        return None

    bbox = actor.bounding_box
    actor_tf = actor.get_transform()
    center_world = actor_tf.transform(bbox.location)
    delta = carla.Location(
        hit_location.x - center_world.x,
        hit_location.y - center_world.y,
        hit_location.z - center_world.z,
    )
    local = _world_offset_in_actor_frame(delta, actor_tf.rotation)
    if bbox.rotation:
        local = _world_offset_in_actor_frame(local, bbox.rotation)

    return _obb_margin_from_local(
        local.x, local.y, local.z,
        bbox.extent.x, bbox.extent.y, bbox.extent.z, inflation_m,
    )


def vehicle_hit_distance_m(world, hit_location, vehicle_snapshot):
    """Deprecated sphere proxy; prefer actor_bbox_margin_m."""
    margin = actor_bbox_margin_m(world, hit_location, vehicle_snapshot, inflation_m=0.0)
    if margin is not None:
        return margin
    return hit_location.distance(vehicle_snapshot["location"])


def actor_rcs_proxy_projected_area_m2(actor_snapshot, sensor_location):
    """
    Sum of (face area × cos θ) for OBB faces visible from the sensor direction — a geometric
    RCS surrogate (m²). Not physical radar cross section; empty if the snapshot is malformed.

    Reads bbox + transform from the cached snapshot dict produced by
    ``TickActorSnapshotter`` / ``RadarActorSnapshotCache`` — no CARLA RPCs.
    """
    if not actor_snapshot:
        return ""
    bbox = actor_snapshot.get("bbox")
    actor_loc = actor_snapshot.get("location")
    actor_rot = actor_snapshot.get("rotation")
    if not bbox or actor_loc is None or actor_rot is None:
        return ""
    extent = bbox.get("extent") or {}
    bbox_loc = bbox.get("location") or {}
    ex = float(extent.get("x", 0.0))
    ey = float(extent.get("y", 0.0))
    ez = float(extent.get("z", 0.0))
    face_specs = [
        ((1.0, 0.0, 0.0), 4.0 * ey * ez),
        ((-1.0, 0.0, 0.0), 4.0 * ey * ez),
        ((0.0, 1.0, 0.0), 4.0 * ex * ez),
        ((0.0, -1.0, 0.0), 4.0 * ex * ez),
        ((0.0, 0.0, 1.0), 4.0 * ex * ey),
        ((0.0, 0.0, -1.0), 4.0 * ex * ey),
    ]

    actor_tf = carla.Transform(
        snapshot_location(actor_snapshot),
        carla.Rotation(
            pitch=float(actor_rot.get("pitch", 0.0)),
            yaw=float(actor_rot.get("yaw", 0.0)),
            roll=float(actor_rot.get("roll", 0.0)),
        ),
    )
    center_world = actor_tf.transform(
        carla.Location(
            x=float(bbox_loc.get("x", 0.0)),
            y=float(bbox_loc.get("y", 0.0)),
            z=float(bbox_loc.get("z", 0.0)),
        )
    )
    vx = sensor_location.x - center_world.x
    vy = sensor_location.y - center_world.y
    vz = sensor_location.z - center_world.z
    vl = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vl < 1e-6:
        return ""
    ux, uy, uz = vx / vl, vy / vl, vz / vl

    bbox_rot_dict = bbox.get("rotation")
    if bbox_rot_dict:
        bbox_rotation = carla.Rotation(
            pitch=float(bbox_rot_dict.get("pitch", 0.0)),
            yaw=float(bbox_rot_dict.get("yaw", 0.0)),
            roll=float(bbox_rot_dict.get("roll", 0.0)),
        )
    else:
        bbox_rotation = carla.Rotation()
    bbox_tf = carla.Transform(carla.Location(), bbox_rotation)
    world_tf = carla.Transform(carla.Location(), actor_tf.rotation)

    projected = 0.0
    for (lx, ly, lz), area in face_specs:
        n_bbox = bbox_tf.transform(carla.Location(x=lx, y=ly, z=lz))
        n_world = world_tf.transform(n_bbox)
        nx, ny, nz = n_world.x, n_world.y, n_world.z
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1e-9:
            continue
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        dot = nx * ux + ny * uy + nz * uz
        if dot > 0:
            projected += area * dot

    return f"{projected:.6f}"


def vehicle_rcs_proxy_projected_area_m2(vehicle_snapshot, sensor_location):
    return actor_rcs_proxy_projected_area_m2(vehicle_snapshot, sensor_location)


def _match_margin(world, hit_location, actor, inflation_m, limit_m):
    """(margin, limit) for one actor with the class-specific tolerance.
    Pedestrians: distance to the TRUE box, limited to ped_match_max_dist_m().
    Vehicles: distance to the box inflated by inflation_m, limited to limit_m."""
    if _is_pedestrian(actor):
        return (actor_bbox_margin_m(world, hit_location, actor, inflation_m=0.0),
                min(limit_m, ped_match_max_dist_m()))
    return actor_bbox_margin_m(world, hit_location, actor, inflation_m=inflation_m), limit_m


def match_detection_to_actor(
    hit_location,
    candidate_actors,
    world=None,
    *,
    max_margin_m=RADAR_HIT_MATCH_MAX_MARGIN_M,
    extent_inflation_m=BBOX_MATCH_EXTENT_INFLATION_M,
):
    """
    Pick the actor with the smallest OBB margin to hit_location within max_margin_m.
    Uses logged bbox when world is None (offline labeling).
    """
    if not candidate_actors:
        return None, None

    best_actor = None
    best_margin = None
    best_center_d = None
    for actor in candidate_actors:
        margin, limit = _match_margin(
            world, hit_location, actor, extent_inflation_m, max_margin_m
        )
        if margin is None:
            continue
        center_d = hit_location.distance(actor["location"])
        if margin > limit:
            continue
        if (
            best_margin is None
            or margin < best_margin
            or (margin == best_margin and (best_center_d is None or center_d < best_center_d))
        ):
            best_margin = margin
            best_center_d = center_d
            best_actor = actor

    if best_actor is None:
        return None, None
    return best_actor, best_margin


def nearest_actor_bbox_margin_m(
    hit_location,
    candidate_actors,
    world=None,
    *,
    extent_inflation_m=BBOX_MATCH_EXTENT_INFLATION_M,
):
    """Smallest OBB margin among candidates (no accept threshold). For labeling failure diagnostics."""
    if not candidate_actors:
        return None
    best_margin = None
    for actor in candidate_actors:
        margin = actor_bbox_margin_m(
            world, hit_location, actor, inflation_m=extent_inflation_m
        )
        if margin is None:
            continue
        if best_margin is None or margin < best_margin:
            best_margin = margin
    return best_margin


def match_radar_detection_to_actor(
    sensor_transform,
    detection,
    candidate_actors,
    world=None,
    *,
    max_margin_m=RADAR_HIT_MATCH_MAX_MARGIN_M,
    single_candidate_max_margin_m=RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M,
    extent_inflation_m=BBOX_MATCH_EXTENT_INFLATION_M,
    use_legacy_hit_fallback=False,
):
    """
    Match a radar return to an actor: primary hit, legacy hit, then single-target fallbacks.
    Uses logged bbox when world is None (offline labeling).
    """
    if not candidate_actors:
        return None, None

    hit_loc = radar_detection_world_location(sensor_transform, detection)
    ma, margin = match_detection_to_actor(
        hit_loc,
        candidate_actors,
        world,
        max_margin_m=max_margin_m,
        extent_inflation_m=extent_inflation_m,
    )
    if ma is not None:
        return ma, margin

    if use_legacy_hit_fallback:
        legacy_hit = radar_detection_world_location_legacy(sensor_transform, detection)
        ma, margin = match_detection_to_actor(
            legacy_hit,
            candidate_actors,
            world,
            max_margin_m=max_margin_m,
            extent_inflation_m=extent_inflation_m,
        )
        if ma is not None:
            return ma, margin

    gated = _actors_in_depth_azimuth_gate(world, sensor_transform, detection, candidate_actors)
    pool = gated if gated else candidate_actors

    if len(pool) == 1:
        actor = pool[0]
        margin, limit = _match_margin(
            world, hit_loc, actor, extent_inflation_m, single_candidate_max_margin_m
        )
        if margin is not None and margin <= limit:
            return actor, margin

    best_actor = None
    best_margin = None
    best_center_d = None
    for actor in pool:
        margin, limit = _match_margin(
            world, hit_loc, actor, extent_inflation_m, single_candidate_max_margin_m
        )
        if margin is None:
            continue
        center_d = hit_loc.distance(actor["location"])
        if margin > limit:
            continue
        if (
            best_margin is None
            or margin < best_margin
            or (margin == best_margin and (best_center_d is None or center_d < best_center_d))
        ):
            best_margin = margin
            best_center_d = center_d
            best_actor = actor

    if best_actor is None and len(pool) == 1:
        actor = pool[0]
        margin, limit = _match_margin(
            world, hit_loc, actor, extent_inflation_m, single_candidate_max_margin_m
        )
        if margin is not None and margin <= limit:
            return actor, margin

    if best_actor is None:
        return None, None
    return best_actor, best_margin


def match_detection_to_vehicle(hit_location, candidate_vehicles, world=None, **kwargs):
    return match_detection_to_actor(hit_location, candidate_vehicles, world, **kwargs)


def get_nearby_actors_in_fov(sensor_transform, actor_snapshots, max_distance, horizontal_fov_deg):
    """Vehicles and pedestrians within camera horizontal FOV and range."""
    in_fov = []
    for actor in actor_snapshots:
        actor_location = actor["location"]
        if not actor_snapshot_in_sensor_fov(
            sensor_transform, actor_location, max_distance, horizontal_fov_deg
        ):
            continue
        sensor_location = sensor_transform.location
        distance = math.hypot(
            actor_location.x - sensor_location.x,
            actor_location.y - sensor_location.y,
        )
        in_fov.append(
            {
                "id": actor["id"],
                "kind": actor["kind"],
                "type_id": actor["type_id"],
                "class_label": actor["class_label"],
                "location": actor_location,
                "distance": distance,
            }
        )

    in_fov.sort(key=lambda item: item["distance"])
    return in_fov


def get_nearby_vehicles_in_fov(sensor_transform, vehicles, max_distance, horizontal_fov_deg):
    return get_nearby_actors_in_fov(sensor_transform, vehicles, max_distance, horizontal_fov_deg)


def evaluate_radar_detection_label(
    world,
    sensor_transform,
    detection,
    actors,
    *,
    range_m,
    hfov_deg,
    labelable_min_speed_mps=RADAR_LABELABLE_MIN_SPEED_MPS,
    hit_match_max_margin_m: float | None = None,
    single_candidate_max_margin_m: float | None = None,
    compare_legacy=False,
):
    """
    Shared radar labeling path (TestRadarLabeling + CaptureRadarCameraData).

    Scores returns with an actor in the detection beam or |velocity| above threshold.
    Matching uses beam/depth candidates, primary + legacy hit, and single-target fallbacks.

    When ``hit_match_max_margin_m`` / ``single_candidate_max_margin_m`` are not
    supplied, the env-overridable defaults (``DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M`` /
    ``DATASET_RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M``) are used.
    """
    velocity_mps = float(detection.velocity)
    candidate_hit_m = radar_candidate_hit_max_bbox_margin_m()
    if hit_match_max_margin_m is None:
        hit_match_max_margin_m = radar_hit_match_max_margin_m_from_env()
    if single_candidate_max_margin_m is None:
        single_candidate_max_margin_m = radar_single_candidate_max_margin_m_from_env()
    # CARLA rays reach past the nominal range on oblique paths (see
    # radar_effective_max_depth_m); gate on the true reach or far-lane hits at
    # 35-55 m never get a candidate (655 of them in capture 20260917_125123).
    effective_range_m = max(
        float(range_m),
        radar_effective_max_depth_m(range_m, hfov_deg, max(radar_vertical_fov_deg_from_env(), 60.0)),
    )
    hit_loc = radar_detection_world_location(sensor_transform, detection)
    match_candidates = actor_snapshots_for_radar_detection(
        sensor_transform,
        detection,
        actors,
        world,
        max_range_m=effective_range_m,
        horizontal_fov_deg=hfov_deg,
        hit_max_bbox_margin_m=candidate_hit_m,
    )
    had_candidates = bool(match_candidates)
    scored = should_score_radar_return(
        velocity_mps,
        had_candidates,
        min_speed_mps=labelable_min_speed_mps,
    )

    matched = False
    legacy_matched = None
    actor_id = None
    actor_kind = ""
    actor_type_id = ""
    actor_class = ""
    actor_snapshot = None
    match_bbox_margin_m = None
    nearest_bbox_margin_m = None
    # Nearest-OBB margin BEFORE matching — survives the matched-row overwrite below
    # so the QA report can plot the true (threshold-independent) margin distribution.
    uncensored_nearest_bbox_margin_m = None

    if had_candidates:
        uncensored_nearest_bbox_margin_m = nearest_actor_bbox_margin_m(
            hit_loc, match_candidates, world
        )
        nearest_bbox_margin_m = uncensored_nearest_bbox_margin_m
        ma, margin = match_radar_detection_to_actor(
            sensor_transform,
            detection,
            match_candidates,
            world,
            max_margin_m=hit_match_max_margin_m,
            single_candidate_max_margin_m=single_candidate_max_margin_m,
        )
        if ma is not None:
            matched = True
            actor_id = ma["id"]
            actor_kind = ma["kind"]
            actor_type_id = ma["type_id"]
            actor_class = ma["class_label"]
            actor_snapshot = ma
            match_bbox_margin_m = margin
            nearest_bbox_margin_m = margin

        if compare_legacy:
            legacy_hit = radar_detection_world_location_legacy(sensor_transform, detection)
            legacy_ma, _ = match_detection_to_actor(
                legacy_hit, match_candidates, world
            )
            legacy_matched = legacy_ma is not None

    if matched:
        return_class = actor_kind or "vehicle"
    elif hit_loc.z <= ROAD_SURFACE_MAX_Z_M:
        return_class = "road"
    elif hit_loc.z >= STRUCTURE_MIN_Z_M:
        return_class = "structure"
    else:
        return_class = "unassigned"

    return {
        "scored": scored,
        "had_candidates": had_candidates,
        "matched": matched,
        "return_class": return_class,
        "hit_world": hit_loc,
        "legacy_matched": legacy_matched,
        "actor_id": actor_id,
        "actor_kind": actor_kind,
        "actor_type_id": actor_type_id,
        "actor_class": actor_class,
        "actor_snapshot": actor_snapshot,
        "match_bbox_margin_m": match_bbox_margin_m,
        "nearest_bbox_margin_m": nearest_bbox_margin_m,
        "uncensored_nearest_bbox_margin_m": uncensored_nearest_bbox_margin_m,
        "velocity_mps": velocity_mps,
    }


def labelable_min_speed_from_env() -> float:
    raw = os.environ.get("DATASET_LABELABLE_MIN_SPEED_MPS", "").strip()
    if not raw:
        return RADAR_LABELABLE_MIN_SPEED_MPS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return RADAR_LABELABLE_MIN_SPEED_MPS


def label_after_capture_from_env() -> bool:
    raw = os.environ.get("DATASET_LABEL_RADAR_AFTER_CAPTURE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def postprocess_after_capture_from_env() -> bool:
    """Auto-run PostProcessDataset (ped Doppler fix + rcs_dBsm) after labeling.

    Default on: without it pedestrian Doppler stays 0 and rcs_dBsm is missing,
    which is the wrong feature set for training and inflates shortcut probes.
    """
    raw = os.environ.get("DATASET_POSTPROCESS_AFTER_CAPTURE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def postprocess_seed_from_env() -> int:
    """Seed for PostProcessDataset's RCS/micro-Doppler noise (recorded in run_meta)."""
    raw = os.environ.get("DATASET_POSTPROCESS_SEED", os.environ.get("DATASET_SEED", "0")).strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def capture_duration_s_from_env() -> float | None:
    """Auto-stop recording after N seconds (wall clock). None = run until Enter.

    Lets the campaign orchestrator run fixed-length unattended captures via
    DATASET_CAPTURE_DURATION_S. Values <= 0 are treated as unlimited.
    """
    raw = os.environ.get("DATASET_CAPTURE_DURATION_S", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None


def _git_commit() -> str:
    """Short git commit of the working tree (best-effort; '' if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def write_run_meta(run_dir, world, *, radar_count, camera_count, phase="start", extra=None):
    """Write/refresh run_meta.json — the single source of truth for a capture.

    Records the map, sensor counts, master seed, every DATASET_* env var actually
    set, git commit, and timestamps. This is what makes leakage-free actor/site-level
    train/eval splits and exact reruns possible. Called once at start and again at
    stop (phase='stop') to stamp the end time. Best-effort: never raises.
    """
    try:
        meta_path = Path(os.path.normpath(run_dir)) / "run_meta.json"
        dataset_env = {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith("DATASET_") or k in ("AUTOPILOT", "MAP")
        }
        try:
            map_name = world.get_map().name
        except Exception:  # noqa: BLE001
            map_name = ""
        meta = {
            "phase": phase,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_dir": os.path.normpath(run_dir),
            "map": map_name,
            "radar_count": radar_count,
            "camera_count": camera_count,
            "seed": os.environ.get("DATASET_SEED", ""),
            "site_id": os.environ.get("DATASET_SITE_ID", ""),
            "rig_pose": {
                "anchor_x": os.environ.get("DATASET_RIG_ANCHOR_X", ""),
                "anchor_y": os.environ.get("DATASET_RIG_ANCHOR_Y", ""),
                "height_m": os.environ.get("DATASET_RIG_HEIGHT_M", ""),
                "radars_south": os.environ.get("DATASET_RADARS_SOUTH", ""),
                "radars_north": os.environ.get("DATASET_RADARS_NORTH", ""),
                "yaw_deg": os.environ.get("DATASET_RIG_YAW_DEG", ""),
                "pitch_deg": os.environ.get("DATASET_RADAR_PITCH_DEG", ""),
            },
            "postprocess_seed": postprocess_seed_from_env(),
            "git_commit": _git_commit(),
            "dataset_env": dataset_env,
        }
        if extra:
            meta.update(extra)
        # Merge start fields if we're stamping the stop phase over an existing file.
        if phase != "start" and meta_path.is_file():
            try:
                prior = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = {**prior, **meta, "started_at": prior.get("timestamp", "")}
            except Exception:  # noqa: BLE001
                pass
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not write run_meta.json: {exc}", file=sys.stderr)


def write_capture_labeling_report(
    collector,
    run_dir: str,
    *,
    labelable_min_speed_mps: float,
) -> None:
    """Write TestRadarLabeling-style QA plots/CSVs into the capture folder."""
    out = Path(os.path.normpath(run_dir)) / "radar_labeling_qa"
    report_kwargs = {
        "min_match_rate": 0.05,
        "expected_radar_labels": EXPECTED_RADAR_LABELS,
        "proximity_m": RADAR_VEHICLE_PROXIMITY_M,
        "hit_match_m": RADAR_HIT_MATCH_MAX_DISTANCE_M,
        "hit_match_max_margin_m": RADAR_HIT_MATCH_MAX_MARGIN_M,
        "bbox_extent_inflation_m": BBOX_MATCH_EXTENT_INFLATION_M,
        "labelable_min_speed_mps": labelable_min_speed_mps,
        "candidate_max_range_m": RADAR_MAX_RANGE_M,
        "candidate_horizontal_fov_deg": RADAR_HORIZONTAL_FOV_DEG,
        "candidate_depth_margin_m": RADAR_CANDIDATE_DEPTH_MARGIN_M,
        "candidate_azimuth_margin_deg": RADAR_CANDIDATE_AZIMUTH_MARGIN_DEG,
        "candidate_hit_max_bbox_margin_m": radar_candidate_hit_max_bbox_margin_m(),
        "single_candidate_max_margin_m": RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M,
    }
    try:
        write_report(collector, out, **report_kwargs)
        print(f"Radar labeling QA report: {out.resolve()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Labeling QA report failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()


def _run_dataset_extrinsic_exports(world, run_dir: str) -> None:
    """
    After CSVs are closed, write camera_extrinsics.* and sensor_extrinsics.* into run_dir.
    Runs in this process (same Python + CARLA as the recorder) so a subprocess is not
    used — that was failing silently when a different python.exe could not import carla.
    """
    out = Path(os.path.normpath(run_dir))
    print("Exporting camera + radar extrinsics into the capture folder...", flush=True)
    try:
        ok_c = write_camera_extrinsics_to_dataset_dir(world, out)
        ok_r = write_radar_extrinsics_live_to_dataset_dir(world, out)
    except Exception as e:
        print(f"Extrinsic export failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return
    if ok_c and ok_r:
        print(
            f"Done. Extrinsic files are in: {out}",
            flush=True,
        )
    else:
        print(
            "Extrinsic export incomplete (see messages above). "
            "Keep CARLA and RadarCameraSetup.py running when you stop recording with Enter.",
            file=sys.stderr,
            flush=True,
        )


def process_radar_measurement_for_capture(
    measurement,
    sensor_id,
    sensor_label,
    radar_actor,
    *,
    world,
    actor_cache: RadarActorSnapshotCache,
    labelable_min_speed_mps: float,
    radar_writer,
    labeling_collector: LabelingStatsCollector,
    lock: threading.Lock,
    counts: dict,
) -> None:
    sensor_transform = measurement.transform
    loc = sensor_transform.location
    rot = sensor_transform.rotation
    actors = actor_cache.get(int(measurement.frame))
    range_m, hfov_deg = radar_sensor_limits(radar_actor)

    rows = []
    qa_records: list[DetectionRecord] = []
    for idx, detection in enumerate(measurement):
        label = evaluate_radar_detection_label(
            world,
            sensor_transform,
            detection,
            actors,
            range_m=range_m,
            hfov_deg=hfov_deg,
            labelable_min_speed_mps=labelable_min_speed_mps,
        )

        matched_actor_id = ""
        matched_actor_kind = ""
        matched_actor_type_id = ""
        matched_actor_class = ""
        matched_actor_bbox_margin = ""
        matched_vehicle_id = ""
        matched_vehicle_type_id = ""
        matched_vehicle_class = ""
        matched_vehicle_distance = ""
        nearest_margin_str = ""
        if label["nearest_bbox_margin_m"] is not None:
            nearest_margin_str = f"{label['nearest_bbox_margin_m']:.6f}"

        if label["matched"] and label["actor_id"] is not None:
            matched_actor_id = str(label["actor_id"])
            matched_actor_kind = label["actor_kind"]
            matched_actor_type_id = label["actor_type_id"]
            matched_actor_class = label["actor_class"]
            if label["match_bbox_margin_m"] is not None:
                matched_actor_bbox_margin = f"{label['match_bbox_margin_m']:.6f}"
            if label["actor_kind"] == "vehicle":
                matched_vehicle_id = matched_actor_id
                matched_vehicle_type_id = matched_actor_type_id
                matched_vehicle_class = matched_actor_class
                matched_vehicle_distance = matched_actor_bbox_margin

        rcs_proxy_m2 = ""
        if matched_actor_id:
            rcs_proxy_m2 = actor_rcs_proxy_projected_area_m2(label["actor_snapshot"], loc)

        rows.append(
            [
                sensor_id,
                sensor_label,
                measurement.frame,
                f"{measurement.timestamp:.6f}",
                idx,
                f"{detection.depth:.6f}",
                f"{detection.azimuth:.6f}",
                f"{detection.altitude:.6f}",
                f"{detection.velocity:.6f}",
                f"{loc.x:.6f}",
                f"{loc.y:.6f}",
                f"{loc.z:.6f}",
                f"{rot.pitch:.6f}",
                f"{rot.yaw:.6f}",
                f"{rot.roll:.6f}",
                matched_actor_id,
                matched_actor_kind,
                matched_actor_type_id,
                matched_actor_class,
                matched_actor_bbox_margin,
                matched_vehicle_id,
                matched_vehicle_type_id,
                matched_vehicle_class,
                matched_vehicle_distance,
                rcs_proxy_m2,
                "1" if label["had_candidates"] else "0",
                "1" if label["scored"] else "0",
                nearest_margin_str,
            ]
        )
        if label["scored"]:
            qa_records.append(
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
                )
            )

    with lock:
        for row in rows:
            radar_writer.writerow(row)
        counts["radar_messages"] += 1
        counts["radar_detections"] += len(rows)
        counts["radar_scored"] += len(qa_records)
        counts["radar_matched"] += sum(1 for r in qa_records if r.matched)
        labeling_collector.record_message(raw_returns=len(measurement))
        for rec in qa_records:
            labeling_collector.record_detection(rec)


def process_radar_measurement_fast(
    measurement,
    sensor_id,
    sensor_label,
    radar_actor,
    *,
    radar_writer,
    lock: threading.Lock,
    counts: dict,
) -> None:
    """Write all CARLA returns to CSV without OBB actor matching (dataset capture throughput).

    Actor frames are logged independently by :class:`TickActorSnapshotter`, so this
    hot path never issues a CARLA RPC and can keep up with high-rate radar streams
    even when draining the queue after Ctrl+C.
    """
    del radar_actor
    sensor_transform = measurement.transform
    loc = sensor_transform.location
    rot = sensor_transform.rotation
    rows = []
    for idx, detection in enumerate(measurement):
        rows.append(
            [
                sensor_id,
                sensor_label,
                measurement.frame,
                f"{measurement.timestamp:.6f}",
                idx,
                f"{detection.depth:.6f}",
                f"{detection.azimuth:.6f}",
                f"{detection.altitude:.6f}",
                f"{detection.velocity:.6f}",
                f"{loc.x:.6f}",
                f"{loc.y:.6f}",
                f"{loc.z:.6f}",
                f"{rot.pitch:.6f}",
                f"{rot.yaw:.6f}",
                f"{rot.roll:.6f}",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "0",
                "1",
                "",
            ]
        )
    with lock:
        for row in rows:
            radar_writer.writerow(row)
        counts["radar_messages"] += 1
        counts["radar_detections"] += len(rows)
        counts["radar_scored"] += len(rows)


class RadarQueueConsumer:
    """Write radar CSV off the thread that owns ``world.tick()``.

    Capture is the CARLA clock owner. The live loop used to empty the whole
    radar queue after every tick before calling ``world.tick()`` again. CSV
    formatting for a full radar set easily exceeds ``fixed_delta_seconds``
    (0.05 s at 20 Hz); that extra wall time delays the next tick, so SUMO
    (``world.wait_for_tick()``) and every sensor slow together.

    Listen callbacks stay O(1) enqueue. This thread is the sole live consumer.
    A slow drain can still grow the bounded per-radar deque (and eventually
    drop), but it no longer stretches the simulation clock.
    """

    def __init__(self, radar_queue, process_fn, *, batch_size: int = 8) -> None:
        self._queue = radar_queue
        self._process = process_fn
        self._batch_size = max(1, int(batch_size))
        self._stop = threading.Event()
        self._write_errors = 0
        self.processed = 0
        self.last_batch_s = 0.0
        self._thread = threading.Thread(
            target=self._loop, name="radar-queue-consumer", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            # One tick's worth (one measurement per listening radar), not
            # drain_all: keeps last_batch_s meaningful and yields the GIL
            # between measurements so the tick owner can call world.tick()
            # even when CSV formatting is slow.
            batch = self._queue.drain(max_items=self._batch_size)
            if not batch:
                time.sleep(0.001)
                continue
            t0 = time.monotonic()
            for item in batch:
                try:
                    self._process(item)
                except Exception as exc:  # noqa: BLE001 - never kill the consumer
                    self._write_errors += 1
                    if self._write_errors <= 3:
                        print(
                            f"[capture] radar consumer: skipping a measurement "
                            f"due to error: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                self.processed += 1
                time.sleep(0)
                if self._stop.is_set():
                    break
            self.last_batch_s = time.monotonic() - t0

    def stop(self, timeout_s: float = 30.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            print(
                f"[capture] radar consumer still running after {timeout_s:.0f}s; "
                "continuing shutdown drain on the main thread.",
                file=sys.stderr,
                flush=True,
            )


def camera_encoder_threads_from_env() -> int:
    raw = os.environ.get("DATASET_CAMERA_ENCODER_THREADS", "").strip()
    try:
        return max(1, min(int(raw), 8)) if raw else 2
    except ValueError:
        return 2


def camera_max_backlog_from_env() -> int:
    """Max camera frames allowed to wait for PNG encoding before NEW frames are
    dropped (each 800x600 frame holds ~2 MB). 0 = never drop (default)."""
    raw = os.environ.get("DATASET_CAMERA_MAX_BACKLOG", "").strip()
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


def _encode_carla_image_png(image, path: str) -> None:
    """Write a carla.Image as PNG. Uses numpy + Pillow (compress_level=1, several
    times faster than CARLA's save_to_disk) when available; falls back otherwise."""
    try:
        import numpy as np  # noqa: WPS433
        from PIL import Image as PILImage  # noqa: WPS433
    except ImportError:
        image.save_to_disk(path)
        return
    buf = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = buf.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    PILImage.fromarray(np.ascontiguousarray(arr)).save(path, format="PNG", compress_level=1)


class CameraFrameWriter:
    """Two-stage camera pipeline: metadata now, PNG encoding in the background.

    Why two stages (capture 20260917_125123 / _162136 wrote exactly 41 camera
    frames then went silent for the rest of the run):

    * The listen callback must stay O(1), so it only enqueues the carla.Image.
    * The OLD single writer thread did the actor lookup AND the PNG encode per
      frame. save_to_disk is slower than one 20 Hz tick on a laptop, so the
      queue fell behind; once it lagged more than TickActorSnapshotter's
      in-memory window (600 frames = 30 s) every lookup returned an EMPTY actor
      list, and the old writer silently skipped frames with no nearby actor.
      From then on nothing was written at all. A faster machine never lags,
      which is why a colleague could not reproduce it.

    Now stage 1 (``_meta_loop``) only does the cheap snapshot lookup + CSV row,
    so it keeps up with the tick and always sees a fresh snapshot; stage 2 (a
    small thread pool) encodes PNGs and may fall behind without affecting
    metadata. Every frame is saved, with or without actors nearby; a frame whose
    snapshot really is missing gets empty actor fields and a counter bump.
    """

    def __init__(
        self,
        snapshotter: TickActorSnapshotter,
        camera_csv_writer,
        camera_file,
        lock: threading.Lock,
        counts: dict,
        segment_fn=None,
    ) -> None:
        import concurrent.futures  # noqa: WPS433

        self._segment_fn = segment_fn
        self._snapshotter = snapshotter
        self._csv_writer = camera_csv_writer
        self._camera_file = camera_file
        self._lock = lock
        self._counts = counts
        self._queue: "queue.Queue" = queue.Queue()
        self._closed = False
        self.dropped = 0
        self.enqueued = 0
        self.missing_snapshot = 0
        self._write_errors = 0
        self._encode_errors = 0
        self._encode_pending = 0
        self._encode_lock = threading.Lock()
        self._max_backlog = camera_max_backlog_from_env()
        self._last_backlog_warn = 0.0
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=camera_encoder_threads_from_env(), thread_name_prefix="camera-png"
        )
        self._thread = threading.Thread(
            target=self._meta_loop, name="camera-frame-meta", daemon=True
        )
        self._thread.start()

    def enqueue(self, image, sid, slabel, folder, sensor_hfov) -> bool:
        if self._closed:
            self.dropped += 1
            return False
        if self._max_backlog and self.encode_pending() >= self._max_backlog:
            self.dropped += 1
            return False
        self._queue.put((image, sid, slabel, folder, sensor_hfov))
        self.enqueued += 1
        return True

    def pending(self) -> int:
        return self._queue.qsize() + self.encode_pending()

    def encode_pending(self) -> int:
        with self._encode_lock:
            return self._encode_pending

    def _actors_for_frame(self, frame_id: int) -> list | None:
        if self._snapshotter.has(frame_id):
            return self._snapshotter.get(frame_id)
        # Camera stream can beat world.on_tick by a few ms. Wait up to one
        # 20 Hz tick on THIS thread only, never on the listen callback.
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            if self._snapshotter.has(frame_id):
                return self._snapshotter.get(frame_id)
            time.sleep(0.001)
        return None

    def _meta_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._process(*item)
            except Exception as exc:  # noqa: BLE001 - never kill the writer
                self._write_errors += 1
                if self._write_errors <= 3:
                    print(
                        f"[capture] camera writer error: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            finally:
                self._queue.task_done()

    def _submit_encode(self, image, image_path: str) -> None:
        with self._encode_lock:
            self._encode_pending += 1
            backlog = self._encode_pending
        if backlog and backlog % 200 == 0 and time.monotonic() - self._last_backlog_warn > 10.0:
            self._last_backlog_warn = time.monotonic()
            print(
                f"[capture] camera PNG backlog {backlog} frames (~{2 * backlog} MB held); "
                "encoding is slower than the tick. Raise DATASET_CAMERA_ENCODER_THREADS, "
                "lower the camera resolution, or set DATASET_CAMERA_MAX_BACKLOG to drop.",
                file=sys.stderr,
                flush=True,
            )
        self._pool.submit(self._encode, image, image_path)

    def _encode(self, image, image_path: str) -> None:
        try:
            _encode_carla_image_png(image, image_path)
        except Exception as exc:  # noqa: BLE001
            self._encode_errors += 1
            if self._encode_errors <= 3:
                print(f"[capture] camera PNG encode error: {exc}", file=sys.stderr, flush=True)
        finally:
            with self._encode_lock:
                self._encode_pending -= 1

    def _process(self, image, sid, slabel, folder, sensor_hfov) -> None:
        actors = self._actors_for_frame(int(image.frame))
        if actors is None:
            self.missing_snapshot += 1
            if self.missing_snapshot <= 3:
                print(
                    f"[capture] camera frame {image.frame}: no actor snapshot for this "
                    "frame (saving the image with empty actor fields).",
                    file=sys.stderr,
                    flush=True,
                )
            actors = []
        nearby_actors = get_nearby_actors_in_fov(
            image.transform,
            actors,
            NEARBY_DISTANCE_M,
            sensor_hfov,
        )

        image_name = f"frame_{image.frame:08d}.png"
        image_path = os.path.join(folder, image_name)
        self._submit_encode(image, image_path)
        seg = self._segment_fn(int(image.frame)) if self._segment_fn else None
        seg_id = "" if seg is None else seg

        nearest = nearby_actors[0] if nearby_actors else None
        nearby_ids = ";".join(str(a["id"]) for a in nearby_actors)
        nearby_kinds = ";".join(a["kind"] for a in nearby_actors)
        nearby_classes = ";".join(a["class_label"] for a in nearby_actors)

        nearby_vehicles = [a for a in nearby_actors if a["kind"] == "vehicle"]
        nearby_peds = [a for a in nearby_actors if a["kind"] == "pedestrian"]
        nearest_vehicle = nearby_vehicles[0] if nearby_vehicles else None
        nearest_ped = nearby_peds[0] if nearby_peds else None

        def _actor_fields(actor):
            if actor is None:
                return ("", "", "", "")
            return (
                actor["id"],
                actor["type_id"],
                actor["class_label"],
                f"{actor['distance']:.6f}",
            )

        n_id, n_type, n_class, n_dist = _actor_fields(nearest)
        n_kind = nearest["kind"] if nearest is not None else ""
        nv_id, nv_type, nv_class, nv_dist = _actor_fields(nearest_vehicle)
        np_id, np_type, np_class, np_dist = _actor_fields(nearest_ped)
        veh_ids = ";".join(str(v["id"]) for v in nearby_vehicles)
        veh_classes = ";".join(v["class_label"] for v in nearby_vehicles)
        ped_ids = ";".join(str(p["id"]) for p in nearby_peds)
        ped_classes = ";".join(p["class_label"] for p in nearby_peds)

        with self._lock:
            if self._camera_file.closed:
                return
            self._csv_writer.writerow(
                [
                    sid,
                    slabel,
                    image.frame,
                    f"{image.timestamp:.6f}",
                    image.width,
                    image.height,
                    image_path,
                    n_id,
                    n_kind,
                    n_type,
                    n_class,
                    n_dist,
                    nearby_ids,
                    nearby_kinds,
                    nearby_classes,
                    nv_id,
                    nv_type,
                    nv_class,
                    nv_dist,
                    veh_ids,
                    veh_classes,
                    np_id,
                    np_type,
                    np_class,
                    np_dist,
                    ped_ids,
                    ped_classes,
                    seg_id,
                ]
            )
            self._counts["camera_frames"] += 1

    def close(self, timeout_s: float = 60.0) -> None:
        if self._closed:
            return
        self._closed = True
        pending = self._queue.qsize()
        if pending:
            print(
                f"[capture] draining {pending} queued camera frame(s) ...",
                flush=True,
            )
        self._queue.put(None)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            print(
                f"[capture] camera writer still running after {timeout_s:.0f}s; "
                "continuing shutdown.",
                file=sys.stderr,
                flush=True,
            )
        backlog = self.encode_pending()
        if backlog:
            print(f"[capture] encoding {backlog} remaining camera PNG(s) ...", flush=True)
        self._pool.shutdown(wait=True)
        if self.missing_snapshot:
            print(
                f"[capture] camera frames saved without actor metadata: {self.missing_snapshot}",
                flush=True,
            )


def _install_stop_signal_handlers():
    """Force SIGINT and SIGTERM to raise KeyboardInterrupt so this capture always
    stops gracefully (drain → extrinsics → offline labeling in main()'s finally).

    Critical for background launches: when Start.py is started in a backgrounded
    shell pipeline, bash sets SIGINT to SIG_IGN and that disposition is INHERITED by
    this child. Without re-installing a handler, Start.py's ``send_signal(SIGINT)``
    (and any ``kill -INT``) is silently ignored, the capture records forever, and
    Start.py blocks in ``capture_proc.wait()`` — the recurring teardown hang. We also
    map SIGTERM to the same path so a plain ``kill`` stops us cleanly too."""
    def _raise_kbd(signum, frame):
        raise KeyboardInterrupt
    # default_int_handler is CPython's built-in that raises KeyboardInterrupt; setting
    # it explicitly overrides any inherited SIG_IGN.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        signal.signal(signal.SIGTERM, _raise_kbd)
    except (ValueError, OSError):
        # Not in the main thread (shouldn't happen for __main__) — skip rather than crash.
        pass


def _run_offline_step(name: str, cmd: list) -> bool:
    """Run a CARLA-free post-capture step as a child process; True on exit 0."""
    print(f"[capture] {name}: starting as a separate process ...", flush=True)
    try:
        code = subprocess.run(cmd, cwd=str(capture_dir().parent)).returncode
    except Exception as exc:  # noqa: BLE001
        print(f"[capture] {name} could not start: {exc}", file=sys.stderr, flush=True)
        return False
    if code != 0:
        print(f"[capture] {name} exited with code {code}. Re-run it manually with:\n"
              f"    {' '.join(str(c) for c in cmd)}", file=sys.stderr, flush=True)
        return False
    return True


def main():
    _install_stop_signal_handlers()
    client, world = get_world()

    print(f"Waiting up to {SENSOR_WAIT_TIMEOUT_S:.1f}s for tagged radar/camera sensors...")
    radar_sensors, camera_sensors = wait_for_sensors(world, SENSOR_WAIT_TIMEOUT_S)

    if not radar_sensors and not camera_sensors:
        print("No tagged dataset sensors found in the world.")
        print("Run RadarCameraSetup.py first, then run this script again.")
        return

    capture_parent = os.environ.get("DATASET_CAPTURE_BASE_DIR", "").strip()
    if capture_parent:
        capture_parent = os.path.normpath(capture_parent)
    else:
        capture_parent = str(data_output_dir())
    run_dir, camera_dir, radar_csv, camera_csv = make_output_paths(capture_parent)

    pointer_path = capture_dir() / ".last_dataset_capture_dir"
    try:
        with open(pointer_path, "w", encoding="utf-8") as pointer_f:
            pointer_f.write(os.path.normpath(run_dir) + "\n")
    except OSError as e:
        print(f"Warning: could not write {pointer_path}: {e}", file=sys.stderr)

    write_run_meta(
        run_dir,
        world,
        radar_count=len(radar_sensors),
        camera_count=len(camera_sensors),
        phase="start",
    )

    radar_file, radar_writer = setup_radar_writer(radar_csv)
    camera_file, camera_writer = setup_camera_writer(camera_csv)

    # Campaign mode (fusion/campaign.py): record only the frame windows the
    # orchestrator asks for; keep ticking in between. See campaign_control.py.
    campaign_gate = None
    campaign_dir = campaign_control_dir_from_env()
    if campaign_dir is not None:
        campaign_gate = CampaignGate(campaign_dir, run_dir)
        campaign_gate.write_status()
        print(f"[capture] CAMPAIGN MODE: control dir {campaign_dir}; nothing is "
              "recorded until the orchestrator opens a segment window.", flush=True)
    segment_fn = campaign_gate.segment_for_frame if campaign_gate is not None else None

    labelable_min_speed_mps = labelable_min_speed_from_env()
    labeling_collector = LabelingStatsCollector(labelable_min_speed_mps=labelable_min_speed_mps)

    lock = threading.Lock()
    counts = {
        "radar_messages": 0,
        "radar_detections": 0,
        "radar_scored": 0,
        "radar_matched": 0,
        "camera_frames": 0,
    }

    vehicle_count = len(world.get_actors().filter("vehicle.*"))
    pedestrian_count = len(world.get_actors().filter("walker.pedestrian.*"))
    print(f"Recording output directory: {run_dir}")
    print(
        f"World actors: {vehicle_count} vehicles, {pedestrian_count} pedestrians "
        f"(radar labels vehicles + pedestrians via OBB)"
    )
    print(f"Radars found: {len(radar_sensors)}")
    print(f"RGB cameras found: {len(camera_sensors)}")
    if radar_sensors:
        found_radar_labels = {
            sensor_label_from_role_name(r.attributes.get("role_name", ""), DATASET_RADAR_ROLE_PREFIX)
            for r in radar_sensors
        }
        missing_radar_labels = sorted(EXPECTED_RADAR_LABELS - found_radar_labels)
        radar_summary = [
            f"{sensor_label_from_role_name(r.attributes.get('role_name', ''), DATASET_RADAR_ROLE_PREFIX)}:{r.id}"
            for r in radar_sensors
        ]
        radar_summary.sort()
        print(f"Tagged radars (label:actor_id): {', '.join(radar_summary)}")
        if missing_radar_labels:
            print(
                "Warning: Missing expected radars: "
                + ", ".join(missing_radar_labels)
            )
    if camera_sensors:
        camera_summary = [
            f"{sensor_label_from_role_name(c.attributes.get('role_name', ''), DATASET_CAMERA_ROLE_PREFIX)}:{c.id}"
            for c in camera_sensors
        ]
        print(f"Tagged cameras (label:actor_id): {', '.join(camera_summary)}")
        print(
            "Camera capture: listen callback enqueues only "
            "(PNG + FOV CSV on writer thread; no per-actor RPCs)."
        )

    capture_fast = radar_capture_fast_from_env()
    if capture_fast:
        print(
            "Radar capture: FAST mode (all CARLA returns written; actor matching skipped). "
            "Actor frames logged for offline labeling after capture. "
            "Set DATASET_RADAR_CAPTURE_FAST=0 for live OBB labeling (much slower)."
        )
    else:
        print("Radar capture: LIVE labeling mode (OBB match per return — lower throughput).")

    # ── Synchronous-mode setup ────────────────────────────────────────────
    # When DATASET_SYNC_MODE=1, the capture script takes ownership of the world
    # clock and ticks at fixed_delta_seconds. This forces every listening radar
    # to fire on the same world tick (instead of each running its own staggered
    # phase in async mode), enabling instantaneous multi-radar fusion on a
    # single frame_id. Default off so legacy async captures still work unchanged.
    sync_mode = sync_mode_from_env()
    sync_min_period_s = sync_min_period_s_from_env() if sync_mode else 0.0
    if sync_min_period_s > 0:
        print(
            f"[capture] tick wall-clock floor {sync_min_period_s:.3f}s "
            "(DATASET_SYNC_MIN_PERIOD_S); simulation step is unchanged.",
            flush=True,
        )
    original_world_settings = world.get_settings()
    sync_traffic_manager = None
    if sync_mode:
        sync_delta_s = sync_fixed_delta_s_from_env()
        tm_port = traffic_manager_port_from_env()
        try:
            new_settings = carla.WorldSettings(
                synchronous_mode=True,
                fixed_delta_seconds=sync_delta_s,
                substepping=getattr(original_world_settings, "substepping", True),
                max_substep_delta_time=getattr(
                    original_world_settings, "max_substep_delta_time", 0.01
                ),
                max_substeps=getattr(original_world_settings, "max_substeps", 10),
            )
            world.apply_settings(new_settings)
            print(
                f"[capture] SYNC MODE ENABLED: fixed_delta_seconds={sync_delta_s:g}s "
                f"(matches DATASET_RADAR_SENSOR_TICK_S). All radars will co-fire on each tick.",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[capture] Failed to apply sync world settings: {exc}",
                file=sys.stderr,
                flush=True,
            )
            sync_mode = False

        if sync_mode:
            try:
                sync_traffic_manager = client.get_trafficmanager(tm_port)
                sync_traffic_manager.set_synchronous_mode(True)
                print(
                    f"[capture] TrafficManager(port={tm_port}) switched to synchronous mode.",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                # Non-fatal: if no TM is in use (e.g. no autopilot vehicles) this is fine.
                # If TM IS in use elsewhere on a different port, vehicles will appear to
                # freeze — set DATASET_TRAFFIC_MANAGER_PORT to match your spawner.
                print(
                    f"[capture] TrafficManager sync-mode attach failed (port={tm_port}): {exc}. "
                    "If your traffic appears frozen during capture, set "
                    "DATASET_TRAFFIC_MANAGER_PORT to your spawner's TM port.",
                    file=sys.stderr,
                    flush=True,
                )
                sync_traffic_manager = None

    # Eagerly capture actor snapshots on every server tick. Both fast and live
    # capture paths look up actors by frame_id via this in-memory cache, so the
    # post-Ctrl+C drain can finish without issuing any new CARLA RPCs and
    # without truncating actor_frames.jsonl. ``make_fast_tick_snapshot_fn``
    # is mandatory here (not the plain ``get_radar_target_snapshots``): the
    # latter issues ~1 RPC per actor per tick and CARLA silently stops dispatching
    # on_tick once a callback overruns its tick budget — see capture 231410,
    # where the actor log went silent for 7.4 minutes once the actor count
    # reached ~60.
    tick_snapshotter = TickActorSnapshotter(
        world,
        run_dir,
        snapshot_fn=make_fast_tick_snapshot_fn(world),
        max_frames_in_memory=2400,
        segment_fn=segment_fn,  # 2 min at 20 Hz; the camera meta stage never lags this much
    )
    radar_queue = make_radar_capture_buffer()
    camera_frame_writer = CameraFrameWriter(
        tick_snapshotter,
        camera_writer,
        camera_file,
        lock,
        counts,
        segment_fn=segment_fn,
    )

    def process_measurement_item(item) -> None:
        if capture_fast:
            process_radar_measurement_fast(
                *item,
                radar_writer=radar_writer,
                lock=lock,
                counts=counts,
            )
        else:
            process_radar_measurement_for_capture(*item, **process_kwargs)

    def drain_radar_queue(*, budget_s: float | None = None) -> int:
        deadline = None if budget_s is None else time.monotonic() + budget_s
        processed = 0
        last_heartbeat = time.monotonic()
        heartbeat_interval_s = 3.0
        start = last_heartbeat
        announced = False
        while deadline is None or time.monotonic() < deadline:
            batch = radar_queue.drain_all()
            if not batch:
                break
            if not announced:
                pending_hint = len(batch)
                print(
                    f"[capture] draining queued radar measurements "
                    f"(initial batch={pending_hint:,}, "
                    f"budget={'unbounded' if budget_s is None else f'{budget_s:.0f}s'})...",
                    flush=True,
                )
                announced = True
            for item in batch:
                try:
                    process_measurement_item(item)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[capture] drain: skipping a measurement due to error: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                processed += 1
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval_s:
                    print(
                        f"[capture] draining... processed={processed:,} "
                        f"elapsed={now - start:0.1f}s",
                        flush=True,
                    )
                    last_heartbeat = now
        if announced:
            print(
                f"[capture] drain complete: processed={processed:,} "
                f"elapsed={time.monotonic() - start:0.1f}s",
                flush=True,
            )
        return processed

    process_kwargs = dict(
        world=world,
        actor_cache=tick_snapshotter,
        labelable_min_speed_mps=labelable_min_speed_mps,
        radar_writer=radar_writer,
        labeling_collector=labeling_collector,
        lock=lock,
        counts=counts,
    )

    # Live radar CSV must not run on the tick-owner thread. Shutdown still uses
    # drain_radar_queue() on the main thread after this consumer is stopped.
    radar_consumer = RadarQueueConsumer(
        radar_queue,
        process_measurement_item,
        batch_size=max(1, len(radar_sensors)),
    )
    print(
        "[capture] radar CSV consumer running off the tick thread "
        "(queue drain no longer gates world.tick / SUMO).",
        flush=True,
    )

    # Per-sensor liveness tracking for the listen() watchdog. Each entry holds
    # (actor, sensor_label, callback, last_world_frame). Integer reads/writes
    # are GIL-atomic in CPython, so we don't need a lock here — the watchdog only
    # cares about relative staleness vs the latest-seen frame (or world.tick()).
    radar_watchdog_stale_ticks = radar_watchdog_stale_ticks_from_env()
    camera_watchdog_stale_ticks = camera_watchdog_stale_ticks_from_env()
    radar_track: dict[int, dict] = {}
    camera_track: dict[int, dict] = {}
    radar_latest_frame: list[int] = [0]
    camera_latest_frame: list[int] = [0]
    radar_watchdog_resets: list[int] = [0]
    camera_watchdog_resets: list[int] = [0]

    try:
        for radar in radar_sensors:
            sensor_id = radar.id
            sensor_label = sensor_label_from_role_name(
                radar.attributes.get("role_name", ""), DATASET_RADAR_ROLE_PREFIX
            )

            def radar_callback(
                measurement,
                sid=sensor_id,
                slabel=sensor_label,
                radar_actor=radar,
            ):
                frame_id = int(measurement.frame)
                entry = radar_track.get(sid)
                if entry is not None:
                    entry["last_frame"] = frame_id
                if frame_id > radar_latest_frame[0]:
                    radar_latest_frame[0] = frame_id
                if campaign_gate is not None and not campaign_gate.is_recorded(frame_id):
                    return   # between campaign segments: tick, but record nothing
                item = (measurement, sid, slabel, radar_actor)
                if is_per_radar_buffer(radar_queue):
                    radar_queue.enqueue(slabel, item)
                else:
                    radar_queue.enqueue(item)

            radar_track[sensor_id] = {
                "actor": radar,
                "label": sensor_label,
                "callback": radar_callback,
                "last_frame": 0,
            }
            radar.listen(radar_callback)

        def _watchdog_check(
            track: dict,
            stale_ticks: int,
            latest: int,
            resets: list,
        ) -> None:
            """Re-attach listen() on any sensor that's fallen behind the baseline."""
            if stale_ticks <= 0 or latest <= 0:
                return
            for sid, entry in track.items():
                last = entry["last_frame"]
                if last == 0:
                    # Sensor hasn't produced anything yet — don't reset until the
                    # baseline has advanced enough to establish that others are live.
                    if latest < stale_ticks:
                        continue
                if latest - last <= stale_ticks:
                    continue
                actor = entry["actor"]
                try:
                    if actor.is_listening:
                        actor.stop()
                    actor.listen(entry["callback"])
                    resets[0] += 1
                    print(
                        f"[capture] watchdog: re-attached listen() on "
                        f"{entry['label']} (sid={sid}) — was {latest - last} "
                        f"ticks behind (latest={latest}, last={last}).",
                        file=sys.stderr,
                        flush=True,
                    )
                    # Seed last_frame to the current latest so we don't immediately
                    # re-trigger if the sensor takes a few ticks to fire again.
                    entry["last_frame"] = latest
                except Exception as exc:  # noqa: BLE001 - best-effort recovery
                    print(
                        f"[capture] watchdog: re-attach failed for "
                        f"{entry['label']} (sid={sid}): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        def radar_watchdog_check(tick_frame: int | None = None) -> None:
            latest = radar_latest_frame[0] if tick_frame is None else tick_frame
            _watchdog_check(
                radar_track, radar_watchdog_stale_ticks, latest, radar_watchdog_resets
            )

        def camera_watchdog_check(tick_frame: int | None = None) -> None:
            latest = (
                tick_frame
                if tick_frame is not None
                else max(camera_latest_frame[0], radar_latest_frame[0])
            )
            _watchdog_check(
                camera_track, camera_watchdog_stale_ticks, latest, camera_watchdog_resets
            )

        for camera in camera_sensors:
            sensor_id = camera.id
            sensor_label = sensor_label_from_role_name(
                camera.attributes.get("role_name", ""), DATASET_CAMERA_ROLE_PREFIX
            )
            sensor_folder = os.path.join(camera_dir, f"camera_{sensor_id}")
            os.makedirs(sensor_folder, exist_ok=True)
            camera_hfov = float(camera.attributes.get("fov", "90.0"))

            def camera_callback(
                image,
                sid=sensor_id,
                slabel=sensor_label,
                folder=sensor_folder,
                sensor_hfov=camera_hfov,
            ):
                # Keep this O(1): the Image object owns the pixel buffer for as
                # long as we hold it. FOV + PNG happen on camera_frame_writer.
                frame_id = int(image.frame)
                entry = camera_track.get(sid)
                if entry is not None:
                    entry["last_frame"] = frame_id
                if frame_id > camera_latest_frame[0]:
                    camera_latest_frame[0] = frame_id
                if campaign_gate is not None and not campaign_gate.is_recorded(frame_id):
                    return
                camera_frame_writer.enqueue(
                    image, sid, slabel, folder, sensor_hfov
                )

            camera_track[sensor_id] = {
                "actor": camera,
                "label": sensor_label,
                "callback": camera_callback,
                "last_frame": 0,
            }
            camera.listen(camera_callback)

        print("Listening to sensors...")
        capture_duration_s = None if campaign_gate is not None else capture_duration_s_from_env()
        if capture_duration_s is not None:
            capture_deadline = time.monotonic() + capture_duration_s
            print(
                f"Auto-stop after {capture_duration_s:.0f}s (or press Enter to stop early)."
            )
        else:
            capture_deadline = None
            print("Press Enter to stop recording.")

        last_print = time.time()
        last_tick_at = None
        tick_rpc_s = 0.0
        tick_period_s = 0.0
        while True:
            # In sync mode the capture script owns the world clock: each
            # iteration ticks the world once, which causes the server to run
            # exactly fixed_delta_seconds of simulation and dispatch every
            # sensor that's due. Every listening radar fires on that frame_id.
            # Do NOT drain the radar queue here. CSV write is on
            # RadarQueueConsumer; emptying the queue on this thread used to
            # delay the next tick whenever drain exceeded 0.05 s, which
            # slowed SUMO and every sensor in lockstep.
            tick_frame = None
            if sync_mode:
                t0 = time.monotonic()
                try:
                    tick_frame = int(world.tick())
                except RuntimeError as exc:
                    print(
                        f"[capture] world.tick failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                tick_rpc_s = time.monotonic() - t0
                if last_tick_at is not None:
                    tick_period_s = t0 - last_tick_at
                last_tick_at = t0
            radar_watchdog_check(tick_frame)
            camera_watchdog_check(tick_frame)
            if campaign_gate is not None:
                campaign_gate.on_tick(tick_frame)
                if campaign_gate.stop_requested:
                    print("[capture] campaign finished; stopping capture.", flush=True)
                    break
            if enter_pressed():
                break
            if capture_deadline is not None and time.monotonic() >= capture_deadline:
                print(
                    f"Auto-stop: reached {capture_duration_s:.0f}s capture duration.",
                    flush=True,
                )
                break

            now = time.time()
            if now - last_print >= 2.0:
                with lock:
                    snap = labeling_collector.snapshot() if not capture_fast else {}
                    wc = snap.get("with_candidates", 0)
                    rate_c = snap.get("match_rate_given_candidates", 0.0)
                    print(
                        "Status | "
                        f"radar_msgs={counts['radar_messages']} "
                        f"radar_detections={counts['radar_detections']} "
                        f"queue={radar_queue.pending()} "
                        f"dropped={radar_queue.dropped} "
                        f"actor_ticks={tick_snapshotter.tick_count()} "
                        f"watchdog_resets={radar_watchdog_resets[0]} "
                        f"cam_watchdog_resets={camera_watchdog_resets[0]} "
                        f"fast={int(capture_fast)} "
                        f"sync={int(sync_mode)} "
                        + (
                            f"tick_ms={1000 * tick_rpc_s:.1f} "
                            f"period_ms={1000 * tick_period_s:.1f} "
                            f"drain_ms={1000 * radar_consumer.last_batch_s:.1f} "
                            if sync_mode
                            else f"drain_ms={1000 * radar_consumer.last_batch_s:.1f} "
                        )
                        + (
                            f"radar_scored={counts['radar_scored']} "
                            f"radar_matched={counts['radar_matched']} "
                            f"label_rate={100 * rate_c:.1f}% ({snap.get('matched_detections', 0)}/{wc} w/ cand) "
                            if not capture_fast
                            else f"pts/msg={counts['radar_detections'] / max(counts['radar_messages'], 1):.1f} "
                        )
                        + f"camera_frames={counts['camera_frames']} "
                        f"cam_q={camera_frame_writer.pending()} "
                        f"cam_dropped={camera_frame_writer.dropped}"
                    )
                last_print = now

            # In async mode, yield to the OS so we don't busy-spin while the
            # radar consumer and listen callbacks run. In sync mode the loop
            # is rate-limited by world.tick(), which blocks until the server
            # reports the frame complete. An optional min period keeps a fast
            # server from outrunning the radar CSV writer.
            if not sync_mode:
                time.sleep(0.005)
            elif sync_min_period_s > 0 and last_tick_at is not None:
                remain = sync_min_period_s - (time.monotonic() - last_tick_at)
                if remain > 0:
                    time.sleep(remain)

    finally:
        print("[capture] shutting down — running post-capture pipeline...", flush=True)
        # Restore async world settings BEFORE draining the queue so the world keeps
        # ticking on its own (and other CARLA clients — TrafficManager, spawners —
        # resume normally) while we finish writing CSVs. If we left sync mode on
        # without anyone calling world.tick(), the simulation would freeze and
        # any other client waiting on a tick (e.g. SpawnPedestriansAcrossMap)
        # would hang.
        if sync_mode:
            try:
                world.apply_settings(original_world_settings)
                print(
                    "[capture] restored original world settings (sync mode off).",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[capture] failed to restore world settings: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if sync_traffic_manager is not None:
                try:
                    sync_traffic_manager.set_synchronous_mode(False)
                    print(
                        "[capture] restored TrafficManager to async mode.",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[capture] failed to restore TrafficManager: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
        # Stop the live consumer first so shutdown drain is single-threaded, then
        # finish the radar queue using the in-memory per-frame actor cache
        # populated by TickActorSnapshotter (no CARLA RPCs needed). Actor frames
        # captured by the on_tick callback before Ctrl+C are already available
        # for every queued radar message, so the drain finishes in seconds and
        # every frame in radar_data.csv keeps a matching actor record.
        try:
            radar_consumer.stop()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[capture] radar_consumer.stop failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        drain_radar_queue(budget_s=30.0)
        print(
            f"[capture] actor frames captured by tick callback: "
            f"{tick_snapshotter.tick_count()}",
            flush=True,
        )
        # Stop cameras before draining PNGs so the writer queue is finite, and
        # keep TickActorSnapshotter alive until that drain finishes (FOV metadata
        # is a cache lookup, not a CARLA RPC).
        for camera in camera_sensors:
            try:
                camera.stop()
            except RuntimeError:
                pass
        try:
            camera_frame_writer.close()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[capture] camera_frame_writer.close failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        print("[capture] closing actor_frames.jsonl...", flush=True)
        try:
            tick_snapshotter.stop()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[capture] tick_snapshotter.stop failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        try:
            _run_dataset_extrinsic_exports(world, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[capture] extrinsic export skipped/failed (sensors may already be gone): {exc}",
                file=sys.stderr,
                flush=True,
            )
        print("[capture] stopping sensor streams...", flush=True)
        for sensor in radar_sensors + camera_sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass

        with lock:
            radar_file.flush()
            camera_file.flush()
        radar_file.close()
        camera_file.close()
        # Belt-and-suspenders: any exception in cosmetic prints must NOT prevent the
        # offline labeling step below from running.
        try:
            print(
                f"[capture] CSVs flushed and closed: "
                f"{os.path.basename(radar_csv)}, {os.path.basename(camera_csv)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] (status print failed: {exc})", file=sys.stderr, flush=True)

        if campaign_gate is not None:
            campaign_gate.finalize()
            campaign_gate.set_state("labeling")

        # Write run_meta BEFORE the (hours-long) offline steps so the capture is
        # fully recorded even if something below fails.
        write_run_meta(
            run_dir,
            world,
            radar_count=len(radar_sensors),
            camera_count=len(camera_sensors),
            phase="stop",
        )

        if capture_fast and label_after_capture_from_env():
            # Run labeling + post-processing in a FRESH interpreter, never in this
            # process. Neither needs CARLA, but this process still owns a live
            # libcarla client; when the simulator is closed or crashes while a
            # 3-hour labeling pass is running, libcarla's threads fail-fast and
            # take the whole process down (Windows exit 0xC0000409, capture
            # 20260917_215125 died at 65% of labeling). A subprocess is immune.
            label_ok = _run_offline_step(
                "Offline radar labeling",
                [sys.executable, str(capture_dir() / "LabelRadarCapture.py"),
                 "--capture-dir", run_dir],
            )
            if label_ok and postprocess_after_capture_from_env():
                if campaign_gate is not None:
                    campaign_gate.set_state("postprocessing")
                _run_offline_step(
                    "Post-processing (Doppler/RCS)",
                    [sys.executable, str(capture_dir() / "PostProcessDataset.py"),
                     "--capture-dir", run_dir, "--seed", str(postprocess_seed_from_env())],
                )
        elif not capture_fast:
            write_capture_labeling_report(
                labeling_collector,
                run_dir,
                labelable_min_speed_mps=labelable_min_speed_mps,
            )

        if campaign_gate is not None:
            campaign_gate.set_state("done")
        print("Recording stopped.")
        print(f"Radar file: {radar_csv}")
        print(f"Camera file: {camera_csv}")
        print(f"Camera frames: {camera_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # main()'s finally block has already drained the radar queue, closed files,
        # and (in fast mode) run offline labeling. Swallow the propagating Ctrl+C so
        # the user doesn't see a scary trailing traceback after everything succeeded.
        sys.exit(0)
