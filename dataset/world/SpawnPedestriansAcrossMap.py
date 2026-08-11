"""
Spawn AI pedestrians across the map using CARLA's built-in navmesh logic.

Uses walker.pedestrian.* + controller.ai.walker (same flow as
PythonAPI/examples/generate_traffic.py). Pedestrians spawn on the navmesh
(sidewalks / crosswalks) and path to random nav targets — they stay on
walkable areas instead of cutting through buildings.

Run:
  python SpawnPedestriansAcrossMap.py
"""

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import os
import random
import sys
import time

import carla

from carla_connect import get_world


DEFAULT_PED_COUNT = 30
PED_RUNNING_FRACTION = 0.15
PED_CROSSING_FACTOR = 0.35
# Occasionally send a new nav target so pedestrians keep roaming the map.
RETARGET_INTERVAL_S = 45.0
# Default pedestrian spawn region: the monitored corridor (~(18.45, -63.0), 60 m).
# Keeps walkers in/around the radar coverage instead of scattered map-wide.
# Override via DATASET_PED_SPAWN_CENTER_X/Y + DATASET_PED_SPAWN_RADIUS_M; set the
# radius env to a huge value (or edit here) to restore map-wide spawning.
DEFAULT_PED_SPAWN_CENTER_X = 18.45
DEFAULT_PED_SPAWN_CENTER_Y = -63.0
DEFAULT_PED_SPAWN_RADIUS_M = 60.0

# --- Mid-block corridor crossers ------------------------------------------------
# CARLA's navmesh AI only crosses roads at map-defined crosswalks, and the corridor
# has exactly one (at the west junction ~(-31,-61)). To get pedestrians crossing the
# ROAD mid-block in radar view, a subset of walkers is driven manually across the
# corridor (direct WalkerControl, no AI controller). Opt-in via
# DATASET_PED_CROSSING_COUNT > 0; those crossers are taken from the total pedestrian
# count (the rest roam on AI).
#
# Each crosser runs a small state machine for *natural* (non-continuous) crossings:
#   WAIT   - stand at a curb for a random dwell (and optionally hold for a gap in
#            traffic) so crossings are intermittent, not a synchronized conveyor.
#   CROSS  - traverse the road once to the opposite curb.
#   STROLL - walk along the far curb to a new random x, so successive crossings
#            happen at different points across the radar zone.
# Per-crosser speed and dwell are randomized, and initial timers are staggered, so
# the crossers desync instead of marching in lockstep (the old shuttle behavior).
DEFAULT_PED_CROSSING_COUNT = 0
DEFAULT_PED_CROSSING_X_MIN = -25.0     # radar zone spans x ~[-29, 66]
DEFAULT_PED_CROSSING_X_MAX = 60.0
DEFAULT_PED_CROSSING_Y_SOUTH = -72.0   # south curb (south radars at y=-73.5)
DEFAULT_PED_CROSSING_Y_NORTH = -54.0   # north curb (north radars at y=-52.5)
DEFAULT_PED_CROSSING_SPEED = 1.4       # m/s; legacy fixed speed (overrides the range)
# Per-crossing speed range (sampled unless DATASET_PED_CROSSING_SPEED pins a value).
DEFAULT_PED_CROSSING_SPEED_MIN = 0.9   # m/s, slow walk
DEFAULT_PED_CROSSING_SPEED_MAX = 1.8   # m/s, brisk walk / light jog
# Random dwell at the curb between crossings (seconds, simulation time).
DEFAULT_PED_CROSSING_DWELL_MIN = 2.0
DEFAULT_PED_CROSSING_DWELL_MAX = 8.0
# Optional gap-acceptance: hold at curb if a vehicle is within GAP_M of the crossing
# column, up to GAP_MAX_WAIT seconds. Off by default — crossing through traffic puts
# pedestrians at vehicle depths/bearings (helps decorrelate class from position).
DEFAULT_PED_CROSSING_GAP_M = 12.0
DEFAULT_PED_CROSSING_GAP_MAX_WAIT = 15.0

# Mid-cross car-yield (default ON): while crossing, a crosser pauses if a *moving*
# vehicle is within YIELD_DIST and closing on it, then resumes once that car has
# passed or stopped. Crossers are invincible + kinematic (immovable), so a car that
# can't brake in time flips on contact and stalls the corridor — this prevents the
# collision. Only MOVING cars trigger it (a stopped/yielding car does not), so it
# avoids the mutual standoff the curb gap-check caused.
DEFAULT_PED_CROSSING_CAR_YIELD = True
DEFAULT_PED_CROSSING_YIELD_DIST_M = 7.0      # how close an approaching car must be
DEFAULT_PED_CROSSING_YIELD_MOVING_MPS = 2.0  # below this the car counts as stopped

# Arrival tolerance (m): per-tick walker step (~speed * dt ≈ 0.07 m) is far below
# this, so a curb / x-target is reliably detected without skipping past it.
_CROSS_ARRIVE_EPS = 0.4


def keep_pedestrians_running() -> bool:
    """When set by Start.py, spawn once and roam until the process is stopped (no prompt)."""
    return os.environ.get("DATASET_KEEP_PEDESTRIANS_RUNNING", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pedestrian_count_from_env(default=DEFAULT_PED_COUNT) -> int:
    raw = os.environ.get("DATASET_PEDESTRIAN_COUNT", "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _clamped_float_from_env(name, default, lo, hi):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(lo, min(hi, float(raw)))
    except ValueError:
        return default


def ped_crossing_factor_from_env() -> float:
    """Fraction of pedestrians allowed to cross roads (set_pedestrians_cross_factor).

    Raising this (e.g. 0.7-0.9 at intersection captures) is the keystone for putting
    pedestrians onto the roadway at vehicle depths/bearings, decorrelating class from
    position/depth. Clamped to [0, 1]; default DATASET-overridable."""
    return _clamped_float_from_env("DATASET_PED_CROSSING_FACTOR", PED_CROSSING_FACTOR, 0.0, 1.0)


def ped_running_fraction_from_env() -> float:
    return _clamped_float_from_env("DATASET_PED_RUNNING_FRACTION", PED_RUNNING_FRACTION, 0.0, 1.0)


def ped_crossing_count_from_env(default=DEFAULT_PED_CROSSING_COUNT) -> int:
    """Number of pedestrians driven manually across the corridor mid-block (no AI).
    Taken from the total DATASET_PEDESTRIAN_COUNT; the remainder roam on the navmesh."""
    raw = os.environ.get("DATASET_PED_CROSSING_COUNT", "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def ped_crossing_span_from_env():
    """(x_min, x_max, y_south, y_north, speed) for the mid-block crossing band."""
    def _f(name, default):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default
    return (
        _f("DATASET_PED_CROSSING_X_MIN", DEFAULT_PED_CROSSING_X_MIN),
        _f("DATASET_PED_CROSSING_X_MAX", DEFAULT_PED_CROSSING_X_MAX),
        _f("DATASET_PED_CROSSING_Y_SOUTH", DEFAULT_PED_CROSSING_Y_SOUTH),
        _f("DATASET_PED_CROSSING_Y_NORTH", DEFAULT_PED_CROSSING_Y_NORTH),
        _f("DATASET_PED_CROSSING_SPEED", DEFAULT_PED_CROSSING_SPEED),
    )


def _float_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def ped_crossing_speed_range_from_env():
    """(lo, hi) m/s for per-crossing speed sampling.

    If DATASET_PED_CROSSING_SPEED is set explicitly, crossers use that fixed speed
    (lo == hi) for back-compat; otherwise [MIN, MAX] is sampled per crossing so the
    crossers walk at varied speeds (some slow, some brisk)."""
    fixed = os.environ.get("DATASET_PED_CROSSING_SPEED", "").strip()
    if fixed:
        try:
            v = float(fixed)
            return v, v
        except ValueError:
            pass
    lo = _float_env("DATASET_PED_CROSSING_SPEED_MIN", DEFAULT_PED_CROSSING_SPEED_MIN)
    hi = _float_env("DATASET_PED_CROSSING_SPEED_MAX", DEFAULT_PED_CROSSING_SPEED_MAX)
    return (hi, lo) if hi < lo else (lo, hi)


def ped_crossing_dwell_range_from_env():
    """(lo, hi) seconds of curb dwell between crossings (simulation time)."""
    lo = max(0.0, _float_env("DATASET_PED_CROSSING_DWELL_MIN", DEFAULT_PED_CROSSING_DWELL_MIN))
    hi = max(lo, _float_env("DATASET_PED_CROSSING_DWELL_MAX", DEFAULT_PED_CROSSING_DWELL_MAX))
    return lo, hi


def ped_crossing_gap_from_env():
    """(enabled, gap_m, max_wait_s) for optional traffic gap-acceptance at the curb."""
    enabled = os.environ.get("DATASET_PED_CROSSING_GAP_CHECK", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    gap_m = _float_env("DATASET_PED_CROSSING_GAP_M", DEFAULT_PED_CROSSING_GAP_M)
    max_wait = _float_env("DATASET_PED_CROSSING_GAP_MAX_WAIT", DEFAULT_PED_CROSSING_GAP_MAX_WAIT)
    return enabled, gap_m, max_wait


def ped_crossing_car_yield_from_env():
    """(enabled, yield_dist_m, moving_mps) for the mid-cross moving-car yield."""
    raw = os.environ.get("DATASET_PED_CROSSING_CAR_YIELD", "").strip().lower()
    enabled = DEFAULT_PED_CROSSING_CAR_YIELD if raw == "" else raw in ("1", "true", "yes", "on")
    dist = _float_env("DATASET_PED_CROSSING_YIELD_DIST_M", DEFAULT_PED_CROSSING_YIELD_DIST_M)
    moving = _float_env("DATASET_PED_CROSSING_YIELD_MOVING_MPS", DEFAULT_PED_CROSSING_YIELD_MOVING_MPS)
    return enabled, dist, moving


def crossing_config_from_env():
    """Bundle all mid-block crossing parameters into one dict (built once per run)."""
    x_min, x_max, y_south, y_north, _legacy_speed = ped_crossing_span_from_env()
    speed_lo, speed_hi = ped_crossing_speed_range_from_env()
    dwell_min, dwell_max = ped_crossing_dwell_range_from_env()
    gap_check, gap_m, gap_max_wait = ped_crossing_gap_from_env()
    car_yield, yield_dist, yield_moving = ped_crossing_car_yield_from_env()
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_south": y_south,
        "y_north": y_north,
        "speed_lo": speed_lo,
        "speed_hi": speed_hi,
        "dwell_min": dwell_min,
        "dwell_max": dwell_max,
        "gap_check": gap_check,
        "gap_m": gap_m,
        "gap_max_wait": gap_max_wait,
        "car_yield": car_yield,
        "yield_dist": yield_dist,
        "yield_moving": yield_moving,
    }


def ped_retarget_interval_s_from_env() -> float:
    return _clamped_float_from_env("DATASET_PED_RETARGET_INTERVAL_S", RETARGET_INTERVAL_S, 1.0, 1e6)


def ped_seed_from_env():
    """Seed for pedestrian RNG (reproducible spawns; recorded in run_meta). None = nondeterministic."""
    raw = os.environ.get("DATASET_PED_SEED", os.environ.get("DATASET_SEED", "")).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def ped_spawn_region_from_env():
    """Return (center_x, center_y, radius_m); defaults to the monitored corridor.

    Both initial navmesh spawn and AI retargeting reject samples outside this 2D
    circle (z ignored). Each of the three values falls back to its corridor default
    independently, so you can override just the radius (etc.) via env.
    """
    def _f(name, default):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default
    return (
        _f("DATASET_PED_SPAWN_CENTER_X", DEFAULT_PED_SPAWN_CENTER_X),
        _f("DATASET_PED_SPAWN_CENTER_Y", DEFAULT_PED_SPAWN_CENTER_Y),
        _f("DATASET_PED_SPAWN_RADIUS_M", DEFAULT_PED_SPAWN_RADIUS_M),
    )


_PED_REGION_MAX_NAV_ATTEMPTS = 200


def get_nav_location_in_region(world, region):
    """Rejection-sample navmesh points until one lands inside region (cx, cy, r_m)."""
    if region is None:
        return world.get_random_location_from_navigation()
    cx, cy, r = region
    r_sq = r * r
    for _ in range(_PED_REGION_MAX_NAV_ATTEMPTS):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        dx = loc.x - cx
        dy = loc.y - cy
        if dx * dx + dy * dy <= r_sq:
            return loc
    return None


def prompt_pedestrian_count(default=DEFAULT_PED_COUNT):
    while True:
        try:
            raw = input(f"How many pedestrians to spawn? [default {default}]: ").strip()
        except EOFError:
            return default

        if not raw:
            return default

        try:
            value = int(raw)
        except ValueError:
            print(f"  '{raw}' is not an integer. Try again.")
            continue

        if value < 0:
            print("  Count must be >= 0. Try again.")
            continue

        return value


def spawn_pedestrians_with_ai(client, world, count):
    """
    CARLA official pedestrian flow (generate_traffic.py).
    """
    SpawnActor = carla.command.SpawnActor

    seed = ped_seed_from_env()
    if seed is None:
        seed = random.randint(0, 0xFFFFFFFF)
    try:
        world.set_pedestrians_seed(seed)
    except RuntimeError:
        pass
    random.seed(seed)

    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        raise RuntimeError("No walker.pedestrian.* blueprints found.")

    controller_bp = bp_lib.find("controller.ai.walker")
    if controller_bp is None:
        raise RuntimeError("controller.ai.walker blueprint not found.")

    # 1) Collect navmesh spawn points (only valid pedestrian locations).
    region = ped_spawn_region_from_env()
    if region is not None:
        print(
            f"[ped] restricting spawn to circle (cx={region[0]:.1f}, cy={region[1]:.1f}, "
            f"r={region[2]:.1f} m)",
            flush=True,
        )
    spawn_points = []
    for _ in range(count):
        loc = get_nav_location_in_region(world, region)
        if loc is not None:
            spawn_points.append(carla.Transform(loc))

    if not spawn_points:
        raise RuntimeError("No navmesh spawn locations found on this map.")

    # 2) Spawn walkers (batch + tick so the server registers them).
    walker_batch = []
    walker_speeds = []
    for transform in spawn_points:
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        if walker_bp.has_attribute("speed"):
            values = walker_bp.get_attribute("speed").recommended_values
            if random.random() < ped_running_fraction_from_env() and len(values) > 2:
                walker_speeds.append(values[2])
            else:
                walker_speeds.append(values[1])
        else:
            walker_speeds.append("1.4")

        walker_batch.append(SpawnActor(walker_bp, transform))

    walker_results = client.apply_batch_sync(walker_batch, True)

    walkers_meta = []
    speeds_meta = []
    for result, speed in zip(walker_results, walker_speeds):
        if result.error:
            print(f"  walker spawn failed: {result.error}", flush=True)
            continue
        walkers_meta.append({"id": result.actor_id})
        speeds_meta.append(speed)

    if not walkers_meta:
        return [], [], count

    # 3) Spawn AI controllers attached to each walker (batch + tick).
    controller_batch = [
        SpawnActor(controller_bp, carla.Transform(), entry["id"]) for entry in walkers_meta
    ]
    controller_results = client.apply_batch_sync(controller_batch, True)

    all_ids = []
    paired_speeds = []
    for entry, ctrl_result, speed in zip(walkers_meta, controller_results, speeds_meta):
        if ctrl_result.error:
            print(f"  controller spawn failed: {ctrl_result.error}", flush=True)
            client.apply_batch([carla.command.DestroyActor(entry["id"])])
            continue
        entry["con"] = ctrl_result.actor_id
        all_ids.append(entry["con"])
        all_ids.append(entry["id"])
        paired_speeds.append(speed)

    skipped = count - len(paired_speeds)

    # 4) Wait for a tick so transforms are ready (required before start()).
    settings = world.get_settings()
    sync_mode = bool(getattr(settings, "synchronous_mode", False))
    if sync_mode:
        world.tick()
    else:
        world.wait_for_tick()

    # 5) Crossing factor: fraction that will use crosswalks / cross streets legally.
    try:
        world.set_pedestrians_cross_factor(ped_crossing_factor_from_env())
    except RuntimeError:
        pass

    # 6) Interleaved list from get_actors: [controller, walker, controller, walker, ...]
    all_actors = world.get_actors(all_ids)
    controllers = []
    walkers = []

    for i in range(0, len(all_ids), 2):
        controller = all_actors[i]
        walker = all_actors[i + 1]
        speed = float(paired_speeds[i // 2])
        target = get_nav_location_in_region(world, region)
        if target is None:
            continue

        controller.start()
        controller.go_to_location(target)
        controller.set_max_speed(speed)

        controllers.append(controller)
        walkers.append(walker)
        loc = walker.get_location()
        print(
            f"  walker_id={walker.id} controller_id={controller.id} "
            f"at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) speed={speed:.2f}",
            flush=True,
        )

    return walkers, controllers, skipped, sync_mode


def _sim_time(world) -> float:
    """Simulation clock (seconds). Robust to sync/async mode — used for crosser dwell
    timers so durations are in simulation time, not wall-clock."""
    try:
        return world.get_snapshot().timestamp.elapsed_seconds
    except RuntimeError:
        return 0.0


def _sample_crossing_speed(cfg) -> float:
    lo, hi = cfg["speed_lo"], cfg["speed_hi"]
    return lo if hi <= lo else random.uniform(lo, hi)


def spawn_corridor_crossers(client, world, count, cfg):
    """Spawn `count` walkers that cross the corridor mid-block, driven by direct
    WalkerControl (no AI controller). Returns a list of per-crosser state dicts.

    Crossers alternate start curb and are spread across the x-band to cover the radar
    zone. Each starts in WAIT with a staggered timer so they don't cross in lockstep.
    They are made invincible so a passing vehicle doesn't ragdoll them mid-crossing
    (TM still brakes for them)."""
    if count <= 0:
        return []

    SpawnActor = carla.command.SpawnActor
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        raise RuntimeError("No walker.pedestrian.* blueprints found.")

    x_min, x_max = cfg["x_min"], cfg["x_max"]
    y_south, y_north = cfg["y_south"], cfg["y_north"]
    # Ground height near the corridor centre (so walkers spawn at road level).
    wp = world.get_map().get_waypoint(
        carla.Location(x=(x_min + x_max) / 2.0, y=(y_south + y_north) / 2.0, z=0.0),
        project_to_road=True,
    )
    base_z = (wp.transform.location.z if wp is not None else 0.0) + 1.0

    batch, sides = [], []
    for i in range(count):
        frac = (i + 0.5) / count
        x = x_min + (x_max - x_min) * frac
        start_south = (i % 2 == 0)
        y = y_south if start_south else y_north
        d = 1.0 if start_south else -1.0  # +y heads toward the north curb
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "true")
        tf = carla.Transform(
            carla.Location(x=x, y=y, z=base_z),
            carla.Rotation(yaw=90.0 if d > 0 else -90.0),
        )
        batch.append(SpawnActor(bp, tf))
        sides.append(start_south)

    results = client.apply_batch_sync(batch, True)

    settings = world.get_settings()
    if bool(getattr(settings, "synchronous_mode", False)):
        world.tick()
    else:
        world.wait_for_tick()

    now = _sim_time(world)
    _, dwell_max = cfg["dwell_min"], cfg["dwell_max"]
    crossers = []
    for res, start_south in zip(results, sides):
        if res.error:
            print(f"  crosser spawn failed: {res.error}", flush=True)
            continue
        w = world.get_actor(res.actor_id)
        if w is None:
            continue
        # Begin standing at the spawn curb; a staggered dwell desyncs the first wave.
        w.apply_control(carla.WalkerControl(speed=0.0))
        crossers.append(
            {
                "walker": w,
                "state": "WAIT",
                "curb": "S" if start_south else "N",
                "speed": _sample_crossing_speed(cfg),   # stroll speed for this crosser
                "cross_speed": _sample_crossing_speed(cfg),
                "x_target": None,
                "y_target": None,
                "wait_until": now + random.uniform(0.0, dwell_max),
            }
        )
        loc = w.get_location()
        print(
            f"  crosser walker_id={w.id} at ({loc.x:.1f}, {loc.y:.1f}) "
            f"start_curb={'S' if start_south else 'N'} speed={crossers[-1]['speed']:.2f}",
            flush=True,
        )

    print(
        f"[ped] {len(crossers)} mid-block crossers over x=[{x_min:.0f},{x_max:.0f}], "
        f"y=[{y_south:.0f},{y_north:.0f}] (WAIT/CROSS/STROLL state machine, no AI).",
        flush=True,
    )
    return crossers


def _road_clear(world, x, cfg) -> bool:
    """True if no vehicle currently sits on the crossing column near `x` (between the
    curbs). Used for optional gap-acceptance before stepping off the curb."""
    gap = cfg["gap_m"]
    y_lo = min(cfg["y_south"], cfg["y_north"]) - 2.0
    y_hi = max(cfg["y_south"], cfg["y_north"]) + 2.0
    try:
        vehicles = world.get_actors().filter("vehicle.*")
    except RuntimeError:
        return True
    for v in vehicles:
        try:
            l = v.get_location()
        except RuntimeError:
            continue
        if y_lo <= l.y <= y_hi and abs(l.x - x) <= gap:
            return False
    return True


def _approaching_car(world, px, py, cfg) -> bool:
    """True if a *moving* vehicle (>= yield_moving m/s) is within yield_dist of
    (px, py) and closing on it. Stopped/yielding cars are ignored on purpose, so a
    crosser resumes once an approaching car has braked (no mutual standoff)."""
    if not cfg["car_yield"]:
        return False
    yd = cfg["yield_dist"]
    yd_sq = yd * yd
    vmin_sq = cfg["yield_moving"] * cfg["yield_moving"]
    try:
        vehicles = world.get_actors().filter("vehicle.*")
    except RuntimeError:
        return False
    for v in vehicles:
        try:
            l = v.get_location()
            vel = v.get_velocity()
        except RuntimeError:
            continue
        dx = px - l.x
        dy = py - l.y
        if dx * dx + dy * dy > yd_sq:
            continue
        if vel.x * vel.x + vel.y * vel.y < vmin_sq:
            continue  # stopped/slow car -> don't yield to it (avoids standoff)
        if vel.x * dx + vel.y * dy > 0.0:  # car velocity points toward the crosser
            return True
    return False


def step_crossers(world, crossers, cfg, now):
    """Advance each crosser's WAIT -> CROSS -> STROLL state machine by one tick."""
    for c in crossers:
        w = c["walker"]
        try:
            if not w.is_alive:
                continue
            loc = w.get_location()
            state = c["state"]

            if state == "WAIT":
                w.apply_control(carla.WalkerControl(speed=0.0))
                if now < c["wait_until"]:
                    continue
                # Dwell elapsed. Optionally hold for a gap in traffic (bounded so a
                # crosser never waits forever on a busy corridor).
                if (
                    cfg["gap_check"]
                    and (now - c["wait_until"]) < cfg["gap_max_wait"]
                    and not _road_clear(world, loc.x, cfg)
                ):
                    continue
                # Step off toward the opposite curb.
                c["y_target"] = cfg["y_north"] if c["curb"] == "S" else cfg["y_south"]
                c["cross_speed"] = _sample_crossing_speed(cfg)
                c["state"] = "CROSS"

            elif state == "CROSS":
                # Yield to an approaching moving car: stand still and let it pass (or
                # brake). Prevents driving a car into the immovable crosser (which
                # flips the car and stalls the corridor). Resumes once clear/stopped.
                if _approaching_car(world, loc.x, loc.y, cfg):
                    w.apply_control(carla.WalkerControl(speed=0.0))
                    continue
                d = 1.0 if c["y_target"] > loc.y else -1.0
                w.apply_control(
                    carla.WalkerControl(
                        direction=carla.Vector3D(0.0, d, 0.0), speed=c["cross_speed"]
                    )
                )
                if abs(loc.y - c["y_target"]) <= _CROSS_ARRIVE_EPS:
                    # Reached the far curb; pick a new x to stroll to for variety.
                    c["curb"] = "N" if c["curb"] == "S" else "S"
                    c["x_target"] = random.uniform(cfg["x_min"], cfg["x_max"])
                    c["state"] = "STROLL"

            elif state == "STROLL":
                dx = c["x_target"] - loc.x
                if abs(dx) <= _CROSS_ARRIVE_EPS:
                    w.apply_control(carla.WalkerControl(speed=0.0))
                    c["state"] = "WAIT"
                    c["wait_until"] = now + random.uniform(cfg["dwell_min"], cfg["dwell_max"])
                else:
                    d = 1.0 if dx > 0 else -1.0
                    w.apply_control(
                        carla.WalkerControl(
                            direction=carla.Vector3D(d, 0.0, 0.0), speed=c["speed"]
                        )
                    )
        except RuntimeError:
            continue


def retarget_controllers(world, controllers):
    """Send walkers to a new random navmesh point (sidewalk / crosswalk)."""
    region = ped_spawn_region_from_env()
    for controller in controllers:
        try:
            if not controller.is_alive:
                continue
            target = get_nav_location_in_region(world, region)
            if target is not None:
                controller.go_to_location(target)
        except RuntimeError:
            continue


def cleanup(walkers, controllers):
    for controller in controllers:
        try:
            if controller.is_alive:
                controller.stop()
        except RuntimeError:
            pass

    destroyed = 0
    for actor in controllers + walkers:
        try:
            if actor.is_alive and actor.destroy():
                destroyed += 1
        except RuntimeError:
            continue
    return destroyed


def main():
    if keep_pedestrians_running():
        count = pedestrian_count_from_env()
    else:
        count = prompt_pedestrian_count()
    if count == 0:
        print("Nothing to spawn. Exiting.")
        return 0

    client, world = get_world()

    # Split the total count: a subset becomes manual mid-block corridor crossers, the
    # rest roam on the navmesh AI.
    crossing_count = min(ped_crossing_count_from_env(), count)
    ai_count = count - crossing_count

    print(
        "Using CARLA built-in pedestrians (controller.ai.walker + navmesh).",
        flush=True,
    )
    print(
        f"Spawning {ai_count} AI pedestrians + {crossing_count} mid-block crossers "
        f"(total {count})...",
        flush=True,
    )

    walkers, controllers, skipped, sync_mode = spawn_pedestrians_with_ai(
        client, world, ai_count
    )

    crossing_cfg = crossing_config_from_env()
    crossers = spawn_corridor_crossers(client, world, crossing_count, crossing_cfg)
    crosser_walkers = [c["walker"] for c in crossers]
    if crossing_count > 0:
        print(
            f"[ped] crossing params: speed=[{crossing_cfg['speed_lo']:.2f},"
            f"{crossing_cfg['speed_hi']:.2f}] m/s, dwell=[{crossing_cfg['dwell_min']:.1f},"
            f"{crossing_cfg['dwell_max']:.1f}] s, gap_check={crossing_cfg['gap_check']}, "
            f"car_yield={crossing_cfg['car_yield']} (dist={crossing_cfg['yield_dist']:.0f}m, "
            f"moving>{crossing_cfg['yield_moving']:.1f}m/s).",
            flush=True,
        )

    print(
        f"Spawned {len(walkers)}/{ai_count} AI + {len(crossers)}/{crossing_count} crossers "
        f"(skipped {skipped}, crossing_factor={ped_crossing_factor_from_env():.2f}).",
        flush=True,
    )
    if not walkers and not crossers:
        return 0

    if keep_pedestrians_running():
        print(
            "Pedestrians roaming (DATASET_KEEP_PEDESTRIANS_RUNNING). "
            "Stop via Start.py Ctrl+C or kill this process.",
            flush=True,
        )
    else:
        print(
            "Pedestrians follow the navmesh (sidewalks / crosswalks). "
            "Press Ctrl+C to stop.",
            flush=True,
        )

    retarget_interval_s = ped_retarget_interval_s_from_env()
    next_retarget = time.time() + retarget_interval_s
    try:
        while True:
            if sync_mode:
                world.tick()
            else:
                world.wait_for_tick()

            # Advance the mid-block crossers' WAIT/CROSS/STROLL state machine.
            step_crossers(world, crossers, crossing_cfg, _sim_time(world))

            now = time.time()
            if now >= next_retarget:
                retarget_controllers(world, controllers)
                next_retarget = now + retarget_interval_s
    except KeyboardInterrupt:
        if not keep_pedestrians_running():
            print("\nCtrl+C received. Cleaning up...", flush=True)
    finally:
        if not keep_pedestrians_running():
            destroyed = cleanup(walkers + crosser_walkers, controllers)
            print(f"Destroyed {destroyed} actors.", flush=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"CARLA error: {exc}", file=sys.stderr)
        sys.exit(1)
