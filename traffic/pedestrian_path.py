"""Pure pedestrian-path decisions shared by the CARLA mirror and the tests.

No CARLA import. SUMO's sidewalk is not CARLA's pavement, so the mirror asks
where a walker should stand and this module answers without teleporting them:
stay on the sidewalk lane already locked, and move only as far as they walked
this tick.
"""

import math


# Extra metres of sidewalk correction allowed on top of the SUMO step.
# A few centimetres per tick closes a real kerb offset while they walk.
CORRECTION_M = 0.05

# Below this, SUMO barely moved (waiting at a light). The body stays put.
STOPPED_M = 0.02

# A crossing or walking area has to last this many ticks before it is real.
# One lane-id flicker at the kerb does not count.
CROSSING_COMMIT_TICKS = 3


def is_crossing_lane(lane_id: str) -> bool:
    """True for a SUMO crossing or walking-area lane."""
    return bool(lane_id) and lane_id.startswith(":") and ("_c" in lane_id or "_w" in lane_id)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _close(a, b, eps: float = 1e-6) -> bool:
    return a is not None and b is not None and _dist(a, b) <= eps


def sidewalk_query_point(raw, locked, prev_raw):
    """Where to ask CARLA for a sidewalk.

    Follow the step the person just took, starting from the pavement already
    locked, so the lookup does not begin at a raw position that has drifted
    into the road.
    """
    if locked is None or prev_raw is None:
        return raw
    return (locked[0] + raw[0] - prev_raw[0], locked[1] + raw[1] - prev_raw[1])


def choose_sidewalk_point(raw, nearest, locked, snap_max,
                          nearest_lane=None, locked_lane=None):
    """Keep a walker on the sidewalk lane they already locked onto.

    ``nearest`` is the sidewalk point found from the locked pavement, or None.
    A candidate on a different lane is rejected while the lock is still within
    ``snap_max`` of the walker.
    """
    if nearest is not None and _dist(nearest, raw) <= snap_max:
        if (locked is not None and locked_lane is not None
                and nearest_lane != locked_lane
                and _dist(locked, raw) <= snap_max):
            return locked
        return nearest
    if locked is not None and _dist(locked, raw) <= snap_max:
        return locked
    return raw


def note_crossing(crossing_ticks: int, lane_id: str) -> int:
    """Consecutive ticks spent on a crossing or walking area."""
    if is_crossing_lane(lane_id):
        return crossing_ticks + 1
    return 0


def crossing_is_real(ticks: int, commit_ticks: int = CROSSING_COMMIT_TICKS) -> bool:
    return ticks >= commit_ticks


def crossing_will_be_real(state, lane_id: str,
                          commit_ticks: int = CROSSING_COMMIT_TICKS) -> bool:
    """True when this tick commits a crossing, so the sidewalk lookup can be skipped."""
    ticks = 0 if not state else state.get("crossing_ticks", 0)
    return crossing_is_real(note_crossing(ticks, lane_id), commit_ticks)


def step_toward(prev, target, sumo_moved: float,
                correction_m: float = CORRECTION_M,
                stopped_m: float = STOPPED_M):
    """Move ``prev`` toward ``target`` by at most the distance SUMO moved.

    A few extra centimetres let a real sidewalk offset close. If SUMO barely
    moved, stay at ``prev``. The first tick (no previous point) uses the target.
    """
    if prev is None:
        return target
    if sumo_moved < stopped_m:
        return prev
    max_step = sumo_moved + correction_m
    dx = target[0] - prev[0]
    dy = target[1] - prev[1]
    dist = math.hypot(dx, dy)
    if dist <= max_step or dist <= 1e-9:
        return target
    scale = max_step / dist
    return (prev[0] + dx * scale, prev[1] + dy * scale)


def yaw_from_step(prev_yaw, prev_xy, xy, min_move: float = STOPPED_M):
    """Heading is the direction of the rendered step. Hold it while stopped.

    CARLA yaw is degrees, 0 along +x, positive toward +y.
    """
    if prev_xy is None or xy is None:
        return prev_yaw
    dx = xy[0] - prev_xy[0]
    dy = xy[1] - prev_xy[1]
    if math.hypot(dx, dy) < min_move:
        return prev_yaw
    return math.degrees(math.atan2(dy, dx))


def next_pose(raw, sumo_yaw, lane_id, nearest, nearest_lane, state, snap_max,
              correction_m: float = CORRECTION_M,
              stopped_m: float = STOPPED_M,
              commit_ticks: int = CROSSING_COMMIT_TICKS):
    """Return ``(x, y, yaw, state)`` for this tick.

    On a sidewalk the target is the locked pavement. Once a crossing has lasted
    ``commit_ticks``, the target is SUMO's position and the body walks toward
    it. The lock is dropped only then, and only once that pavement is farther
    than ``snap_max``. The first tick a person appears uses the target directly.
    """
    state = state or {}
    prev_raw = state.get("prev_raw")
    locked = state.get("locked")
    locked_lane = state.get("locked_lane")
    rendered = state.get("rendered")
    yaw = state.get("yaw")
    sumo_moved = 0.0 if prev_raw is None else _dist(raw, prev_raw)
    crossing_ticks = note_crossing(state.get("crossing_ticks", 0), lane_id)
    real = crossing_is_real(crossing_ticks, commit_ticks)

    if real and locked is not None and _dist(locked, raw) > snap_max:
        locked = None
        locked_lane = None

    if real:
        target = raw
    else:
        chosen = choose_sidewalk_point(
            raw, nearest, locked, snap_max, nearest_lane, locked_lane)
        target = chosen
        # Adopt a sidewalk point only when it is the one we kept. Falling
        # through to the raw position does not drop the lock: that happens
        # above, once a crossing is real and the pavement is out of range.
        if nearest is not None and _close(chosen, nearest):
            locked = nearest
            locked_lane = nearest_lane

    if rendered is None:
        new_xy = target
        new_yaw = sumo_yaw
    else:
        new_xy = step_toward(rendered, target, sumo_moved, correction_m, stopped_m)
        new_yaw = yaw_from_step(
            sumo_yaw if yaw is None else yaw, rendered, new_xy, stopped_m)

    new_state = {
        "locked": locked,
        "locked_lane": locked_lane,
        "rendered": new_xy,
        "yaw": new_yaw,
        "prev_raw": raw,
        "crossing_ticks": crossing_ticks,
    }
    return new_xy[0], new_xy[1], new_yaw, new_state
