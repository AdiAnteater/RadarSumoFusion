"""Pure pedestrian-path decisions shared by the CARLA mirror and the tests.

No CARLA import. The mirror asks where a walker should stand; this module
answers from the SUMO lane and the sidewalk point already locked for that walker.
"""

import math


def is_crossing_lane(lane_id: str) -> bool:
    """True for a SUMO crossing or walking-area lane."""
    return bool(lane_id) and lane_id.startswith(":") and ("_c" in lane_id or "_w" in lane_id)


def choose_sidewalk_point(raw, nearest, locked, snap_max, continuity_m):
    """Keep a walker on the sidewalk they already locked onto.

    ``nearest`` is the closest CARLA sidewalk point, or None. A new nearest
    that jumps farther than ``continuity_m`` from the lock is rejected while
    the lock is still within ``snap_max`` of the walker.
    """
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    if nearest is not None and dist(nearest, raw) <= snap_max:
        if locked is not None and dist(nearest, locked) > continuity_m and dist(locked, raw) <= snap_max:
            return locked
        return nearest
    if locked is not None and dist(locked, raw) <= snap_max:
        return locked
    return raw


def blend_crossing(raw, sidewalk, blend_m):
    """Move from the locked sidewalk point toward the SUMO crossing position."""
    if sidewalk is None or blend_m <= 0:
        return raw
    dx = raw[0] - sidewalk[0]
    dy = raw[1] - sidewalk[1]
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return raw
    t = min(1.0, dist / blend_m)
    return (sidewalk[0] + t * dx, sidewalk[1] + t * dy)
