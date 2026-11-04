"""Shared radar placement helpers (no imports from other capture modules)."""

import math
import os

import carla

# Radar tilt. CARLA/Unreal rotators use POSITIVE pitch = nose UP (the forward
# vector is (cos p cos y, cos p sin y, sin p), so +pitch raises z). The earlier
# +8 deg default therefore tilted every radar 8 deg UP (35% of returns landed
# above 4.5 m, i.e. building facades). NEGATIVE pitch looks DOWN at the road.
# raycast_sensor_testing.py uses pitch=+5 because its radar sits on a car roof
# and is tilted UP to avoid the bonnet; that is the opposite of a pole mount.
# Default -6 deg from a 3 m pole: boresight meets the road ~28 m out; with a
# 30 deg VFOV the lower edge (-21 deg) reaches the ground at ~8 m and the
# upper edge (+9 deg) stays low enough that far-lane vehicles (~40 m) are
# still inside the cone.
RADAR_PITCH_DEG = -6.0


def radar_pitch_deg_from_env() -> float:
    raw = os.environ.get("DATASET_RADAR_PITCH_DEG", "").strip()
    if raw:
        try:
            return max(-45.0, min(float(raw), 30.0))
        except ValueError:
            pass
    return RADAR_PITCH_DEG


# Mounting height (m) of every radar above the road. Live knob via DATASET_RIG_HEIGHT_M
# (the setup scripts apply it to each radar's z). Clamped to a sane pole range.
RADAR_HEIGHT_M = 3.0


def radar_height_m_from_env() -> float:
    raw = os.environ.get("DATASET_RIG_HEIGHT_M", "").strip()
    if raw:
        try:
            return max(0.3, min(float(raw), 12.0))
        except ValueError:
            pass
    return RADAR_HEIGHT_M


def apply_radar_pitch(radar_positions):
    """Apply shared down-tilt pitch to radar transforms (yaw and roll unchanged)."""
    pitch_deg = radar_pitch_deg_from_env()
    for name, tr in list(radar_positions.items()):
        radar_positions[name] = carla.Transform(
            tr.location,
            carla.Rotation(pitch=pitch_deg, yaw=tr.rotation.yaw, roll=tr.rotation.roll),
        )
    return radar_positions


# ---------------------------------------------------------------------------
# Monitored-stretch rig placement.
#
# The SUMO-governed traffic runs along the east-west boulevard. Derived from the
# SUMO net + the runtime transform (carla_x = sumo_x - 109.34,
# carla_y = -(sumo_y - 135.96)):
#   - the boulevard is one straight road along CARLA +X made of edges
#     18 / 19 / 20 / 21 (WB) and -18 / -19 / -20 / -21 (EB),
#   - it runs from the stop line of perimeter junction 719 (west end of edge 18,
#     x = -88.6) to the stop line of perimeter junction 532 (east end of edge
#     21, x = 83.1), ~172 m,
#   - both carriageways span CARLA y in [5.4 (WB) .. 36.0 (EB)],
#   - cross-section mid ~y=20.7.
# The original 55 m stretch (edges 20 / -20 only) is x in [-28.7, 26.4].
#
# Layout per radar count:
#   - 8, 12, 14 radars: stations spread end to end over the FULL road
#     (FULL_STRETCH_X_WEST .. FULL_STRETCH_X_EAST). First and last stations sit
#     on the junction stop lines.
#   - 4 radars: kept on the original 55 m stretch (2 stations only, a 172 m
#     spread would leave nothing covered in the middle).
# The rig straddles the road: one radar row on the south kerb, one on the north
# kerb, N/2 stations along the length. Every number below is env-tunable so the
# rig can be nudged live in CARLA without editing code. DATASET_RIG_ANCHOR_X /
# DATASET_RIG_LENGTH_M override the per-count defaults for ALL counts.
# ---------------------------------------------------------------------------
FULL_STRETCH_X_WEST = -88.6    # west end of edge 18 (junction 719 stop line)
FULL_STRETCH_X_EAST = 83.1     # east end of edge 21 (junction 532 stop line)
FULL_STRETCH_MIN_RADARS = 8    # counts >= this use the full road

LEGACY_RIG_ANCHOR_X = -1.0     # original 55 m stretch mid X
LEGACY_RIG_LENGTH_M = 52.0     # original along-stretch coverage

RIG_ANCHOR_X = (FULL_STRETCH_X_WEST + FULL_STRETCH_X_EAST) / 2.0   # -2.75
RIG_ANCHOR_Y = 20.7        # boulevard cross-section mid Y (CARLA world)
RIG_HEADING_DEG = 0.0      # boulevard direction (0 = +X / east-west)
RIG_LENGTH_M = FULL_STRETCH_X_EAST - FULL_STRETCH_X_WEST          # 171.7
RIG_HALF_WIDTH_M = 20.5    # centre -> each radar row (rows ~y0.2 and ~y41.2)
CAM_HEIGHT_M = 6.5         # camera mount height
CAM_END_MARGIN_M = 16.0    # camera set-back beyond the stretch end
# The overview camera stays framed on the ORIGINAL 55 m stretch for every radar
# count (camera pipeline unchanged). Override with DATASET_CAM_ANCHOR_X /
# DATASET_CAM_STRETCH_LENGTH_M.
CAM_ANCHOR_X = LEGACY_RIG_ANCHOR_X
CAM_STRETCH_LENGTH_M = LEGACY_RIG_LENGTH_M

# Spawn exclusion zone. No car or pedestrian may spawn (or despawn) inside the
# radar-monitored road: the full stretch x range, between the two radar rows,
# grown by this margin on every side. Mirrored in traffic/ambient_traffic.py
# (the traffic package does not import dataset code) -- keep the two in sync.
SPAWN_EXCLUSION_MARGIN_M = 10.0


# Radar aiming. Every radar looks ACROSS the boulevard toward the far kerb and
# is skewed ALONG the stretch by RIG_SKEW_DEG so that all radars share one
# along-stretch look direction (RIG_LOOK_DIR). "east" = +heading (the direction
# the overview camera looks); "west" = -heading.
#   south row yaw = heading + (90 - skew)  [east]   or  heading + 90 + skew  [west]
#   north row yaw = heading - (90 - skew)  [east]   or  heading - 90 - skew  [west]
# With heading 0 / skew 40 this gives 50 / -50 (east) or 130 / -130 (west).
# This replaces the old per-radar compute_radar_yaw_toward_road() pass in the
# setup scripts: that pass picked between +40 and -40 by comparing angular
# distances that are an EXACT tie on a straight road, so which side won was
# decided by floating-point noise in the projected waypoint (R5/R6/R8 ended up
# facing west while R1/R3/R7/R2/R4 faced east in capture 20260917_125123).
RIG_LOOK_DIR = "east"
RIG_SKEW_DEG = 40.0


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def uses_full_stretch(count):
    """True when a rig of ``count`` radars spans the full road (see above)."""
    return count is None or int(count) >= FULL_STRETCH_MIN_RADARS


def rig_anchor_from_env(count=None):
    """(x, y) centre of the rig in CARLA world coords for a ``count``-radar rig
    (None = full-stretch default). Override with DATASET_RIG_ANCHOR_X /
    DATASET_RIG_ANCHOR_Y."""
    default_x = RIG_ANCHOR_X if uses_full_stretch(count) else LEGACY_RIG_ANCHOR_X
    return (_env_float("DATASET_RIG_ANCHOR_X", default_x),
            _env_float("DATASET_RIG_ANCHOR_Y", RIG_ANCHOR_Y))


def rig_heading_deg_from_env():
    return _env_float("DATASET_RIG_HEADING_DEG", RIG_HEADING_DEG)


def rig_length_m_from_env(count=None):
    default = RIG_LENGTH_M if uses_full_stretch(count) else LEGACY_RIG_LENGTH_M
    return max(1.0, _env_float("DATASET_RIG_LENGTH_M", default))


def rig_half_width_m_from_env():
    return max(1.0, _env_float("DATASET_RIG_HALF_WIDTH_M", RIG_HALF_WIDTH_M))


def rig_look_dir_from_env():
    raw = os.environ.get("DATASET_RIG_LOOK_DIR", "").strip().lower()
    if raw in ("east", "west"):
        return raw
    return RIG_LOOK_DIR


def rig_skew_deg_from_env():
    return max(0.0, min(_env_float("DATASET_RIG_SKEW_DEG", RIG_SKEW_DEG), 85.0))


def _normalize_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def stretch_radar_yaws(heading_deg=None, look_dir=None, skew_deg=None):
    """(south_row_yaw, north_row_yaw) in CARLA world degrees. Deterministic."""
    H = rig_heading_deg_from_env() if heading_deg is None else float(heading_deg)
    look = rig_look_dir_from_env() if look_dir is None else look_dir
    skew = rig_skew_deg_from_env() if skew_deg is None else float(skew_deg)
    if look == "west":
        south = H + 90.0 + skew
        north = H - 90.0 - skew
    else:
        south = H + 90.0 - skew
        north = H - 90.0 + skew
    return _normalize_deg(south), _normalize_deg(north)


def stretch_radar_positions(count, height=None):
    """Two-row straddle rig on the monitored stretch.

    8+ radars span the full road (junction 719 stop line to junction 532 stop
    line); 4 radars stay on the original 55 m stretch.

    Returns {"R1": carla.Transform, ...} with ``count`` radars: count//2 stations
    spaced along the stretch, each with one radar on the south kerb (odd R#) and
    one on the north kerb (even R#). Yaws are FINAL (see stretch_radar_yaws):
    every radar looks across the road toward the far kerb, skewed along the
    stretch so all radars share the same look direction. Pitch is applied
    afterward by apply_radar_pitch(). ``count`` must be even.
    """
    if count % 2 != 0:
        raise ValueError(f"radar count must be even (got {count})")
    ax, ay = rig_anchor_from_env(count)
    H = math.radians(rig_heading_deg_from_env())
    length = rig_length_m_from_env(count)
    half_w = rig_half_width_m_from_env()
    z = radar_height_m_from_env() if height is None else float(height)
    south_yaw, north_yaw = stretch_radar_yaws()

    dx, dy = math.cos(H), math.sin(H)      # along the stretch
    nx, ny = -math.sin(H), math.cos(H)     # across the stretch (toward north row)

    n_stations = max(1, count // 2)
    if n_stations == 1:
        offsets = [0.0]
    else:
        step = length / (n_stations - 1)
        offsets = [-length / 2.0 + i * step for i in range(n_stations)]

    positions = {}
    for i, t in enumerate(offsets):
        cx, cy = ax + t * dx, ay + t * dy
        south = (cx - half_w * nx, cy - half_w * ny)
        north = (cx + half_w * nx, cy + half_w * ny)
        s_id, n_id = 2 * i + 1, 2 * i + 2
        positions[f"R{s_id}"] = carla.Transform(
            carla.Location(x=south[0], y=south[1], z=z),
            carla.Rotation(0.0, south_yaw, 0.0),
        )
        positions[f"R{n_id}"] = carla.Transform(
            carla.Location(x=north[0], y=north[1], z=z),
            carla.Rotation(0.0, north_yaw, 0.0),
        )
    return positions


# Horizontal slew off the inward (toward-road) vector. Both rows keep the
# candidate closer to DATASET_RADAR_LOOK_HEADING_DEG (falls back to
# DATASET_RIG_HEADING_DEG) so they look down-stretch together. Override look
# heading to flip facing without moving the rows (180 = west on this boulevard).
RADAR_YAW_OFFSET_DEG = 40.0


def radar_yaw_offset_deg_from_env():
    return max(0.0, min(_env_float("DATASET_RADAR_YAW_OFFSET_DEG", RADAR_YAW_OFFSET_DEG), 80.0))


def radar_look_heading_deg_from_env():
    """Along-road direction every radar slews toward. Does not move the rig."""
    return _env_float("DATASET_RADAR_LOOK_HEADING_DEG", rig_heading_deg_from_env())


def _normalize_angle_deg(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def _angular_distance_deg(a_deg, b_deg):
    return abs(_normalize_angle_deg(a_deg - b_deg))


def _yaw_toward_road_deg(location, current_map):
    """CARLA yaw of the vector from the radar toward the road.

    Prefers the nearest driving waypoint (inward across the carriageways).
    If that snap is missing or on top of the radar, face the rig centerline.
    Never uses waypoint *heading* — that is along-road and unstable at junctions.
    """
    dx = dy = None
    if current_map is not None:
        wp = current_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is not None:
            dx = wp.transform.location.x - location.x
            dy = wp.transform.location.y - location.y
    if dx is None or dy is None or (abs(dx) < 1e-6 and abs(dy) < 1e-6):
        ax, ay = rig_anchor_from_env()
        heading = math.radians(rig_heading_deg_from_env())
        nx, ny = -math.sin(heading), math.cos(heading)
        across = (location.x - ax) * nx + (location.y - ay) * ny
        dx, dy = -across * nx, -across * ny
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return rig_heading_deg_from_env()
    return math.degrees(math.atan2(dy, dx))


def stretch_radar_yaw(location, current_map=None, offset_deg=None):
    """Look at the road, then slew ±offset along DATASET_RADAR_LOOK_HEADING_DEG.

    Default look heading 0 on this boulevard: south ~+50° (N+E), north ~-50° (S+E).
    Look heading 180: south ~+130° (N+W), north ~-130° (S+W). Rows stay put.
    """
    if offset_deg is None:
        offset_deg = radar_yaw_offset_deg_from_env()
    yaw_to_road = _yaw_toward_road_deg(location, current_map)
    plus = yaw_to_road + offset_deg
    minus = yaw_to_road - offset_deg
    look = radar_look_heading_deg_from_env()
    chosen = min((plus, minus), key=lambda c: _angular_distance_deg(c, look))
    return _normalize_angle_deg(chosen)


def apply_stretch_radar_yaws(radar_positions, current_map=None, offset_deg=None):
    """Overwrite each radar's yaw in-place; pitch and roll are left unchanged."""
    for name, tr in list(radar_positions.items()):
        new_yaw = stretch_radar_yaw(tr.location, current_map, offset_deg=offset_deg)
        radar_positions[name] = carla.Transform(
            tr.location,
            carla.Rotation(pitch=tr.rotation.pitch, yaw=new_yaw, roll=tr.rotation.roll),
        )
    return radar_positions


def radar_yaw_summary(radar_positions):
    def _key(name):
        try:
            return int(name.lstrip("R"))
        except ValueError:
            return name
    yaws = ", ".join(
        f"{n}={radar_positions[n].rotation.yaw:.1f}°"
        for n in sorted(radar_positions, key=_key)
    )
    return f"look={radar_look_heading_deg_from_env():.1f}° | {yaws}"


def stretch_camera_transform(height=None):
    """A single overview camera set back beyond the west end of the ORIGINAL
    55 m stretch,
    looking down its length. Returns the RAW transform the setup scripts expect
    (they apply a +180 deg yaw convention flip afterward), so the final camera
    boresight ends up along the stretch heading."""
    # Framed on the original 55 m stretch regardless of radar count.
    ax = _env_float("DATASET_CAM_ANCHOR_X", CAM_ANCHOR_X)
    ay = _env_float("DATASET_RIG_ANCHOR_Y", RIG_ANCHOR_Y)
    heading = rig_heading_deg_from_env()
    H = math.radians(heading)
    length = max(1.0, _env_float("DATASET_CAM_STRETCH_LENGTH_M", CAM_STRETCH_LENGTH_M))
    margin = _env_float("DATASET_CAM_END_MARGIN_M", CAM_END_MARGIN_M)
    z = _env_float("DATASET_CAM_HEIGHT_M", CAM_HEIGHT_M) if height is None else float(height)
    dx, dy = math.cos(H), math.sin(H)
    cx = ax - (length / 2.0 + margin) * dx
    cy = ay - (length / 2.0 + margin) * dy
    # Setup scripts do yaw = normalize(input + 180); we want final = heading.
    input_yaw = heading - 180.0
    return carla.Transform(
        carla.Location(x=cx, y=cy, z=z),
        carla.Rotation(pitch=-15.0, yaw=input_yaw, roll=0.0),
    )


def spawn_exclusion_margin_m_from_env():
    return max(0.0, _env_float("DATASET_SPAWN_EXCLUSION_MARGIN_M", SPAWN_EXCLUSION_MARGIN_M))


def spawn_exclusion_box(margin_m=None):
    """(x_min, x_max, y_min, y_max) in CARLA world coords: the full radar-
    monitored road between the two radar rows, grown by ``margin_m``.
    Always the FULL road, whatever the radar count, so spawn behaviour does not
    change with the rig."""
    m = spawn_exclusion_margin_m_from_env() if margin_m is None else float(margin_m)
    ay = _env_float("DATASET_RIG_ANCHOR_Y", RIG_ANCHOR_Y)
    half_w = rig_half_width_m_from_env()
    return (FULL_STRETCH_X_WEST - m, FULL_STRETCH_X_EAST + m,
            ay - half_w - m, ay + half_w + m)


def in_spawn_exclusion_zone(x, y, margin_m=None):
    """True if CARLA world point (x, y) is inside the spawn exclusion zone."""
    x0, x1, y0, y1 = spawn_exclusion_box(margin_m)
    return x0 <= x <= x1 and y0 <= y <= y1
