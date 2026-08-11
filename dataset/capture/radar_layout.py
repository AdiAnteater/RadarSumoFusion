"""Shared radar placement helpers (no imports from other capture modules)."""

import math
import os

import carla

# Down-tilt toward road traffic. CARLA radars use positive pitch to look down
# (see PythonAPI/util/raycast_sensor_testing.py: Rotation(pitch=5) on radar mounts).
RADAR_PITCH_DEG = 8.0


def radar_pitch_deg_from_env() -> float:
    raw = os.environ.get("DATASET_RADAR_PITCH_DEG", "").strip()
    if raw:
        try:
            return max(-30.0, min(float(raw), 30.0))
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
# The SUMO-governed traffic runs through the monitored stretch = the east-west
# boulevard (net edges 20 / -20). Derived from the SUMO net + the runtime
# transform (carla_x = sumo_x - 109.34, carla_y = -(sumo_y - 135.96)):
#   - the boulevard runs along CARLA +X, x in [-28.7, 26.4] (~55 m),
#   - both carriageways span CARLA y in [5.4 (WB/edge20) .. 36.0 (EB/edge-20)],
#   - cross-section mid ~y=20.7, along-stretch mid ~x=-1.0.
# The rig straddles the WHOLE boulevard: one radar row on the south kerb, one on
# the north kerb, N/2 stations along the length -- the same two-row pattern the
# old rig used, moved onto the stretch. Every number below is env-tunable so the
# rig can be nudged live in CARLA without editing code.
# ---------------------------------------------------------------------------
RIG_ANCHOR_X = -1.0        # stretch mid X (CARLA world)
RIG_ANCHOR_Y = 20.7        # boulevard cross-section mid Y (CARLA world)
RIG_HEADING_DEG = 0.0      # boulevard direction (0 = +X / east-west)
RIG_LENGTH_M = 52.0        # along-stretch coverage (stations spread over this)
RIG_HALF_WIDTH_M = 20.5    # centre -> each radar row (rows ~y0.2 and ~y41.2)
CAM_HEIGHT_M = 6.5         # camera mount height
CAM_END_MARGIN_M = 16.0    # camera set-back beyond the stretch end


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def rig_anchor_from_env():
    """(x, y) centre of the rig in CARLA world coords.
    Override with DATASET_RIG_ANCHOR_X / DATASET_RIG_ANCHOR_Y."""
    return (_env_float("DATASET_RIG_ANCHOR_X", RIG_ANCHOR_X),
            _env_float("DATASET_RIG_ANCHOR_Y", RIG_ANCHOR_Y))


def rig_heading_deg_from_env():
    return _env_float("DATASET_RIG_HEADING_DEG", RIG_HEADING_DEG)


def rig_length_m_from_env():
    return max(1.0, _env_float("DATASET_RIG_LENGTH_M", RIG_LENGTH_M))


def rig_half_width_m_from_env():
    return max(1.0, _env_float("DATASET_RIG_HALF_WIDTH_M", RIG_HALF_WIDTH_M))


def stretch_radar_positions(count, height=None):
    """Two-row straddle rig on the monitored stretch.

    Returns {"R1": carla.Transform, ...} with ``count`` radars: count//2 stations
    spaced along the stretch, each with one radar on the south kerb (odd R#, base
    yaw 0) and one on the north kerb (even R#, base yaw 180). These base yaws only
    SEED the setup scripts' compute_radar_yaw_toward_road() pass, which then aims
    each radar at the real lane direction from the map -- so moving the rig here
    auto-reorients it to the boulevard. ``count`` must be even.
    """
    if count % 2 != 0:
        raise ValueError(f"radar count must be even (got {count})")
    ax, ay = rig_anchor_from_env()
    H = math.radians(rig_heading_deg_from_env())
    length = rig_length_m_from_env()
    half_w = rig_half_width_m_from_env()
    z = radar_height_m_from_env() if height is None else float(height)

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
            carla.Rotation(0.0, 0.0, 0.0),
        )
        positions[f"R{n_id}"] = carla.Transform(
            carla.Location(x=north[0], y=north[1], z=z),
            carla.Rotation(0.0, 180.0, 0.0),
        )
    return positions


def stretch_north_row_names(count):
    """North-kerb radar names (even R#). These get the opposite +/-40 deg cone
    side in the setup scripts, mirroring the south row -- replaces the old
    hand-tuned per-name flip sets with a symmetric rule."""
    return {f"R{2 * i + 2}" for i in range(max(1, count // 2))}


def stretch_camera_transform(height=None):
    """A single overview camera set back beyond the start end of the stretch,
    looking down its length. Returns the RAW transform the setup scripts expect
    (they apply a +180 deg yaw convention flip afterward), so the final camera
    boresight ends up along the stretch heading."""
    ax, ay = rig_anchor_from_env()
    heading = rig_heading_deg_from_env()
    H = math.radians(heading)
    length = rig_length_m_from_env()
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
