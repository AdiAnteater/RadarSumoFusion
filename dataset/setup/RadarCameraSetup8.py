import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import colorsys
import math
from _kbhit_compat import enter_pressed
import os
import time

import carla

from carla_connect import get_world
from capture.CaptureRadarCameraData import (
    RADAR_HORIZONTAL_FOV_DEG,
    RADAR_MAX_RANGE_M,
    RADAR_VERTICAL_FOV_DEG,
    apply_radar_pitch,
    configure_dataset_radar_blueprint,
    destroy_dataset_radars,
    radar_sensor_tick_s_from_env,
)
from capture.radar_layout import (
    radar_pitch_deg_from_env,
    radar_height_m_from_env,
    stretch_radar_positions,
    stretch_north_row_names,
    stretch_camera_transform,
)
from dataset_paths import config_dir

KEEP_SENSORS_RUNNING = os.environ.get("DATASET_KEEP_SENSORS_RUNNING", "").lower() in (
    "1",
    "true",
    "yes",
)

DRAW_DEBUG_MARKERS = True
DATASET_RADAR_ROLE_PREFIX = "dataset_radar_"
DATASET_CAMERA_ROLE_PREFIX = "dataset_camera_"
DISABLE_CAMERA_POSTPROCESS_EFFECTS = True


def make_transform(x, y, z, pitch, yaw, roll):
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
    )


def normalize_angle(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def angular_distance(a_deg, b_deg):
    return abs(normalize_angle(a_deg - b_deg))


def compute_radar_yaw_toward_road(
    current_map, location, fallback_yaw, offset_deg=40.0, use_opposite_side=False
):
    road_wp = current_map.get_waypoint(
        location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if road_wp is None:
        return fallback_yaw

    road_loc = road_wp.transform.location
    dx = road_loc.x - location.x
    dy = road_loc.y - location.y

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        yaw_to_road = road_wp.transform.rotation.yaw
    else:
        yaw_to_road = math.degrees(math.atan2(dy, dx))

    candidates = [yaw_to_road + offset_deg, yaw_to_road - offset_deg]
    lane_yaws = [road_wp.transform.rotation.yaw, road_wp.transform.rotation.yaw + 180.0]

    chosen = min(candidates, key=lambda c: min(angular_distance(c, ly) for ly in lane_yaws))

    if use_opposite_side:
        chosen = candidates[1] if abs(normalize_angle(chosen - candidates[0])) < 1e-6 else candidates[0]

    return normalize_angle(chosen)


def radar_debug_color_for_name(name, num_radars=8):
    try:
        n = int(name.lstrip("R"))
        idx = (n - 1) % num_radars
    except:
        idx = 0

    h = idx / float(num_radars)
    r, g, b = colorsys.hsv_to_rgb(h, 0.88, 1.0)
    return carla.Color(int(r * 255), int(g * 255), int(b * 255))


def draw_radar_fov(
    world,
    transform,
    radar_range,
    horizontal_fov_deg,
    vertical_fov_deg,
    life_time,
    color,
):
    origin = transform.location + carla.Location(z=0.2)
    center_yaw = math.radians(transform.rotation.yaw)
    half_hfov = math.radians(horizontal_fov_deg / 2.0)

    for yaw in [center_yaw - half_hfov, center_yaw, center_yaw + half_hfov]:
        end = carla.Location(
            x=origin.x + radar_range * math.cos(yaw),
            y=origin.y + radar_range * math.sin(yaw),
            z=origin.z,
        )
        world.debug.draw_line(origin, end, 0.06, color, life_time)

    # Horizontal arc (ground plane)
    arc_points = []
    for i in range(11):
        t = i / 10
        yaw = (center_yaw - half_hfov) + t * (2 * half_hfov)
        arc_points.append(
            carla.Location(
                x=origin.x + radar_range * math.cos(yaw),
                y=origin.y + radar_range * math.sin(yaw),
                z=origin.z,
            )
        )

    for i in range(len(arc_points) - 1):
        world.debug.draw_line(arc_points[i], arc_points[i + 1], 0.04, color, life_time)

    # Vertical slice through center azimuth (sensor-local XZ plane).
    sensor_origin = transform.location
    half_vfov = vertical_fov_deg / 2.0
    v_color = carla.Color(
        min(255, color.r + 40),
        min(255, color.g + 40),
        min(255, color.b + 40),
    )

    for elevation_deg in [-half_vfov, 0.0, half_vfov]:
        el_rad = math.radians(elevation_deg)
        local = carla.Location(
            x=radar_range * math.cos(el_rad),
            y=0.0,
            z=radar_range * math.sin(el_rad),
        )
        end = transform.transform(local)
        world.debug.draw_line(sensor_origin, end, 0.06, v_color, life_time)

    vertical_arc = []
    for i in range(11):
        t = i / 10
        elevation_deg = (-half_vfov) + t * (2 * half_vfov)
        el_rad = math.radians(elevation_deg)
        local = carla.Location(
            x=radar_range * math.cos(el_rad),
            y=0.0,
            z=radar_range * math.sin(el_rad),
        )
        vertical_arc.append(transform.transform(local))

    for i in range(len(vertical_arc) - 1):
        world.debug.draw_line(vertical_arc[i], vertical_arc[i + 1], 0.04, v_color, life_time)


def format_transform(transform):
    loc = transform.location
    rot = transform.rotation
    return (
        "Transform("
        f"Location(x={loc.x:.6f}, y={loc.y:.6f}, z={loc.z:.6f}), "
        f"Rotation(pitch={rot.pitch:.6f}, yaw={rot.yaw:.6f}, roll={rot.roll:.6f})"
        ")"
    )


def draw_camera_fov(world, transform, camera_range, horizontal_fov_deg, life_time):
    color = carla.Color(80, 220, 255)
    origin = transform.location + carla.Location(z=0.1)
    center_yaw = math.radians(transform.rotation.yaw)
    half_fov = math.radians(horizontal_fov_deg / 2.0)

    for yaw in [center_yaw - half_fov, center_yaw, center_yaw + half_fov]:
        end = carla.Location(
            x=origin.x + camera_range * math.cos(yaw),
            y=origin.y + camera_range * math.sin(yaw),
            z=origin.z,
        )
        world.debug.draw_line(origin, end, 0.06, color, life_time)

    arc_points = []
    for i in range(12):
        t = i / 11
        yaw = (center_yaw - half_fov) + t * (2 * half_fov)
        arc_points.append(
            carla.Location(
                x=origin.x + camera_range * math.cos(yaw),
                y=origin.y + camera_range * math.sin(yaw),
                z=origin.z,
            )
        )
    for i in range(len(arc_points) - 1):
        world.debug.draw_line(arc_points[i], arc_points[i + 1], 0.04, color, life_time)


def main():
    client, world = get_world()
    try:
        removed = destroy_dataset_radars(world)
        if removed:
            print(f"Removed {removed} stale dataset radar(s) from a prior run.")
    except ImportError:
        pass

    current_map = world.get_map()
    bp_lib = world.get_blueprint_library()

    camera_bp = bp_lib.find("sensor.camera.rgb")
    if DISABLE_CAMERA_POSTPROCESS_EFFECTS:
        if camera_bp.has_attribute("enable_postprocess_effects"):
            camera_bp.set_attribute("enable_postprocess_effects", "false")
        if camera_bp.has_attribute("bloom_intensity"):
            camera_bp.set_attribute("bloom_intensity", "0.0")
        if camera_bp.has_attribute("lens_flare_intensity"):
            camera_bp.set_attribute("lens_flare_intensity", "0.0")

    # Overview camera set back beyond the stretch, looking down the boulevard.
    camera_positions = {"C10": stretch_camera_transform()}
    for name, tr in list(camera_positions.items()):
        camera_positions[name] = carla.Transform(
            tr.location,
            carla.Rotation(
                pitch=tr.rotation.pitch,
                yaw=normalize_angle(tr.rotation.yaw + 180.0),
                roll=tr.rotation.roll,
            ),
        )
    camera_output_path = str(config_dir() / "camera_points_dataset.txt")
    with open(camera_output_path, "w", encoding="utf-8") as f:
        f.write("Camera points used in DatasetCreation.py\n\n")
        for name, transform in camera_positions.items():
            f.write(f"{name}: {format_transform(transform)}\n")
    print(f"Saved camera transforms to {camera_output_path}")

    radar_bp = bp_lib.find("sensor.other.radar")
    radar_pps = configure_dataset_radar_blueprint(radar_bp)
    print(
        f"Radar blueprint: range={int(RADAR_MAX_RANGE_M)} m, "
        f"HFOV={int(RADAR_HORIZONTAL_FOV_DEG)}°, "
        f"VFOV={int(RADAR_VERTICAL_FOV_DEG)}°, "
        f"pitch={radar_pitch_deg_from_env():.1f}°, height={radar_height_m_from_env():.1f}m, "
        f"points_per_second={radar_pps}, sensor_tick={radar_sensor_tick_s_from_env():g}",
    )

    # ~5 m from the nearest driving lane (was ~11 m — weak returns at shallow grazing angle).
    rh = radar_height_m_from_env()   # mounting height (m); DATASET_RIG_HEIGHT_M (default 3.0)
    # Radars straddle the monitored stretch (SUMO edges 20/-20). All coordinates
    # come from the shared, env-tunable helper in capture/radar_layout.py; the
    # compute_radar_yaw_toward_road pass below then aims each radar at the road.
    radar_positions = stretch_radar_positions(8, height=rh)

    # North-kerb row takes the opposite +/-40 deg cone (symmetric mirror rule).
    flipped_40_deg_names = stretch_north_row_names(8)

    for name in radar_positions:
        tr = radar_positions[name]
        new_yaw = compute_radar_yaw_toward_road(
            current_map,
            tr.location,
            tr.rotation.yaw,
            offset_deg=40.0,
            use_opposite_side=name in flipped_40_deg_names,
        )
        radar_positions[name] = carla.Transform(
            tr.location,
            carla.Rotation(tr.rotation.pitch, new_yaw, tr.rotation.roll),
        )

    # Same first-column fix as RadarCameraSetup12.py: R1/R2 share x; copy R2's aligned yaw to R1.
    tr1 = radar_positions["R1"]
    tr2 = radar_positions["R2"]
    r1_yaw = normalize_angle(tr2.rotation.yaw + 90)
    radar_positions["R1"] = carla.Transform(
        tr1.location,
        carla.Rotation(tr1.rotation.pitch, r1_yaw, tr1.rotation.roll),
    )

    apply_radar_pitch(radar_positions)

    camera_hfov = (
        float(camera_bp.get_attribute("fov").as_float())
        if camera_bp.has_attribute("fov")
        else 90.0
    )
    radar_range = (
        float(radar_bp.get_attribute("range").as_float())
        if radar_bp.has_attribute("range")
        else RADAR_MAX_RANGE_M
    )
    radar_hfov = (
        float(radar_bp.get_attribute("horizontal_fov").as_float())
        if radar_bp.has_attribute("horizontal_fov")
        else RADAR_HORIZONTAL_FOV_DEG
    )
    radar_vfov = (
        float(radar_bp.get_attribute("vertical_fov").as_float())
        if radar_bp.has_attribute("vertical_fov")
        else RADAR_VERTICAL_FOV_DEG
    )
    camera_range = 90.0
    camera_debug = list(camera_positions.items())

    spawned = []

    try:
        for name, transform in camera_positions.items():
            if camera_bp.has_attribute("role_name"):
                camera_bp.set_attribute("role_name", f"{DATASET_CAMERA_ROLE_PREFIX}{name}")
            actor = world.try_spawn_actor(camera_bp, transform)
            if actor is None:
                print(f"Failed to spawn camera at {name}: {transform}")
            else:
                spawned.append(actor)
                print(f"Spawned camera {name}")

        for name, transform in radar_positions.items():
            if radar_bp.has_attribute("role_name"):
                radar_bp.set_attribute("role_name", f"{DATASET_RADAR_ROLE_PREFIX}{name}")
            actor = world.try_spawn_actor(radar_bp, transform)

            if actor:
                spawned.append(actor)
                attrs = actor.attributes
                role_name = attrs.get("role_name", "")
                tr = actor.get_transform()
                print(
                    f"Spawned {name} -> id={actor.id} role={role_name!r} "
                    f"pps={attrs.get('points_per_second', '?')} "
                    f"hfov={attrs.get('horizontal_fov', '?')} "
                    f"vfov={attrs.get('vertical_fov', '?')} "
                    f"tick={attrs.get('sensor_tick', '?')} "
                    f"pitch={tr.rotation.pitch:.1f}°"
                )
            else:
                print(f"Failed to spawn radar at {name}: {transform}")

        if KEEP_SENSORS_RUNNING:
            print(
                "Sensors running (dataset test mode). "
                "Stop with Start.py Ctrl+C — do not press Enter in this console."
            )
        else:
            print("Press ENTER to exit...")

        while True:
            if DRAW_DEBUG_MARKERS:
                for name, transform in camera_debug:
                    world.debug.draw_point(
                        transform.location,
                        size=0.2,
                        color=carla.Color(0, 120, 255),
                        life_time=0.5,
                    )
                    world.debug.draw_string(
                        transform.location + carla.Location(z=0.5),
                        f"CAM {name}",
                        draw_shadow=False,
                        color=carla.Color(0, 120, 255),
                        life_time=0.5,
                    )
                    draw_camera_fov(
                        world, transform, camera_range, camera_hfov, 0.5
                    )
            for name, transform in radar_positions.items():
                color = radar_debug_color_for_name(name)

                world.debug.draw_point(
                    transform.location,
                    size=0.2,
                    color=color,
                    life_time=0.5,
                )

                world.debug.draw_string(
                    transform.location + carla.Location(z=0.5),
                    f"RAD {name}",
                    draw_shadow=False,
                    color=color,
                    life_time=0.5,
                )

                draw_radar_fov(
                    world,
                    transform,
                    radar_range,
                    radar_hfov,
                    radar_vfov,
                    0.5,
                    color,
                )

            if not KEEP_SENSORS_RUNNING and enter_pressed():
                break

            time.sleep(0.25)

    finally:
        for actor in spawned:
            actor.destroy()
        print("Cleaned up sensors")


if __name__ == "__main__":
    main()