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
    apply_radar_pitch,
    configure_dataset_radar_blueprint,
    destroy_dataset_radars,
)
from dataset_paths import config_dir
from capture.radar_layout import (
    apply_stretch_radar_yaws,
    radar_yaw_summary,
    stretch_radar_positions,
    stretch_camera_transform,
)

KEEP_SENSORS_RUNNING = os.environ.get("DATASET_KEEP_SENSORS_RUNNING", "").lower() in (
    "1",
    "true",
    "yes",
)

DRAW_DEBUG_MARKERS = True
DISABLE_CAMERA_POSTPROCESS_EFFECTS = True
DATASET_RADAR_ROLE_PREFIX = "dataset_radar_"
DATASET_CAMERA_ROLE_PREFIX = "dataset_camera_"


def make_transform(x, y, z, pitch, yaw, roll):
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
    )


def format_transform(transform):
    loc = transform.location
    rot = transform.rotation
    return (
        "Transform("
        f"Location(x={loc.x:.6f}, y={loc.y:.6f}, z={loc.z:.6f}), "
        f"Rotation(pitch={rot.pitch:.6f}, yaw={rot.yaw:.6f}, roll={rot.roll:.6f})"
        ")"
    )


def normalize_angle(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def radar_debug_color_for_name(name, num_radars=12):
    """Distinct hue per radar (R1..R12) for debug visualization."""
    try:
        n = int(name.lstrip("R"))
        idx = (n - 1) % max(num_radars, 1)
    except ValueError:
        idx = 0
    h = idx / float(max(num_radars, 1))
    r, g, b = colorsys.hsv_to_rgb(h, 0.88, 1.0)
    return carla.Color(int(r * 255), int(g * 255), int(b * 255))


def draw_radar_fov(world, transform, radar_range, horizontal_fov_deg, life_time, color=None):
    if color is None:
        color = carla.Color(255, 180, 0)
    origin = transform.location + carla.Location(z=0.2)
    center_yaw = math.radians(transform.rotation.yaw)
    half_fov = math.radians(horizontal_fov_deg / 2.0)

    for yaw in [center_yaw - half_fov, center_yaw, center_yaw + half_fov]:
        end = carla.Location(
            x=origin.x + radar_range * math.cos(yaw),
            y=origin.y + radar_range * math.sin(yaw),
            z=origin.z,
        )
        world.debug.draw_line(
            origin,
            end,
            thickness=0.06,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )

    arc_points = []
    segments = 10
    for i in range(segments + 1):
        t = i / segments
        yaw = (center_yaw - half_fov) + t * (2.0 * half_fov)
        arc_points.append(
            carla.Location(
                x=origin.x + radar_range * math.cos(yaw),
                y=origin.y + radar_range * math.sin(yaw),
                z=origin.z,
            )
        )
    for i in range(len(arc_points) - 1):
        world.debug.draw_line(
            arc_points[i],
            arc_points[i + 1],
            thickness=0.04,
            color=color,
            life_time=life_time,
            persistent_lines=False,
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
        world.debug.draw_line(
            origin,
            end,
            thickness=0.06,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )

    arc_points = []
    segments = 12
    for i in range(segments + 1):
        t = i / segments
        yaw = (center_yaw - half_fov) + t * (2.0 * half_fov)
        arc_points.append(
            carla.Location(
                x=origin.x + camera_range * math.cos(yaw),
                y=origin.y + camera_range * math.sin(yaw),
                z=origin.z,
            )
        )
    for i in range(len(arc_points) - 1):
        world.debug.draw_line(
            arc_points[i],
            arc_points[i + 1],
            thickness=0.04,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )


def unit_xy(dx, dy):
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return 0.0, 0.0
    return dx / mag, dy / mag


def yaw_from_vector(dx, dy, fallback_yaw):
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback_yaw
    return math.degrees(math.atan2(dy, dx))


def orient_pair_like_diagram(radar_positions, upper_pair, lower_pair):
    """
    Make each roadside pair look inward and slightly toward the opposite pair,
    matching the diagonal-crossing style in the user diagram.
    """
    upper_center = carla.Location(
        x=(radar_positions[upper_pair[0]].location.x + radar_positions[upper_pair[1]].location.x) * 0.5,
        y=(radar_positions[upper_pair[0]].location.y + radar_positions[upper_pair[1]].location.y) * 0.5,
        z=(radar_positions[upper_pair[0]].location.z + radar_positions[upper_pair[1]].location.z) * 0.5,
    )
    lower_center = carla.Location(
        x=(radar_positions[lower_pair[0]].location.x + radar_positions[lower_pair[1]].location.x) * 0.5,
        y=(radar_positions[lower_pair[0]].location.y + radar_positions[lower_pair[1]].location.y) * 0.5,
        z=(radar_positions[lower_pair[0]].location.z + radar_positions[lower_pair[1]].location.z) * 0.5,
    )

    # Match the figure: upper row points down/inward, lower row points up/inward.
    # For each radar, aim toward the opposite row center.
    row_targets = {
        upper_pair[0]: lower_center,
        upper_pair[1]: lower_center,
        lower_pair[0]: upper_center,
        lower_pair[1]: upper_center,
    }

    for name, opposite_row_center in row_targets.items():
        tr = radar_positions[name]
        loc = tr.location
        aim_dx, aim_dy = unit_xy(
            opposite_row_center.x - loc.x, opposite_row_center.y - loc.y
        )
        new_yaw = yaw_from_vector(aim_dx, aim_dy, tr.rotation.yaw)
        radar_positions[name] = carla.Transform(
            tr.location,
            carla.Rotation(pitch=tr.rotation.pitch, yaw=new_yaw, roll=tr.rotation.roll),
        )


def main():
    client, world = get_world()
    try:
        removed = destroy_dataset_radars(world)
        if removed:
            print(f"Removed {removed} stale dataset radar(s) from a prior run.")
    except ImportError:
        pass

    bp_lib = world.get_blueprint_library()
    current_map = world.get_map()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    if DISABLE_CAMERA_POSTPROCESS_EFFECTS:
        if camera_bp.has_attribute("enable_postprocess_effects"):
            camera_bp.set_attribute("enable_postprocess_effects", "false")
        if camera_bp.has_attribute("bloom_intensity"):
            camera_bp.set_attribute("bloom_intensity", "0.0")
        if camera_bp.has_attribute("lens_flare_intensity"):
            camera_bp.set_attribute("lens_flare_intensity", "0.0")
    radar_bp = bp_lib.find("sensor.other.radar")
    radar_pps = configure_dataset_radar_blueprint(radar_bp)
    print(f"Radar blueprint: points_per_second={radar_pps}")

    # Camera: C10
    # Overview camera set back beyond the stretch, looking down the boulevard.
    camera_positions = {"C10": stretch_camera_transform()}
    # Flip camera look direction 180° about vertical (yaw), keep pitch/roll.
    for name, tr in camera_positions.items():
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

    # Radars (manual placement, evenly spaced over 111.54 m stretch).
    # Stretch: x=-33.825321 to x=77.715952 (6 intervals => ~18.590212 m spacing).
    # Radars straddle the monitored stretch (SUMO edges 20/-20); env-tunable.
    # Coordinates from the shared helper in capture/radar_layout.py.
    radar_positions = stretch_radar_positions(12)
    # Yaws are final and deterministic (capture/radar_layout.py:
    # stretch_radar_yaws). The former compute_radar_yaw_toward_road() pass
    # and the R1 := R2+90 hack are gone: the pass tie-broke +/-40 deg on
    # floating-point noise, which is why R5/R6/R8 faced the other way.
    # Override with DATASET_RIG_LOOK_DIR=east|west, DATASET_RIG_SKEW_DEG.
    apply_radar_pitch(radar_positions)
    print(f"Radar yaws: {radar_yaw_summary(radar_positions)}")

    aligned_radar_names = ["R2", "R1", "R4", "R3"]
    aligned_output_path = str(config_dir() / "aligned_radar_points.txt")
    with open(aligned_output_path, "w", encoding="utf-8") as f:
        f.write("Aligned radar points used in DatasetCreation.py\n\n")
        for name in aligned_radar_names:
            f.write(f"{name}: {format_transform(radar_positions[name])}\n")
    print(f"Saved aligned radar transforms to {aligned_output_path}")

    spawned = []
    camera_markers = []
    radar_markers = []
    marker_lifetime = 1.0
    marker_refresh_s = 0.25
    radar_range = (
        float(radar_bp.get_attribute("range").as_float())
        if radar_bp.has_attribute("range")
        else 30.0
    )
    radar_hfov = (
        float(radar_bp.get_attribute("horizontal_fov").as_float())
        if radar_bp.has_attribute("horizontal_fov")
        else 40.0
    )
    camera_range = 90.0
    camera_hfov = (
        float(camera_bp.get_attribute("fov").as_float())
        if camera_bp.has_attribute("fov")
        else 90.0
    )
    try:
        for name, transform in camera_positions.items():
            if camera_bp.has_attribute("role_name"):
                camera_bp.set_attribute("role_name", f"{DATASET_CAMERA_ROLE_PREFIX}{name}")
            actor = world.try_spawn_actor(camera_bp, transform)
            if actor is None:
                print(f"Failed to spawn camera at {name}: {transform}")
                continue
            spawned.append(actor)
            camera_markers.append((name, transform))
            role_name = actor.attributes.get("role_name", "")
            print(f"Spawned camera {name} -> actor_id={actor.id} role_name={role_name}")

        for name, transform in radar_positions.items():
            if radar_bp.has_attribute("role_name"):
                radar_bp.set_attribute("role_name", f"{DATASET_RADAR_ROLE_PREFIX}{name}")
            actor = world.try_spawn_actor(radar_bp, transform)
            if actor is None:
                print(f"Failed to spawn radar at {name}: {transform}")
                continue
            spawned.append(actor)
            radar_markers.append((name, transform))
            role_name = actor.attributes.get("role_name", "")
            print(f"Spawned radar {name} -> actor_id={actor.id} role_name={role_name}")

        print(f"Total sensors spawned: {len(spawned)}")
        if DRAW_DEBUG_MARKERS:
            print("Markers active: blue=CAM; each radar has its own color.")
        else:
            print("Debug markers disabled (clean camera capture mode).")
        if KEEP_SENSORS_RUNNING:
            print(
                "Sensors running (dataset test mode). "
                "Stop with Start.py Ctrl+C — do not press Enter in this console."
            )
        else:
            print("Press Enter to destroy sensors and clear markers...")

        while True:
            if DRAW_DEBUG_MARKERS:
                for name, transform in camera_markers:
                    world.debug.draw_point(
                        transform.location,
                        size=0.2,
                        color=carla.Color(0, 120, 255),
                        life_time=marker_lifetime,
                        persistent_lines=False,
                    )
                    world.debug.draw_string(
                        transform.location + carla.Location(z=0.35),
                        f"CAM {name}",
                        draw_shadow=False,
                        color=carla.Color(0, 120, 255),
                        life_time=marker_lifetime,
                        persistent_lines=False,
                    )
                    draw_camera_fov(
                        world,
                        transform,
                        camera_range=camera_range,
                        horizontal_fov_deg=camera_hfov,
                        life_time=marker_lifetime,
                    )

                for name, transform in radar_markers:
                    rcolor = radar_debug_color_for_name(name)
                    world.debug.draw_point(
                        transform.location,
                        size=0.2,
                        color=rcolor,
                        life_time=marker_lifetime,
                        persistent_lines=False,
                    )
                    world.debug.draw_string(
                        transform.location + carla.Location(z=0.35),
                        f"RAD {name}",
                        draw_shadow=False,
                        color=rcolor,
                        life_time=marker_lifetime,
                        persistent_lines=False,
                    )
                    draw_radar_fov(
                        world,
                        transform,
                        radar_range=radar_range,
                        horizontal_fov_deg=radar_hfov,
                        life_time=marker_lifetime,
                        color=rcolor,
                    )

            if not KEEP_SENSORS_RUNNING and enter_pressed():
                break

            time.sleep(marker_refresh_s)

    finally:
        for actor in spawned:
            if actor.is_alive:
                actor.destroy()
        print("Destroyed spawned sensors. Markers will fade out in about 1 second.")


if __name__ == "__main__":
    main()
