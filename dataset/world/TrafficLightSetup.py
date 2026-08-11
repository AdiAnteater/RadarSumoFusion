import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import math
import os
import time

import carla

from carla_connect import get_world


PERIMETER_MARGIN_M = 25.0
TURN_ANGLE_THRESHOLD_DEG = 35.0
TURN_LOOKAHEAD_M = 12.0
# Anchor near CAM C10 from your dataset setup.
C10_ANCHOR_LOCATION = carla.Location(x=-45.741562, y=-68.056618, z=0.0)
LABEL_REFRESH_S = 0.25
LABEL_LIFETIME_S = 1.0
LABEL_Z_OFFSET_M = 2.5
# Standard phase lengths so unfrozen lights actually cycle in CARLA.
DEFAULT_GREEN_TIME_S = 12.0
DEFAULT_YELLOW_TIME_S = 2.0
DEFAULT_RED_TIME_S = 12.0


def yaw_delta_deg(target_yaw, current_yaw):
    return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0


def get_driving_bounds(world_map, waypoint_step=8.0):
    waypoints = world_map.generate_waypoints(waypoint_step)
    if not waypoints:
        raise RuntimeError("No driving waypoints found; cannot compute map bounds.")

    xs = [wp.transform.location.x for wp in waypoints]
    ys = [wp.transform.location.y for wp in waypoints]
    return min(xs), max(xs), min(ys), max(ys)


def is_perimeter_light(light, min_x, max_x, min_y, max_y, margin_m):
    """
    Treat a light as perimeter if any affected lane lies close to the map edge.

    If CARLA returns no stop waypoints for this light, do not classify as inner (that would force
    red on every such light and effectively kill the junction network).
    """
    stop_wps = light.get_stop_waypoints()
    if not stop_wps:
        return True

    for wp in stop_wps:
        loc = wp.transform.location
        if (
            loc.x <= (min_x + margin_m)
            or loc.x >= (max_x - margin_m)
            or loc.y <= (min_y + margin_m)
            or loc.y >= (max_y - margin_m)
        ):
            return True
    return False


def light_controls_turning_movement(
    light,
    angle_threshold_deg=TURN_ANGLE_THRESHOLD_DEG,
    lookahead_m=TURN_LOOKAHEAD_M,
):
    """
    Return True if any stop waypoint controlled by this light diverges into a turn.
    """
    stop_wps = light.get_stop_waypoints()
    for stop_wp in stop_wps:
        # At/after the stop line inside junction, next() often contains route branches.
        for next_wp in stop_wp.next(lookahead_m):
            delta = abs(yaw_delta_deg(next_wp.transform.rotation.yaw, stop_wp.transform.rotation.yaw))
            if angle_threshold_deg <= delta <= (180.0 - angle_threshold_deg):
                return True
    return False


def distance_sq(loc_a, loc_b):
    dx = loc_a.x - loc_b.x
    dy = loc_a.y - loc_b.y
    dz = loc_a.z - loc_b.z
    return dx * dx + dy * dy + dz * dz


def reset_light_phase_times(
    light,
    *,
    green_s=DEFAULT_GREEN_TIME_S,
    yellow_s=DEFAULT_YELLOW_TIME_S,
    red_s=DEFAULT_RED_TIME_S,
):
    light.set_green_time(green_s)
    light.set_yellow_time(yellow_s)
    light.set_red_time(red_s)


def set_light_near_location_always_green(lights, anchor_location):
    if not lights:
        return None

    selected = min(lights, key=lambda light: distance_sq(light.get_location(), anchor_location))
    selected.set_state(carla.TrafficLightState.Green)
    selected.set_green_time(99999.0)
    selected.set_yellow_time(0.1)
    selected.set_red_time(0.1)
    selected.freeze(True)
    return selected


def color_for_light_state(state):
    if state == carla.TrafficLightState.Red:
        return carla.Color(255, 80, 80)
    if state == carla.TrafficLightState.Yellow:
        return carla.Color(255, 220, 0)
    if state == carla.TrafficLightState.Green:
        return carla.Color(80, 255, 80)
    return carla.Color(180, 180, 180)


def label_all_traffic_lights(world, always_green_light_id=None):
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    for light in lights:
        state = light.get_state()
        extra = " | ALWAYS_GREEN" if light.id == always_green_light_id else ""
        world.debug.draw_string(
            light.get_location() + carla.Location(z=LABEL_Z_OFFSET_M),
            f"TL {light.id} | {state}{extra}",
            draw_shadow=False,
            color=color_for_light_state(state),
            life_time=LABEL_LIFETIME_S,
            persistent_lines=False,
        )
    return len(lights)


def automatic_traffic_lights_from_env() -> bool:
    """When true, unfreeze all lights. Default off — legacy perimeter rules work better in Town10HD."""
    raw = os.environ.get("DATASET_AUTOMATIC_TRAFFIC_LIGHTS", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def force_all_green_from_env() -> bool:
    """When true, force EVERY traffic light to a frozen green so cars never stop at a
    signal — they flow continuously toward and through the monitored corridor. Maximizes
    moving-car occupancy of the corridor; the TM still avoids collisions at junctions
    (ignore_vehicles=0). Override via DATASET_TRAFFIC_LIGHTS_ALL_GREEN."""
    raw = os.environ.get("DATASET_TRAFFIC_LIGHTS_ALL_GREEN", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def configure_traffic_lights_all_green(world):
    """Force all lights green + frozen so traffic never stops at a signal."""
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        print("No traffic lights found.")
        return None
    for light in lights:
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(99999.0)
        light.set_yellow_time(0.1)
        light.set_red_time(0.1)
        light.freeze(True)
    print(
        f"All traffic lights forced GREEN and frozen ({len(lights)}) — "
        "continuous flow toward/through the corridor."
    )
    return None


def green_wave_from_env() -> bool:
    """Corridor green-wave: green every light whose movement runs ALONG the monitored
    corridor and red the conflicting cross movements, so the corridor flows continuously
    without the all-green junction deadlock. Override via DATASET_TRAFFIC_LIGHTS_GREEN_WAVE."""
    raw = os.environ.get("DATASET_TRAFFIC_LIGHTS_GREEN_WAVE", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _principal_axis_yaw_deg(points):
    """Direction (deg) of maximum spread of the points = corridor through-axis."""
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    sxx = sum((p[0] - cx) ** 2 for p in points)
    syy = sum((p[1] - cy) ** 2 for p in points)
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in points)
    return math.degrees(0.5 * math.atan2(2.0 * sxy, sxx - syy))


def _dataset_radar_xy(world):
    return [
        (a.get_transform().location.x, a.get_transform().location.y)
        for a in world.get_actors().filter("sensor.other.radar")
        if a.attributes.get("role_name", "").startswith("dataset_radar_")
    ]


def configure_traffic_lights_green_wave(world, align_deg=45.0):
    """Green the corridor's through-movement, red the cross movements (a 'green wave').

    A light is greened iff its controlled lanes run roughly PARALLEL to the corridor
    axis (within ``align_deg``); otherwise it is forced red. Because only one axis
    flows at each junction, the corridor streams continuously without the all-green
    deadlock (where every direction goes and cars mutually block the junction box).

    The corridor axis is the principal axis of the dataset radar layout (falls back to
    the road direction at the C10 anchor if no radars are spawned yet).
    """
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        print("No traffic lights found.")
        return None

    radar_xy = _dataset_radar_xy(world)
    if len(radar_xy) >= 2:
        corridor_yaw = _principal_axis_yaw_deg(radar_xy)
        src = f"radar layout ({len(radar_xy)} radars)"
    else:
        wp = world.get_map().get_waypoint(C10_ANCHOR_LOCATION, project_to_road=True)
        corridor_yaw = wp.transform.rotation.yaw if wp is not None else 0.0
        src = "C10 anchor road"
    cos_thresh = math.cos(math.radians(align_deg))

    def route_feeds_corridor(wp, reach_m=25.0, step_m=4.0, max_steps=90):
        """True if driving FORWARD from this stop waypoint reaches the radar zone.

        The corridor curves north at both ends, so its drive-in approaches are NOT
        axis-parallel — a purely orientation-based rule freezes those feeder lights
        solid red, stalling every car that drives in along the curve (the recurring
        'stuck at the intersection' jam). Greening any movement that actually FEEDS
        the corridor keeps both eastbound and westbound drive-in traffic flowing."""
        if not radar_xy:
            return False
        cur = wp
        for _ in range(max_steps):
            loc = cur.transform.location
            if min((loc.x - rx) ** 2 + (loc.y - ry) ** 2 for rx, ry in radar_xy) <= reach_m * reach_m:
                return True
            nxt = cur.next(step_m)
            if not nxt:
                break
            cur = nxt[0]
        return False

    n_green = n_red = n_feed = 0
    for light in lights:
        stop_wps = light.get_stop_waypoints()
        if stop_wps:
            aligns = [
                abs(math.cos(math.radians(wp.transform.rotation.yaw - corridor_yaw)))
                for wp in stop_wps
            ]
            parallel = (sum(aligns) / len(aligns)) >= cos_thresh
        else:
            parallel = False  # unknown movement -> treat as cross (red) to avoid conflicts

        # Green = corridor through-movement OR a movement that drives into the corridor.
        feeds = (not parallel) and any(route_feeds_corridor(wp) for wp in stop_wps)
        if parallel or feeds:
            light.set_state(carla.TrafficLightState.Green)
            light.set_green_time(99999.0)
            light.set_yellow_time(0.1)
            light.set_red_time(0.1)
            n_green += 1
            n_feed += int(feeds)
        else:
            light.set_state(carla.TrafficLightState.Red)
            light.set_red_time(99999.0)
            light.set_yellow_time(0.1)
            light.set_green_time(0.1)
            n_red += 1
        light.freeze(True)

    print(
        f"Green-wave traffic lights: corridor axis {corridor_yaw:+.0f}° (from {src}); "
        f"{n_green} green ({n_green - n_feed} through + {n_feed} feeder) / "
        f"{n_red} red (cross) of {len(lights)}.",
        flush=True,
    )
    return None


def configure_traffic_lights_free_automatic(world):
    """
    Let CARLA run the traffic-light state machine (unfreeze all lights).
    Optionally keeps one light near the dataset anchor always green for corridor flow.
    """
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        print("No traffic lights found.")
        return None

    for light in lights:
        reset_light_phase_times(light)
        light.freeze(False)

    always_green_light = set_light_near_location_always_green(lights, C10_ANCHOR_LOCATION)
    print(
        f"Automatic traffic lights: unfroze {len(lights)} lights "
        "(CARLA cycles red/yellow/green)."
    )
    if always_green_light is not None:
        light_loc = always_green_light.get_location()
        print(
            "Dataset corridor override (always green):",
            f"id={always_green_light.id} "
            f"at ({light_loc.x:.2f}, {light_loc.y:.2f}, {light_loc.z:.2f})",
        )
        return always_green_light.id
    return None


def configure_traffic_lights(world, margin_m=PERIMETER_MARGIN_M):
    world_map = world.get_map()
    min_x, max_x, min_y, max_y = get_driving_bounds(world_map)

    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    if not lights:
        print("No traffic lights found.")
        return

    perimeter_lights = []
    perimeter_turn_lights = []
    forced_red_lights = []

    for light in lights:
        is_perimeter = is_perimeter_light(light, min_x, max_x, min_y, max_y, margin_m)
        is_turn_light = light_controls_turning_movement(light)

        if is_perimeter and not is_turn_light:
            perimeter_lights.append(light)
            # Let perimeter lights run normally with sane phase times.
            reset_light_phase_times(light)
            light.freeze(False)
        else:
            if is_perimeter and is_turn_light:
                perimeter_turn_lights.append(light)
            forced_red_lights.append(light)
            light.set_state(carla.TrafficLightState.Red)
            light.set_red_time(99999.0)
            light.set_yellow_time(0.1)
            light.set_green_time(0.1)
            light.freeze(True)

    print(
        f"Traffic lights total={len(lights)} | "
        f"perimeter_straight={len(perimeter_lights)} | "
        f"perimeter_turn_forced_red={len(perimeter_turn_lights)} | "
        f"forced_red_total={len(forced_red_lights)}"
    )
    print(f"Perimeter margin used: {margin_m:.1f} m")
    print(
        f"Turn detection: threshold={TURN_ANGLE_THRESHOLD_DEG:.1f} deg, "
        f"lookahead={TURN_LOOKAHEAD_M:.1f} m"
    )
    if perimeter_lights:
        print("Perimeter straight light IDs:", ", ".join(str(light.id) for light in perimeter_lights))
    if perimeter_turn_lights:
        print(
            "Perimeter turn light IDs (forced red):",
            ", ".join(str(light.id) for light in perimeter_turn_lights),
        )

    always_green_light = set_light_near_location_always_green(lights, C10_ANCHOR_LOCATION)
    if always_green_light is not None:
        light_loc = always_green_light.get_location()
        print(
            "Always-green override light:",
            f"id={always_green_light.id} "
            f"at ({light_loc.x:.2f}, {light_loc.y:.2f}, {light_loc.z:.2f})",
        )
        return always_green_light.id
    return None


def main():
    _, world = get_world()

    if green_wave_from_env():
        always_green_light_id = configure_traffic_lights_green_wave(world)
    elif force_all_green_from_env():
        always_green_light_id = configure_traffic_lights_all_green(world)
    elif automatic_traffic_lights_from_env():
        always_green_light_id = configure_traffic_lights_free_automatic(world)
    else:
        always_green_light_id = configure_traffic_lights(
            world=world, margin_m=PERIMETER_MARGIN_M
        )

    print("Drawing traffic light labels. Press Ctrl+C to stop.")
    try:
        while True:
            count = label_all_traffic_lights(world, always_green_light_id=always_green_light_id)
            print(f"Labeled traffic lights: {count}", end="\r", flush=True)
            time.sleep(LABEL_REFRESH_S)
    except KeyboardInterrupt:
        print("\nStopped traffic light labeling.")


if __name__ == "__main__":
    main()
