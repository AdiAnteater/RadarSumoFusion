"""
Print world-frame radar extrinsics (x,y,z m; yaw,pitch,roll deg) for each layout
(4 / 8 / 12 / 14) using the same math as RadarCameraSetup4/8/12/14.py.

Requires CARLA running (map waypoints define yaw alignment). Writes by default
(next to this script):
  - radar_layout_extrinsics_<mapname>.json
  - radar_layout_extrinsics_<mapname>.csv

Usage:
  python PrintRadarLayoutExtrinsics.py
  python PrintRadarLayoutExtrinsics.py --out-dir C:\\path
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
import csv
import json
import math
import sys
from pathlib import Path

import carla

from capture.radar_layout import apply_radar_pitch
from dataset_paths import config_dir


def make_transform(x, y, z, pitch, yaw, roll):
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
    )


def normalize_angle(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def angular_distance(a_deg: float, b_deg: float) -> float:
    return abs(normalize_angle(a_deg - b_deg))


def compute_radar_yaw_toward_road(
    current_map,
    location,
    fallback_yaw,
    offset_deg: float = 40.0,
    use_opposite_side: bool = False,
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
        chosen = (
            candidates[1] if abs(normalize_angle(chosen - candidates[0])) < 1e-6 else candidates[0]
        )

    return normalize_angle(chosen)


def apply_road_alignment(
    current_map,
    radar_positions: dict[str, carla.Transform],
    flipped_40_deg_names: set[str],
) -> dict[str, carla.Transform]:
    out = {}
    for name, tr in radar_positions.items():
        new_yaw = compute_radar_yaw_toward_road(
            current_map,
            tr.location,
            tr.rotation.yaw,
            offset_deg=40.0,
            use_opposite_side=name in flipped_40_deg_names,
        )
        out[name] = carla.Transform(
            tr.location,
            carla.Rotation(tr.rotation.pitch, new_yaw, tr.rotation.roll),
        )
    return out


def transform_to_row(name: str, tr: carla.Transform) -> dict:
    loc, rot = tr.location, tr.rotation
    return {
        "sensor_label": name,
        "x_m": round(float(loc.x), 6),
        "y_m": round(float(loc.y), 6),
        "z_m": round(float(loc.z), 6),
        "yaw_deg": round(float(rot.yaw), 6),
        "pitch_deg": round(float(rot.pitch), 6),
        "roll_deg": round(float(rot.roll), 6),
    }


def layout_4(current_map) -> dict[str, carla.Transform]:
    radar_positions = {
        "R1": make_transform(-33.825321, -52.015091, 1.0, 0.0, 0.0, 0.0),
        "R2": make_transform(-33.825321, -74.745476, 1.0, 0.0, 180.0, 0.0),
        "R3": make_transform(3.355103, -52.015091, 1.0, 0.0, 0.0, 0.0),
        "R4": make_transform(3.355103, -74.745476, 1.0, 0.0, 180.0, 0.0),
    }
    radar_positions = apply_road_alignment(
        current_map, radar_positions, flipped_40_deg_names={"R4"}
    )
    tr1 = radar_positions["R1"]
    tr2 = radar_positions["R2"]
    r1_yaw = normalize_angle(tr2.rotation.yaw + 90)
    radar_positions["R1"] = carla.Transform(
        tr1.location,
        carla.Rotation(tr1.rotation.pitch, r1_yaw, tr1.rotation.roll),
    )
    apply_radar_pitch(radar_positions)
    return radar_positions


def layout_8(current_map) -> dict[str, carla.Transform]:
    y_upper = -52.5
    y_lower = -73.5
    radar_positions = {
        "R1": make_transform(-28.825321, y_upper, 3.0, 0.0, 0.0, 0.0),
        "R2": make_transform(-28.825321, y_lower, 3.0, 0.0, 180.0, 0.0),
        "R3": make_transform(3.355103, y_upper, 3.0, 0.0, 0.0, 0.0),
        "R4": make_transform(3.355103, y_lower, 3.0, 0.0, 180.0, 0.0),
        "R5": make_transform(38.535528, y_upper, 3.0, 0.0, 0.0, 0.0),
        "R6": make_transform(38.535528, y_lower, 3.0, 0.0, 180.0, 0.0),
        "R7": make_transform(65.715952, y_upper, 3.0, 0.0, 0.0, 0.0),
        "R8": make_transform(65.715952, y_lower, 3.0, 0.0, 180.0, 0.0),
    }
    radar_positions = apply_road_alignment(
        current_map, radar_positions, flipped_40_deg_names={"R4", "R5", "R8"}
    )
    tr1 = radar_positions["R1"]
    tr2 = radar_positions["R2"]
    r1_yaw = normalize_angle(tr2.rotation.yaw + 90)
    radar_positions["R1"] = carla.Transform(
        tr1.location,
        carla.Rotation(tr1.rotation.pitch, r1_yaw, tr1.rotation.roll),
    )
    apply_radar_pitch(radar_positions)
    return radar_positions


def layout_12(current_map) -> dict[str, carla.Transform]:
    radar_positions = {
        "R1": make_transform(-33.825321, -52.015091, 1.0, 0.0, 0.0, 0.0),
        "R2": make_transform(-33.825321, -74.745476, 1.0, 0.0, -179.403259, 0.0),
        "R3": make_transform(-15.235109, -52.015091, 1.0, 0.0, 360.020691, 0.0),
        "R4": make_transform(-15.235109, -74.745476, 1.0, 0.0, -179.085037, 0.0),
        "R5": make_transform(3.355103, -52.015091, 1.0, 0.0, -179.085037, 0.0),
        "R6": make_transform(3.355103, -74.745476, 1.0, 0.0, -179.085037, 0.0),
        "R7": make_transform(21.945315, -52.015091, 1.0, 0.0, 359.976562, 0.0),
        "R8": make_transform(21.945315, -74.745476, 1.0, 0.0, 179.976578, 0.0),
        "R9": make_transform(40.535528, -52.015091, 1.0, 0.0, 1.382248, 0.0),
        "R10": make_transform(40.535528, -74.745476, 1.0, 0.0, -151.803711, 0.0),
        "R11": make_transform(59.125740, -52.015091, 1.0, 0.0, -179.085037, 0.0),
        "R12": make_transform(59.125740, -74.745476, 1.0, 0.0, -179.085037, 0.0),
    }
    radar_positions = apply_road_alignment(
        current_map, radar_positions, flipped_40_deg_names={"R3", "R7", "R9", "R11"}
    )
    tr1 = radar_positions["R1"]
    tr2 = radar_positions["R2"]
    r1_yaw = normalize_angle(tr2.rotation.yaw)
    radar_positions["R1"] = carla.Transform(
        tr1.location,
        carla.Rotation(tr1.rotation.pitch, r1_yaw, tr1.rotation.roll),
    )
    apply_radar_pitch(radar_positions)
    return radar_positions


def layout_14(current_map) -> dict[str, carla.Transform]:
    x_start = -33.825321
    x_end = 77.715952
    y_upper = -52.015091
    y_lower = -74.745476
    z = 1.0
    n_cols = 7
    span = x_end - x_start
    xs = [x_start + i * span / (n_cols - 1) for i in range(n_cols)]

    radar_positions: dict[str, carla.Transform] = {}
    for col, x in enumerate(xs):
        upper_id = 2 * col + 1
        lower_id = 2 * col + 2
        radar_positions[f"R{upper_id}"] = make_transform(x, y_upper, z, 0.0, 0.0, 0.0)
        radar_positions[f"R{lower_id}"] = make_transform(x, y_lower, z, 0.0, 180.0, 0.0)
    radar_positions = apply_road_alignment(
        current_map, radar_positions, flipped_40_deg_names={"R3", "R7", "R9", "R11"}
    )
    apply_radar_pitch(radar_positions)
    return radar_positions


LAYOUTS = {
    4: ("RadarCameraSetup4 (4 radars)", layout_4),
    8: ("RadarCameraSetup8 (8 radars)", layout_8),
    12: ("RadarCameraSetup12 (12 radars)", layout_12),
    14: ("RadarCameraSetup14 (14 radars)", layout_14),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder (default: folder containing this script).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Only print to stdout; do not write JSON/CSV files.",
    )
    args = p.parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir = args.out_dir if args.out_dir is not None else config_dir()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
    except Exception as e:
        print("Could not connect to CARLA. Start the simulator, then re-run this script.", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        return 1

    current_map = world.get_map()
    map_name = current_map.name.split("/")[-1] if current_map.name else "unknown"
    world_snapshot = world.get_snapshot()
    print(
        f"Map: {map_name}  |  CARLA world snapshot frame: {world_snapshot.frame}\n",
        flush=True,
    )

    all_layouts: dict = {}

    for n, (title, fn) in LAYOUTS.items():
        trs: dict[str, carla.Transform] = fn(current_map)
        rows = [transform_to_row(name, trs[name]) for name in sorted(trs.keys(), key=lambda s: int(s[1:]))]
        all_layouts[str(n)] = rows
        print(f"=== {title} ===", flush=True)
        for r in rows:
            print(
                f"  {r['sensor_label']}:  x={r['x_m']}  y={r['y_m']}  z={r['z_m']}  |  "
                f"yaw={r['yaw_deg']}  pitch={r['pitch_deg']}  roll={r['roll_deg']}",
                flush=True,
            )
        print(flush=True)

    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"radar_layout_extrinsics_{map_name.replace(' ', '_')}"
        json_path = out_dir / f"{base}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "map": map_name,
                    "frame": world_snapshot.frame,
                    "note": "World frame; same logic as RadarCameraSetup4/8/12/14. Yaw is map-defined.",
                    "layouts": all_layouts,
                },
                f,
                indent=2,
            )
        csv_path = out_dir / f"{base}.csv"
        fieldnames = [
            "layout_radars",
            "sensor_label",
            "x_m",
            "y_m",
            "z_m",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for n in sorted(int(k) for k in all_layouts):
                for row in all_layouts[str(n)]:
                    w.writerow(
                        {
                            "layout_radars": n,
                            "sensor_label": row["sensor_label"],
                            "x_m": row["x_m"],
                            "y_m": row["y_m"],
                            "z_m": row["z_m"],
                            "yaw_deg": row["yaw_deg"],
                            "pitch_deg": row["pitch_deg"],
                            "roll_deg": row["roll_deg"],
                        }
                    )
        print(f"Wrote: {json_path.resolve()}", flush=True)
        print(f"Wrote: {csv_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
