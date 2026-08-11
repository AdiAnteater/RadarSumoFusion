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
from typing import Any

import carla

from capture.radar_layout import apply_radar_pitch
from dataset_paths import data_output_dir

DATASET_RADAR_ROLE_PREFIX = "dataset_radar_"


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


def build_manual_radar_positions():
    # Mirrors the manual placement used in RadarCameraSetup.py.
    return {
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


def apply_road_alignment(radar_positions, current_map):
    flipped_40_deg_names = {"R3", "R7", "R9", "R11"}
    for name, tr in list(radar_positions.items()):
        yaw = compute_radar_yaw_toward_road(
            current_map,
            tr.location,
            tr.rotation.yaw,
            offset_deg=40.0,
            use_opposite_side=name in flipped_40_deg_names,
        )
        radar_positions[name] = carla.Transform(
            tr.location,
            carla.Rotation(pitch=tr.rotation.pitch, yaw=yaw, roll=tr.rotation.roll),
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


def sensor_label_from_radar_role(role_name: str) -> str:
    if role_name.startswith(DATASET_RADAR_ROLE_PREFIX):
        return role_name[len(DATASET_RADAR_ROLE_PREFIX) :]
    return ""


def collect_radar_extrinsics_from_world(world: carla.World) -> list[dict[str, Any]]:
    """World-frame poses for spawned dataset radars (matches live session used for capture)."""
    rows: list[dict[str, Any]] = []
    for actor in world.get_actors().filter("sensor.other.radar"):
        role_name = actor.attributes.get("role_name", "")
        label = sensor_label_from_radar_role(role_name)
        if not label:
            continue
        tr = actor.get_transform()
        loc, rot = tr.location, tr.rotation
        rows.append(
            {
                "sensor_id": int(actor.id),
                "sensor_label": label,
                "x": round(float(loc.x), 6),
                "y": round(float(loc.y), 6),
                "z": round(float(loc.z), 6),
                "yaw": round(float(rot.yaw), 6),
                "pitch": round(float(rot.pitch), 6),
                "roll": round(float(rot.roll), 6),
            }
        )
    rows.sort(key=lambda r: r["sensor_id"])
    return rows


def read_sensor_mapping(radar_csv_path):
    sensor_by_label = {}
    with radar_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sensor_id", "sensor_label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{radar_csv_path} must contain columns: {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            label = str(row["sensor_label"]).strip()
            if not label:
                continue
            sensor_id = int(str(row["sensor_id"]).strip())
            if label in sensor_by_label and sensor_by_label[label] != sensor_id:
                raise ValueError(
                    f"Conflicting sensor_id for label '{label}': "
                    f"{sensor_by_label[label]} vs {sensor_id}"
                )
            sensor_by_label[label] = sensor_id

    return sensor_by_label


def find_latest_capture_dir(base_dir):
    candidates = [
        p for p in base_dir.glob("sensor_capture_*") if p.is_dir() and (p / "radar_data.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No sensor_capture_* folder with radar_data.csv found in {base_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_extrinsic_rows(sensor_by_label, radar_positions):
    missing = sorted(label for label in sensor_by_label if label not in radar_positions)
    if missing:
        raise KeyError(
            "These sensor labels were found in radar_data.csv but not in configured radar transforms: "
            + ", ".join(missing)
        )

    rows = []
    for label, sensor_id in sensor_by_label.items():
        tr = radar_positions[label]
        rows.append(
            {
                "sensor_id": int(sensor_id),
                "sensor_label": label,
                "x": round(float(tr.location.x), 6),
                "y": round(float(tr.location.y), 6),
                "z": round(float(tr.location.z), 6),
                "yaw": round(float(tr.rotation.yaw), 6),
                "pitch": round(float(tr.rotation.pitch), 6),
                "roll": round(float(tr.rotation.roll), 6),
            }
        )
    rows.sort(key=lambda r: r["sensor_id"])
    return rows


def write_outputs(rows, output_dir: Path) -> tuple[Path, Path]:
    """Write sensor_extrinsics.* (legacy name) and radar_extrinsics.* (clearer for datasets)."""
    field_order = ["sensor_id", "sensor_label", "x", "y", "z", "yaw", "pitch", "roll"]
    for base in ("sensor_extrinsics", "radar_extrinsics"):
        json_path = output_dir / f"{base}.json"
        csv_path = output_dir / f"{base}.csv"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(rows)
    # Return the radar_* names as primary for callers
    rjson = output_dir / "radar_extrinsics.json"
    rcsv = output_dir / "radar_extrinsics.csv"
    return rjson, rcsv


def _read_sensor_ids_sample_for_radar_check(radar_csv_path: Path, max_rows: int = 5000) -> dict[str, int]:
    """
    Light scan: first max_rows data rows (not full 1M+ file). Used only for id-set warnings.
    """
    out: dict[str, int] = {}
    with radar_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"sensor_id", "sensor_label"}.issubset(set(reader.fieldnames)):
            return out
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            label = str(row["sensor_label"]).strip()
            if not label:
                continue
            out[label] = int(str(row["sensor_id"]).strip())
    return out


def write_radar_extrinsics_live_to_dataset_dir(world: carla.World, output_dir: Path) -> bool:
    """
    Write radar_extrinsics.* + sensor_extrinsics.* from live dataset_radar_* actors.
    Does not read the full radar_data.csv (can be very large) — that used to break export.
    """
    output_dir = output_dir.resolve()
    radar_csv_path = output_dir / "radar_data.csv"
    if not radar_csv_path.is_file():
        print(
            f"Radar extrinsics: missing {radar_csv_path.name} in {output_dir}.",
            file=sys.stderr,
        )
        return False

    print("Radar extrinsics: reading live dataset_radar_* actors...", flush=True)
    rows = collect_radar_extrinsics_from_world(world)
    if not rows:
        print(
            "Radar extrinsics: no dataset_radar_ actors in the world. "
            "Export runs before sensor.stop() in capture; keep RadarCameraSetup* running.",
            file=sys.stderr,
        )
        return False

    try:
        sample_map = _read_sensor_ids_sample_for_radar_check(radar_csv_path)
    except OSError as e:
        print(f"Warning: could not read sample of {radar_csv_path.name}: {e}", file=sys.stderr)
        sample_map = {}
    if sample_map:
        in_world = {r["sensor_id"] for r in rows}
        expected = set(sample_map.values())
        if expected != in_world and expected and in_world:
            missing = sorted(expected - in_world)
            extra = sorted(in_world - expected)
            if missing or extra:
                print(
                    "Note: radar_data.csv id set vs. live actors (from first ~5k rows): "
                    f"missing {missing!r} extra {extra!r} (export uses live world).",
                    file=sys.stderr,
                )
    rjson, rcsv = write_outputs(rows, output_dir)
    print(
        f"Wrote {rjson.name}, {rcsv.name} (and sensor_extrinsics.*) -> {output_dir} ({len(rows)} radars)",
        flush=True,
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export radar sensor extrinsics (x,y,z,yaw,pitch,roll) "
            "for each unique sensor_id/sensor_label found in radar_data.csv."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Path to a sensor_capture_* folder containing radar_data.csv. "
            "If omitted, the latest capture folder under this script directory is used."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="CARLA host for map-based yaw alignment (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA port for map-based yaw alignment (default: 2000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="CARLA client timeout seconds (default: 10.0)",
    )
    parser.add_argument(
        "--live-actors",
        action="store_true",
        help=(
            "Read extrinsics from live dataset_radar_* actors in CARLA (any radar count). "
            "Use with Start.py on shutdown while sensors are still loaded. "
            "If omitted, uses manual R1..R12 transforms + map alignment (legacy)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = data_output_dir()
    dataset_dir = args.dataset_dir if args.dataset_dir else find_latest_capture_dir(script_dir)
    dataset_dir = dataset_dir.resolve()
    radar_csv_path = dataset_dir / "radar_data.csv"
    if not radar_csv_path.exists():
        print(f"Missing radar_data.csv in {dataset_dir}", file=sys.stderr)
        return 1

    try:
        sensor_by_label = read_sensor_mapping(radar_csv_path)
    except ValueError as e:
        print(f"Error reading {radar_csv_path}: {e}", file=sys.stderr)
        return 1

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    if args.live_actors:
        if not write_radar_extrinsics_live_to_dataset_dir(world, dataset_dir):
            return 1
        return 0

    radar_positions = build_manual_radar_positions()
    current_map = world.get_map()
    radar_positions = apply_road_alignment(radar_positions, current_map)
    try:
        rows = build_extrinsic_rows(sensor_by_label, radar_positions)
    except KeyError as e:
        print(
            f"{e}. For layouts outside fixed R1..R12, use --live-actors with CARLA running.",
            file=sys.stderr,
        )
        return 1
    rjson, rcsv = write_outputs(rows, dataset_dir)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Sensors exported: {len(rows)}")
    print(
        f"Wrote {rjson.name}, {rcsv.name} and matching sensor_extrinsics.* (radar = same data)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
