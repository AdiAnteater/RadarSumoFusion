"""
Query CARLA for dataset RGB cameras (role_name dataset_camera_*) and print/write extrinsics.

World-frame pose matches ExportRadarExtrinsics.csv (x,y,z,yaw,pitch,roll in meters / degrees).

Run while the RadarCameraSetup* process is still alive (e.g. from start.py before stopping children).
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
import sys
from pathlib import Path

import carla

from dataset_paths import data_output_dir

DATASET_CAMERA_ROLE_PREFIX = "dataset_camera_"


def sensor_label_from_role(role_name: str) -> str:
    if role_name.startswith(DATASET_CAMERA_ROLE_PREFIX):
        return role_name[len(DATASET_CAMERA_ROLE_PREFIX) :]
    return ""


def find_latest_capture_dir(base_dir: Path) -> Path | None:
    candidates = [
        p
        for p in base_dir.glob("sensor_capture_*")
        if p.is_dir() and (p / "camera_data.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_camera_sensor_labels(camera_csv_path: Path) -> dict[int, str]:
    """sensor_id -> sensor_label from camera_data.csv (first occurrence wins)."""
    mapping: dict[int, str] = {}
    with camera_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sensor_id", "sensor_label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{camera_csv_path} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            label = str(row["sensor_label"]).strip()
            if not label:
                continue
            sid = int(str(row["sensor_id"]).strip())
            if sid not in mapping:
                mapping[sid] = label
    return mapping


def collect_camera_extrinsics(world: carla.World) -> list[dict]:
    rows: list[dict] = []
    for actor in world.get_actors().filter("sensor.camera.rgb"):
        role_name = actor.attributes.get("role_name", "")
        label = sensor_label_from_role(role_name)
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


def write_outputs(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "camera_extrinsics.json"
    csv_path = output_dir / "camera_extrinsics.csv"
    field_order = ["sensor_id", "sensor_label", "x", "y", "z", "yaw", "pitch", "roll"]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def write_camera_extrinsics_to_dataset_dir(world: carla.World, output_dir: Path) -> bool:
    """
    Write camera_extrinsics.json / .csv into a sensor_capture folder.
    For use from CaptureRadarCameraData (same process + world as the recorder).
    """
    output_dir = output_dir.resolve()
    rows = collect_camera_extrinsics(world)
    if not rows:
        print(
            "Camera extrinsics: no dataset_camera_* RGB sensors in the world. "
            "Is RadarCameraSetup* still running?",
            file=sys.stderr,
        )
        return False

    print("--- Camera extrinsics (world frame) ---", flush=True)
    for r in rows:
        print(
            f"  {r['sensor_label']} (id={r['sensor_id']}): "
            f"x={r['x']}, y={r['y']}, z={r['z']}, "
            f"yaw={r['yaw']}, pitch={r['pitch']}, roll={r['roll']}",
            flush=True,
        )

    camera_csv = output_dir / "camera_data.csv"
    if camera_csv.exists():
        try:
            expected = read_camera_sensor_labels(camera_csv)
            for r in rows:
                exp_label = expected.get(r["sensor_id"])
                if exp_label is not None and exp_label != r["sensor_label"]:
                    print(
                        f"Warning: camera_data.csv label for id {r['sensor_id']} "
                        f"is {exp_label!r} but actor has {r['sensor_label']!r}.",
                        file=sys.stderr,
                    )
        except ValueError as e:
            print(f"Warning: could not validate labels: {e}", file=sys.stderr)

    json_path, csv_path = write_outputs(rows, output_dir)
    print(f"Wrote {json_path.name} and {csv_path.name} -> {output_dir}", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export CARLA world-frame RGB camera extrinsics for sensors with "
            "role_name prefix dataset_camera_."
        )
    )
    p.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port")
    p.add_argument("--timeout", type=float, default=10.0, help="Client timeout (seconds)")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="sensor_capture_* folder to write camera_extrinsics.* into (default: latest with camera_data.csv)",
    )
    p.add_argument(
        "--no-files",
        action="store_true",
        help="Print only; do not write JSON/CSV files",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = data_output_dir()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    if args.no_files:
        rows = collect_camera_extrinsics(world)
        print("--- Camera extrinsics (CARLA world frame: x,y,z meters; yaw,pitch,roll degrees) ---")
        if not rows:
            print("No RGB cameras with role_name prefix 'dataset_camera_' found.")
            print(
                "(Spawn a layout that includes the dataset camera: RadarCameraSetup4/8/12/14.py.)"
            )
            return 0
        for r in rows:
            print(
                f"  {r['sensor_label']} (id={r['sensor_id']}): "
                f"x={r['x']}, y={r['y']}, z={r['z']}, "
                f"yaw={r['yaw']}, pitch={r['pitch']}, roll={r['roll']}"
            )
        return 0

    output_dir: Path | None = args.dataset_dir
    if output_dir is None:
        output_dir = find_latest_capture_dir(data_dir)
    if output_dir is None:
        print(
            "No sensor_capture_* folder with camera_data.csv found; use --dataset-dir. ",
            file=sys.stderr,
        )
        return 1
    if not write_camera_extrinsics_to_dataset_dir(world, output_dir):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
