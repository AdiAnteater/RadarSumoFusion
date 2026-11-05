"""
Print world-frame radar extrinsics (x,y,z m; yaw,pitch,roll deg) for a stretch rig
with --south / --north radars per kerb, using the same placement as
setup/RadarCameraSetup.py.

Requires CARLA running (map waypoints define the inward vector). Writes by default
to dataset/config/:
  - radar_layout_extrinsics_<mapname>.json  (layouts keyed by total radar count)
  - radar_layout_extrinsics_<mapname>.csv

Usage:
  python PrintRadarLayoutExtrinsics.py --south 4 --north 4
  python PrintRadarLayoutExtrinsics.py --south 3 --north 2 --height 3.0 --out-dir C:\\path
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

import carla

from capture.radar_layout import (
    apply_radar_pitch,
    apply_stretch_radar_yaws,
    radars_per_side_from_env,
    stretch_radar_positions,
)
from dataset_paths import config_dir


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


def layout_for(south: int, north: int, current_map, height=None) -> dict[str, carla.Transform]:
    radar_positions = stretch_radar_positions(south, north, height=height)
    apply_stretch_radar_yaws(radar_positions, current_map)
    apply_radar_pitch(radar_positions)
    return radar_positions


def main() -> int:
    env_south, env_north = radars_per_side_from_env()
    p = argparse.ArgumentParser()
    p.add_argument("--south", type=int, default=env_south,
                   help="radars on the south kerb (default: DATASET_RADARS_SOUTH or 4)")
    p.add_argument("--north", type=int, default=env_north,
                   help="radars on the north kerb (default: DATASET_RADARS_NORTH or 4)")
    p.add_argument("--height", type=float, default=None,
                   help="mount height in m (default: DATASET_RIG_HEIGHT_M or 3.0)")
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

    try:
        trs: dict[str, carla.Transform] = layout_for(
            args.south, args.north, current_map, height=args.height)
    except ValueError as e:
        print(f"Invalid rig: {e}", file=sys.stderr)
        return 1
    n = len(trs)
    rows = [transform_to_row(name, trs[name]) for name in sorted(trs.keys(), key=lambda s: int(s[1:]))]
    all_layouts[str(n)] = rows
    print(f"=== {args.south} south + {args.north} north ({n} radars) ===", flush=True)
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
                    "note": "World frame; same stretch placement as setup/RadarCameraSetup.py.",
                    "radars_south": args.south,
                    "radars_north": args.north,
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
