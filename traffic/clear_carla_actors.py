"""
clear_carla_actors.py
=====================
Standalone manual wipe: removes all vehicle (and optionally pedestrian)
actors from a running CARLA server. Run this any time the world has leftover
cars -- e.g. after a run was hard-killed (taskkill / closing the window)
instead of stopped with Ctrl+C, so the finally-block cleanup never ran.

Usage (from an ACTIVATED venv):
    python clear_carla_actors.py
    python clear_carla_actors.py --walkers
    python clear_carla_actors.py --host 127.0.0.1 --port 2000

Only vehicle.* / walker.* actors are removed. sensor.* actors (dataset
radars/cameras) are left untouched.
"""

import argparse
import sys

try:
    import carla
except ImportError:
    sys.exit("carla not found - activate your venv and: pip install carla==0.9.16")

from carla_cleanup import destroy_all_vehicles


def main():
    parser = argparse.ArgumentParser(
        description="Remove all vehicles (and optionally pedestrians) from a running CARLA world."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="CARLA server host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA server port (default 2000)")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Connection timeout in seconds (default 10)")
    parser.add_argument("--walkers", action="store_true",
                        help="Also remove pedestrians and their AI controllers")
    args = parser.parse_args()

    print(f"[clear] Connecting to CARLA at {args.host}:{args.port} ...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
    except Exception as e:
        sys.exit(f"[clear] Could not connect to CARLA: {e}")

    print(f"[clear] Connected. Map: {world.get_map().name}")
    removed = destroy_all_vehicles(client, world, include_walkers=args.walkers, verbose=True)
    print(f"[clear] Done. {removed} actor(s) removed.")


if __name__ == "__main__":
    main()
