import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import carla

from carla_connect import get_world


def despawn_pedestrians(world):
    destroyed = 0
    for pattern in ("walker.pedestrian.*", "controller.ai.walker"):
        for actor in world.get_actors().filter(pattern):
            try:
                if actor.is_alive and actor.destroy():
                    destroyed += 1
            except RuntimeError:
                continue
    return destroyed


def despawn_all_cars(world):
    destroyed_cars = 0
    skipped_motorcycles = 0
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            if not actor.is_alive:
                continue
            type_id = actor.type_id.lower()
            if "motorcycle" in type_id or "vespa" in type_id or "yamaha" in type_id:
                skipped_motorcycles += 1
                continue
            if actor.destroy():
                destroyed_cars += 1
        except RuntimeError:
            continue

    return destroyed_cars, skipped_motorcycles


def main():
    _, world = get_world()

    destroyed_cars, skipped_motorcycles = despawn_all_cars(world)
    destroyed_peds = despawn_pedestrians(world)
    print(f"Destroyed cars: {destroyed_cars}")
    print(f"Skipped motorcycles: {skipped_motorcycles}")
    print(f"Destroyed pedestrians/controllers: {destroyed_peds}")


if __name__ == "__main__":
    main()
