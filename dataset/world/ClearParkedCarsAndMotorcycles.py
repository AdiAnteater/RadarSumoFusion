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


def disable_environment_objects(world, label_name):
    label = getattr(carla.CityObjectLabel, label_name, None)
    if label is None:
        return 0

    objects = world.get_environment_objects(label)
    ids = {obj.id for obj in objects}
    if not ids:
        return 0

    world.enable_environment_objects(ids, False)
    return len(ids)


def destroy_vehicle_actors(world):
    cars_destroyed = 0
    motorcycles_destroyed = 0

    for actor in world.get_actors().filter("vehicle.*"):
        bp = actor.blueprint
        wheels_attr = bp.get_attribute("number_of_wheels") if bp.has_attribute("number_of_wheels") else None
        wheels = int(wheels_attr.as_int()) if wheels_attr is not None else 4

        if wheels == 2:
            actor.destroy()
            motorcycles_destroyed += 1
            continue

        # Treat very low-speed vehicles as parked and remove them.
        vel = actor.get_velocity()
        speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
        if speed < 0.15:
            actor.destroy()
            cars_destroyed += 1

    return cars_destroyed, motorcycles_destroyed


def main():
    _, world = get_world()

    env_cars_disabled = disable_environment_objects(world, "Car")
    env_motorcycles_disabled = disable_environment_objects(world, "Motorcycle")
    cars_destroyed, motorcycles_destroyed = destroy_vehicle_actors(world)

    print(f"Disabled environment parked cars: {env_cars_disabled}")
    print(f"Disabled environment motorcycles: {env_motorcycles_disabled}")
    print(f"Destroyed parked car actors: {cars_destroyed}")
    print(f"Destroyed motorcycle actors: {motorcycles_destroyed}")


if __name__ == "__main__":
    main()
