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


# Match both American/British spellings and common prop naming.
ENV_OBJECT_KEYWORDS = ("trash", "garbage", "bin", "mailbox", "mail_box", "letterbox")
ACTOR_KEYWORDS = ("trash", "garbage", "bin", "mailbox", "mail_box", "letterbox")


def disable_matching_environment_objects(world, keywords):
    matched_objects = []
    all_objects = world.get_environment_objects(carla.CityObjectLabel.Any)

    for obj in all_objects:
        object_name = (obj.name or "").lower()
        if any(keyword in object_name for keyword in keywords):
            matched_objects.append(obj)

    ids = {obj.id for obj in matched_objects}
    if ids:
        world.enable_environment_objects(ids, False)

    return len(ids), matched_objects


def destroy_matching_prop_actors(world, keywords):
    destroyed = 0

    for actor in world.get_actors():
        blueprint_id = actor.type_id.lower()
        if not any(keyword in blueprint_id for keyword in keywords):
            continue
        actor.destroy()
        destroyed += 1

    return destroyed


def main():
    _, world = get_world()

    env_disabled, matched_objects = disable_matching_environment_objects(
        world, ENV_OBJECT_KEYWORDS
    )
    actors_destroyed = destroy_matching_prop_actors(world, ACTOR_KEYWORDS)

    print(f"Disabled environment objects: {env_disabled}")
    if matched_objects:
        print("Matched environment objects:")
        for obj in matched_objects:
            print(f"  - id={obj.id} type={obj.type} name={obj.name}")
    else:
        print("No matching environment objects found by name.")

    print(f"Destroyed actor props: {actors_destroyed}")


if __name__ == "__main__":
    main()
