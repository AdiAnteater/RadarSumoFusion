"""
Standalone utility: query CARLA for the x/y/z dimensions of every available
vehicle and pedestrian (walker) blueprint, plus any actors currently spawned
in the world.

CARLA exposes sizing through Actor.bounding_box (a carla.BoundingBox). Its
`extent` field is the HALF size on each axis, in meters, in the actor's local
frame:
    x -> forward  (length)
    y -> right    (width)
    z -> up       (height)

So full dimensions are 2 * extent.x, 2 * extent.y, 2 * extent.z.

Blueprints themselves do not expose a bounding box; you only get it after the
actor is spawned. To enumerate every blueprint we briefly spawn each one high
in the air (out of the way of the dataset pipeline), read the bounding box,
then destroy it.

This script is intentionally NOT wired into the main DatasetCreation pipeline.
Run it on its own while the CARLA server is up:

    python FetchActorSizing.py

It also renders each bounding box in the simulation using world.debug.draw_box:
  - Probe actors get their box drawn briefly while they are alive at the probe
    location (lift PROBE_LOCATION.z down to ground level if you want to see them
    inside the regular camera view).
  - Currently-spawned vehicles and pedestrians are visualised in a watch loop
    after the catalog finishes. Press ENTER in the console to exit the loop.
"""

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

from _kbhit_compat import enter_pressed
import os
import time
from collections import defaultdict

import carla

from carla_connect import get_world
from dataset_paths import config_dir


PROBE_LOCATION = carla.Location(x=0.0, y=0.0, z=500.0)
PROBE_SPACING_M = 8.0
PROBE_SETTLE_S = 0.02
OUTPUT_FILENAME = "actor_sizing_catalog.txt"
INCLUDE_CURRENTLY_SPAWNED = True

DRAW_BOUNDING_BOXES = True
BBOX_THICKNESS_M = 0.05
BBOX_VEHICLE_COLOR = carla.Color(80, 220, 255)
BBOX_WALKER_COLOR = carla.Color(255, 200, 0)
BBOX_PROBE_VEHICLE_COLOR = carla.Color(0, 255, 120)
BBOX_PROBE_WALKER_COLOR = carla.Color(255, 120, 200)
PROBE_DRAW_LIFETIME_S = 0.5
WATCH_REFRESH_S = 0.25
WATCH_LIFETIME_S = 0.5


def full_dimensions(actor):
    """Return (length_x, width_y, height_z) in meters for an actor."""
    extent = actor.bounding_box.extent
    return 2.0 * extent.x, 2.0 * extent.y, 2.0 * extent.z


def dim_label(length, width, height):
    return f"L={length:.2f} W={width:.2f} H={height:.2f}"


def draw_actor_bounding_box(world, actor, color, life_time, label=None):
    """
    Draw the actor's bounding box in world space.

    actor.bounding_box is expressed in the actor's local frame: bbox.location is
    the box center as an offset from the actor origin, bbox.extent is the half
    size on each axis, and bbox.rotation is usually identity. We transform the
    center into world coordinates and reuse the actor's rotation for the box
    orientation, which is the standard CARLA idiom for vehicles/walkers since
    their bboxes are axis-aligned in the actor's local frame.
    """
    bbox = actor.bounding_box
    actor_tf = actor.get_transform()
    world_center = actor_tf.transform(bbox.location)
    world_box = carla.BoundingBox(world_center, bbox.extent)
    world.debug.draw_box(
        world_box,
        actor_tf.rotation,
        BBOX_THICKNESS_M,
        color,
        life_time,
    )

    if label:
        world.debug.draw_string(
            world_center + carla.Location(z=bbox.extent.z + 0.3),
            label,
            draw_shadow=False,
            color=color,
            life_time=life_time,
        )


def bp_base_type(blueprint):
    if blueprint.has_attribute("base_type"):
        return blueprint.get_attribute("base_type").as_str() or "unknown"
    return "unknown"


def bp_special_type(blueprint):
    if blueprint.has_attribute("special_type"):
        value = blueprint.get_attribute("special_type").as_str()
        return value if value else ""
    return ""


def actor_base_type(actor):
    attr = actor.attributes.get("base_type", "")
    return attr if attr else "unknown"


def probe_blueprint(world, blueprint, probe_index, draw_color):
    """Spawn the blueprint at a safe altitude, read its bbox, then destroy it."""
    offset_x = (probe_index % 25) * PROBE_SPACING_M
    offset_y = (probe_index // 25) * PROBE_SPACING_M
    transform = carla.Transform(
        carla.Location(
            x=PROBE_LOCATION.x + offset_x,
            y=PROBE_LOCATION.y + offset_y,
            z=PROBE_LOCATION.z,
        )
    )

    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        return None

    try:
        if PROBE_SETTLE_S > 0:
            time.sleep(PROBE_SETTLE_S)
        length, width, height = full_dimensions(actor)
        if DRAW_BOUNDING_BOXES:
            draw_actor_bounding_box(
                world,
                actor,
                draw_color,
                PROBE_DRAW_LIFETIME_S,
                label=f"{blueprint.id}\n{dim_label(length, width, height)}",
            )
        return length, width, height
    finally:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def collect_blueprint_sizes(world, pattern, draw_color):
    bp_lib = world.get_blueprint_library()
    blueprints = list(bp_lib.filter(pattern))
    results = []
    skipped = []
    for i, bp in enumerate(blueprints):
        dims = probe_blueprint(world, bp, i, draw_color)
        if dims is None:
            skipped.append(bp.id)
            continue
        length, width, height = dims
        results.append(
            {
                "type_id": bp.id,
                "base_type": bp_base_type(bp),
                "special_type": bp_special_type(bp),
                "length_m": length,
                "width_m": width,
                "height_m": height,
            }
        )
    return results, skipped


def collect_currently_spawned(world):
    actors = world.get_actors()
    vehicles = []
    for actor in actors.filter("vehicle.*"):
        length, width, height = full_dimensions(actor)
        vehicles.append(
            {
                "actor_id": actor.id,
                "type_id": actor.type_id,
                "base_type": actor_base_type(actor),
                "length_m": length,
                "width_m": width,
                "height_m": height,
            }
        )

    walkers = []
    for actor in actors.filter("walker.pedestrian.*"):
        length, width, height = full_dimensions(actor)
        walkers.append(
            {
                "actor_id": actor.id,
                "type_id": actor.type_id,
                "length_m": length,
                "width_m": width,
                "height_m": height,
            }
        )

    return vehicles, walkers


def print_vehicle_table(rows, header):
    print(f"\n{header} ({len(rows)} entries)")
    if not rows:
        print("  (none)")
        return

    by_base = defaultdict(list)
    for row in rows:
        by_base[row["base_type"]].append(row)

    for base in sorted(by_base):
        bucket = by_base[base]
        print(f"\n  [base_type={base}]  ({len(bucket)})")
        for row in sorted(bucket, key=lambda r: r["type_id"]):
            special = row.get("special_type", "")
            tag = f"  (special={special})" if special else ""
            print(
                "    {tid:<46s}  L={l:6.3f}m  W={w:6.3f}m  H={h:6.3f}m{tag}".format(
                    tid=row["type_id"],
                    l=row["length_m"],
                    w=row["width_m"],
                    h=row["height_m"],
                    tag=tag,
                )
            )


def print_walker_table(rows, header):
    print(f"\n{header} ({len(rows)} entries)")
    if not rows:
        print("  (none)")
        return
    for row in sorted(rows, key=lambda r: r["type_id"]):
        print(
            "    {tid:<46s}  L={l:6.3f}m  W={w:6.3f}m  H={h:6.3f}m".format(
                tid=row["type_id"],
                l=row["length_m"],
                w=row["width_m"],
                h=row["height_m"],
            )
        )


def write_catalog(path, vehicle_rows, walker_rows, skipped_vehicles, skipped_walkers):
    with open(path, "w", encoding="utf-8") as f:
        f.write("CARLA actor sizing catalog\n")
        f.write(
            "All dimensions are full extents in meters "
            "(length = 2*extent.x, width = 2*extent.y, height = 2*extent.z).\n"
        )
        f.write("Axes are in the actor's local frame: x=forward, y=right, z=up.\n\n")

        by_base = defaultdict(list)
        for row in vehicle_rows:
            by_base[row["base_type"]].append(row)

        f.write("=== Vehicle blueprints ===\n")
        for base in sorted(by_base):
            f.write(f"\n[base_type: {base}]\n")
            for row in sorted(by_base[base], key=lambda r: r["type_id"]):
                special = row.get("special_type", "")
                tag = f"  special={special}" if special else ""
                f.write(
                    "  {tid:<46s}  L={l:6.3f}  W={w:6.3f}  H={h:6.3f}{tag}\n".format(
                        tid=row["type_id"],
                        l=row["length_m"],
                        w=row["width_m"],
                        h=row["height_m"],
                        tag=tag,
                    )
                )

        f.write("\n=== Walker (pedestrian) blueprints ===\n")
        for row in sorted(walker_rows, key=lambda r: r["type_id"]):
            f.write(
                "  {tid:<46s}  L={l:6.3f}  W={w:6.3f}  H={h:6.3f}\n".format(
                    tid=row["type_id"],
                    l=row["length_m"],
                    w=row["width_m"],
                    h=row["height_m"],
                )
            )

        if skipped_vehicles or skipped_walkers:
            f.write("\n=== Skipped (could not be spawned for probing) ===\n")
            for tid in skipped_vehicles:
                f.write(f"  vehicle: {tid}\n")
            for tid in skipped_walkers:
                f.write(f"  walker:  {tid}\n")


def watch_currently_spawned(world):
    """
    Continuously redraw bounding boxes around live vehicles and walkers so the
    user can move the spectator camera around and inspect sizes in-sim.

    Each draw uses life_time = WATCH_LIFETIME_S, and we refresh every
    WATCH_REFRESH_S seconds. The loop ends when ENTER is pressed in the console.
    """
    print(
        "\nDrawing bounding boxes on currently-spawned vehicles and pedestrians."
    )
    print("Move the CARLA spectator camera to view them. Press ENTER here to exit...")

    while True:
        actors = world.get_actors()

        for actor in actors.filter("vehicle.*"):
            try:
                length, width, height = full_dimensions(actor)
            except RuntimeError:
                continue
            label = f"{actor.type_id}\n{dim_label(length, width, height)}"
            draw_actor_bounding_box(
                world, actor, BBOX_VEHICLE_COLOR, WATCH_LIFETIME_S, label=label
            )

        for actor in actors.filter("walker.pedestrian.*"):
            try:
                length, width, height = full_dimensions(actor)
            except RuntimeError:
                continue
            label = f"{actor.type_id}\n{dim_label(length, width, height)}"
            draw_actor_bounding_box(
                world, actor, BBOX_WALKER_COLOR, WATCH_LIFETIME_S, label=label
            )

        if enter_pressed():
            break

        time.sleep(WATCH_REFRESH_S)


def main():
    _, world = get_world()

    if INCLUDE_CURRENTLY_SPAWNED:
        spawned_vehicles, spawned_walkers = collect_currently_spawned(world)
        print_vehicle_table(spawned_vehicles, "Currently spawned vehicles")
        print_walker_table(spawned_walkers, "Currently spawned walkers")

    print("\nProbing all vehicle blueprints (spawn -> measure -> destroy)...")
    vehicle_rows, skipped_vehicles = collect_blueprint_sizes(
        world, "vehicle.*", BBOX_PROBE_VEHICLE_COLOR
    )
    print(f"  Measured {len(vehicle_rows)} / skipped {len(skipped_vehicles)}")

    print("Probing all walker blueprints (spawn -> measure -> destroy)...")
    walker_rows, skipped_walkers = collect_blueprint_sizes(
        world, "walker.pedestrian.*", BBOX_PROBE_WALKER_COLOR
    )
    print(f"  Measured {len(walker_rows)} / skipped {len(skipped_walkers)}")

    print_vehicle_table(vehicle_rows, "Vehicle blueprint catalog")
    print_walker_table(walker_rows, "Walker blueprint catalog")

    output_path = str(config_dir() / OUTPUT_FILENAME)
    write_catalog(output_path, vehicle_rows, walker_rows, skipped_vehicles, skipped_walkers)
    print(f"\nSaved catalog to {output_path}")

    if DRAW_BOUNDING_BOXES:
        watch_currently_spawned(world)


if __name__ == "__main__":
    main()
