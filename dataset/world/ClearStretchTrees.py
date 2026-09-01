"""Hide trees over the monitored stretch (median + kerbs of the boulevard).

The fusion orchestrator (fusion/orchestrator.py :: _clear_stretch_trees) calls
this script best-effort before spawning the sensor rig. It hides only CARLA
Vegetation environment objects whose world position falls inside a box around
the monitored stretch, so palm trees on the median stop occluding the radars
and camera. Traffic lights, poles, buildings and all other geometry are left
untouched because only the Vegetation label is considered.

Trees are HIDDEN (world.enable_environment_objects(..., False)), not destroyed,
so a fresh map reload brings them back. This runs on the live world before any
sensor exists, which is safe (see the DatasetCreation notes on why clearing
env objects mid-run near live GPU sensors can crash CARLA).

The stretch box comes from the verified net geometry:
    boulevard runs along +X, x in [-28.7, 26.4] (~55 m),
    carriageways span y in [5.4 .. 36.0], mid ~(-1.0, 20.7).
A few metres of margin are added so trees just off the ends/kerbs are caught.
All bounds are env-tunable so the region can be nudged without editing code:
    DATASET_TREE_CLEAR_X_MIN   default -35.0
    DATASET_TREE_CLEAR_X_MAX   default  33.0
    DATASET_TREE_CLEAR_Y_MIN   default   3.0
    DATASET_TREE_CLEAR_Y_MAX   default  38.0
Set DATASET_TREE_CLEAR_DRY_RUN=1 to only report matches without hiding them.
"""

import importlib.util
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import carla

from carla_connect import get_world


# Default stretch box (CARLA world coords). See module docstring for the source.
DEFAULT_X_MIN = -35.0
DEFAULT_X_MAX = 33.0
DEFAULT_Y_MIN = 3.0
DEFAULT_Y_MAX = 38.0


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name):
    raw = os.environ.get(name, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def stretch_box_from_env():
    x_min = _env_float("DATASET_TREE_CLEAR_X_MIN", DEFAULT_X_MIN)
    x_max = _env_float("DATASET_TREE_CLEAR_X_MAX", DEFAULT_X_MAX)
    y_min = _env_float("DATASET_TREE_CLEAR_Y_MIN", DEFAULT_Y_MIN)
    y_max = _env_float("DATASET_TREE_CLEAR_Y_MAX", DEFAULT_Y_MAX)
    # Guard against a swapped min/max so the box is always well formed.
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    return x_min, x_max, y_min, y_max


def _object_world_xy(obj):
    """Return the (x, y) world position of an environment object.

    EnvironmentObject exposes a world-space bounding_box; its location is the
    object's world centre. Fall back to transform.location if needed.
    """
    try:
        loc = obj.bounding_box.location
        return float(loc.x), float(loc.y)
    except (AttributeError, RuntimeError):
        pass
    try:
        loc = obj.transform.location
        return float(loc.x), float(loc.y)
    except (AttributeError, RuntimeError):
        return None


def vegetation_over_stretch(world, box):
    x_min, x_max, y_min, y_max = box
    matched = []
    try:
        veg_objects = world.get_environment_objects(carla.CityObjectLabel.Vegetation)
    except (RuntimeError, AttributeError):
        # Older/newer label enums may differ; fall back to scanning all objects.
        veg_objects = [
            o for o in world.get_environment_objects(carla.CityObjectLabel.Any)
            if "tree" in (o.name or "").lower() or "veg" in (o.name or "").lower()
        ]

    for obj in veg_objects:
        xy = _object_world_xy(obj)
        if xy is None:
            continue
        x, y = xy
        if x_min <= x <= x_max and y_min <= y <= y_max:
            matched.append(obj)
    return matched


def main():
    _, world = get_world()
    box = stretch_box_from_env()
    dry_run = _env_flag("DATASET_TREE_CLEAR_DRY_RUN")

    matched = vegetation_over_stretch(world, box)
    ids = {obj.id for obj in matched}

    x_min, x_max, y_min, y_max = box
    print(
        f"Tree clear region (CARLA world): x[{x_min:.1f}, {x_max:.1f}] "
        f"y[{y_min:.1f}, {y_max:.1f}]"
    )
    print(f"Vegetation objects over the stretch: {len(ids)}")

    if not ids:
        print("No vegetation found in the stretch region; nothing to hide.")
        return

    if dry_run:
        print("DRY RUN: not hiding anything (DATASET_TREE_CLEAR_DRY_RUN=1).")
    else:
        try:
            world.enable_environment_objects(ids, False)
            print(f"Hid {len(ids)} vegetation object(s) over the stretch.")
        except RuntimeError as exc:
            print(f"Failed to hide vegetation objects: {exc}")
            return

    for obj in matched:
        xy = _object_world_xy(obj)
        pos = f"({xy[0]:.1f}, {xy[1]:.1f})" if xy else "(unknown)"
        print(f"  - id={obj.id} name={obj.name} pos={pos}")


if __name__ == "__main__":
    main()
