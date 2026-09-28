"""
carla_cleanup.py
================
Shared helper to remove vehicle (and optionally pedestrian) actors from a
running CARLA world.

Used by:
  - carla_sync.py    -> clears leftover/parked vehicles when the sync manager
                        connects (start) and sweeps again on shutdown (stop)
  - clear_carla_actors.py -> standalone manual wipe you can run any time

Why this exists: the CARLA server keeps running independently of runner.py.
If a run crashes or is hard-killed, its spawned actors LEAK and stay in the
world. Town10HD_Opt also ships with parked-vehicle actors. Either one sitting
on a spawn point makes try_spawn_actor() report a collision, so the SUMO
mirror vehicle never appears (the "spawn collision ... attempt N" cascade),
and a leftover physics-on car looks "stuck" while the physics-off mirror cars
teleport straight through it. Clearing vehicles before the run removes both.

SAFETY: only vehicle.* and (optionally) walker.* / controller.ai.walker
actors are destroyed. sensor.* actors -- the dataset radars and cameras --
are NEVER touched, so this is safe to run while the capture rig is set up.
"""

import carla


def _default_do_tick() -> bool:
    """Tick after the batch only when nobody else owns the clock.

    In a fused / campaign run the capture process owns world.tick() and every
    other client sets DATASET_EXTERNAL_TICK=1. apply_batch_sync(batch, True)
    TICKS the world in synchronous mode, i.e. a second ticker would inject extra
    frames into the capture's clock (every runner start and every campaign
    cleanup did this). With an external ticker the destroys are applied on the
    capture's next tick anyway.
    """
    import os
    return os.environ.get("DATASET_EXTERNAL_TICK", "").strip() not in ("1", "true", "yes")


def destroy_all_vehicles(client, world, include_walkers=False, verbose=True,
                         do_tick=None):
    """Destroy every vehicle actor in the world (batched). Returns the count.

    include_walkers=True also stops walker AI controllers and destroys both
    the controllers and the pedestrians they drive.
    """
    actors = world.get_actors()

    vehicles = list(actors.filter("vehicle.*"))
    batch = [carla.command.DestroyActor(a.id) for a in vehicles]

    walker_count = 0
    if include_walkers:
        # Stop the AI controllers before destroying, or CARLA logs orphaned
        # controller warnings.
        controllers = list(actors.filter("controller.ai.walker"))
        for c in controllers:
            try:
                c.stop()
            except RuntimeError:
                pass
        walkers = list(actors.filter("walker.pedestrian.*"))
        walker_count = len(walkers)
        batch += [carla.command.DestroyActor(a.id) for a in controllers]
        batch += [carla.command.DestroyActor(a.id) for a in walkers]

    if not batch:
        if verbose:
            print("[cleanup] Nothing to remove (no vehicles/pedestrians present).")
        return 0

    # apply_batch_sync with do_tick=True guarantees the destroys are applied
    # before we return, even if no one else is ticking the world right now.
    client.apply_batch_sync(batch, _default_do_tick() if do_tick is None else bool(do_tick))

    if verbose:
        msg = f"[cleanup] Removed {len(vehicles)} vehicle(s)"
        if include_walkers:
            msg += f" and {walker_count} pedestrian(s)"
        print(msg + ".")

    return len(vehicles) + walker_count
