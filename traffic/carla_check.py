"""
carla_check.py
==============
Twenty-second diagnostic for "actors are not showing up in CARLA".

    python carla_check.py            # report only, changes nothing
    python carla_check.py --async    # ALSO force the world back to
                                     # asynchronous mode

WHY THIS EXISTS

carla_sync.py never calls world.tick(). The tick rate is owned by
DatasetCreation/capture/CaptureRadarCameraData.py, which switches the server
into synchronous mode when DATASET_SYNC_MODE=1 and restores the original
settings on the way out.

If that capture process dies without reaching its cleanup path, the SERVER
STAYS IN SYNCHRONOUS MODE. Nothing in the SUMO pipeline ticks it, so the world
is frozen: try_spawn_actor still succeeds over RPC, the runner log happily
prints "Spawned ...", and yet nothing ever appears or moves in the viewport.
It looks exactly like a broken transform or a broken spawn path, and it is
neither.

Same thing happens if you export DATASET_SYNC_MODE=1 and then run the SUMO
runner on its own, without the capture script alongside it to drive the clock.

Run this any time actors go missing, BEFORE editing carla_sync.py.
"""

import argparse
import sys

try:
    import carla
except ImportError:
    sys.exit("ERROR: carla not found - activate your venv "
             "(pip install carla==0.9.16)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--async", dest="force_async", action="store_true",
                    help="Force the world back to asynchronous mode "
                         "(fixes a server left stuck in sync mode)")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
    except Exception as e:
        sys.exit(f"ERROR: could not reach CARLA at {args.host}:{args.port}: {e}\n"
                 f"Is CarlaUE4 running?")

    settings = world.get_settings()
    carla_map = world.get_map()

    print()
    print("=" * 64)
    print("  CARLA WORLD STATE")
    print("=" * 64)
    print(f"  map                 : {carla_map.name}")
    print(f"  synchronous_mode    : {settings.synchronous_mode}")
    print(f"  fixed_delta_seconds : {settings.fixed_delta_seconds}")

    actors = world.get_actors()
    vehicles = [a for a in actors if a.type_id.startswith("vehicle.")]
    walkers = [a for a in actors if a.type_id.startswith("walker.")]
    sensors = [a for a in actors if a.type_id.startswith("sensor.")]
    mirrored = [a for a in vehicles
                if a.attributes.get("role_name", "").startswith("sumo_")]

    print(f"  vehicles in world   : {len(vehicles)} "
          f"({len(mirrored)} mirrored from SUMO)")
    print(f"  walkers in world    : {len(walkers)}")
    print(f"  sensors in world    : {len(sensors)}")
    print("=" * 64)

    if "Town10HD" not in carla_map.name:
        print()
        print("  PROBLEM: this is not Town10HD_Opt. The SUMO net was built from")
        print("  Town10HD_Opt's OpenDRIVE export, so the coordinate transform")
        print("  only means anything on that map. Load it, then re-run:")
        print("      python -c \"import carla; "
              "carla.Client('127.0.0.1',2000).load_world('Town10HD_Opt')\"")

    if settings.synchronous_mode:
        print()
        print("  PROBLEM: the world is in SYNCHRONOUS mode.")
        print()
        print("  In this mode the server only advances when something calls")
        print("  world.tick(). carla_sync.py never does that on purpose -- the")
        print("  tick belongs to CaptureRadarCameraData.py. So unless the")
        print("  capture script is running RIGHT NOW alongside the runner, the")
        print("  world is frozen and nothing you spawn will appear or move.")
        print()
        print("  Two valid setups:")
        print("    a) Capture script running with DATASET_SYNC_MODE=1, and the")
        print("       SUMO runner started with --step-length matching")
        print("       DATASET_SYNC_FIXED_DELTA_S (0.05 by default).")
        print("    b) No capture script: leave DATASET_SYNC_MODE unset/0 so the")
        print("       server free-runs, and the runner mirrors into it.")
        print()
        if args.force_async:
            # KICK the frozen world forward first. When a prior capture died in
            # sync mode, the server is stuck waiting for a tick; a fresh scenario
            # then connects onto a frozen world and its spawns never appear. Tick
            # a few frames here to flush that pending state, THEN hand the clock
            # back to asynchronous free-run.
            for _ in range(5):
                try:
                    world.tick()
                except RuntimeError:
                    break
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("  FIXED: ticked the world forward and reset it to asynchronous "
                  "mode.")
            print("  Re-run your scenario; vehicles should appear again.")
        else:
            print("  To reset it now:  python carla_check.py --async")
    else:
        print()
        print("  Sync mode is OFF, so a frozen clock is not the problem.")
        print("  Next things to check, in order:")
        print("    1. Does the runner log say '[CarlaSyncManager] Connected'?")
        print("       If not, it is running SUMO-only and never mirrors.")
        print("    2. Does it say 'Spawned ...' or 'Spawn collision ...'?")
        print("       Collisions mean leftover actors are occupying the road --")
        print("       run clear_carla_actors.py and try again.")
        print("    3. Neither message at all means no SUMO vehicle got within")
        print("       the 120 m render radius. Check the scenario is actually")
        print("       producing traffic, or pass --no-cull to mirror the whole")
        print("       city and see if they show up further out.")

    print()


if __name__ == "__main__":
    main()