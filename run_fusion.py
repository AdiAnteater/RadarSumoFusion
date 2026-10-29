"""RadarSumoFusion -- headless fused run.

SUMO governs traffic; DatasetCreation spawns the radar/camera rig and records.
Capture owns the CARLA tick; the SUMO runner subscribes to it.

Example:
    python run_fusion.py --scenario 3 --density 60 --duration 300 \
        --radars 8 --rate-hz 20 --pedestrians 20 --bicycles 10

Campaign (several scenarios, one capture) from a file saved by fusion_gui.py:
    python run_fusion.py --campaign my_campaign.json

Prereqs: CarlaUE4 running on Town10HD_Opt, SUMO_HOME set, venv active
(carla, traci, sumolib installed). Run once first:  python traffic/build_network.py
"""

from __future__ import annotations

import argparse
import sys

from fusion.config import FusionConfig, MAX_RADARS_PER_SIDE, split_radar_count
from fusion.orchestrator import run
from fusion.campaign import CampaignConfig, run_campaign


def _bool_flag(parser, name, default, help_on, help_off):
    dest = name.replace("-", "_")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=dest, action="store_true", help=help_on)
    grp.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_off)
    parser.set_defaults(**{dest: default})


def main() -> int:
    # UTF-8 stdout on Windows consoles.
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # traffic
    p.add_argument("--scenario", type=int, default=1, help="SUMO scenario 1-11")
    p.add_argument("--density", type=int, default=50, help="traffic density 1-100")
    p.add_argument("--duration", type=int, default=300, help="capture seconds")
    p.add_argument("--direction", default="BOTH", choices=["WB", "EB", "BOTH"])
    p.add_argument("--ambient-vehicles", type=int, default=20)
    p.add_argument("--pedestrians", type=int, default=20)
    p.add_argument("--bicycles", type=int, default=10)
    p.add_argument("--ambient-seed", type=int, default=42)
    p.add_argument("--render-radius", type=float, default=150.0)
    p.add_argument("--no-cull", action="store_true", help="mirror the whole city")
    p.add_argument("--sumo-gui", action="store_true", help="show sumo-gui")
    # sensors / capture
    p.add_argument("--radars", type=int, default=None,
                   help="total radars, split evenly between the kerbs (south takes "
                        "the odd one); default 8. Overridden per side by "
                        "--radars-south / --radars-north")
    p.add_argument("--radars-south", type=int, default=None,
                   help=f"radars on the south kerb row (0-{MAX_RADARS_PER_SIDE})")
    p.add_argument("--radars-north", type=int, default=None,
                   help=f"radars on the north kerb row (0-{MAX_RADARS_PER_SIDE})")
    p.add_argument("--radar-height", type=float, default=3.0,
                   help="radar mount height above the road (m)")
    _bool_flag(p, "label", True,
               "run radar labeling after capture", "skip radar labeling")
    _bool_flag(p, "postprocess", True,
               "run post-processing after capture", "skip post-processing")
    p.add_argument("--capture-base-dir", default="", help="override output root")
    # shared clock / carla
    p.add_argument("--rate-hz", type=float, default=20.0,
                   help="tick rate: CARLA fixed_delta = radar tick = SUMO step")
    p.add_argument("--carla-host", default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000, help="TrafficManager port")
    # orchestration
    _bool_flag(p, "scene-cleanup", True,
               "clear parked cars/trash first", "skip scene cleanup")
    _bool_flag(p, "clear-trees", True,
               "hide trees over the monitored stretch", "keep trees")
    p.add_argument("--sync-timeout", type=float, default=180.0)
    p.add_argument("--campaign", default="",
                   help="run a campaign JSON (saved from fusion_gui.py); all other "
                        "flags are ignored except --carla-host/--carla-port")
    args = p.parse_args()

    if args.campaign:
        ccfg = CampaignConfig.load(args.campaign)
        ccfg.carla_host, ccfg.carla_port = args.carla_host, args.carla_port
        return 0 if run_campaign(ccfg).ok else 1

    south, north = split_radar_count(8 if args.radars is None else args.radars)
    if args.radars_south is not None:
        south = args.radars_south
    if args.radars_north is not None:
        north = args.radars_north

    cfg = FusionConfig(
        scenario=args.scenario, density=args.density, duration=args.duration,
        direction=args.direction, ambient_vehicles=args.ambient_vehicles,
        pedestrians=args.pedestrians, bicycles=args.bicycles,
        ambient_seed=args.ambient_seed, render_radius=args.render_radius,
        no_cull=args.no_cull, sumo_gui=args.sumo_gui,
        radars_south=south, radars_north=north, radar_height_m=args.radar_height,
        label=args.label, postprocess=args.postprocess,
        capture_base_dir=args.capture_base_dir,
        rate_hz=args.rate_hz, carla_host=args.carla_host, carla_port=args.carla_port,
        traffic_manager_port=args.tm_port,
        scene_cleanup=args.scene_cleanup, clear_trees=args.clear_trees,
        sync_timeout=args.sync_timeout,
    )
    result = run(cfg)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
