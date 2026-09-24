"""Headless validation of the 11 traffic scenarios (SUMO only, no CARLA).

Builds every scenario exactly the way runner.py does (density substitution,
map-edge routes, ambient city layer, bus stop, scenario controllers) and runs it
in headless SUMO, then reports what actually happened on the monitored stretch:
occupancy, speeds, stopped fraction, lane changes, pedestrians and bicycles,
SUMO teleports / collisions / emergency braking, and whether each scenario's
signature mechanism fired (shockwave seeds held, blocker parked, occlusion pairs
locked, buses dwelling at the stop).

    python validate_scenarios.py                       # all 11, defaults
    python validate_scenarios.py --scenarios 4 8 11    # subset
    python validate_scenarios.py --duration 240 --density 60 --pedestrians 20 \
        --bicycles 10 --ambient-vehicles 30 --out validation_report

Writes <out>.json and <out>.md (a table you can paste into the methods
section). Exit code 1 if any scenario fails a hard check.

Runs on Windows or Linux; needs SUMO_HOME or `sumo` on PATH and traci/sumolib.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import traci  # noqa: E402

import runner  # noqa: E402
import ambient_traffic  # noqa: E402

MONITORED = set(runner.MONITORED_EDGES)
STOPPED_MPS = 0.1

# Scenario-specific expectations (hard checks). Everything else is reported.
EXPECT = {
    4: "shockwave seeds must enter the stretch and be held (StopAndGoController)",
    5: "at least one bus must dwell at BS_main",
    8: "lane_blocker_wb must park on edge 20 lane 3",
    11: "at least one occ_pair must be locked on the stretch",
}


def sumo_binary() -> str:
    home = os.environ.get("SUMO_HOME", "")
    for cand in (os.path.join(home, "bin", "sumo.exe"), os.path.join(home, "bin", "sumo")):
        if home and os.path.isfile(cand):
            return cand
    return "sumo"


def build_files(scenario_id, density, duration, direction, ambient_vehicles,
                pedestrians, bicycles, seed, step_length):
    meta = runner.SCENARIOS[scenario_id]
    vph = runner.density_to_vph(density)
    density_map = runner.build_density_map(vph, direction)
    route_file = runner.substitute_density(
        os.path.join(runner.ROUTES_DIR, meta["file"]), density_map, duration)
    city_mode = bool(ambient_vehicles or pedestrians or bicycles)
    if city_mode:
        with open(route_file, "r", encoding="utf-8") as f:
            rc = f.read()
        amb = ambient_traffic.generate_ambient(
            runner.NET_FILE, seed, ambient_vehicles, pedestrians, bicycles, duration)
        rc = ambient_traffic.inject_ambient(rc, amb)
        with open(route_file, "w", encoding="utf-8") as f:
            f.write(rc)
    add_files = [runner.ADD_FILE]
    busstop = None
    if scenario_id == 5:
        busstop = runner.write_busstop_additional()
        add_files.append(busstop)
    cfg = runner.write_sumocfg(runner.NET_FILE, route_file, add_files, duration,
                               step_length=step_length,
                               ignore_route_errors=city_mode, seed=seed)
    tmp = [route_file, cfg] + ([busstop] if busstop else [])
    return cfg, tmp


def run_one(scenario_id, args, log_path):
    cfg, tmp_files = build_files(
        scenario_id, args.density, args.duration, args.direction,
        args.ambient_vehicles, args.pedestrians, args.bicycles, args.seed,
        args.step_length)
    res = {
        "scenario": scenario_id,
        "name": runner.SCENARIOS[scenario_id]["name"],
        "ok": True, "problems": [], "notes": [],
    }
    cmd = [sumo_binary(), "-c", cfg, "--start", "--log", log_path,
           "--no-step-log", "--duration-log.disable", "true"]
    t0 = time.time()
    try:
        traci.start(cmd, label=f"val{scenario_id}")
    except Exception as exc:  # noqa: BLE001
        res["ok"] = False
        res["problems"].append(f"SUMO failed to start: {exc}")
        _cleanup(tmp_files)
        return res

    if args.stretch_signals == "green":
        import stretch_signals
        stretch_signals.apply_corridor_priority(runner.NET_FILE, verbose=False)
    controller = runner.get_controller(scenario_id, args.duration)
    controller.on_start()
    if scenario_id == 5:
        runner.spawn_buses_with_stop(args.duration)

    subscribed = set()
    last_lane = {}          # vid -> lane index while on stretch
    on_stretch_prev = {}    # vid -> bool
    crossed = set()
    lane_changes = 0
    occ_samples = []
    speed_sum = 0.0
    veh_steps = 0
    stopped_steps = 0
    ped_stretch_max = 0
    ped_seen = set()
    bike_seen = set()
    ped_on_crossing_steps = 0
    ped_steps = 0
    veh_ids_seen = set()
    departed = arrived = 0
    teleports = collisions = 0
    buses_dwelled = set()
    blocker_parked = False
    passes = 0
    pos_prev = {}
    sub_vars = [traci.constants.VAR_ROAD_ID, traci.constants.VAR_SPEED,
                traci.constants.VAR_LANE_INDEX, traci.constants.VAR_TYPE,
                traci.constants.VAR_LANEPOSITION]
    steps = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and \
                traci.simulation.getTime() < args.duration:
            traci.simulationStep()
            controller.step()
            steps += 1
            for vid in traci.simulation.getDepartedIDList():
                departed += 1
                veh_ids_seen.add(vid)
                if vid not in subscribed:
                    traci.vehicle.subscribe(vid, sub_vars)
                    subscribed.add(vid)
            arrived += traci.simulation.getArrivedNumber()
            teleports += traci.simulation.getStartingTeleportNumber()
            collisions += traci.simulation.getCollidingVehiclesNumber()
            results = traci.vehicle.getAllSubscriptionResults()
            n_on = 0
            on_now = {}
            for vid, vals in results.items():
                road = vals.get(traci.constants.VAR_ROAD_ID, "")
                vtype = vals.get(traci.constants.VAR_TYPE, "")
                if vtype in runner_bike_types():
                    bike_seen.add(vid)
                on = road in MONITORED
                if on:
                    n_on += 1
                    crossed.add(vid)
                    spd = float(vals.get(traci.constants.VAR_SPEED, 0.0))
                    speed_sum += spd
                    veh_steps += 1
                    if spd < STOPPED_MPS:
                        stopped_steps += 1
                    li = vals.get(traci.constants.VAR_LANE_INDEX, -1)
                    if on_stretch_prev.get(vid) and last_lane.get(vid, li) != li:
                        lane_changes += 1
                    last_lane[vid] = li
                    on_now[vid] = (road, li, float(vals.get(traci.constants.VAR_LANEPOSITION, 0.0)))
                    if scenario_id == 8 and vid == runner.BottleneckController.BLOCKER_ID \
                            and road == "20" and li == 3 and spd < STOPPED_MPS:
                        blocker_parked = True
                    if scenario_id == 5 and vid.startswith("scheduled_bus_") and spd < STOPPED_MPS:
                        try:
                            if traci.vehicle.getStopState(vid) & 16:   # at bus stop
                                buses_dwelled.add(vid)
                        except traci.TraCIException:
                            pass
                on_stretch_prev[vid] = on
            occ_samples.append(n_on)
            # Passing events: two vehicles on the same stretch edge in different
            # lanes swap longitudinal order between consecutive steps.
            for a, (ra, la, pa) in on_now.items():
                for b, (rb, lb, pb) in on_now.items():
                    if a >= b or ra != rb or la == lb:
                        continue
                    prev = pos_prev.get((a, b))
                    now_sign = pa > pb
                    if prev is not None and prev != now_sign:
                        passes += 1
                    pos_prev[(a, b)] = now_sign
            for key in [k for k in pos_prev if k[0] not in on_now or k[1] not in on_now]:
                pos_prev.pop(key, None)

            if args.pedestrians and steps % 20 == 0:   # 1 Hz sampling for peds
                try:
                    pids = traci.person.getIDList()
                except traci.TraCIException:
                    pids = ()
                n_ped_near = 0
                for pid in pids:
                    ped_seen.add(pid)
                    lane = traci.person.getLaneID(pid)
                    ped_steps += 1
                    if lane.startswith(":") and ("_c" in lane or "_w" in lane):
                        ped_on_crossing_steps += 1
                    if traci.person.getRoadID(pid) in MONITORED:
                        n_ped_near += 1
                ped_stretch_max = max(ped_stretch_max, n_ped_near)
    except Exception as exc:  # noqa: BLE001
        res["ok"] = False
        res["problems"].append(f"exception during run: {exc}")
    finally:
        try:
            pending = len(traci.simulation.getPendingVehicles())
        except Exception:  # noqa: BLE001
            pending = -1
        try:
            controller.on_stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            traci.close()
        except Exception:  # noqa: BLE001
            pass
        _cleanup(tmp_files)

    wall = time.time() - t0
    warn = parse_log(log_path)
    res.update({
        "wall_s": round(wall, 1),
        "sim_s": args.duration,
        "vehicles_loaded": len(veh_ids_seen),
        "departed": departed, "arrived": arrived,
        "insertion_backlog_at_end": pending,
        "crossed_stretch": len(crossed),
        "stretch_occupancy_mean": round(sum(occ_samples) / max(len(occ_samples), 1), 2),
        "stretch_occupancy_max": max(occ_samples) if occ_samples else 0,
        "stretch_mean_speed_mps": round(speed_sum / max(veh_steps, 1), 2),
        "stretch_stopped_fraction": round(stopped_steps / max(veh_steps, 1), 3),
        "stretch_lane_changes": lane_changes,
        "stretch_passes": passes,
        "teleports": teleports, "collisions": collisions,
        "pedestrians_seen": len(ped_seen),
        "ped_max_on_stretch": ped_stretch_max,
        "ped_crossing_fraction": round(ped_on_crossing_steps / max(ped_steps, 1), 3),
        "bicycles_seen": len(bike_seen),
        "sumo_warnings": warn,
    })

    # Hard checks.
    if scenario_id == 4:
        entered = len(getattr(controller, "_entered", ()))
        res["shockwave_seeds_entered"] = entered
        if entered == 0:
            res["problems"].append(EXPECT[4])
        elif res["stretch_stopped_fraction"] < 0.05:
            res["notes"].append("seeds entered but stretch stopped fraction is low")
    if scenario_id == 5:
        res["buses_dwelled_at_stop"] = len(buses_dwelled)
        if not buses_dwelled:
            res["problems"].append(EXPECT[5])
    if scenario_id == 8:
        res["blocker_parked"] = blocker_parked
        if not blocker_parked:
            res["problems"].append(EXPECT[8])
    if scenario_id == 11:
        matched = controller_total_matches(controller)
        res["occlusion_pairs_locked"] = matched
        if matched == 0:
            res["problems"].append(EXPECT[11])
    if res["crossed_stretch"] == 0:
        res["problems"].append("no vehicle crossed the monitored stretch")
    if warn.get("route errors", 0) and not (args.ambient_vehicles or args.pedestrians or args.bicycles):
        res["problems"].append(f"{warn['route errors']} route error(s)")
    if collisions:
        res["notes"].append(f"{collisions} collision step(s) reported by SUMO")
    if teleports:
        res["notes"].append(f"{teleports} teleport(s)")
    if pending and pending > 0:
        res["notes"].append(f"{pending} vehicles still waiting to be inserted at the end "
                            "(demand above what the entry edges can absorb)")
    if res["problems"]:
        res["ok"] = False
    return res


def controller_total_matches(controller):
    return len(getattr(controller, "_ever_matched", ()))


def runner_bike_types():
    return {"amb_bike", "bike", "bicycle"}


def parse_log(path):
    counts = {}
    if not os.path.isfile(path):
        return counts
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Warning:" not in line and "Error:" not in line:
                continue
            key = "other"
            for k, pat in (("teleport", "Teleporting"), ("collision", "collision"),
                           ("emergency braking", "emergency braking"),
                           ("route errors", "route"), ("person", "person"),
                           ("bus stop", "busStop"), ("insertion", "insert")):
                if pat.lower() in line.lower():
                    key = k
                    break
            counts[key] = counts.get(key, 0) + 1
    try:
        os.remove(path)
    except OSError:
        pass
    return counts


def _cleanup(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def write_markdown(results, args, path):
    cols = ["scenario", "name", "ok", "crossed_stretch", "stretch_occupancy_mean",
            "stretch_occupancy_max", "stretch_mean_speed_mps", "stretch_stopped_fraction",
            "stretch_lane_changes", "stretch_passes", "pedestrians_seen", "bicycles_seen", "teleports",
            "collisions", "insertion_backlog_at_end"]
    lines = [f"Scenario validation (density={args.density}, direction={args.direction}, "
             f"duration={args.duration}s, signals={args.stretch_signals}, ambient={args.ambient_vehicles}, "
             f"pedestrians={args.pedestrians}, bicycles={args.bicycles}, seed={args.seed})", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in results:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    for r in results:
        extra = {k: v for k, v in r.items() if k in (
            "shockwave_seeds_entered", "buses_dwelled_at_stop", "blocker_parked",
            "occlusion_pairs_locked", "sumo_warnings")}
        if extra or r["problems"] or r["notes"]:
            lines.append(f"- S{r['scenario']:02d} {r['name']}: {extra}")
            for p in r["problems"]:
                lines.append(f"    - PROBLEM: {p}")
            for n in r["notes"]:
                lines.append(f"    - note: {n}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenarios", type=int, nargs="*", default=sorted(runner.SCENARIOS))
    p.add_argument("--density", type=int, default=60)
    p.add_argument("--duration", type=int, default=240)
    p.add_argument("--direction", default="BOTH", choices=("WB", "EB", "BOTH"))
    p.add_argument("--ambient-vehicles", type=int, default=30)
    p.add_argument("--pedestrians", type=int, default=20)
    p.add_argument("--bicycles", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step-length", type=float, default=0.05)
    p.add_argument("--stretch-signals", default="green", choices=("green", "static"))
    p.add_argument("--out", default=os.path.join(HERE, "validation_report"))
    args = p.parse_args()

    if not os.path.isfile(runner.NET_FILE):
        sys.exit(f"ERROR: {runner.NET_FILE} missing. Run build_network.py first.")

    results = []
    for sid in args.scenarios:
        log_path = tempfile.mktemp(suffix=f"_s{sid}.log")
        print(f"[validate] scenario {sid:02d} {runner.SCENARIOS[sid]['name']} ...", flush=True)
        r = run_one(sid, args, log_path)
        status = "OK " if r["ok"] else "FAIL"
        print(f"[validate]   {status} crossed={r.get('crossed_stretch')} "
              f"occ={r.get('stretch_occupancy_mean')}/{r.get('stretch_occupancy_max')} "
              f"v={r.get('stretch_mean_speed_mps')} stopped={r.get('stretch_stopped_fraction')} "
              f"lc={r.get('stretch_lane_changes')} passes={r.get('stretch_passes')} ped={r.get('pedestrians_seen')} "
              f"bike={r.get('bicycles_seen')} tele={r.get('teleports')} "
              f"coll={r.get('collisions')} warn={r.get('sumo_warnings')} "
              f"({r.get('wall_s')}s)", flush=True)
        for prob in r["problems"]:
            print(f"[validate]   PROBLEM: {prob}", flush=True)
        for note in r["notes"]:
            print(f"[validate]   note: {note}", flush=True)
        results.append(r)

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    write_markdown(results, args, args.out + ".md")
    print(f"[validate] wrote {args.out}.json and {args.out}.md", flush=True)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
