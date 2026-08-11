#!/usr/bin/env python3
"""Diversity campaign launcher for the CARLA radar dataset.

Runs N captures with **Latin-Hypercube-sampled** configs to maximize diversity at small
N (10-12). Map (Town10HD_Opt) and radar count are FIXED — varying them would change the
fusion model's input structure (per-sensor positional encoding / topology mask are sized
by M), so they are pinned per campaign, not sampled. Everything M-safe is varied:

  traffic density / speed / following-gap, truck fraction, crosser count, AI crossing
  fraction, running fraction, pedestrian spawn spread, the crossing-band x-position
  (slid across the radar zone to decorrelate ped position from class), and the
  traffic-light regime (green-wave vs automatic cycle).

Each run auto-stops (DATASET_CAPTURE_DURATION_S), auto-labels and auto-post-processes
(canonical PostProcessDataset realism), then the launcher tears down the pipeline (CARLA
server is left up) and starts the next config. A manifest CSV records run -> config -> dir.

Session fixes baked in as fixed env: ACTOR_MAX_DWELL_S=25 (no single-instance point
domination), gap-check OFF + car-yield ON (no flipped/stuck cars), label+postprocess on.

Usage:
  python tools/run_campaign.py --n 12 --dry-run            # print the sampled design, exit
  python tools/run_campaign.py --n 12 --duration 300       # run the campaign
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

DC_ROOT = Path(__file__).resolve().parents[1]          # .../DatasetCreation
LAST_DIR_PTR = DC_ROOT / "capture" / ".last_dataset_capture_dir"

MAP = "Town10HD_Opt"
RADAR_COUNT = 8

# Continuous diversity axes (env_key, lo, hi, fmt). Sampled by Latin Hypercube.
CONT_AXES = [
    ("DATASET_TARGET_CAR_COUNT",            12,   36,   "int"),
    ("DATASET_VEHICLE_SPEED_REDUCTION_PCT",  0,   45,   "f1"),
    ("DATASET_SAFE_FOLLOWING_DISTANCE_M",    2.0,  6.0, "f1"),
    ("DATASET_VEHICLE_TRUCK_FRACTION",       0.05, 0.20,"f2"),
    ("DATASET_PED_CROSSING_COUNT",           0,    8,   "int"),
    ("DATASET_PED_CROSSING_FACTOR",          0.2,  0.8, "f2"),
    ("DATASET_PED_RUNNING_FRACTION",         0.0,  0.4, "f2"),
    ("CROSS_BAND_CENTER_X",                 -14,   51,   "f1"),   # -> X_MIN/X_MAX
]
CROSS_BAND_HALF_WIDTH = 15.0          # 30 m crossing band, slid across radar zone [-29,66]

# Fixed for every run: the session's quality fixes + pinned behavior.
FIXED_ENV = {
    "DATASET_ACTOR_MAX_DWELL_S":        "25",   # recycle long-lived actors (no 1-truck=22% domination)
    "DATASET_PED_CROSSING_GAP_CHECK":   "0",    # OFF (curb gap-check causes yield standoff)
    "DATASET_PED_CROSSING_CAR_YIELD":   "1",    # ON  (mid-cross yield -> no flipped/stuck cars)
    "DATASET_PED_SPAWN_RADIUS_M":       "60",   # FIXED (not swept) — matches sensor_capture_20260608_060024 default
    "DATASET_LABEL_RADAR_AFTER_CAPTURE":"1",
    "DATASET_POSTPROCESS_AFTER_CAPTURE":"1",
    "DATASET_AUTOMATIC_TRAFFIC_LIGHTS": "1",    # lights cycle so the corridor flows
    # Pinned rig geometry (not swept): low mount + shallow tilt; VFOV per user.
    "DATASET_RIG_HEIGHT_M":             "3",
    "DATASET_RADAR_PITCH_DEG":          "8",
    "DATASET_RADAR_VERTICAL_FOV_DEG":   "40",
}

# Teardown patterns (kept narrow so the CARLA server + other sessions survive).
TEARDOWN_PATTERNS = [
    "Start.py --radar-count", "setup/RadarCameraSetup", "world/SpawnCarsAtPosition14.py",
    "world/SpawnPedestriansAcrossMap.py", "world/TrafficLightSetup.py",
    "world/TrafficLightControl.py", "run_sim.sh",
]


def lhs(n: int, k: int, rng: random.Random) -> list[list[float]]:
    """Latin Hypercube in [0,1]^k: each column stratified into n bins, independently shuffled."""
    cols = []
    for _ in range(k):
        perm = list(range(n))
        rng.shuffle(perm)
        cols.append([(perm[i] + rng.random()) / n for i in range(n)])
    return [[cols[j][i] for j in range(k)] for i in range(n)]


def _fmt(val: float, fmt: str) -> str:
    if fmt == "int":
        return str(int(round(val)))
    if fmt == "f1":
        return f"{val:.1f}"
    return f"{val:.2f}"


def build_configs(n: int, base_seed: int) -> list[dict]:
    rng = random.Random(base_seed)
    U = lhs(n, len(CONT_AXES), rng)
    # Balanced traffic-light regime: green-wave (smooth flow) vs automatic cycle (red/green queueing).
    waves = ([1, 0] * (n // 2 + 1))[:n]
    rng.shuffle(waves)
    configs = []
    for i in range(n):
        env = dict(FIXED_ENV)
        for (key, lo, hi, fmt), u in zip(CONT_AXES, U[i]):
            val = lo + u * (hi - lo)
            if key == "CROSS_BAND_CENTER_X":
                env["DATASET_PED_CROSSING_X_MIN"] = f"{val - CROSS_BAND_HALF_WIDTH:.1f}"
                env["DATASET_PED_CROSSING_X_MAX"] = f"{val + CROSS_BAND_HALF_WIDTH:.1f}"
            else:
                env[key] = _fmt(val, fmt)
        env["DATASET_TRAFFIC_LIGHTS_GREEN_WAVE"] = str(waves[i])
        env["DATASET_SEED"] = str(base_seed + 1001 + i)   # unique + reproducible per run
        configs.append(env)
    return configs


def manifest_rows(configs: list[dict]) -> list[dict]:
    keys = (["DATASET_SEED", "DATASET_TARGET_CAR_COUNT", "DATASET_VEHICLE_SPEED_REDUCTION_PCT",
             "DATASET_SAFE_FOLLOWING_DISTANCE_M", "DATASET_VEHICLE_TRUCK_FRACTION",
             "DATASET_PED_CROSSING_COUNT", "DATASET_PED_CROSSING_FACTOR",
             "DATASET_PED_RUNNING_FRACTION", "DATASET_PED_SPAWN_RADIUS_M",
             "DATASET_PED_CROSSING_X_MIN", "DATASET_PED_CROSSING_X_MAX",
             "DATASET_TRAFFIC_LIGHTS_GREEN_WAVE"])
    return [{"run": i + 1, **{k: c[k] for k in keys}} for i, c in enumerate(configs)]


def print_design(configs: list[dict]) -> None:
    rows = manifest_rows(configs)
    cols = list(rows[0].keys())
    short = {c: c.replace("DATASET_", "").replace("PED_CROSSING_", "X").replace("VEHICLE_", "")[:10] for c in cols}
    print("  ".join(f"{short[c]:>10}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>10}" for c in cols))


def _pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL).returncode == 0


def teardown() -> None:
    for sig in ("-INT", "-9"):
        for pat in TEARDOWN_PATTERNS:
            subprocess.run(["pkill", sig, "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)


def run_one(env_overrides: dict, idx: int, duration: int) -> str | None:
    env = dict(os.environ)
    env.update(env_overrides)
    env["DATASET_CAPTURE_DURATION_S"] = str(duration)
    env["KEEP_SERVER"] = "1"
    env["MAP"] = MAP
    log_path = f"/tmp/campaign_run{idx:02d}.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ["setsid", "bash", "-c", f'printf "1\\n" | ./run_sim.sh --radar-count {RADAR_COUNT}'],
            cwd=str(DC_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
    # 1) wait for the capture process to appear (startup: map load + radar setup + spawns)
    t0 = time.time()
    while not _pgrep("capture/CaptureRadarCameraData.py"):
        if time.time() - t0 > 300:
            print(f"  [run {idx}] capture never started; tearing down.", flush=True)
            teardown()
            return None
        time.sleep(5)
    # 2) wait for it to finish: records `duration` s, then drains/exports/labels/post-processes, then exits
    deadline = time.time() + duration + 3600          # generous: labeling 5 min of data can take ~25 min
    while _pgrep("capture/CaptureRadarCameraData.py"):
        if time.time() > deadline:
            print(f"  [run {idx}] capture exceeded deadline; tearing down.", flush=True)
            break
        time.sleep(10)
    run_dir = LAST_DIR_PTR.read_text(encoding="utf-8").strip() if LAST_DIR_PTR.is_file() else ""
    teardown()
    return run_dir or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=12, help="number of captures (10-12 recommended)")
    ap.add_argument("--duration", type=int, default=300, help="record seconds per capture")
    ap.add_argument("--seed", type=int, default=20260608, help="campaign design seed (reproducible)")
    ap.add_argument("--dry-run", action="store_true", help="print the sampled design and exit")
    args = ap.parse_args()

    configs = build_configs(args.n, args.seed)
    print(f"Campaign: {args.n} captures, map={MAP}, radar_count={RADAR_COUNT}, "
          f"duration={args.duration}s, design_seed={args.seed}\n", flush=True)
    print_design(configs)
    if args.dry_run:
        print("\n(dry run — nothing launched)")
        return 0

    if not _pgrep("CarlaUE4-Linux-Shipping"):
        print("CARLA server not running. Start it (or run_sim.sh will), then re-run.", flush=True)
    manifest = DC_ROOT / "config" / f"campaign_manifest_{args.seed}.csv"
    rows = manifest_rows(configs)
    for i, (cfg, row) in enumerate(zip(configs, rows), start=1):
        print(f"\n=== [{i}/{args.n}] launching (seed={cfg['DATASET_SEED']}) ===", flush=True)
        t0 = time.time()
        run_dir = run_one(cfg, i, args.duration)
        row["capture_dir"] = run_dir or "FAILED"
        row["minutes"] = f"{(time.time() - t0) / 60:.1f}"
        with manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows[: i])
        print(f"  [{i}/{args.n}] -> {run_dir} ({row['minutes']} min). manifest: {manifest}", flush=True)
    print(f"\nCampaign done. Manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
