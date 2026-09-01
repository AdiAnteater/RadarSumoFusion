# RadarSumoFusion

A fresh merge of two components into one dataset-capture system:

- **`traffic/`** - the SUMO traffic prototype. SUMO governs ALL traffic
  (scenario vehicles, ambient city vehicles, pedestrians, bicycles) and mirrors
  every actor into CARLA. It owns no clock.
- **`dataset/`** - CarlaDatasetCreation. It spawns the radar + camera rig and
  records the data (radar CSV, camera frames, actor frames), then labels and
  post-processes.
- **`fusion/`** - the new orchestration layer that runs both together, plus two
  entry points (`run_fusion.py`, `fusion_gui.py`).

## The clock model (why this doesn't freeze)

Exactly one process owns the CARLA tick: **the capture process**. When it starts
with `DATASET_SYNC_MODE=1` it puts CARLA into synchronous mode and calls
`world.tick()` at a fixed rate, so all radars co-fire on one frame.

The **SUMO runner subscribes** to that tick. It is launched with
`--external-tick`: it never calls `world.tick()`; instead it waits on
`world.wait_for_tick()` and does exactly one `simulationStep()` + one mirror per
CARLA frame. At startup it reads CARLA's `fixed_delta_seconds` and adopts it as
its SUMO step-length, so the two clocks cannot drift.

One rate drives everything (`--rate-hz`, default 20 Hz -> 0.05 s):
CARLA `fixed_delta_seconds` = radar `sensor_tick` = SUMO step-length.

Safety nets carried over from the tick-rate work:
- **Preflight heal**: every run first resets a world a dead capture may have left
  frozen in synchronous mode (via `traffic/carla_check.py --async`).
- **Standalone runner** (no `--external-tick`) self-heals and wall-clock-paces so
  it never floods CARLA with teleport RPCs.
- **Polite teardown**: children are stopped with CTRL_BREAK/SIGINT so their
  finally-blocks run (runner un-mirrors actors; the sensor rig destroys sensors;
  capture restores async mode).

## Prerequisites

1. CarlaUE4 running, map **Town10HD_Opt**.
2. `SUMO_HOME` set; venv active with `carla`, `traci`, `sumolib` installed.
3. Build the SUMO network once:
   ```
   python traffic/build_network.py
   ```

## Run it

### Headless (recommended for campaigns)
```
python run_fusion.py --scenario 3 --density 60 --duration 300 \
    --radars 8 --rate-hz 20 --pedestrians 20 --bicycles 10
```
Useful flags: `--direction WB|EB|BOTH`, `--ambient-vehicles N`, `--sumo-gui`,
`--no-cull`, `--no-label`, `--no-postprocess`, `--no-scene-cleanup`,
`--capture-base-dir PATH`, `--carla-port`, `--tm-port`. See `--help`.

### GUI
```
python fusion_gui.py
```
One panel for traffic + sensors + rate; Start; log streams in the console box.

## What the orchestrator does (order)

1. Preflight heal (reset leftover sync mode).
2. Optional scene cleanup (`dataset/world/Clear*.py`).
3. Spawn the sensor rig (`dataset/setup/RadarCameraSetupN.py`, keep-alive).
4. Start capture (owns the tick; auto-stops after `--duration`; then labels +
   post-processes per DatasetCreation defaults).
5. Start the SUMO runner (`--external-tick`, subscriber) governing traffic.
6. Wait for capture to finish, then tear everything down.

## The original GUIs are preserved

- `traffic/gui_launcher.py` - SUMO traffic only (standalone, async; still works
  for visual validation in CARLA without capture).
- `dataset/Start.py` - the DatasetCreation menu. NOTE: its "full pipeline" still
  uses CARLA-native traffic (`SpawnCarsAtPosition14`, `SpawnPedestriansAcrossMap`,
  `TrafficLight*`). For SUMO-governed fused runs use `run_fusion.py` /
  `fusion_gui.py` instead; those native-traffic scripts are left in place but are
  not part of the fused flow.

## Sensor rig placement (on the monitored stretch)

All four rigs (`RadarCameraSetup{4,8,12,14}.py`) now spawn on the monitored
stretch - the east-west boulevard (SUMO edges 20 / -20) the SUMO traffic drives
through - instead of the old native-traffic road ~77 m south. The rig straddles
the whole boulevard: one radar row on the south kerb, one on the north kerb,
`count/2` stations along the ~55 m length, with a single overview camera set back
at the west end looking east down the stretch. Radar aiming still uses each
setup's `compute_radar_yaw_toward_road()` pass, so the radars auto-orient to the
boulevard lanes at the new location.

All placement numbers live in one place (`dataset/capture/radar_layout.py`) and
are env-tunable, so you can nudge the rig live in CARLA without editing code:

| Env var | Default | Meaning |
|---|---|---|
| `DATASET_RIG_ANCHOR_X` | -1.0 | stretch mid X (CARLA world) |
| `DATASET_RIG_ANCHOR_Y` | 20.7 | boulevard cross-section mid Y |
| `DATASET_RIG_HEADING_DEG` | 0.0 | stretch direction (0 = east-west) |
| `DATASET_RIG_LENGTH_M` | 52.0 | along-stretch coverage |
| `DATASET_RIG_HALF_WIDTH_M` | 20.5 | centre -> each radar row (rows ~y0.2 / y41.2) |
| `DATASET_RIG_HEIGHT_M` | per-layout | radar mount height |
| `DATASET_CAM_HEIGHT_M` | 6.5 | camera height |
| `DATASET_CAM_END_MARGIN_M` | 16.0 | camera set-back beyond the stretch end |

Derived from the net: boulevard runs along +X, x in [-28.7, 26.4] (~55 m),
carriageways span y in [5.4 .. 36.0], mid ~(-1.0, 20.7). Lower
`DATASET_RIG_HALF_WIDTH_M` (e.g. 12) to pull the rows closer to the lanes.

## Post-run freeze fix

The orchestrator now forces CARLA back to asynchronous mode at the END of every
run (not just the start). This closes the case behind the "freeze / process dies
when I start another process after capture" report: if capture ever exits without
its `finally` restoring async (hard kill, crash, Windows Ctrl+C), the world would
otherwise stay in synchronous mode with nobody ticking, and the next process to
connect would open onto a frozen world and hang.

## Post-capture behavior

Honors DatasetCreation defaults: radar labeling and post-processing both run
after capture. Disable with `--no-label` / `--no-postprocess`.

## Notes / not done

- Component internals were preserved. The only edits to `traffic/` are the
  tick-subscriber upgrades in `runner.py` and `carla_sync.py` (external-tick
  pacing, rate alignment, standalone wall-clock pacing, sync-heal). The
  capture/sensor code in `dataset/` is unchanged.
- `dataset/dataset_paths.venv_site_packages()` walks up from `dataset/` until it
  finds a `.venv` (Windows `Lib/site-packages` or Linux `lib/python*/site-packages`).
  If your venv lives elsewhere, run with it activated — children inherit it via
  the interpreter; the lookup is only a convenience.
- Validated with `py_compile` across the whole tree; NOT run against a live CARLA
  server here. Do a short visual check in CARLA before recording a full campaign.
- Two processes means a constant one-frame (one tick) offset between SUMO state
  and the sampled sensor frame; sensors and the CARLA actor-frame log are read
  from the same frame, so labels stay self-consistent. A single-process merge
  would remove even that, at the cost of restructuring the capture loop.
```
