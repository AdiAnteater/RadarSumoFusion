# RadarSumoFusion

Dataset creation and traffic orchestration for autonomous-driving radar/camera
datasets in CARLA, using SUMO as the traffic simulator.

## Project Structure

```
RadarSumoFusion/
├── traffic/
│   ├── runner.py
│   ├── carla_sync.py
│   └── ...
│
├── dataset/
│   ├── capture/
│   ├── setup/
│   ├── world/
│   ├── Start.py
│   └── ...
│
├── fusion/
│   └── orchestration layer
│
├── run_fusion.py
└── fusion_gui.py
```

## Components

- **`traffic/`** — SUMO traffic integration. SUMO controls scenario vehicles,
  ambient vehicles, pedestrians, and bicycles, and mirrors them into CARLA.
- **`dataset/`** — Radar/camera sensor setup, data capture, actor-frame
  recording, labeling, and post-processing.
- **`fusion/`** — Coordinates SUMO traffic and dataset capture.
- **`run_fusion.py`** — Headless entry point for dataset-generation campaigns.
- **`fusion_gui.py`** — GUI entry point for configuring and starting fused runs.

The original standalone tools remain available:

- **`traffic/gui_launcher.py`** — SUMO traffic validation without dataset capture.
- **`dataset/Start.py`** — Original DatasetCreation pipeline. Its full pipeline
  uses CARLA-native traffic and is not part of the SUMO-fused workflow.

## System Architecture

The fused pipeline uses a single CARLA clock owner.

**Dataset capture owns the CARLA tick.** In synchronous mode, capture configures
CARLA with the requested fixed timestep and calls `world.tick()` at that rate;
radar sensors and cameras operate on that same simulation timeline.

**The SUMO runner subscribes to the tick.** Started with `--external-tick`, it
does not call `world.tick()`; for every CARLA frame it performs exactly one
`simulationStep()` and mirrors the resulting actors into CARLA.

```
Capture process
      │
      ├── world.tick()
      │
      ├── Radar / Camera capture
      │
      └── SUMO runner waits for tick
                    │
                    └── simulationStep() + actor mirroring
```

The default rate is **20 Hz (0.05 s timestep)**. The following are aligned so the
clocks cannot drift:

- CARLA `fixed_delta_seconds`
- Radar `sensor_tick`
- SUMO step-length

## How the Issues Are Fixed

- **Single tick owner.** Only the capture process calls `world.tick()`. The SUMO
  runner starts with `--external-tick` and waits on the tick instead of driving
  its own clock (`traffic/runner.py`, `traffic/carla_sync.py:wait_for_tick`),
  removing the multi-process clock contention that caused freezes.
- **Clock alignment.** The runner reads CARLA's `fixed_delta_seconds` at startup
  and adopts it as the SUMO step-length; the same rate drives the radar
  `sensor_tick`, so the three clocks stay locked.
- **Standalone pacing.** Without `--external-tick`, the runner uses wall-clock
  pacing instead of flooding CARLA with actor updates.
- **Stale sync recovery.** The orchestrator resets CARLA to asynchronous mode
  before each run and again at the end (`fusion/orchestrator.py:_heal_async`,
  `traffic/carla_check.py --async`), so a crashed prior run cannot leave the
  world frozen.
- **Graceful shutdown.** Child processes are stopped with signals that let their
  cleanup handlers run (`fusion/orchestrator.py:_interrupt`) — un-mirroring
  actors, destroying sensor rigs, and restoring CARLA's simulation mode.
- **Sensor rig placement.** Rig coordinates are centralized and env-tunable in
  `dataset/capture/radar_layout.py`.

## Prerequisites

1. Start CARLA with map **Town10HD_Opt**.
2. Set `SUMO_HOME`.
3. Activate the Python virtual environment with `carla`, `traci`, and `sumolib`.
4. Build the SUMO network once:

   ```
   python traffic/build_network.py
   ```

## Running the Fused Pipeline

### Headless

Recommended for dataset campaigns:

```
python run_fusion.py \
    --scenario 3 \
    --density 60 \
    --duration 300 \
    --radars 8 \
    --rate-hz 20 \
    --pedestrians 20 \
    --bicycles 10
```

Useful options:

- `--direction WB|EB|BOTH`
- `--ambient-vehicles N`
- `--sumo-gui`
- `--no-cull`
- `--no-label`
- `--no-postprocess`
- `--no-scene-cleanup`
- `--capture-base-dir PATH`
- `--carla-port`
- `--tm-port`

Run `python run_fusion.py --help` for the complete option list.

### GUI

```
python fusion_gui.py
```

The GUI provides configuration for traffic, sensors, and simulation rate, with
process logs displayed in the console panel.

## Fusion Run Sequence

1. Reset any stale CARLA synchronous-mode state.
2. Optionally clean the previous scene.
3. Start the radar/camera sensor rig.
4. Start dataset capture and make it the CARLA tick owner.
5. Start the SUMO runner with `--external-tick`.
6. SUMO advances once per CARLA frame and mirrors traffic.
7. Capture runs until the requested duration, then labels and post-processes.
8. Stop the SUMO runner and sensor rig.
9. Restore CARLA to asynchronous mode.

## Sensor Rig Configuration

`dataset/setup/RadarCameraSetup.py` places the radar/camera rig on the
SUMO-monitored east-west boulevard. The rig straddles the stretch with one radar
row on each side and a camera set back at the west end looking down its length.
Placement is centralized in `dataset/capture/radar_layout.py`.

The radar count per row is variable (0-16 per side, at least one total). Set it
in the GUI ("Radars south" / "Radars north"), in the campaign JSON
(`radars_south`, `radars_north`, `radar_height_m`), or headless with
`--radars N` (even split) or `--radars-south S --radars-north N`. Each row
spreads its radars evenly along the stretch; radars are numbered `R1..RN` west
to east, south before north.

Default monitored stretch:

- Direction: East-West
- X range: approximately -28.7 to 26.4
- Y range: approximately 5.4 to 36.0
- Center: approximately (-1.0, 20.7)

The rig position can be adjusted through environment variables without modifying
the layout code:

| Environment variable       | Default         | Description                        |
|----------------------------|-----------------|------------------------------------|
| `DATASET_RIG_ANCHOR_X`     | -1.0            | Center X position                  |
| `DATASET_RIG_ANCHOR_Y`     | 20.7            | Center Y position                  |
| `DATASET_RIG_HEADING_DEG`  | 0.0             | Rig heading                        |
| `DATASET_RIG_LENGTH_M`     | 52.0            | Coverage length                    |
| `DATASET_RIG_HALF_WIDTH_M` | 20.5            | Distance from center to radar rows |
| `DATASET_RIG_HEIGHT_M`     | 3.0             | Radar mounting height              |
| `DATASET_RADARS_SOUTH`     | 4               | Radars on the south row            |
| `DATASET_RADARS_NORTH`     | 4               | Radars on the north row            |
| `DATASET_CAM_HEIGHT_M`     | 6.5             | Camera height                      |
| `DATASET_CAM_END_MARGIN_M` | 16.0            | Camera setback                     |

For example, `DATASET_RIG_HALF_WIDTH_M=12` moves the radar rows closer to the
traffic lanes.

## Data Capture

The fused pipeline records:

- Radar data
- Camera frames
- CARLA actor frames
- SUMO-controlled traffic state

Radar labeling and post-processing run automatically after capture using the
existing DatasetCreation defaults. They can be disabled with `--no-label` and
`--no-postprocess`.

## Known Issues

### CARLA freezes on consecutive runs

CARLA can still freeze when a second scenario is started without restarting the
CARLA server. The end-of-run async reset reduces this but does not eliminate it.
Run a short visual check before launching long unattended campaigns.

Places to investigate:

- **Async reset may not fully take on a frozen world.** After
  `apply_settings(async)`, the server can need one `world.tick()` (or a
  `client.reload_world()`) before it resumes free-running. Look at
  `fusion/orchestrator.py:_heal_async` and `traffic/carla_check.py`; consider
  ticking once after clearing sync mode and verifying `get_settings().
  synchronous_mode` is actually `False`.
- **TrafficManager state carries over.** Capture puts the TrafficManager on
  `--tm-port` into synchronous mode. If it isn't fully reset, the next run can
  stall. Check the TM restore in the capture `finally` block and TM-port reuse
  between runs.
- **Windows hard-kill skips cleanup.** If a child is killed instead of receiving
  `CTRL_BREAK_EVENT`, its `finally` never runs, leaving sensors and world
  settings behind. Confirm the signal is delivered and the child exits in
  `fusion/orchestrator.py:_interrupt`; consider an explicit dataset-sensor sweep
  after teardown.
- **Not enough settle time between runs.** The next run's preflight may connect
  before the previous teardown has released actors and the world. Consider a
  short delay or a world-ready poll at the start of `orchestrator.run()`.
- **Server-side degradation after long sync sessions.** As a heavier reset
  between scenarios, try `client.reload_world()` before starting the next run.
