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
| `DATASET_RIG_LOOK_DIR` | east | along-stretch look direction shared by ALL radars (`east` = +heading, the way the camera looks; `west`) |
| `DATASET_RIG_SKEW_DEG` | 40 | how far each radar is skewed from "straight across the road" toward the look direction (south row yaw = 90-skew, north row = -(90-skew)) |
| `DATASET_RADAR_PITCH_DEG` | -6 | radar tilt; NEGATIVE = down (CARLA/Unreal: +pitch is nose UP; the old +8 default tilted the radars up into the facades) |
| `DATASET_RADAR_VERTICAL_FOV_DEG` | 30 | radar elevation FOV (was 60) |
| `DATASET_CAMERA_ENCODER_THREADS` | 2 | PNG encoder threads for the camera writer |
| `DATASET_CAMERA_MAX_BACKLOG` | 0 | drop new camera frames once this many wait for encoding (0 = never drop) |

Radar yaws are deterministic now: every radar looks across the boulevard toward the
far kerb and is skewed along the stretch so all eight share one look direction.
The former `compute_radar_yaw_toward_road()` pass chose between +40 and -40 by an
angular-distance comparison that is an exact tie on a straight road, so the side
was decided by floating-point noise (R5/R6/R8 faced west while the rest faced east
in the 2026-09-17 captures).

## Traffic scenarios: validation, signal timing, routing (2026-09-18)

`traffic/validate_scenarios.py` runs all 11 scenarios headless in SUMO (no CARLA),
with the real scenario controllers and the city layer, and reports what happens on
the monitored stretch: vehicles crossed, occupancy, mean speed, stopped fraction,
lane changes, passing events, pedestrians/bicycles, teleports, collisions, SUMO
warnings, and whether each scenario's mechanism fired (shockwave seeds held, bus
dwelling at BS_main, blocker parked on 20_3, occlusion pairs locked). It writes
`validation_report.md/.json` (the committed copy is density 60, BOTH, 240 s,
ambient 30 / ped 20 / bike 10, seed 42). Run it after any change to the net,
routes, controllers or signals:

```
python traffic/validate_scenarios.py                 # all 11
python traffic/validate_scenarios.py --scenarios 4 8 11 --duration 300
```

What that validation found, and what changed because of it:

1. **Signals masked every scenario.** The stretch (55 m) sits between lights 532
   (25 m west) and 189 (at its east end), with 719 another 19 m on. The imported
   two-phase programs gave the corridor ~34 s green per 100 s at each light, so a
   red at any of them queued traffic back across the whole stretch: all 11
   scenarios, "free flow" included, ran at ~1.3 m/s with vehicles stopped ~70% of
   the time, and 82% of vehicle-frames in capture 20260917_215125 were stationary.
   `traffic/stretch_signals.py` (runner `--stretch-signals green`, the default) now
   re-times the imported programs (state strings and phase order are kept, so the
   junction logic stays valid): the through lights 189/719 get a long corridor
   phase, the split light 532 (the corridor turns there, so westbound entry and
   eastbound exit sit in different phases) gets a short balanced cycle, the
   eastbound right-turn exit is permitted (yielding, with amber) during the entry
   phase, through lights run at exactly twice the split cycle and are offset so
   their red falls between the platoons 532 releases. Result at density 60: free
   flow 8-9.5 m/s, 10-30% stopped, zero teleports/collisions. `--stretch-signals
   static` restores the old behaviour for comparison.
2. **One usable lane per direction.** After the stretch, lane 3 (curb) only leads
   to exit -6 (WB) / 2 (EB) and lane 4 only to 5 (WB) / -3 (EB), with 9-25 m to
   sort it out, so with a single exit route SUMO pulled every vehicle into one lane
   ON the stretch and no overtaking could happen there (0 lane changes in the
   overtaking scenarios, dozens on the 100 m approach edges). `ambient_traffic.py`
   now defines a second exit per direction (`WB_route_alt`, `EB_route_alt`) and
   50/50 (WB) / 75/25 (EB) route distributions; flows and vehicles pinned to a
   lane (departLane first/last/3/4) get the matching exit, buses take the curb
   exit. Overtaking on this block is a between-lanes speed differential, not a
   lane change: `SlowLeaderController` (scenarios 7/9/10) ramps `car_slow`
   vehicles down to 4-5 m/s on the stretch so the inner lane passes them in view
   (`stretch_passes` in the report).
3. **Occlusion pairs never met** under free flow (car 13.9 m/s vs truck 11.1 m/s
   separate on the approach). `OcclusionController` now shepherds the car's max
   speed to the truck's from departure and lane-locks both on the stretch.
4. `time-to-teleport` 60 -> 150 s (side-street vehicles legitimately wait through
   the long corridor phase).

Density notes: the slider sets veh/h per direction identically for every scenario;
"heavy demand" and "stop-and-go" only differ from "free flow" through their vehicle
mix and controllers, so run them at density 80-100 and free flow at 30-50. The
westbound entry at 532 is a single lane (-1_4 -> -2_4 -> 21), so westbound demand
above ~500 veh/h backs up out of view (reported as `insertion_backlog_at_end`).

## Pedestrians (2026-09-18)

A CARLA walker's transform is the centre of its capsule (bounding_box.extent.z =
0.93 m adult, 0.55-0.65 m child), not its feet. The mirror placed every walker at
surface + 0.5 m, so adults stood 0.43 m deep in the pavement. `carla_sync.py` now
places each walker at surface + its own extent.z (read from the spawned actor).
The ground raycast only accepts ground-like hit labels (road, sidewalk, ground,
terrain...): a ray landing on a car roof or a bus shelter used to be cached as
the pavement height, so walkers near cars popped up and down. Walkers that SUMO
reports on a crossing or walkingarea are no longer snapped back onto the sidewalk
(default snap distance 6 -> 3 m), which removed the back-and-forth at the kerb in
front of waiting cars.

## Radar labeling: what the numbers mean (2026-09-17 diagnosis)

Capture `sensor_capture_20260917_125123` re-analysed against exact OBB geometry
(every return reconstructed with the true spherical-to-world transform, membership
tested against the un-inflated CARLA bounding box):

| | old labeler | new labeler |
|---|---|---|
| label precision (labeled return really inside the actor OBB) | 63% | 98.5% |
| recall of true on-body returns | 96% | 100% |
| QA "matched / with candidates" | 35% (7 m candidate bubble) | 89% (2 m bubble) |

The old 35% was a denominator artifact: 65% of "unmatched with candidates" were
road-surface returns (z = 0) up to 7 m from a car. The 37% of wrong old labels were
road returns beside cars admitted by the 0.75 m box inflation, the 1.0 m
single-candidate margin and the Euler-addition hit reconstruction (wrong once the
mount pitch is non-zero). Changes: exact reconstruction, inflation 0.2 m, ground
rejection below the actor's own ground plane, single-candidate margin 0.5 m,
candidate bubble 2 m, candidate range gate uses CARLA's real ray reach
(`range * sqrt(1 + tan^2(hfov/2) + tan^2(vfov/2))`, 73 m for 35/120/60, so far-lane
hits at 35-55 m are no longer dropped).

Every labeled row now also carries `hit_world_{x,y,z}_m` and `return_class`
(`vehicle | pedestrian | road | structure | unassigned`), and the labeler writes
`radar_labeling_qa/return_class_summary.{txt,json}`. That breakdown is the
honest dataset statistic; "match rate given candidates" is only a QA check.

Doppler: CARLA's radar takes the hit actor's PHYSICS velocity, which is exactly 0
for SUMO-mirrored (physics-off, teleported) vehicles. `PostProcessDataset.py` now
synthesizes the radial velocity for every matched return from the logged
trajectory (central difference, projected on the sensor->hit line of sight,
CARLA sign convention: positive = receding) and keeps the raw value in
`velocity_raw_mps`. Radar returns on unmatched (static) geometry keep 0.

Camera: the writer is two-stage (metadata + CSV row immediately, PNG encoding on
a thread pool). The old single thread fell behind on slower machines, outran the
600-frame snapshot cache and then silently skipped every frame ("41 frames then
nothing"). Every frame is saved now, with or without actors nearby.

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
