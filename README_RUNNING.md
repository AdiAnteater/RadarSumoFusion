# Running scenarios from the terminal

Every command works on Windows and Linux. Run them from the repo root with your
venv active. On Windows use `python`, on Linux `python3` (or `python` inside the venv).
Paths with `/` work on both.

## 1. One-time setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate
pip install -r dataset/requirements.txt
pip install traci sumolib carla==0.9.16
```

SUMO 1.27+. Any of these works:

| OS | Install | SUMO_HOME |
|---|---|---|
| Windows | Eclipse installer | set by the installer |
| Linux | `sudo apt install sumo sumo-tools` | optional (`/usr/share/sumo`) |
| Any | `pip install eclipse-sumo` | not needed |

The code finds `sumo`, `sumo-gui` and `netconvert` from `SUMO_HOME/bin` or `PATH`.
The GUIs need Tk (Linux: `sudo apt install python3-tk`). BEV videos need ffmpeg
(Linux: `sudo apt install ffmpeg`).

Start CARLA 0.9.16 before anything that touches CARLA:

```bash
# Windows
CarlaUE4.exe
# Linux (add -RenderOffScreen on a headless server)
./CarlaUE4.sh -RenderOffScreen
```

The SUMO network is already built. Only rebuild it if you change the .xodr:
`python traffic/build_network.py`.

## 2. Scenarios

| # | Scenario | # | Scenario |
|---|---|---|---|
| 1 | Free flow | 7 | Aggressive lane changing |
| 2 | Moderate demand | 8 | Bottleneck / work zone |
| 3 | Heavy demand | 9 | Overtaking |
| 4 | Stop and go | 10 | Simultaneous multi-lane overtake |
| 5 | Mixed vehicles (bus stop) | 11 | Occlusion |
| 6 | Directional rush hour | | |

Common traffic flags (all commands below): `--density 1-100`, `--duration <s>`,
`--direction WB|EB|BOTH`, `--ambient-vehicles 0-100`, `--pedestrians 0-100`,
`--bicycles 0-100`.

## 3. SUMO only (no CARLA)

Check that every scenario behaves (headless, about 1 min for all 11):

```bash
python traffic/validate_scenarios.py --density 60 --duration 240 --ambient-vehicles 30 --pedestrians 20 --bicycles 10
python traffic/validate_scenarios.py --scenarios 4 8 11
```

Writes `traffic/validation_report.md` and `.json`.

Watch one scenario in sumo-gui:

```bash
python traffic/runner.py --scenario 3 --density 60 --duration 300 --gui
```

Unit tests: `python traffic/tests/test_demand.py` (same for the other `test_*.py` files).

## 4. SUMO mirrored into CARLA (no recording)

With CARLA running:

```bash
python traffic/runner.py --scenario 3 --density 60 --duration 300 --pedestrians 20 --bicycles 10
```

The runner ticks CARLA itself. Ctrl+C stops it and removes its actors.
GUI version: `python traffic/gui_launcher.py`.

If a hard-killed run left actors behind: `python traffic/clear_carla_actors.py --walkers`.

## 5. Record a dataset (one scenario)

```bash
python run_fusion.py --scenario 3 --density 60 --duration 300 --radars 8 --rate-hz 20 --pedestrians 20 --bicycles 10
```

Output goes to `dataset/Data/sensor_capture_<timestamp>/`. Labeling and
post-processing run automatically (`--no-label`, `--no-postprocess` to skip).

## 6. Record a campaign (several scenarios, one capture)

Write a campaign file (or save one from `python fusion_gui.py`).
`campaigns/example_campaign.json` runs scenarios 1, 3 and 4 back to back:

```bash
python run_fusion.py --campaign campaigns/example_campaign.json
```

Actors are cleared between scenarios. Each row gets a `segment_id` /
`scenario_id` column in `radar_data_labeled.csv`.

## 7. After a capture

Run these from `dataset/`:

```bash
cd dataset
# Re-label or re-post-process an existing capture
python capture/LabelRadarCapture.py --capture-dir Data/sensor_capture_XXXX
python capture/PostProcessDataset.py --capture-dir Data/sensor_capture_XXXX
# BEV video -> Data/sensor_capture_XXXX/bev.mp4
python tools/render_bev_video.py Data/sensor_capture_XXXX
# RCS and Doppler distributions -> Data/sensor_capture_XXXX/radar_distributions/
python tools/plot_rcs_doppler.py Data/sensor_capture_XXXX
```

## Stopping

Ctrl+C in the terminal. On Linux, `kill <pid>` also stops the runner cleanly.
The GUI Stop button asks the runner to finish its current step and remove its
CARLA actors, then force-stops it after 10 s.
