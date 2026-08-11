"""RadarSumoFusion orchestration package.

Fuses two components into one dataset run:
  - traffic/  : SUMO governs all traffic (vehicles, pedestrians, bicycles) and
                mirrors actors into CARLA. It owns NO clock -- it subscribes.
  - dataset/  : CarlaDatasetCreation spawns the radar + camera rig and records
                the data. The capture process OWNS the CARLA tick (synchronous
                mode), so all sensors co-fire on one frame; the SUMO runner
                paces itself to that tick via world.wait_for_tick().

Entry points:
  - run_fusion.py   (headless CLI)
  - fusion_gui.py   (Tkinter GUI)
The original per-component GUIs are preserved:
  - traffic/gui_launcher.py   (SUMO-only)
  - dataset/Start.py          (DatasetCreation menu)
"""

from .config import FusionConfig  # noqa: F401
