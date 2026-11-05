"""FusionConfig: one struct describing a single fused capture run.

The tick rate is the single source of truth for the whole run. rate_hz sets:
  - CARLA fixed_delta_seconds (capture owns the tick at this rate),
  - the radar sensor_tick (so every radar produces one return per frame),
  - the SUMO step-length (the runner adopts CARLA's fixed_delta at startup).
Keeping all three equal is what stops the two clocks from drifting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


MAX_RADARS_PER_SIDE = 16
SETUP_SCRIPT_NAME = "RadarCameraSetup.py"


def split_radar_count(total: int) -> tuple:
    """Even split of a total radar count into (south, north); south takes the odd one."""
    total = max(0, int(total))
    return (total + 1) // 2, total // 2


def radar_rig_errors(south: int, north: int, height_m: float) -> list:
    errs = []
    for side, n in (("south", south), ("north", north)):
        if not (0 <= n <= MAX_RADARS_PER_SIDE):
            errs.append(f"radars_{side} must be 0-{MAX_RADARS_PER_SIDE} (got {n})")
    if south + north < 1:
        errs.append("need at least one radar (south + north >= 1)")
    if not (0.3 <= height_m <= 12.0):
        errs.append(f"radar_height_m must be 0.3-12 (got {height_m})")
    return errs


@dataclass
class FusionConfig:
    # --- Traffic (SUMO) ---
    scenario: int = 1                 # 1-11 (see traffic/runner.py SCENARIOS)
    density: int = 50                 # 1-100 -> veh/h/dir
    duration: int = 300               # capture seconds (SUMO covers this + margin)
    direction: str = "BOTH"           # WB | EB | BOTH
    ambient_vehicles: int = 20        # 0-100 city ambient vehicles (0=off)
    pedestrians: int = 20             # 0-100 pedestrians (0=off)
    bicycles: int = 10                # 0-100 bicycles (0=off)
    ambient_seed: int = 42
    render_radius: float = 150.0
    no_cull: bool = False
    sumo_gui: bool = False            # show sumo-gui alongside the run

    # --- Sensors / capture (DatasetCreation) ---
    radars_south: int = 4             # radars on the south kerb row (0-16)
    radars_north: int = 4             # radars on the north kerb row (0-16)
    radar_height_m: float = 3.0       # radar mount height above the road
    label: bool = True                # run radar labeling after capture (DC default)
    postprocess: bool = True          # run post-processing after capture (DC default)
    capture_base_dir: str = ""        # optional override for the output root

    # --- Shared clock / CARLA ---
    rate_hz: float = 20.0             # 20 Hz -> fixed_delta 0.05 s
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    traffic_manager_port: int = 8000

    # --- Orchestration knobs ---
    scene_cleanup: bool = True        # clear parked cars / trash before spawning
    clear_trees: bool = True          # hide trees over the monitored stretch
    sync_timeout: float = 180.0       # runner wait for capture to enable sync
    sensor_settle_s: float = 5.0      # head-start for the sensor rig to spawn
    capture_start_gap_s: float = 2.0  # gap between capture up and runner up

    # populated by validate()
    _errors: list = field(default_factory=list, repr=False)

    @property
    def fixed_delta_s(self) -> float:
        """CARLA fixed_delta_seconds / SUMO step / radar sensor_tick."""
        return round(1.0 / float(self.rate_hz), 6)

    @property
    def radar_count(self) -> int:
        return int(self.radars_south) + int(self.radars_north)

    def validate(self) -> "FusionConfig":
        errs = []
        if not (1 <= self.scenario <= 11):
            errs.append(f"scenario must be 1-11 (got {self.scenario})")
        if not (1 <= self.density <= 100):
            errs.append(f"density must be 1-100 (got {self.density})")
        if self.duration <= 0:
            errs.append(f"duration must be > 0 (got {self.duration})")
        if self.direction not in ("WB", "EB", "BOTH"):
            errs.append(f"direction must be WB|EB|BOTH (got {self.direction})")
        errs += radar_rig_errors(self.radars_south, self.radars_north, self.radar_height_m)
        if not (1.0 <= self.rate_hz <= 200.0):
            errs.append(f"rate_hz must be 1-200 (got {self.rate_hz})")
        # CARLA rejects fixed_delta_seconds above ~0.1 s in some builds; keep sane.
        if self.fixed_delta_s > 0.5:
            errs.append(f"fixed_delta {self.fixed_delta_s}s too large; raise rate_hz")
        self._errors = errs
        return self

    @property
    def ok(self) -> bool:
        return not self._errors

    @property
    def errors(self) -> list:
        return list(self._errors)

    def setup_script_name(self) -> str:
        return SETUP_SCRIPT_NAME

    def as_capture_base(self) -> str:
        return str(Path(self.capture_base_dir).expanduser()) if self.capture_base_dir else ""
