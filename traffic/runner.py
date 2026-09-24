"""
runner.py
=========
TraCI-based simulation controller for the SUMO traffic scenario system.

Called by gui_launcher.py with:
    python runner.py --scenario <1-11> --density <1-100> --duration <seconds>
                     --direction <WB|EB|BOTH> --gui

Responsibilities:
  - Build a temporary .sumocfg and .rou.xml from the scenario template
    (substituting DENSITY_* placeholders, applying the map-edge routes)
  - Launch SUMO (or sumo-gui) via TraCI
  - Apply scenario-specific real-time logic (shockwave seeding,
    bottleneck lane closure, occlusion speed matching)
  - Cleanly shut down on KeyboardInterrupt or simulation end

CHANGES IN THIS VERSION
-----------------------
1. NETWORK. Every run now uses network/Town10HD_Opt_fixed.net.xml from
   build_network.py, not just pedestrian runs. The original net left all 46
   sidewalk lanes drivable, so departLane="random" put a third of traffic on
   the sidewalk and departLane="first" put ALL of it there; 20 of those lanes
   dead-end at the first real junction with no connection and no TL link. That
   was the stuck-car bug. If the rebuilt net is missing we now stop with
   instructions instead of silently running on the broken one.

2. ROUTES. The map-edge routes from ambient_traffic.py are applied on EVERY
   run, not only in city mode. Scenario traffic now enters at a perimeter
   corner, crosses the monitored stretch, and continues to the opposite corner
   before despawning. The applied edge lists are echoed in the run header so
   every log records the routing that was actually used.

3. CONTROLLERS. All three were inactive or misfiring:
     - StopAndGoController tested `"monitored" in road`. Edges are numeric
       ("20"/"-20"), so it never matched and scenario 4 did nothing. It also
       never released its setSpeed override, so a seed leaving the stretch
       mid-stop would have frozen at 0 forever and blocked the lane.
     - OcclusionController looked for occ_pair_NB_* / occ_pair_SB_*, but the
       route file defines occ_pair_WB_* / occ_pair_EB_*, so scenario 11 did
       nothing either.
     - BottleneckController parked its blocker at a fixed step count. With
       map-edge routes the blocker is still out at the SE corner at that point,
       ~130 m from the sensors. It now parks on arrival at the stretch instead.

4. SEED. --ambient-seed is now handed to SUMO as well as to the ambient
   generator, so a given seed reproduces the whole run, vehicle insertion
   jitter included.
"""

import os
import sys
import re
import argparse
import tempfile
import subprocess
import time

# --- TraCI import ---
try:
    import traci
    import sumolib
except ImportError:
    sys.exit(
        "ERROR: traci / sumolib not found. "
        "Activate your venv and run: pip install traci sumolib"
    )

# --- CARLA sync import ---
try:
    from carla_sync import CarlaSyncManager
    CARLA_SYNC_AVAILABLE = True
except ImportError:
    CARLA_SYNC_AVAILABLE = False

# --- Ambient city-traffic import ---
try:
    import ambient_traffic
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False

# --- Corridor-priority signal programs (see stretch_signals.py) ---
try:
    import stretch_signals
    SIGNALS_AVAILABLE = True
except ImportError:
    SIGNALS_AVAILABLE = False

# Centre of the CARLA render radius and its default size (see carla_sync).
RENDER_RADIUS_DEFAULT = 120.0

# SUMO's simulation step. Kept at the historical 0.1 s so nothing changes
# unless you ask for it.
DEFAULT_STEP_LENGTH = 0.1

# What the capture script ticks CARLA at, for the mismatch warning below.
# This is RADAR_SENSOR_TICK_S in DatasetCreation/capture/CaptureRadarCameraData.py
# (line 59), which is what DATASET_SYNC_FIXED_DELTA_S falls back to. We do NOT
# read it from there -- this module must not import the capture package -- so if
# you change it there, change it here too. It is only used for a warning; it
# never drives anything.
CAPTURE_RADAR_SENSOR_TICK_S = 0.05

# The monitored stretch, as SUMO edge ids. Kept in sync with ambient_traffic.
# There is no "monitored" substring anywhere in this network -- any check
# against road names must compare against these numeric ids.
MONITORED_EDGES = (ambient_traffic.MONITORED_EDGES if AMBIENT_AVAILABLE
                   else {"20", "-20"})

# --- Paths ---
HERE          = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR   = os.path.join(HERE, "network")
SCENARIOS_DIR = os.path.join(HERE, "scenarios")
ROUTES_DIR    = os.path.join(SCENARIOS_DIR, "routes")
ADD_DIR       = os.path.join(SCENARIOS_DIR, "additional")
# The rebuilt net: sidewalks restricted to pedestrians, crossings and
# walkingareas present. Required for every run. See build_network.py.
NET_FILE      = os.path.join(NETWORK_DIR, "Town10HD_Opt_fixed.net.xml")
RAW_NET_FILE  = os.path.join(NETWORK_DIR, "Town10HD_Opt.net.xml")
ADD_FILE      = os.path.join(ADD_DIR,     "vehicle_types.add.xml")

SUMO_HOME     = os.environ.get("SUMO_HOME", "")
SUMO_BIN      = os.path.join(SUMO_HOME, "bin", "sumo.exe")
SUMO_GUI_BIN  = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

# --- Scenario metadata ---
SCENARIOS = {
    1:  {"name": "Free Flow Traffic",               "file": "s01_free_flow.rou.xml"},
    2:  {"name": "Moderate Demand",                 "file": "s02_moderate_demand.rou.xml"},
    3:  {"name": "Heavy Demand",                    "file": "s03_heavy_demand.rou.xml"},
    4:  {"name": "Stop and Go Traffic",             "file": "s04_stop_and_go.rou.xml"},
    5:  {"name": "Mixed Vehicles",                  "file": "s05_mixed_vehicles.rou.xml"},
    6:  {"name": "Directional Rush Hour",           "file": "s06_rush_hour.rou.xml"},
    7:  {"name": "Aggressive Lane Changing",        "file": "s07_aggressive_lc.rou.xml"},
    8:  {"name": "Bottleneck / Work Zone",          "file": "s08_bottleneck.rou.xml"},
    9:  {"name": "Overtaking",                      "file": "s09_overtaking.rou.xml"},
    10: {"name": "Simultaneous Multi-Lane Overtake","file": "s10_multilane_overtaking.rou.xml"},
    11: {"name": "Occlusion",                       "file": "s11_occlusion.rou.xml"},
}


# --- Density scaling ---
# density_slider: 1-100
# Maps to vehicles-per-hour per direction
# Low=50, Mid=400, High=900 (urban 2-lane boulevard capacity ~1000 veh/h/dir)
def density_to_vph(density_slider: int) -> int:
    """Map 1-100 slider value to vehicles per hour (per direction)."""
    min_vph = 50
    max_vph = 900
    return int(min_vph + (max_vph - min_vph) * (density_slider - 1) / 99)


def build_density_map(vph: int, direction: str) -> dict:
    """
    Build the full DENSITY_* substitution map.
    direction: 'WB', 'EB', 'BOTH'
    WB = Westbound (edge 20, y~130), EB = Eastbound (edge -20, y~100)
    """
    wb_vph = vph if direction in ("WB", "BOTH") else int(vph * 0.2)
    eb_vph = vph if direction in ("EB", "BOTH") else int(vph * 0.2)

    def split(total, pct): return max(1, int(total * pct / 100))

    return {
        "DENSITY_WB":    str(wb_vph),
        "DENSITY_EB":    str(eb_vph),
        "DENSITY_WB_80": str(split(wb_vph, 80)),
        "DENSITY_WB_70": str(split(wb_vph, 70)),
        "DENSITY_WB_60": str(split(wb_vph, 60)),
        "DENSITY_WB_55": str(split(wb_vph, 55)),
        "DENSITY_WB_50": str(split(wb_vph, 50)),
        "DENSITY_WB_40": str(split(wb_vph, 40)),
        "DENSITY_WB_30": str(split(wb_vph, 30)),
        "DENSITY_WB_25": str(split(wb_vph, 25)),
        "DENSITY_WB_20": str(split(wb_vph, 20)),
        "DENSITY_WB_15": str(split(wb_vph, 15)),
        "DENSITY_WB_10": str(split(wb_vph, 10)),
        "DENSITY_EB_80": str(split(eb_vph, 80)),
        "DENSITY_EB_70": str(split(eb_vph, 70)),
        "DENSITY_EB_60": str(split(eb_vph, 60)),
        "DENSITY_EB_55": str(split(eb_vph, 55)),
        "DENSITY_EB_50": str(split(eb_vph, 50)),
        "DENSITY_EB_40": str(split(eb_vph, 40)),
        "DENSITY_EB_30": str(split(eb_vph, 30)),
        "DENSITY_EB_25": str(split(eb_vph, 25)),
        "DENSITY_EB_20": str(split(eb_vph, 20)),
        "DENSITY_EB_15": str(split(eb_vph, 15)),
        "DENSITY_EB_10": str(split(eb_vph, 10)),
    }


def substitute_density(template_path: str, density_map: dict, duration: int) -> str:
    """Read route template, substitute placeholders, apply the map-edge routes,
    write to a temp file. Returns the path."""
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace all DENSITY_* placeholders LONGEST KEY FIRST.
    #
    # The keys share prefixes -- "DENSITY_WB" is a prefix of "DENSITY_WB_80" --
    # and str.replace() does NOT respect token boundaries. Replacing the short
    # key first rewrites the leading part of the long placeholder and leaves
    # its suffix dangling: "DENSITY_WB_80" becomes "<wb_vph>_80" (e.g. 633_80),
    # which SUMO then rejects with
    #     Attribute 'vehsPerHour' ... Invalid Number Format (double) 633_80.
    # Sorting keys by length descending guarantees DENSITY_WB_80 is consumed
    # before DENSITY_WB, so every placeholder is replaced exactly once.
    for key in sorted(density_map, key=len, reverse=True):
        content = content.replace(key, density_map[key])

    # Update end time to match simulation duration
    content = re.sub(r'end="\d+"', f'end="{duration}"', content)

    # Map-edge routing, applied on every run. All 11 scenario files share the
    # WB_route / EB_route ids, so this rewrites two lines and nothing else.
    # ambient_traffic.py holds the authoritative edge lists; whatever the .rou
    # file on disk says is only a fallback.
    if AMBIENT_AVAILABLE:
        content = ambient_traffic.apply_map_edge_routes(content)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".rou.xml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def write_busstop_additional() -> str:
    """Write a temp additional file with just the bus stop. Returns path."""
    # The bus stop sits on the westbound monitored edge (SUMO edge 20). Edge 20
    # lanes are: 20_0 sidewalk (pedestrian-only after the net rebuild),
    # 20_1/20_2 shoulder, 20_3/20_4 driving, 20_5+ shoulder/median -- so 20_3 is
    # the rightmost DRIVING (curb-side) lane, the correct one for a bus bay.
    # Edge 20 is ~55 m long after the rebuild (was 52.8; netconvert reclaims a
    # little junction area when it builds walkingareas), so startPos/endPos of
    # 20/35 remain comfortably inside. The old lane="NB_monitored_0" and
    # 160-200 m positions were leftovers from the retired custom 4-node network
    # and caused:
    #     Error: The lane NB_monitored_0 to use within the busStop 'BS_main'
    #     is not known.
    # friendlyPos keeps us safe if SUMO needs to nudge the bay onto the lane.
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<additional>',
        '    <busStop id="BS_main"',
        '             lane="20_3"',
        '             startPos="20"',
        '             endPos="35"',
        '             friendlyPos="true"',
        '             name="Main Street Bus Stop"/>',
        '</additional>',
    ]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".add.xml", delete=False, encoding="utf-8"
    )
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


def write_sumocfg(net_file: str, route_file: str, add_files: list,
                  duration: int, step_length: float = 0.1,
                  ignore_route_errors: bool = False, seed: int = 42) -> str:
    """Write a temporary .sumocfg. add_files is a list of paths.

    ignore_route_errors: in city mode, a single randomly-generated ambient trip
    that can't be routed should be skipped with a warning, not abort the whole
    simulation. Left off for plain scenario runs so genuine route errors still
    surface.

    seed: handed to SUMO so a run is reproducible end to end, not just in its
    ambient gateway pairing.
    """
    add_str = ",".join(add_files)
    processing = [
        '    <processing>',
        '        <collision.action value="warn"/>',
        # 150 s (was 60): with the 90 s corridor-priority phase a side-street
        # vehicle can legitimately wait ~95 s for its turn; at 60 s SUMO was
        # teleporting them ("waited too long (yield)") out of the side streets.
        '        <time-to-teleport value="150"/>',
        '        <lanechange.duration value="3"/>',
        # route-steps -1 loads the ENTIRE route file up front instead of
        # streaming it in 200 s windows. This is not a tuning knob, it fixes
        # silent data loss.
        #
        # ambient_traffic.inject_ambient appends its flows and personFlows just
        # before </routes>, i.e. AFTER the scenario's own entries. Scenarios 4,
        # 8, 10 and 11 contain <vehicle depart="30/120/210"> or flows with
        # begin="20". With the default incremental loader SUMO refuses to go
        # backwards in time and drops everything that departs earlier than what
        # it has already read, warning:
        #     Route file should be sorted by departure time, ignoring 'ped_...'
        # "ignoring X" means the element is DISCARDED, not that the sorting rule
        # is waived. Measured on scenario 4 at ped=60: 0 pedestrians with the
        # default, 24 with route-steps -1. Same for the ambient vehicles and
        # bikes. Scenarios that contain only begin="0" flows were unaffected,
        # which is why this hid for so long.
        '        <route-steps value="-1"/>',
    ]
    if ignore_route_errors:
        processing.append('        <ignore-route-errors value="true"/>')
    processing.append('    </processing>')
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<configuration>',
        '    <input>',
        '        <net-file value="' + net_file + '"/>',
        '        <route-files value="' + route_file + '"/>',
        '        <additional-files value="' + add_str + '"/>',
        '    </input>',
        '    <time>',
        '        <begin value="0"/>',
        '        <end value="' + str(duration) + '"/>',
        '        <step-length value="' + str(step_length) + '"/>',
        '    </time>',
    ] + processing + [
        '    <random_number>',
        '        <seed value="' + str(int(seed)) + '"/>',
        '    </random_number>',
        '    <report>',
        '        <verbose value="false"/>',
        '        <no-step-log value="true"/>',
        '    </report>',
        '</configuration>',
    ]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sumocfg", delete=False, encoding="utf-8"
    )
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


# --- Scenario-specific TraCI logic ---

class ScenarioController:
    """Base class -- override step() for scenario-specific logic."""

    def __init__(self, scenario_id: int, duration: int):
        self.scenario_id = scenario_id
        self.duration    = duration
        self.step_count  = 0

    def on_start(self):
        """Called once after SUMO connects."""
        pass

    def step(self):
        """Called every simulation step. Override in subclasses."""
        pass

    def on_stop(self):
        """Called before TraCI closes."""
        pass


class StopAndGoController(ScenarioController):
    """Scenario 4: commands the shockwave seed vehicles to stop and go while
    they are on the monitored stretch.

    Two fixes over the previous version:
      - The trigger was `if "monitored" in road`. Road ids on this net are
        numeric ("20"/"-20"), so the test never matched and the whole scenario
        was inert. Now compares against MONITORED_EDGES.
      - The override was never released. traci.vehicle.setSpeed() persists
        until explicitly cleared, so a seed that left the stretch during a stop
        phase stayed frozen at 0 m/s on edge 19 for the rest of the run and
        blocked the lane behind it. Now setSpeed(-1) hands control back to the
        car-following model on exit, and during every "go" phase, so the seed
        cannot rear-end the queue it just created.
    """

    STOP_SPEED    = 0.0
    CYCLE_SECONDS = 15.0   # 15 s stopped, 15 s moving

    # This used to be CYCLE_STEPS = 150, described as "150 steps at 0.1 s".
    # That comment was the whole problem: it silently hard-coded the step
    # length into the scenario. Now that --step-length is settable (to line
    # SUMO up with the capture script's tick), a step count would have
    # rescaled the shockwave period behind your back -- 0.05 s steps would
    # have turned 15 s stop/go into 7.5 s. Everything here is now keyed off
    # simulation time, so the scenario means the same thing at any step length.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._held    = set()
        self._entered = set()

    def step(self):
        self.step_count += 1
        try:
            vehicles = traci.vehicle.getIDList()
            now      = traci.simulation.getTime()
        except traci.TraCIException:
            return

        for vid in vehicles:
            if not vid.startswith("shockwave_seed_"):
                continue
            road = traci.vehicle.getRoadID(vid)
            if road in MONITORED_EDGES:
                if vid not in self._entered:
                    self._entered.add(vid)
                    print(f"[StopAndGo] {vid} entered stretch (edge {road}) "
                          f"at t={now:.1f}s")
                phase = int(now // self.CYCLE_SECONDS) % 2
                if phase == 0:
                    traci.vehicle.setSpeed(vid, self.STOP_SPEED)
                else:
                    traci.vehicle.setSpeed(vid, -1)   # back to car-following
                self._held.add(vid)
            elif vid in self._held:
                traci.vehicle.setSpeed(vid, -1)
                self._held.discard(vid)


class BottleneckController(ScenarioController):
    """Scenario 8: parks the blocker vehicle in the curb lane of the monitored
    stretch, closing it to through traffic.

    Three problems fixed here, all found by watching a real 260 s run:

    1. The blocker was frozen at a fixed step count (5 s in). That was tuned for
       the old routes, where it spawned on edge 21 one junction from the
       stretch. With map-edge routes it spawns at the SE corner, so at 5 s it is
       still ~130 m away and would have parked out of sensor range entirely.

    2. Holding it with setSpeed(0) does NOT exempt it from SUMO's jam teleport.
       With time-to-teleport at 60 s the blocker was yanked off the stretch one
       minute after parking:
           Warning: Teleporting vehicle 'lane_blocker_wb'; waited too long
           (yield), lane='20_3', time=164.80
       and the work zone quietly disappeared. A scheduled stop (setStop) is
       exempt, so we use that instead. It also lets SUMO drive the blocker to
       the spot itself, which is route-independent.

    3. Pinning setLaneChangeMode(0) for the whole journey stops the STRATEGIC
       lane changes the blocker needs to follow its own route, producing
           Warning: Teleporting vehicle 'lane_blocker_wb'; waited too long
           (wrong lane), lane='-2_3', time=94.40
       before it ever reached the stretch. setStop handles lane selection, so
       the lane change mode is left alone.
    """

    BLOCKER_ID   = "lane_blocker_wb"
    BLOCKER_EDGE = "20"     # westbound monitored edge
    BLOCKER_LANE = 3        # curb-side driving lane (0 is the sidewalk)
    BLOCKER_POS  = 30.0     # ~mid-stretch; edge 20 is ~55 m after the rebuild

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scheduled = False

    def step(self):
        self.step_count += 1
        if self._scheduled:
            return
        try:
            vehicles = traci.vehicle.getIDList()
        except traci.TraCIException:
            return
        if self.BLOCKER_ID not in vehicles:
            return

        # Schedule the stop the moment the blocker exists. SUMO drives it to
        # edge 20, changes into the curb lane on the way, and holds it there for
        # the rest of the run without teleporting it.
        try:
            traci.vehicle.setStop(
                vehID=self.BLOCKER_ID,
                edgeID=self.BLOCKER_EDGE,
                pos=self.BLOCKER_POS,
                laneIndex=self.BLOCKER_LANE,
                duration=float(self.duration),   # outlives the simulation
                flags=0,                         # on-road stop; blocks the lane
            )
            self._scheduled = True
            print(f"[Bottleneck] Work zone scheduled: {self.BLOCKER_ID} will "
                  f"hold lane {self.BLOCKER_EDGE}_{self.BLOCKER_LANE} at "
                  f"pos {self.BLOCKER_POS:.0f} m for {self.duration} s")
        except traci.TraCIException as e:
            print(f"[Bottleneck] WARNING: could not schedule the work zone "
                  f"stop ({e}) -- scenario 8 will run as ordinary traffic")
            self._scheduled = True   # do not retry every step


class OcclusionController(ScenarioController):
    """Scenario 11: keeps each paired car hidden beside its truck while both
    cross the monitored stretch.

    History: the first version matched ids that did not exist (NB/SB instead
    of WB/EB), the second only locked the pair once BOTH were already on the
    stretch. That worked only because signal queuing kept them together; with
    corridor-priority signals the car (13.9 m/s) leaves the truck (11.1 m/s)
    ~50 m behind on the 130 m approach and the pair never meets on the
    stretch (headless validation: 0 pairs locked).

    Now the car is shepherded from departure: its max speed is set each step
    to the truck's current speed, nudged up when it has fallen behind and down
    when it has crept ahead (route distance from traci.vehicle.getDistance).
    setMaxSpeed keeps the car-following model and all safety checks active, so
    there is no rear-end risk. On the stretch both vehicles additionally get
    lane changing disabled so the car stays in the inner lane beside the truck
    (truck departLane 3 / route via -6, car departLane 4 / route via 5).
    Everything is released when the pair leaves the stretch or either vehicle
    despawns.
    """

    DIRECTIONS      = ("wb", "eb")
    MAX_PAIRS       = 10
    DEFAULT_LC_MODE = 1621
    ALIGN_TOL_M     = 1.5     # keep the car within this of the truck (route distance)
    NUDGE_MPS       = 0.6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matched = set()        # cars currently lane-locked on the stretch
        self._ever_matched = set()   # every car id ever locked (validation/reporting)
        self._shepherded = set()     # cars whose max speed is being driven
        self._car_max = {}

    def _release(self, truck_id, car_id, vehicles):
        for vid in (car_id, truck_id):
            if vid not in vehicles:
                continue
            try:
                traci.vehicle.setLaneChangeMode(vid, self.DEFAULT_LC_MODE)
            except traci.TraCIException:
                pass
        if car_id in vehicles and car_id in self._shepherded:
            try:
                traci.vehicle.setMaxSpeed(car_id, self._car_max.get(car_id, 16.67))
            except traci.TraCIException:
                pass
        self._shepherded.discard(car_id)

    def step(self):
        self.step_count += 1
        try:
            vehicles = set(traci.vehicle.getIDList())
        except traci.TraCIException:
            return
        for i in range(1, self.MAX_PAIRS):
            for direction in self.DIRECTIONS:
                truck_id = f"occ_pair_{direction}_truck_{i}"
                car_id   = f"occ_pair_{direction}_car_{i}"
                if not (truck_id in vehicles and car_id in vehicles):
                    if car_id in self._matched or car_id in self._shepherded:
                        self._release(truck_id, car_id, vehicles)
                        self._matched.discard(car_id)
                    continue
                try:
                    truck_road = traci.vehicle.getRoadID(truck_id)
                    car_road = traci.vehicle.getRoadID(car_id)
                    v_truck = traci.vehicle.getSpeed(truck_id)
                    gap = traci.vehicle.getDistance(car_id) - traci.vehicle.getDistance(truck_id)
                except traci.TraCIException:
                    continue
                if car_id in self._matched and (truck_road not in MONITORED_EDGES
                                                 or car_road not in MONITORED_EDGES):
                    # Pair has crossed the stretch: hand the car back.
                    self._release(truck_id, car_id, vehicles)
                    self._matched.discard(car_id)
                    continue
                if car_id in self._matched:
                    pass
                elif truck_road in MONITORED_EDGES and car_road in MONITORED_EDGES:
                    self._matched.add(car_id)
                    self._ever_matched.add(car_id)
                    for vid in (car_id, truck_id):
                        try:
                            traci.vehicle.setLaneChangeMode(vid, 0)
                        except traci.TraCIException:
                            pass
                    print(f"[Occlusion] pairing {truck_id} / {car_id} on the stretch "
                          f"at t={traci.simulation.getTime():.1f}s")
                if car_id not in self._shepherded:
                    try:
                        self._car_max[car_id] = traci.vehicletype.getMaxSpeed(
                            traci.vehicle.getTypeID(car_id).split("@", 1)[0])
                    except traci.TraCIException:
                        self._car_max[car_id] = 16.67
                    self._shepherded.add(car_id)
                # Shepherd: car max speed tracks the truck, corrected by the gap.
                if gap > self.ALIGN_TOL_M:
                    cap = max(0.0, v_truck - self.NUDGE_MPS)
                elif gap < -self.ALIGN_TOL_M:
                    cap = v_truck + self.NUDGE_MPS
                else:
                    cap = v_truck
                cap = max(cap, 0.3)
                try:
                    traci.vehicle.setMaxSpeed(car_id, min(cap, self._car_max.get(car_id, 16.67)))
                except traci.TraCIException:
                    pass

    def on_stop(self):
        try:
            vehicles = set(traci.vehicle.getIDList())
        except traci.TraCIException:
            return
        for i in range(1, self.MAX_PAIRS):
            for direction in self.DIRECTIONS:
                self._release(f"occ_pair_{direction}_truck_{i}",
                              f"occ_pair_{direction}_car_{i}", vehicles)


class SlowLeaderController(ScenarioController):
    """Scenarios 7, 9, 10: make the overtaking happen IN FRONT OF THE SENSORS.

    Headless validation (validate_scenarios.py) showed zero lane changes on the
    monitored stretch in all three overtaking scenarios: the slow/fast mix does
    produce lane changes, but on the 130 m / 97 m approach edges (-1, -5), so by
    the time a platoon reaches the 55 m stretch it is already sorted and the
    "aggressive" cars simply follow. Lane connectivity after the stretch also
    pins each vehicle to its lane there (see ambient_traffic.WB_ROUTE_ALT_EDGES),
    so on this block "overtaking" is a between-lanes speed differential. To make
    it happen where the sensors look, slow-type vehicles (car_slow, curb lane)
    are ramped down to SLOW_ON_STRETCH_MPS while on the stretch via setMaxSpeed
    (car-following and safety checks stay active) and restored to their type's
    maximum on exit; fast vehicles in the inner lane then pass them in view
    (Kesting MOBIL speed-gain motive, expressed as passing rather than as a
    lane change).
    """

    SLOW_TYPE = "car_slow"
    SLOW_ON_STRETCH_MPS = {7: 5.0, 9: 4.0, 10: 4.0}
    CAP_DECEL_MPS2 = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capped = {}
        self._type_max = {}
        self._base_type = {}

    def _restore_speed(self, vid):
        entry = self._capped.pop(vid, None)
        if entry is None:
            return
        try:
            traci.vehicle.setMaxSpeed(vid, self._type_max[entry[0]])
        except traci.TraCIException:
            pass

    def step(self):
        self.step_count += 1
        try:
            vehicles = traci.vehicle.getIDList()
        except traci.TraCIException:
            return
        cap = self.SLOW_ON_STRETCH_MPS.get(self.scenario_id, 4.0)
        live = set(vehicles)
        for vid in list(self._capped):
            if vid not in live:
                self._capped.pop(vid, None)
        if len(self._base_type) > 4 * max(len(live), 1) + 64:
            self._base_type = {k: v for k, v in self._base_type.items() if k in live}
        for vid in vehicles:
            vtype = self._base_type.get(vid)
            if vtype is None:
                try:
                    # setMaxSpeed gives the vehicle a private type copy named
                    # "<type>@<vid>"; compare on the base name. Cached per
                    # vehicle so the per-step TraCI cost stays small.
                    vtype = traci.vehicle.getTypeID(vid).split("@", 1)[0]
                except traci.TraCIException:
                    continue
                self._base_type[vid] = vtype
            if vtype != self.SLOW_TYPE:
                continue
            on = traci.vehicle.getRoadID(vid) in MONITORED_EDGES
            if on:
                if vtype not in self._type_max:
                    self._type_max[vtype] = traci.vehicletype.getMaxSpeed(vtype)
                # Ramp the cap down at a comfortable 2 m/s^2 instead of clamping
                # (an instant clamp is logged by SUMO as emergency braking).
                cur = traci.vehicle.getSpeed(vid)
                dt = traci.simulation.getDeltaT()
                target = max(cap, cur - self.CAP_DECEL_MPS2 * dt)
                if vid not in self._capped or target != self._capped[vid][1]:
                    traci.vehicle.setMaxSpeed(vid, target)
                    self._capped[vid] = (vtype, target)
            elif vid in self._capped:
                self._restore_speed(vid)

    def on_stop(self):
        for vid in list(self._capped):
            self._restore_speed(vid)


def get_controller(scenario_id: int, duration: int) -> ScenarioController:
    """Return the appropriate controller for the scenario."""
    mapping = {
        4:  StopAndGoController,
        7:  SlowLeaderController,
        8:  BottleneckController,
        9:  SlowLeaderController,
        10: SlowLeaderController,
        11: OcclusionController,
    }
    cls = mapping.get(scenario_id, ScenarioController)
    return cls(scenario_id, duration)


# --- Bus spawning for scenario 5 ---

def spawn_buses_with_stop(duration: int, interval: int = 120):
    """
    Dynamically add bus vehicles with a stop at BS_main.
    Called after TraCI connects for scenario 5.
    interval: seconds between buses

    departLane="first" now resolves to lane 20_3 -- the curb driving lane -- as
    intended. Before the net rebuild it resolved to the sidewalk lane _0, so
    every scheduled bus started on the pavement and dead-ended at the next
    junction.
    """
    bus_count = max(1, duration // interval)
    for i in range(bus_count):
        depart_time = i * interval + 10
        bus_id = f"scheduled_bus_wb_{i}"
        traci.vehicle.add(
            vehID=bus_id,
            # Curb-lane exit (18_3 -> -6): the bus dwells in lane 20_3 and must
            # not have to cross to lane 4 within 19 m after the stop.
            routeID="WB_route_alt",
            typeID="bus",
            depart=str(depart_time),
            departLane="first",
            departSpeed="0",
        )
        traci.vehicle.setBusStop(
            vehID=bus_id,
            stopID="BS_main",
            duration=20.0,    # 20 s dwell time
            until=-1,
        )


# --- Main simulation loop ---

def run(scenario_id: int, density: int, duration: int,
        direction: str, use_gui: bool,
        carla_host: str = "127.0.0.1", carla_port: int = 2000,
        ambient_vehicles: int = 0, pedestrians: int = 0, bicycles: int = 0,
        ambient_seed: int = 42, render_radius: float = RENDER_RADIUS_DEFAULT,
        cull: bool = True, extend_routes: bool = False,
        step_length: float = DEFAULT_STEP_LENGTH,
        external_tick: bool = False, sync_timeout: float = 60.0,
        stretch_signals_mode: str = "green"):

    meta = SCENARIOS.get(scenario_id)
    if not meta:
        sys.exit(f"ERROR: Unknown scenario {scenario_id}")

    # extend_routes is retained for CLI compatibility but no longer gates
    # anything: map-edge routing is now unconditional.
    city_mode = bool(ambient_vehicles or pedestrians or bicycles)

    # --- Network. The rebuilt net is mandatory, not an option. ---
    if not os.path.isfile(NET_FILE):
        sys.exit(
            "ERROR: rebuilt network not found:\n"
            f"    {NET_FILE}\n\n"
            "Run this once from an activated venv:\n"
            "    python build_network.py\n\n"
            "The raw netconvert output leaves all 46 sidewalk lanes drivable, so\n"
            "departLane=\"random\" puts about a third of traffic on the pavement and\n"
            "departLane=\"first\" puts all of it there. Those lanes dead-end at the\n"
            "first real junction with no connection and no traffic-light link --\n"
            "that is the stuck-car bug. Pedestrians cannot route on it at all.\n"
            "Running on it would silently poison the dataset, so we stop here."
        )
    net_file = NET_FILE

    print(f"\n{'='*60}")
    print(f"  Scenario {scenario_id:02d}: {meta['name']}")
    print(f"  Density  : {density}/100 -> {density_to_vph(density)} veh/h/dir")
    print(f"  Direction: {direction}")
    print(f"  Duration : {duration}s")
    print(f"  GUI      : {use_gui}")
    print(f"  Seed     : {ambient_seed}")
    print(f"  Step     : {step_length}s  ({1.0/step_length:.0f} Hz)")
    print(f"  Tick     : {'EXTERNAL (capture owns world.tick)' if external_tick else 'standalone (no capture; SUMO-paced)'}")
    # In external-tick mode the SUMO step is realigned to the CARLA fixed_delta
    # below, so this hand-tuning warning only applies to standalone runs.
    if not external_tick and abs(step_length - CAPTURE_RADAR_SENSOR_TICK_S) > 1e-9:
        print(f"  NOTE     : CaptureRadarCameraData.py ticks CARLA at "
              f"{CAPTURE_RADAR_SENSOR_TICK_S}s when DATASET_SYNC_MODE=1.")
        print(f"             SUMO is stepping at {step_length}s, so each mirrored "
              f"position is held for")
        print(f"             {step_length / CAPTURE_RADAR_SENSOR_TICK_S:.1f} CARLA "
              f"frames. Use --external-tick (recommended) to lock them, or pass "
              f"--step-length {CAPTURE_RADAR_SENSOR_TICK_S}.")
    if AMBIENT_AVAILABLE:
        print(f"  WB route : {ambient_traffic.WB_ROUTE_EDGES}")
        print(f"  EB route : {ambient_traffic.EB_ROUTE_EDGES}")
    if city_mode:
        print(f"  City mode: ON  (ambient veh={ambient_vehicles} "
              f"ped={pedestrians} bike={bicycles})")
        print(f"  Render   : radius={render_radius:.0f}m cull={cull}")
    print(f"{'='*60}\n")

    if not AMBIENT_AVAILABLE:
        print("WARNING: ambient_traffic.py is not importable -- map-edge routes\n"
              "         cannot be applied and the .rou.xml routes on disk will be\n"
              "         used as-is. Fix the import before recording data.\n")

    # --- CARLA sync manager, created UP FRONT -----------------------------
    # Built before the SUMO config so external-tick mode can read the world
    # clock (the capture process's fixed_delta_seconds) and align the SUMO
    # step-length to it -- one authoritative rate rather than two guesses.
    # start() also wipes leftover vehicles before we mirror anything.
    sync = None
    if CARLA_SYNC_AVAILABLE:
        sync = CarlaSyncManager(
            carla_host=carla_host, carla_port=carla_port,
            render_radius=render_radius, cull=cull,
            mirror_pedestrians=bool(pedestrians),
        )
        sync.start()

    if external_tick:
        # Fused run: the capture process owns the world tick. We do not tick and
        # do not touch world settings -- we subscribe. Wait for capture to enable
        # synchronous mode, then adopt its fixed_delta_seconds as THE step rate.
        if sync is None or not sync.connected:
            sys.exit(
                "ERROR: --external-tick needs a reachable CARLA server, but the "
                "sync manager is not connected.\n"
                "Start CarlaUE4 (Town10HD_Opt) and the capture script, or run the "
                "runner without --external-tick for a SUMO-only visual check."
            )
        print(f"  Waiting up to {sync_timeout:.0f}s for the capture process to "
              f"enable CARLA synchronous mode (DATASET_SYNC_MODE=1) ...")
        delta = sync.wait_for_sync_mode(timeout=sync_timeout)
        if delta is None:
            sync.stop()
            sys.exit(
                f"ERROR: timed out after {sync_timeout:.0f}s waiting for the "
                "capture process to enable synchronous mode.\n"
                "Either start CaptureRadarCameraData.py with DATASET_SYNC_MODE=1, "
                "or run the runner without --external-tick."
            )
        if abs(delta - step_length) > 1e-9:
            print(f"  RATE     : aligning SUMO step-length {step_length}s -> CARLA "
                  f"fixed_delta_seconds {delta}s (one clock, capture-owned).")
            step_length = float(delta)
        else:
            print(f"  RATE     : SUMO step-length already matches CARLA "
                  f"fixed_delta_seconds ({delta}s).")
    else:
        # Standalone run: no capture, so nobody should be ticking. If a previous
        # capture died and left the world frozen in synchronous mode, our mirrored
        # actors would spawn into a stopped clock and never move -- heal it.
        if sync is not None and sync.connected:
            sync.reset_to_async()

    # --- Build temporary files ---
    vph         = density_to_vph(density)
    density_map = build_density_map(vph, direction)
    route_tmpl  = os.path.join(ROUTES_DIR, meta["file"])
    route_file  = substitute_density(route_tmpl, density_map, duration)

    # --- City layer: ambient vehicles / bikes / pedestrians ---
    if city_mode and AMBIENT_AVAILABLE:
        with open(route_file, "r", encoding="utf-8") as f:
            rc = f.read()
        ambient_xml = ambient_traffic.generate_ambient(
            net_file, ambient_seed,
            ambient_vehicles, pedestrians, bicycles, duration)
        rc = ambient_traffic.inject_ambient(rc, ambient_xml)
        with open(route_file, "w", encoding="utf-8") as f:
            f.write(rc)

    # Bus stop additional is written at runtime only for scenario 5
    add_files    = [ADD_FILE]
    busstop_file = None
    if scenario_id == 5:
        busstop_file = write_busstop_additional()
        add_files.append(busstop_file)

    cfg_file  = write_sumocfg(net_file, route_file, add_files, duration,
                              step_length=step_length,
                              ignore_route_errors=city_mode,
                              seed=ambient_seed)
    tmp_files = [route_file, cfg_file]
    if busstop_file:
        tmp_files.append(busstop_file)

    print(f"  Config   : {cfg_file}")
    print(f"  Routes   : {route_file}")
    print(f"  Network  : {net_file}")
    print(f"  Additional: {', '.join(add_files)}\n")

    # --- Verify files exist ---
    for label, path in [("Network", net_file), ("Additional", ADD_FILE),
                        ("Routes", route_file), ("Config", cfg_file)]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {label} file not found: {path}")

    # --- Pre-flight: run headless sumo to catch config errors ---
    sumo_exe = os.path.join(SUMO_HOME, "bin", "sumo.exe")
    if not os.path.isfile(sumo_exe):
        sumo_exe = "sumo"
    preflight = subprocess.run(
        [sumo_exe, "-c", cfg_file, "--no-step-log", "--duration-log.disable",
         "true", "--end", "1"],
        capture_output=True, text=True
    )
    if preflight.returncode != 0:
        print("ERROR: SUMO pre-flight check failed. SUMO output:")
        print(preflight.stdout)
        print(preflight.stderr)
        for f in tmp_files:
            try: os.remove(f)
            except OSError: pass
        sys.exit(1)
    print("Pre-flight check passed.\n")

    # --- Choose binary ---
    sumo_bin = SUMO_GUI_BIN if use_gui else SUMO_BIN
    if not os.path.isfile(sumo_bin):
        sumo_bin = "sumo-gui" if use_gui else "sumo"

    cmd = [sumo_bin, "-c", cfg_file, "--start"]
    if use_gui:
        cmd += ["--quit-on-end", "--delay", "50"]  # 50ms delay = near real-time

    # --- Connect TraCI ---
    # (The CARLA sync manager was already created and started above so that
    # external-tick mode could read the world clock before this point.)
    traci.start(cmd)
    # Corridor-priority signals: without this the imported two-phase programs
    # queue traffic back over the 55 m stretch about half the time and every
    # scenario degenerates into the same signal queue (see stretch_signals.py).
    if stretch_signals_mode == "green" and SIGNALS_AVAILABLE:
        stretch_signals.apply_corridor_priority(net_file)
    elif stretch_signals_mode == "green":
        print("WARNING: stretch_signals.py not importable; imported TLS programs left as-is.")
    controller = get_controller(scenario_id, duration)
    controller.on_start()

    # Spawn buses for scenario 5
    if scenario_id == 5:
        spawn_buses_with_stop(duration)

    # Wall-clock pacing is used only in standalone async mode: without it the loop
    # spins as fast as headless SUMO can step and floods CARLA with teleport RPCs
    # (the async firehose that stutters/crashes the server). GUI runs self-pace
    # via sumo-gui --delay, and external-tick runs are paced by the CARLA tick, so
    # both skip the sleep.
    wall_pace = (not external_tick) and (not use_gui)
    next_deadline = time.monotonic()
    tick_timeouts = 0

    print("Simulation running - press Ctrl+C to stop early.\n")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            if external_tick and sync is not None:
                # Block until the capture process completes the next CARLA frame.
                # One SUMO step per CARLA tick keeps the two clocks locked. We
                # never call world.tick() ourselves -- capture owns it.
                if not sync.wait_for_tick(timeout=sync_timeout):
                    tick_timeouts += 1
                    print(f"[runner] WARNING: no CARLA tick within "
                          f"{sync_timeout:.0f}s (x{tick_timeouts}) -- is the "
                          f"capture process still running? Holding SUMO here "
                          f"until ticks resume.", flush=True)
                    continue
                tick_timeouts = 0

            traci.simulationStep()
            controller.step()
            if sync:
                sync.step()

            if wall_pace:
                next_deadline += step_length
                sleep_for = next_deadline - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # Fell behind real time (heavy step); reset the cadence so we
                    # do not accumulate a growing sleep debt.
                    next_deadline = time.monotonic()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        controller.on_stop()
        if sync:
            sync.stop()
        traci.close()
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass
        print("Simulation complete.\n")


# --- CLI entry point ---

def main():
    # Force UTF-8 output on Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="SUMO traffic scenario runner (called by GUI or directly)"
    )
    parser.add_argument("--scenario",  type=int,   required=True,
                        help="Scenario number 1-11")
    parser.add_argument("--density",   type=int,   default=50,
                        help="Traffic density 1-100 (default 50)")
    parser.add_argument("--duration",  type=int,   default=600,
                        help="Simulation duration in seconds (default 600)")
    parser.add_argument("--direction", type=str,   default="BOTH",
                        choices=["WB", "EB", "BOTH"],
                        help="Dominant traffic direction: WB=Westbound EB=Eastbound (default BOTH)")
    parser.add_argument("--gui",       action="store_true",
                        help="Launch sumo-gui instead of headless sumo")
    parser.add_argument("--carla-host", type=str, default="127.0.0.1",
                        help="CARLA server host (default 127.0.0.1)")
    parser.add_argument("--carla-port", type=int, default=2000,
                        help="CARLA server port (default 2000)")

    # --- City / ambient layer ---
    parser.add_argument("--ambient-vehicles", type=int, default=0,
                        help="Ambient city vehicle density 0-100 (0=off)")
    parser.add_argument("--pedestrians", type=int, default=0,
                        help="Pedestrian density 0-100 (0=off)")
    parser.add_argument("--bicycles", type=int, default=0,
                        help="Ambient bicycle density 0-100 (0=off)")
    parser.add_argument("--ambient-seed", type=int, default=42,
                        help="Seed for a reproducible run (default 42)")
    parser.add_argument("--render-radius", type=float, default=RENDER_RADIUS_DEFAULT,
                        help="Only render CARLA actors within this many metres "
                             "of the stretch (default %(default)s)")
    parser.add_argument("--no-cull", action="store_true",
                        help="Disable the render-radius cull (mirror the whole city)")
    parser.add_argument("--step-length", type=float, default=DEFAULT_STEP_LENGTH,
                        help="SUMO simulation step in seconds (default "
                             "%(default)s). Set this to match CARLA's "
                             "fixed_delta_seconds when the capture script runs "
                             "with DATASET_SYNC_MODE=1, otherwise mirrored "
                             "actors advance in visible steps.")
    parser.add_argument("--extend-routes", action="store_true",
                        help="Deprecated and ignored -- map-edge routing is now "
                             "always on. Kept so existing GUI calls still work.")
    parser.add_argument("--external-tick", action="store_true", default=None,
                        help="Fused mode: the capture process owns CARLA's clock. "
                             "The runner subscribes to world.wait_for_tick() (one "
                             "SUMO step per CARLA frame) and adopts CARLA's "
                             "fixed_delta_seconds as the SUMO step-length. Also "
                             "settable via DATASET_EXTERNAL_TICK=1. Omit for a "
                             "standalone SUMO-only visual run.")
    parser.add_argument("--stretch-signals", type=str, default="green",
                        choices=("green", "static"),
                        help="green: corridor-priority signal programs on the boulevard "
                             "lights (default). static: keep the imported two-phase "
                             "programs (stretch queues ~half the time).")
    parser.add_argument("--sync-timeout", type=float, default=60.0,
                        help="Seconds to wait for the capture process to enable "
                             "sync mode, and per-frame tick timeout in external "
                             "mode (default %(default)s).")
    args = parser.parse_args()

    # --external-tick can also come from the environment (how the orchestrator
    # sets it). Explicit --external-tick on the CLI wins; else fall back to env.
    if args.external_tick is None:
        env_flag = os.environ.get("DATASET_EXTERNAL_TICK", "0").strip().lower()
        external_tick = env_flag in ("1", "true", "yes", "on")
    else:
        external_tick = bool(args.external_tick)

    run(
        scenario_id=args.scenario,
        density=args.density,
        duration=args.duration,
        direction=args.direction,
        use_gui=args.gui,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        ambient_vehicles=args.ambient_vehicles,
        pedestrians=args.pedestrians,
        bicycles=args.bicycles,
        ambient_seed=args.ambient_seed,
        render_radius=args.render_radius,
        cull=not args.no_cull,
        extend_routes=args.extend_routes,
        step_length=args.step_length,
        external_tick=external_tick,
        sync_timeout=args.sync_timeout,
        stretch_signals_mode=args.stretch_signals,
    )


if __name__ == "__main__":
    main()
