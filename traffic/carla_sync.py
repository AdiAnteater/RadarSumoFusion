"""
carla_sync.py
=============
Synchronises SUMO vehicle positions into CARLA every simulation step.

Called from runner.py after traci.start(). Not run standalone.

Coordinate transform (netconvert netOffset 109.34, 135.96). The SUMO net was
imported from the CARLA OpenDRIVE export with plain netconvert, so no CARLA-side
Y-flip is baked into the net -- we apply the flip here at runtime:
    carla_x   =  sumo_x - OFFSET_X
    carla_y   = -(sumo_y - OFFSET_Y)
    carla_yaw =  sumo_heading - 90.0

YAW DERIVATION (why sumo_heading - 90, and NOT -(sumo_heading - 90)):
    SUMO heading H is degrees clockwise from north, so the SUMO forward unit
    vector is (sin H, cos H). The runtime Y-flip (carla_y = -sumo_y + off)
    maps that vector to (sin H, -cos H) in CARLA's left-handed frame, and
    CARLA reports yaw = atan2(forward_y, forward_x) = atan2(-cos H, sin H),
    which simplifies to H - 90. Cardinal check:
        east  H=90  -> 0     west  H=270 -> 180
        north H=0   -> -90   south H=180 -> +90
    The earlier -(H - 90) form negated this. Negation leaves east/west
    unchanged (they are self-symmetric: 0 -> 0, 180 -> -180 == 180) but flips
    north/south-heading vehicles by 180 degrees. That is exactly why cars
    looked correct on the east-west monitored stretch yet reversed / faced
    backward on the north-south approach and its stop line. This is the fix
    for that symptom -- confirm it by watching a vehicle on a straight N-S
    segment: its CARLA actor yaw should read about -90 when SUMO angle is ~0.

THIS VERSION also carries three fixes from earlier debugging passes:

1. FLOATING -- ground clearance was too generous (0.3m) for some vehicle
   blueprints whose collision origin isn't at wheel-bottom. Reduced to
   0.08m. Spawn log now also prints the blueprint name + final z so if
   one specific vehicle type still floats we can see exactly which one.

2. GHOST ACTORS -- if traci.vehicle.getIDList() ever raises or returns
   incomplete results for a single step, the previous code's
   "carla_ids - sumo_ids" destroy logic would silently skip cleanup for
   that step and never get a second chance, since the vehicle is gone
   from SUMO for good on the next successful call. Fixed by wrapping the
   SUMO ID fetch so a failure does NOT proceed with an empty/partial
   sumo_ids set, and by adding an explicit destroy-count vs spawn-count
   audit on stop() so leaks are immediately visible in the log.

3. DIAGONAL STOPPING AT JUNCTIONS -- caused by get_waypoint()'s
   project_to_road snap picking a DIFFERENT lane/road than intended when
   near a junction, where multiple lanes are spatially close together.
   This was visible in the last DELTA log as ~7-8m jumps for vehicles
   near junctions. Fixed by rejecting snaps that jump too far (those
   indicate we landed on the wrong road segment, not that our transform
   is wrong) and, when that happens, searching nearby waypoints for the
   one whose lane direction best matches our SUMO heading, since at
   junctions multiple roads can be spatially close but pointing
   different directions. (This heuristic compares lane yaw against the
   converted SUMO yaw, so it also benefits from the yaw-sign correction
   above -- the heading match is now meaningful for N-S approaches too.)
"""

import math
import time
import traceback

try:
    import traci
except ImportError:
    raise ImportError("traci not found - activate your venv")

try:
    import carla
except ImportError:
    raise ImportError("carla not found - run: pip install carla==0.9.16")

# Shared vehicle-cleanup helper. Guarded so carla_sync still imports even if
# the file is missing -- in that case start/stop just skip the world wipe.
try:
    from carla_cleanup import destroy_all_vehicles
except ImportError:
    destroy_all_vehicles = None


# ---------------------------------------------------------------------------
# WORLD CLOCK OWNERSHIP -- read before adding anything to this module.
#
# This module NEVER calls world.tick(). Exactly one process may own the tick,
# and that owner is dataset/capture/CaptureRadarCameraData.py, which takes the
# world clock when DATASET_SYNC_MODE=1 and ticks at DATASET_SYNC_FIXED_DELTA_S
# (defaulting to the radar sensor tick, RADAR_SENSOR_TICK_S = 0.05 s).
#
# In the fused run the runner does NOT tick and does NOT change world settings.
# It paces itself on world.wait_for_tick() (see wait_for_sync_mode /
# wait_for_tick below) so each SUMO step lines up with exactly one CARLA frame
# at whatever rate the capture clock dictates -- one clock, no drift.
#
# The single apply_settings() this module is allowed to make is reset_to_async():
# it runs ONLY in the runner's standalone path (no capture process present), to
# heal a world a dead capture left frozen in synchronous mode. It is never
# called while capture is running, so the one-owner rule always holds.
# ---------------------------------------------------------------------------

OFFSET_X = 109.34
OFFSET_Y = 135.96

# Reduced from 0.3 -- was causing visible floating for some blueprints.
GROUND_CLEARANCE = 0.08

# Used only if no drivable waypoint at all is found near our computed
# position (e.g. transform put us totally off the map).
FALLBACK_Z = 0.5

WAYPOINT_SEARCH_Z = 50.0

# If the nearest drivable waypoint is farther than this from our computed
# (x, y), treat it as a wrong-road snap (typically at junctions) rather
# than trusting it blindly. Mid-lane error has been sub-0.3m in testing,
# so 2.0m gives headroom without accepting junction mis-snaps (which were
# 7+m in the last log).
MAX_SNAP_DISTANCE = 2.0

# When the nearest snap is rejected, search this far forward/back along
# the road graph (at each of these distances) for a waypoint whose
# heading matches better. A single fixed distance wasn't enough near
# some junctions/curves -- widened to a list of progressively larger
# search distances.
HEADING_MATCH_SEARCH_DISTANCES = [1.0, 3.0, 6.0, 10.0]

# Extra Z heights to try (added on top of the resolved ground clearance)
# if the initial spawn attempt collides. CARLA's spawn collision check is
# sensitive to exact Z -- a vehicle queued just ahead at a stop line can
# cause a same-spot collision that a slightly higher spawn clears.
SPAWN_RETRY_EXTRA_Z = [0.0, 0.15, 0.35, 0.6]

# After this many consecutive failed spawn attempts for the same SUMO
# vehicle, log a loud one-time warning so a permanently-deadlocked
# vehicle (invisible in CARLA but still moving in SUMO) is obvious
# instead of silently retrying for the rest of the run.
SPAWN_FAIL_WARN_THRESHOLD = 15

BLUEPRINT_MAP = {
    "car":            "vehicle.tesla.model3",
    "car_aggressive": "vehicle.bmw.grandtourer",
    "car_slow":       "vehicle.audi.a2",
    "bus":            "vehicle.volkswagen.t2",
    "truck":          "vehicle.carlamotors.carlacola",
    "truck_large":    "vehicle.carlamotors.carlacola",
    "motorcycle":     "vehicle.kawasaki.ninja",
    # Ambient bicycles arrive as SUMO vehicles (vClass bicycle, vType
    # "amb_bike") so they flow through the normal vehicle path -- they just
    # need a two-wheeled blueprint here.
    "amb_bike":       "vehicle.bh.crossbike",
    "bike":           "vehicle.bh.crossbike",
    "bicycle":        "vehicle.bh.crossbike",
}
DEFAULT_BLUEPRINT = "vehicle.tesla.model3"

# Bicycle blueprints, tried in order if the mapped one is unavailable.
BIKE_BLUEPRINTS = [
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
]
BIKE_VTYPES = {"amb_bike", "bike", "bicycle"}

# Pedestrian mirroring.
WALKER_FILTER = "walker.pedestrian.*"
# Walkers stand on the sidewalk; their origin is at the feet, so only a small
# clearance is needed above the resolved ground surface.
PED_GROUND_CLEARANCE = 0.5

# Refine walker ground height with a real downward raycast against the mesh
# (world.ground_projection) instead of trusting the OpenDRIVE sidewalk lane
# elevation. This is what actually puts feet on top of the kerb rather than
# on the reference surface underneath it. Results are cached on a grid (see
# GROUND_PROBE_GRID_M) so the per-step cost collapses to a dict lookup after
# the first pedestrian visits a given patch of pavement. Set False if the
# raycast RPCs ever show up in profiling.
USE_GROUND_PROJECTION = True

# Cache resolution for the walker ground raycast, in metres. Small enough to
# resolve a kerb edge, large enough that a walking pedestrian reuses a cell
# for several consecutive steps.
GROUND_PROBE_GRID_M = 0.25

# Centre of the CARLA render radius (monitored-stretch midpoint in CARLA
# world coords) and default radius. SUMO simulates the WHOLE city so traffic
# flows naturally into the stretch, but CARLA only renders actors within this
# radius of the sensors -- keeping actor counts down and ensuring the SUMO
# spawn/despawn gateways (>115 m away) are never rendered near the sensors.
DEFAULT_CENTER_CARLA  = (-0.85, 14.49)
DEFAULT_RENDER_RADIUS = 120.0


def sumo_to_carla_xy_yaw(sumo_x: float, sumo_y: float, sumo_heading: float):
    cx  =  sumo_x - OFFSET_X
    cy  = -(sumo_y - OFFSET_Y)
    yaw =  sumo_heading - 90.0
    return cx, cy, yaw


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees,
    accounting for wraparound (e.g. 179 vs -179 should be 2, not 358)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


class CarlaSyncManager:

    def __init__(self, carla_host: str = "127.0.0.1", carla_port: int = 2000,
                 timeout: float = 10.0,
                 center=DEFAULT_CENTER_CARLA,
                 render_radius: float = DEFAULT_RENDER_RADIUS,
                 cull: bool = True,
                 mirror_pedestrians: bool = True):
        self.host    = carla_host
        self.port    = carla_port
        self.timeout = timeout

        # Render-radius cull. When cull is True, only actors within
        # render_radius metres of center (CARLA world coords) are spawned in
        # CARLA; actors leaving the radius are destroyed and re-spawned if they
        # come back. SUMO still simulates every actor across the whole city.
        self._center_x, self._center_y = center
        self._render_radius = float(render_radius)
        self._cull = bool(cull)
        self._mirror_peds = bool(mirror_pedestrians)

        self._client     = None
        self._world      = None
        self._map        = None
        self._blueprints = None
        self._actors     = {}
        self._vtype_map  = {}
        self._connected  = False

        # Pedestrian actors, tracked separately from vehicles.
        self._ped_actors = {}
        self._ped_warned = False

        self._delta_logged  = set()
        # One-shot diagnostics and the walker ground-height cache (see
        # _pedestrian_ground_z).
        self._far_wp_warned    = False
        self._ped_z_logged     = False
        self._ped_ground_cache = {}
        self._spawn_count   = 0
        self._destroy_count = 0
        # Tracks how many consecutive spawn attempts have failed for a
        # given SUMO vehicle id. Used to detect vehicles that are stuck
        # in a permanent collision deadlock (e.g. always landing on top
        # of a queued vehicle ahead at a junction stop line) rather than
        # retrying silently forever.
        self._spawn_fail_count = {}

    def start(self):
        print(f"[CarlaSyncManager] Connecting to CARLA at {self.host}:{self.port} ...")
        try:
            self._client     = carla.Client(self.host, self.port)
            self._client.set_timeout(self.timeout)
            self._world      = self._client.get_world()
            self._map        = self._world.get_map()
            self._blueprints = self._world.get_blueprint_library()
            self._connected  = True
            print(f"[CarlaSyncManager] Connected. Map: {self._map.name}")

            # Wipe any leftover/parked vehicles BEFORE we mirror anything.
            # Leaked actors from a prior crashed run, or Town10HD_Opt's own
            # parked cars, sit on our spawn points and cause the repeated
            # "Spawn collision ... attempt N" cascade (the mirror vehicle
            # then never appears, and a physics-on leftover looks "stuck"
            # while our physics-off cars teleport through it).
            if destroy_all_vehicles is not None:
                try:
                    destroy_all_vehicles(self._client, self._world,
                                         include_walkers=self._mirror_peds,
                                         verbose=True)
                except Exception as e:
                    print(f"[CarlaSyncManager] WARNING: start cleanup failed: {e}")
        except Exception as e:
            print(f"[CarlaSyncManager] WARNING: Could not connect to CARLA: {e}")
            print("[CarlaSyncManager] Running in SUMO-only mode.")
            self._connected = False

    # --- World-clock subscription helpers (used by runner.py) ------------
    # These let the runner act as a passive tick subscriber in the fused run
    # (capture owns the tick) and heal a frozen world in the standalone run.
    # None of them ever calls world.tick().

    @property
    def connected(self) -> bool:
        return self._connected

    def world_settings(self):
        """Return the live carla.WorldSettings, or None if not connected."""
        if not self._connected:
            return None
        try:
            return self._world.get_settings()
        except Exception as e:  # noqa: BLE001
            print(f"[CarlaSyncManager] get_settings failed: {e}")
            return None

    def wait_for_sync_mode(self, timeout: float = 60.0, poll: float = 0.25):
        """Block until another client (the capture process) puts the world into
        synchronous mode, then return its fixed_delta_seconds. Returns None on
        timeout. This is how the runner discovers the single authoritative tick
        rate instead of guessing one."""
        if not self._connected:
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self.world_settings()
            if s is not None and s.synchronous_mode and s.fixed_delta_seconds:
                return float(s.fixed_delta_seconds)
            time.sleep(poll)
        return None

    def wait_for_tick(self, timeout: float = 10.0) -> bool:
        """Block until the server publishes the next world snapshot (produced by
        whichever client owns the tick -- the capture process here). Returns True
        on a tick, False on timeout. NEVER calls world.tick(): a second client
        that ticked too would fight the capture clock. This is the passive-client
        pacing pattern for synchronous mode."""
        if not self._connected:
            return False
        try:
            self._world.wait_for_tick(timeout)
            return True
        except RuntimeError:
            return False

    def reset_to_async(self) -> bool:
        """Standalone-only heal: if the world is stuck in synchronous mode (a dead
        capture left it that way, so nobody is ticking and everything is frozen),
        put it back to asynchronous free-run. No-op if already async. Must NOT be
        called while the capture process is running."""
        if not self._connected:
            return False
        try:
            s = self._world.get_settings()
            if s.synchronous_mode:
                s.synchronous_mode = False
                s.fixed_delta_seconds = None
                self._world.apply_settings(s)
                print("[CarlaSyncManager] Healed a world left in synchronous mode "
                      "-- reset to asynchronous free-run.")
                return True
        except Exception as e:  # noqa: BLE001
            print(f"[CarlaSyncManager] reset_to_async failed: {e}")
        return False

    def _within_radius(self, cx: float, cy: float) -> bool:
        """True if a CARLA (x, y) is inside the render radius (or if culling
        is disabled, always True)."""
        if not self._cull:
            return True
        return math.hypot(cx - self._center_x,
                          cy - self._center_y) <= self._render_radius

    def _visible_subset(self, ids, getpos):
        """From a set of SUMO ids, return only those whose CARLA position is
        within the render radius. Uses the cheap raw transform (offset + flip,
        no waypoint snap) -- exact enough for a radius test that's far from the
        sensors, and avoids a second expensive snap per actor."""
        visible = set()
        for aid in ids:
            try:
                x, y = getpos(aid)
            except Exception:
                continue
            cx = x - OFFSET_X
            cy = -(y - OFFSET_Y)
            if self._within_radius(cx, cy):
                visible.add(aid)
        return visible

    def step(self):
        if not self._connected:
            return
        self._sync_vehicles()
        if self._mirror_peds:
            # Pedestrian mirroring must never take down vehicle sync -- if
            # anything in the person path fails, log once and carry on.
            try:
                self._sync_persons()
            except Exception as e:
                if not self._ped_warned:
                    self._ped_warned = True
                    print(f"[CarlaSyncManager] WARNING: pedestrian sync "
                          f"failed ({e}) -- continuing without pedestrians. "
                          f"(Is the ped network loaded? build_ped_network.py)")

    def _sync_vehicles(self):
        # Fetch the SUMO ID list defensively. If this call fails, we must
        # NOT proceed with an empty/partial set -- that was the root cause of
        # ghost actors, since any vehicle that should have been destroyed this
        # step would otherwise never get a second chance once it's truly gone
        # from SUMO on the next successful call.
        try:
            sumo_ids = set(traci.vehicle.getIDList())
        except Exception as e:
            print(f"[CarlaSyncManager] WARNING: getIDList() failed this "
                  f"step ({e}) -- skipping sync this step to avoid "
                  f"orphaning actors")
            return

        # Only mirror vehicles within the render radius. Vehicles outside it
        # (spawning at far gateways, or crossing the far side of the city) are
        # simulated by SUMO but not drawn in CARLA; they appear once they enter
        # the radius and are destroyed when they leave it.
        visible = self._visible_subset(sumo_ids, traci.vehicle.getPosition)
        carla_ids = set(self._actors.keys())
        try:
            for vid in visible - carla_ids:
                self._spawn(vid)
            for vid in visible & carla_ids:
                self._move(vid)
            # Destroy actors that are gone from SUMO OR have left the radius.
            for vid in carla_ids - visible:
                self._destroy(vid)
        except Exception as e:
            print(f"[CarlaSyncManager] Step error: {e}")
            traceback.print_exc()

    def _sync_persons(self):
        try:
            person_ids = set(traci.person.getIDList())
        except Exception as e:
            # traci.person raises if no pedestrians are in the scenario; treat
            # that as "nothing to mirror" rather than an error.
            person_ids = set()
        visible = self._visible_subset(person_ids, traci.person.getPosition)
        carla_ids = set(self._ped_actors.keys())
        for pid in visible - carla_ids:
            self._spawn_person(pid)
        for pid in visible & carla_ids:
            self._move_person(pid)
        for pid in carla_ids - visible:
            self._destroy_person(pid)

    def stop(self):
        if not self._connected:
            return
        remaining = len(self._actors) + len(self._ped_actors)
        print(f"[CarlaSyncManager] Destroying {remaining} remaining actors "
              f"({len(self._actors)} vehicles, {len(self._ped_actors)} pedestrians) ...")
        for actor in list(self._actors.values()) + list(self._ped_actors.values()):
            try:
                actor.destroy()
                self._destroy_count += 1
            except Exception:
                pass
        self._actors.clear()
        self._ped_actors.clear()
        self._vtype_map.clear()
        self._delta_logged.clear()

        # Audit: every spawned actor should eventually be destroyed
        # (either during the run when SUMO removed it, or here at
        # shutdown). If destroy_count < spawn_count, actors were lost
        # somewhere without going through _destroy() -- that's the ghost
        # signature to watch for.
        status = "OK" if self._spawn_count == self._destroy_count else "MISMATCH -- possible leaked actors"
        print(f"[CarlaSyncManager] Audit: spawned={self._spawn_count} "
              f"destroyed={self._destroy_count} {status}")

        # Final safety sweep: destroy ANY vehicle still in the world, not just
        # the ones we tracked. Catches leftovers so the map is clean when the
        # process stops or the simulation ends. (Hard-kills that skip this
        # finally block are handled by running clear_carla_actors.py.)
        if destroy_all_vehicles is not None:
            try:
                destroy_all_vehicles(self._client, self._world,
                                     include_walkers=self._mirror_peds,
                                     verbose=True)
            except Exception as e:
                print(f"[CarlaSyncManager] WARNING: stop cleanup failed: {e}")

        print("[CarlaSyncManager] Done.")

    def _get_blueprint(self, vid: str) -> carla.ActorBlueprint:
        vtype = self._vtype_map.get(vid)
        if vtype is None:
            try:
                vtype = traci.vehicle.getTypeID(vid)
            except Exception:
                vtype = "car"
            self._vtype_map[vid] = vtype
        is_bike = vtype in BIKE_VTYPES
        bp = None
        if is_bike:
            # Try the bike blueprints in order; some builds are missing one.
            for name in BIKE_BLUEPRINTS:
                bp = self._blueprints.find(name)
                if bp is not None:
                    break
            if bp is None:
                found = self._blueprints.filter("vehicle.*.crossbike")
                bp = found[0] if found else None
        else:
            bp = self._blueprints.find(
                BLUEPRINT_MAP.get(vtype, DEFAULT_BLUEPRINT))

        if bp is None:
            bp = self._blueprints.filter("vehicle.tesla.*")[0]
        if bp.has_attribute("color") and not is_bike:
            bp.set_attribute("color", "255,255,255")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"sumo_{vid}")
        return bp

    def _resolve_ground(self, x: float, y: float, sumo_yaw: float):
        """Resolve the road surface height at (x, y).

        Returns (x, y, z, wp, snap_ok). NOTE: x and y are returned UNCHANGED.

        This function used to return the waypoint's own x/y, i.e. it snapped
        every vehicle onto the lane centreline. That was wrong and it was the
        cause of the abrupt lane changes:

          SUMO moves a vehicle across the lane boundary continuously. Sampling
          getLateralLanePosition() at 0.1 s through a forced change gives

            t=3.4 lat=-0.350   t=3.8 lat=-0.817   t=4.2 lat=-1.283
            t=4.6 lat=-1.750 [lane -1_4] -> t=5.0 lat=+1.283 [lane -1_3]
            t=5.4 lat=+0.817   t=5.8 lat=+0.350   t=6.2 lat= 0.000

          a smooth 1.17 m/s slide. Snapping x/y to the lane centre discarded
          all of it: the car stayed pinned to lane -1_4's centreline until
          get_waypoint() flipped to -1_3, then jumped 3.5 m sideways in one
          frame. Meanwhile the yaw below still came from SUMO's getAngle(),
          which DOES include the lane-change drift -- so the car was rendered
          rotated diagonally while travelling straight down the lane centre,
          then teleported across. Both symptoms, one line.

        The snap was a workaround from when the negated-yaw bug was making
        vehicles drift; that bug is fixed and the transform is verified to
        0.01 m, so there is nothing left for it to correct. We now trust the
        computed position -- exactly as the internal-lane branch in
        _sumo_transform already did -- and use the waypoint for Z only.

        The distance check is kept, but purely as a diagnostic: a far-away
        nearest waypoint means our transform and CARLA's road graph disagree
        somewhere, which is worth knowing even though we no longer act on it.
        """
        wp = self._map.get_waypoint(
            carla.Location(x=x, y=y, z=WAYPOINT_SEARCH_Z),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp is None:
            return x, y, FALLBACK_Z, None, False

        loc = wp.transform.location
        dist = math.hypot(loc.x - x, loc.y - y)

        if dist > MAX_SNAP_DISTANCE and not self._far_wp_warned:
            self._far_wp_warned = True
            print(f"[CarlaSyncManager] NOTE: nearest drivable waypoint is "
                  f"{dist:.2f}m from computed ({x:.2f}, {y:.2f}) "
                  f"(road={wp.road_id} lane={wp.lane_id}). Position is used "
                  f"as computed; only Z is taken from the waypoint. Worth a "
                  f"look if this fires away from a junction.")

        return x, y, loc.z + GROUND_CLEARANCE, wp, dist <= MAX_SNAP_DISTANCE

    def _sumo_transform(self, vid: str, log_delta: bool = False):
        x, y    = traci.vehicle.getPosition(vid)
        heading = traci.vehicle.getAngle(vid)
        raw_cx, raw_cy, yaw = sumo_to_carla_xy_yaw(x, y, heading)

        # SUMO models the actual turning path through an intersection
        # using auto-generated "internal lanes" (IDs starting with ':',
        # e.g. ":719_7_0"). These only exist inside SUMO's junction
        # geometry and have no equivalent named road in CARLA's road
        # graph -- snapping against the nearest external road (the one
        # the vehicle is about to turn onto, or just came from) produces
        # a consistent multi-meter mis-snap, which is what was showing
        # up as vehicles "stuck" mid-turn at junctions. While on an
        # internal lane we trust our own computed transform directly
        # (still ground-snapped for Z) instead of forcing a lane-center
        # snap that doesn't apply here.
        try:
            on_internal_lane = traci.vehicle.getLaneID(vid).startswith(":")
        except Exception:
            on_internal_lane = False

        if on_internal_lane:
            wp_for_z = self._map.get_waypoint(
                carla.Location(x=raw_cx, y=raw_cy, z=WAYPOINT_SEARCH_Z),
                project_to_road=True,
                lane_type=carla.LaneType.Any
            )
            final_x, final_y = raw_cx, raw_cy
            final_z = (wp_for_z.transform.location.z if wp_for_z is not None else FALLBACK_Z) + GROUND_CLEARANCE
            if log_delta and vid not in self._delta_logged:
                self._delta_logged.add(vid)
                print(f"[CarlaSyncManager] DELTA {vid}: on internal junction "
                      f"lane -- using raw transform ({raw_cx:.2f}, {raw_cy:.2f}) "
                      f"without road-snap")
        else:
            final_x, final_y, final_z, wp, snap_ok = self._resolve_ground(raw_cx, raw_cy, yaw)

            if log_delta and vid not in self._delta_logged:
                self._delta_logged.add(vid)
                if not snap_ok or wp is None:
                    print(f"[CarlaSyncManager] DELTA {vid}: no drivable waypoint "
                          f"near computed ({raw_cx:.2f}, {raw_cy:.2f}) -- "
                          f"falling back to flat z, check transform near here")
                else:
                    # x/y are no longer altered, so there is no dx/dy to report.
                    # What is still worth logging is how far the computed
                    # position sits from the lane centre (that is the vehicle's
                    # real lateral offset, and it should be near 0 except during
                    # a lane change) and whether our yaw agrees with the lane.
                    lat = math.hypot(wp.transform.location.x - raw_cx,
                                     wp.transform.location.y - raw_cy)
                    print(f"[CarlaSyncManager] DELTA {vid}: pos=({raw_cx:.2f}, "
                          f"{raw_cy:.2f}) z={final_z:.2f} "
                          f"road={wp.road_id} lane={wp.lane_id} "
                          f"lat_offset={lat:.2f}m "
                          f"yaw_sumo={yaw:.1f} yaw_lane={wp.transform.rotation.yaw:.1f}")

        transform = carla.Transform(
            carla.Location(x=final_x, y=final_y, z=final_z),
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        )
        return transform

    def _spawn(self, vid: str):
        try:
            bp        = self._get_blueprint(vid)
            transform = self._sumo_transform(vid, log_delta=True)

            actor = None
            for extra_z in SPAWN_RETRY_EXTRA_Z:
                attempt = carla.Transform(
                    carla.Location(
                        x=transform.location.x,
                        y=transform.location.y,
                        z=transform.location.z + extra_z
                    ),
                    transform.rotation
                )
                actor = self._world.try_spawn_actor(bp, attempt)
                if actor is not None:
                    transform = attempt
                    break

            if actor is not None:
                actor.set_simulate_physics(False)
                self._actors[vid] = actor
                self._spawn_count += 1
                self._spawn_fail_count.pop(vid, None)
                print(f"[CarlaSyncManager] Spawned {vid} ({bp.id}) at "
                      f"x={transform.location.x:.1f} y={transform.location.y:.1f} "
                      f"z={transform.location.z:.2f} yaw={transform.rotation.yaw:.1f}")
            else:
                fails = self._spawn_fail_count.get(vid, 0) + 1
                self._spawn_fail_count[vid] = fails
                print(f"[CarlaSyncManager] Spawn collision for {vid} at "
                      f"x={transform.location.x:.1f} y={transform.location.y:.1f} "
                      f"z={transform.location.z:.2f} (attempt {fails}) -- will retry next step")
                if fails == SPAWN_FAIL_WARN_THRESHOLD:
                    print(f"[CarlaSyncManager] WARNING: {vid} has failed to spawn "
                          f"{fails} consecutive times -- it is moving in SUMO but "
                          f"INVISIBLE in CARLA. Likely a same-spot collision with "
                          f"another actor queued at a stop line. Consider raising "
                          f"SPAWN_RETRY_EXTRA_Z values or checking for a stalled "
                          f"actor at this position.")
        except Exception as e:
            print(f"[CarlaSyncManager] Spawn error for {vid}: {e}")

    def _move(self, vid: str):
        try:
            self._actors[vid].set_transform(self._sumo_transform(vid))
        except Exception as e:
            print(f"[CarlaSyncManager] Move error for {vid}: {e}")

    def _destroy(self, vid: str):
        try:
            self._actors[vid].destroy()
            self._destroy_count += 1
        except Exception:
            pass
        self._actors.pop(vid, None)
        self._vtype_map.pop(vid, None)
        self._delta_logged.discard(vid)
        self._spawn_fail_count.pop(vid, None)

    # --- Pedestrian mirroring ---

    def _pedestrian_ground_z(self, cx: float, cy: float) -> float:
        """Height of the walkable surface under (cx, cy), in CARLA world coords.

        This is the fix for pedestrians buried in the pavement. The previous
        lookup asked for

            lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder
                      | carla.LaneType.Any

        which looks like "prefer the sidewalk, fall back to shoulder, then
        anything". It is not. carla.LaneType is a bitmask and LaneType.Any is
        0xFFFFFFFE -- every bit set. OR-ing anything into it yields Any, so the
        expression collapses to plain LaneType.Any and the sidewalk preference
        is silently annihilated. get_waypoint then returned the NEAREST lane of
        any type, which for a pedestrian standing near the kerb is the driving
        lane. Every walker got road-surface Z plus 0.10 m, and the pavement sits
        a kerb height above that, so they stood shin-deep in it.

        Two layers now:

        1. Ask for LaneType.Sidewalk on its own, then Shoulder, then Any --
           each as a separate call, never OR-ed with Any.
        2. If USE_GROUND_PROJECTION, refine with a real downward raycast onto
           the rendered mesh. This matters because the OpenDRIVE sidewalk lane
           carries a reference elevation that does not necessarily include the
           kerb height, and because pedestrians spend roughly half their time on
           walkingareas and crossings where no sidewalk lane exists at all and
           the projected waypoint can be metres away. The raycast asks the map
           what is actually under the pedestrian's feet.

        Raycasts are server round-trips, so results are cached on a
        GROUND_PROBE_GRID_M grid. Pavement does not move; after a pedestrian has
        walked a stretch once, every later query there is a dict hit.
        """
        key = (round(cx / GROUND_PROBE_GRID_M), round(cy / GROUND_PROBE_GRID_M))
        cached = self._ped_ground_cache.get(key)
        if cached is not None:
            return cached

        base = None
        for lane_type in (carla.LaneType.Sidewalk,
                          carla.LaneType.Shoulder,
                          carla.LaneType.Any):
            wp = self._map.get_waypoint(
                carla.Location(x=cx, y=cy, z=WAYPOINT_SEARCH_Z),
                project_to_road=True,
                lane_type=lane_type,
            )
            if wp is not None:
                base = wp.transform.location.z
                break
        if base is None:
            base = FALLBACK_Z

        if USE_GROUND_PROJECTION:
            try:
                hit = self._world.ground_projection(
                    carla.Location(x=cx, y=cy, z=base + 2.0), 6.0)
                if hit is not None:
                    if not self._ped_z_logged:
                        self._ped_z_logged = True
                        print(f"[CarlaSyncManager] Pedestrian ground: lane "
                              f"elevation z={base:.3f}, raycast to mesh "
                              f"z={hit.location.z:.3f} (delta "
                              f"{hit.location.z - base:+.3f} m). Walkers are "
                              f"placed on the raycast surface + "
                              f"{PED_GROUND_CLEARANCE:.2f} m.")
                    base = hit.location.z
            except Exception as e:
                if not self._ped_z_logged:
                    self._ped_z_logged = True
                    print(f"[CarlaSyncManager] NOTE: ground_projection "
                          f"unavailable ({e}); falling back to sidewalk lane "
                          f"elevation for walker height.")

        z = base + PED_GROUND_CLEARANCE
        self._ped_ground_cache[key] = z
        return z

    def _person_transform(self, pid: str) -> carla.Transform:
        """Same offset/flip/yaw transform as vehicles. x/y are used exactly as
        computed -- SUMO's lateral placement on the pavement is already correct
        (measured: 0.0% of sidewalk pedestrian samples fall inside a drivable
        lane) -- and only Z is resolved against the world."""
        x, y = traci.person.getPosition(pid)
        heading = traci.person.getAngle(pid)
        cx, cy, yaw = sumo_to_carla_xy_yaw(x, y, heading)
        return carla.Transform(
            carla.Location(x=cx, y=cy, z=self._pedestrian_ground_z(cx, cy)),
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )

    def _walker_blueprint(self, pid: str) -> carla.ActorBlueprint:
        walkers = self._blueprints.filter(WALKER_FILTER)
        # Deterministic pick per pedestrian id so a given ped keeps one look.
        bp = walkers[hash(pid) % len(walkers)]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"sumo_{pid}")
        return bp

    def _spawn_person(self, pid: str):
        try:
            bp = self._walker_blueprint(pid)
            transform = self._person_transform(pid)
            actor = None
            for extra_z in SPAWN_RETRY_EXTRA_Z:
                attempt = carla.Transform(
                    carla.Location(transform.location.x, transform.location.y,
                                   transform.location.z + extra_z),
                    transform.rotation,
                )
                actor = self._world.try_spawn_actor(bp, attempt)
                if actor is not None:
                    break
            if actor is not None:
                actor.set_simulate_physics(False)
                self._ped_actors[pid] = actor
                self._spawn_count += 1
        except Exception as e:
            print(f"[CarlaSyncManager] Person spawn error for {pid}: {e}")

    def _move_person(self, pid: str):
        try:
            self._ped_actors[pid].set_transform(self._person_transform(pid))
        except Exception as e:
            print(f"[CarlaSyncManager] Person move error for {pid}: {e}")

    def _destroy_person(self, pid: str):
        try:
            self._ped_actors[pid].destroy()
            self._destroy_count += 1
        except Exception:
            pass
        self._ped_actors.pop(pid, None)
