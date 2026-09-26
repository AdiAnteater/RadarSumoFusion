"""
ambient_traffic.py
==================
Single source of truth for:

1. The map-edge scenario routes (WB_route / EB_route). Applied by runner.py on
   EVERY run now, not just city mode.
2. Whole-city ambient background traffic: vehicles, bicycles, pedestrians.
3. Pedestrian corridors along the monitored stretch.

Requires network/Town10HD_Opt_fixed.net.xml (see build_network.py). On the
un-rebuilt net the sidewalk lanes are drivable and everything below misbehaves.


MAP-EDGE ROUTES
---------------
Town10HD_Opt is a closed loop with no dead-ends, so "the edge of the map" is the
outer perimeter ring. Its four corner links are also its four longest edges:
5/-5 (SW), 1/-1 (SE), 10/-10 (NE), 6/-6 (NW). The two corners farthest from the
sensors are SE (node 8.1) and SW (node 468), so scenario traffic runs corner to
corner between them, passing through the monitored stretch mid-journey:

  WB: node 8.1 (SE) -> -1 -2 21 [20] 19 18 5 -> node 468 (SW)
  EB: node 468 (SW) -> -5 -18 -19 [-20] -21 2 1 -> node 8.1 (SE)

Verified against Town10HD_Opt_fixed.net.xml with sumolib -- both routes are
fully connected on passenger lanes:

  route  total   spawn dist   despawn dist   spawn edge
  WB     357 m   131.1 m      146.6 m        -1  (130 m, lanes -1_3/-1_4)
  EB     315 m   131.3 m      115.3 m        -5  ( 97 m, lanes -5_3/-5_4)

Both corners are used in both directions (WB despawns where EB spawns and vice
versa), on opposite carriageways, so nothing collides.

This replaces the previous extended routes, which had two problems:
  - WB started on edge -2, which is 11 m long. A 12 m bus cannot be inserted on
    an 11 m edge, so scenario 5's buses could never depart there.
  - EB started on edge 6, whose spawn point is 108.5 m from the stretch centre,
    not the >115 m that was claimed.


PEDESTRIANS ON THE MONITORED STRETCH
------------------------------------
Six sidewalk corridors, all of which traverse edge 20 or -20 and all of which
use a real crossing. Verified on the rebuilt net: 100% departure rate, 100%
reach the stretch, nearest spawn point 91 m from the stretch centre.
"""

import random

try:
    import sumolib
except ImportError:
    sumolib = None


# --- Shared city constants (verified against Town10HD_Opt_fixed.net.xml) ---

# The monitored stretch. WB traffic uses edge 20, EB uses edge -20.
# Numeric ids -- there is no "monitored" substring anywhere in this net. Any
# controller testing road names must compare against THIS set.
MONITORED_EDGES = {"20", "-20"}

# Map-edge to map-edge scenario routes. Applied to every run by runner.py.
WB_ROUTE_EDGES = "-1 -2 21 20 19 18 5"
EB_ROUTE_EDGES = "-5 -18 -19 -20 -21 2 1"

# Second exit per direction, so BOTH driving lanes of the stretch are through
# lanes. Lane connectivity after the stretch (from the net):
#     WB  18_4 -> 5      18_3 -> -6        EB  -21_3 -> 2     -21_4 -> -3
# With a single exit route every WB vehicle had to be in lane 4 within 19 m
# of leaving the stretch and every EB vehicle in lane 3 within 25 m, so SUMO's
# strategic lane changing pulled everything into one lane ON the stretch and
# no speed-gain lane change (overtaking) ever happened there: headless
# validation of scenarios 7/9/10 counted 0 lane changes on the stretch while
# the same runs had dozens on the 100+ m approach edges. Lane connectivity
# after the stretch (19/18 and -21 are 9-25 m) also means a vehicle cannot
# leave its lane on the stretch and get back, so overtaking here is a
# between-lanes speed differential, not a lane-change manoeuvre. Flows and
# vehicles with departLane first/last/3/4 keep the exit that matches that
# lane; random/best/free flows sample a 50/50 routeDistribution (WB_mix /
# EB_mix), so both lanes carry through traffic. The alternative exits
# despawn ~96 m (WB, junction 23) and ~119 m (EB, node 10.0) from the stretch
# centre, beyond radar reach and behind the camera.
WB_ROUTE_ALT_EDGES = "-1 -2 21 20 19 18 -6"
EB_ROUTE_ALT_EDGES = "-5 -18 -19 -20 -21 -3 -0"
# departLane -> route id, per direction (lane 3 = curb, lane 4 = inner).
WB_LANE_ROUTE = {"3": "WB_route_alt", "4": "WB_route"}
EB_LANE_ROUTE = {"3": "EB_route", "4": "EB_route_alt"}
# Share of un-pinned flow that takes the ALT exit. Westbound both exits are
# green together at 719, so 50/50. Eastbound the alt exit (-21_4 -> -3) is a
# LEFT turn at 532 that is red during the westbound entry phase, while the
# main exit (-21_3 -> 2) is a right turn that is permitted throughout; a 50%
# share queued the inner lane back over the stretch, 25% fits in edge -21.
WB_ALT_SHARE = 0.5
EB_ALT_SHARE = 0.25

# Perimeter gateways for ambient vehicles and bikes. All eight are outer-ring
# edges at least 74 m long (so insertion always has room, including for buses)
# and at least ~100 m from the stretch centre. The old list included -2 (11 m)
# and -8 (19 m), which are too short to insert anything large.
AMBIENT_GATEWAYS = ["-1", "1", "5", "-5", "6", "-6", "10", "-10"]

# Sidewalk walks along the monitored boulevard. Each entry is
# (id_suffix, edge list). Pedestrians walk these end to end; SUMO routes them
# over the walking areas built by build_network.py.
#
# The lists have to be connected. Writing the same one-way edge ids backwards
# is not a sidewalk in the other direction: SUMO fills each gap with a path
# through the junction, and that path waited out the red at 719 (the corridor
# green is about 90 s). Corridors that crossed the boulevard are gone for the
# same reason. These walks start on the stretch (edges 20 and -20) and continue
# along that sidewalk until the next traffic light, without taking the turn
# through it. The two directions are the two sidewalks; each is named twice so
# the pedestrian budget stays on them. People still pass through a walking area
# where the sidewalk changes edge. They are not parked there for the car green.
STRETCH_PED_CORRIDORS = [
    ("n_wb", "20 19 18"),
    ("n_eb", "-20 -21"),
    ("s_eb", "-20 -21"),
    ("s_wb", "20 19 18"),
]

# Monitored-stretch midpoint in CARLA world coordinates -- centre of the CARLA
# render radius (SUMO 108.5,121.5 -> CARLA via x-off 109.34 and the y flip).
STRETCH_CENTER_CARLA = (-0.85, 14.49)
DEFAULT_RENDER_RADIUS = 120.0

# Density scaling at slider value 100 (totals across the whole map).
AMBIENT_VEH_MAX_VPH   = 500.0   # total ambient passenger veh/h
AMBIENT_BIKE_MAX_VPH  = 120.0   # total ambient bicycle veh/h
AMBIENT_PED_MAX_PERHOUR = 400.0 # total pedestrians/h

# Share of the pedestrian budget spent on the monitored stretch corridors.
# Random city-to-city trips spent most of their time inside junctions, which
# is what pushed the crossing fraction to ~0.57. Stretch walks are the paths
# the sensors can see.
PED_STRETCH_SHARE = 1.0


def apply_map_edge_routes(content: str) -> str:
    """Rewrite the WB_route / EB_route edge lists so every scenario vehicle
    spawns at a map-edge corner and despawns at the opposite one, passing
    through the monitored stretch in between.

    All 11 scenario files share these two route ids, and every flow/vehicle in
    them references the ids rather than the edges, so this is the only place
    routing needs to change. Idempotent and safe if the ids are absent.
    """
    import re
    content = re.sub(
        r'(<route\s+id="WB_route"\s+edges=")[^"]*(")',
        lambda m: m.group(1) + WB_ROUTE_EDGES + m.group(2), content,
    )
    content = re.sub(
        r'(<route\s+id="EB_route"\s+edges=")[^"]*(")',
        lambda m: m.group(1) + EB_ROUTE_EDGES + m.group(2), content,
    )
    return _apply_two_exit_routing(content)


def _apply_two_exit_routing(content: str) -> str:
    """Add the alternative-exit routes + 50/50 distributions and point flows at
    the distributions; explicit vehicles get the route matching their lane.
    See WB_ROUTE_ALT_EDGES. Idempotent."""
    import re
    if "WB_route_alt" in content:
        return content
    for base, alt_id, alt_edges, mix_id, alt_share in (
        ("WB_route", "WB_route_alt", WB_ROUTE_ALT_EDGES, "WB_mix", WB_ALT_SHARE),
        ("EB_route", "EB_route_alt", EB_ROUTE_ALT_EDGES, "EB_mix", EB_ALT_SHARE),
    ):
        pat = re.compile(r'(<route\s+id="%s"\s+edges="[^"]*"\s*/>)' % base)
        if not pat.search(content):
            continue
        extra = (
            '\n    <route id="%s" edges="%s"/>'
            '\n    <routeDistribution id="%s">'
            '\n        <route refId="%s" probability="%.2f"/>'
            '\n        <route refId="%s" probability="%.2f"/>'
            '\n    </routeDistribution>' % (alt_id, alt_edges, mix_id, base, 1.0 - alt_share,
                                             alt_id, alt_share)
        )
        content = pat.sub(lambda m: m.group(1) + extra, content, count=1)
        lane_map = WB_LANE_ROUTE if base == "WB_route" else EB_LANE_ROUTE

        def _route_for(el):
            """Route id consistent with the element's departLane: an explicit
            lane, "first" (curb, lane 3) or "last" (inner, lane 4) pins the
            exit; random/best/free sample the 50/50 distribution."""
            lane = re.search(r'departLane="([^"]*)"', el)
            if not lane:
                return mix_id
            v = lane.group(1)
            if v == "first":
                v = "3"
            elif v == "last":
                v = "4"
            return lane_map.get(v, mix_id)

        def _fix(m):
            el = m.group(0)
            return re.sub(r'\broute="%s"' % base, 'route="%s"' % _route_for(el), el)
        content = re.sub(r'<(?:flow|vehicle)\b[^>]*?\broute="%s"[^>]*>' % base,
                         _fix, content, flags=re.S)
    return content


# Kept so older call sites and notes keep working.
extend_scenario_routes = apply_map_edge_routes


def _edge_pools(net_file: str):
    """Return (drivable_edges, pedestrian_edges), excluding internal (:) edges.

    The monitored edges are excluded from the DRIVABLE pool so ambient vehicles
    never spawn or despawn on the stretch. They are deliberately KEPT in the
    pedestrian pool -- pedestrians on the stretch are wanted, and a person
    appearing on a sidewalk is not the visual artefact that a car materialising
    in a live traffic lane is.
    """
    if sumolib is None:
        raise ImportError("sumolib is required for ambient traffic "
                          "(activate venv, then: pip install sumolib)")
    net = sumolib.net.readNet(net_file)
    drivable, ped = [], []
    for e in net.getEdges():
        eid = e.getID()
        if eid.startswith(":"):
            continue
        if e.allows("passenger") and eid not in MONITORED_EDGES:
            drivable.append(eid)
        if e.allows("pedestrian"):
            ped.append(eid)
    return drivable, ped


def _flow(fid, vtype, frm, to, vph, duration):
    # departSpeed="0" everywhere, per the project rule -- gateway edges are all
    # 74 m or longer so there is plenty of room to accelerate.
    return (f'    <flow id="{fid}" type="{vtype}" begin="0" end="{duration}" '
            f'vehsPerHour="{vph:.1f}" from="{frm}" to="{to}" '
            f'departLane="free" departSpeed="0"/>')


def generate_ambient(net_file: str, seed: int, veh_level: int,
                     ped_level: int, bike_level: int, duration: int) -> str:
    """Build the ambient XML fragment (vTypes + flows + personFlows) to inject
    into <routes>. Levels are 0-100 sliders; 0 disables that class.

    Reproducibility: this function's gateway pairing is seeded here, and
    runner.py additionally hands the same seed to SUMO itself, so insertion
    jitter is reproducible too.
    """
    rng = random.Random(seed)
    drivable, ped_edges = _edge_pools(net_file)
    gateways = [g for g in AMBIENT_GATEWAYS if g in set(drivable)]
    lines = []

    # Bicycle vType (the base additional file has none). Slow and narrow so
    # cars can overtake instead of stacking up behind it. Bikes ride the
    # driving lanes, which permit the bicycle vClass already.
    lines.append(
        '    <vType id="amb_bike" vClass="bicycle" color="0.9,0.7,0.1" '
        'maxSpeed="6.0" length="1.7" width="0.65" minGap="0.5" sigma="0.5"/>'
    )

    def rand_pair(pool):
        a = rng.choice(pool)
        b = rng.choice(pool)
        tries = 0
        while b == a and tries < 10:
            b = rng.choice(pool)
            tries += 1
        return a, b

    if veh_level > 0 and len(gateways) >= 2:
        total = AMBIENT_VEH_MAX_VPH * veh_level / 100.0
        n = min(6, len(gateways))
        per = total / n
        for i in range(n):
            a, b = rand_pair(gateways)
            lines.append(_flow(f"amb_veh_{i}", "car", a, b, per, duration))

    if bike_level > 0 and len(gateways) >= 2:
        total = AMBIENT_BIKE_MAX_VPH * bike_level / 100.0
        n = min(4, len(gateways))
        per = total / n
        for i in range(n):
            a, b = rand_pair(gateways)
            lines.append(_flow(f"amb_bike_{i}", "amb_bike", a, b, per, duration))

    if ped_level > 0 and len(ped_edges) >= 2:
        total_ph = AMBIENT_PED_MAX_PERHOUR * ped_level / 100.0

        # Pedestrians ON the monitored stretch: fixed sidewalk corridors, so
        # they are guaranteed to walk past the sensors rather than left to the
        # router's discretion.
        stretch_ph = total_ph * PED_STRETCH_SHARE / len(STRETCH_PED_CORRIDORS)
        for suffix, edges in STRETCH_PED_CORRIDORS:
            lines.append(
                f'    <personFlow id="ped_stretch_{suffix}" begin="0" '
                f'end="{duration}" perHour="{stretch_ph:.1f}" departPos="0">\n'
                f'        <walk edges="{edges}"/>\n'
                f'    </personFlow>'
            )

        # Pedestrians elsewhere in the city: random sidewalk-to-sidewalk trips.
        city_ph = total_ph * (1.0 - PED_STRETCH_SHARE)
        if city_ph >= 1.0:
            n_city = 6
            for i in range(n_city):
                a, b = rand_pair(ped_edges)
                lines.append(
                    f'    <personFlow id="ped_city_{i}" begin="0" end="{duration}" '
                    f'perHour="{city_ph / n_city:.1f}">\n'
                    f'        <personTrip from="{a}" to="{b}"/>\n'
                    f'    </personFlow>'
                )

    return "\n".join(lines)


def inject_ambient(route_content: str, ambient_xml: str) -> str:
    """Insert the ambient fragment just before the closing </routes>."""
    if not ambient_xml.strip():
        return route_content
    idx = route_content.rfind("</routes>")
    if idx == -1:
        return route_content
    return route_content[:idx] + ambient_xml + "\n" + route_content[idx:]
