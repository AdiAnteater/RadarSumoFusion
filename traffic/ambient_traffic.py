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
Four sidewalk corridors, one per sidewalk and direction, each walking the FULL
road (edges 18-21 / -18..-21) end to end. Every corridor starts and ends on a
perimeter sidewalk outside the radar zone, so nobody spawns or despawns inside
it. Verified headless (SUMO 1.28, rebuilt net): 60/60 walkers inserted, every
walker traverses all four boulevard edges, nearest spawn/despawn > 15 m outside
the zone.


SPAWN EXCLUSION ZONE
--------------------
The radar rig spans the whole boulevard: junction 719 stop line (CARLA x -88.6)
to junction 532 stop line (x 83.1), between the radar rows (y 0.2 .. 41.2).
No vehicle or pedestrian may spawn or despawn inside that box grown by
SPAWN_EXCLUSION_MARGIN_M. check_spawn_exclusion() verifies a finished route
file against it. Mirrors dataset/capture/radar_layout.py -- keep in sync.
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

# Sidewalk walks along the monitored boulevard. Pedestrians walk these end to
# end; SUMO routes them over the walking areas built by build_network.py.
#
# The lists have to be connected. Writing the same one-way edge ids backwards
# is not a sidewalk in the other direction: SUMO fills each gap with a path
# through the junction, and a path that crosses the road waits out the red at
# 719 (the corridor green is about 90 s). So each corridor stays on ONE
# boulevard sidewalk and joins the perimeter on the same side through a
# walking area only (no boulevard crossing at 719 or 532). The side-street
# crossings at 189 and 841 are on the way, as before.
# (id_suffix, walk edges, departPos on first edge, arrivalPos on last edge).
# Each corridor walks one boulevard sidewalk end to end. The first and last
# edges are inner-ring perimeter sidewalks that meet that boulevard sidewalk at
# junction 719 / 532 without crossing the boulevard:
#   north sidewalk (WB edges 18-21): 10 0 3 (NE)  <->  -6 (NW)
#   south sidewalk (EB edges -18..-21): 2 1 (SE)  <->  -5 (SW)
# departPos / arrivalPos put every spawn and despawn well outside the zone:
#   10 @75 -> CARLA (82.5, -43.8)    -6 @35 -> (-92.0, -36.6)
#   1  @35 -> (82.9, 115.0)          -5 @45 -> (-95.4, 88.9)
STRETCH_PED_CORRIDORS = [
    ("n_wb", "10 0 3 21 20 19 18 -6", 75.0, 35.0),
    ("n_eb", "-6 18 19 20 21 3 0 10", 35.0, 75.0),
    ("s_eb", "-5 -18 -19 -20 -21 2 1", 45.0, 35.0),
    ("s_wb", "1 2 -21 -20 -19 -18 -5", 35.0, 45.0),
]

# --- Spawn exclusion zone (CARLA world coords) -- see docstring ---
# Full radar-monitored road. Mirrors dataset/capture/radar_layout.py.
RADAR_ZONE_X_WEST = -88.6     # west end of edge 18 (junction 719 stop line)
RADAR_ZONE_X_EAST = 83.1      # east end of edge 21 (junction 532 stop line)
RADAR_ZONE_Y_SOUTH = 0.2      # radar row (rig anchor y 20.7 - half width 20.5)
RADAR_ZONE_Y_NORTH = 41.2     # radar row (rig anchor y 20.7 + half width 20.5)
SPAWN_EXCLUSION_MARGIN_M = 10.0
# Boulevard edges fully inside the zone. Never used as an ambient endpoint.
RADAR_ZONE_EDGES = {"18", "19", "20", "21", "-18", "-19", "-20", "-21"}

SUMO_NET_OFFSET = (109.34, 135.96)


def sumo_to_carla_xy(x, y):
    """Verified runtime transform (do not re-derive)."""
    return x - SUMO_NET_OFFSET[0], -(y - SUMO_NET_OFFSET[1])


def in_radar_zone_carla(x, y, margin=SPAWN_EXCLUSION_MARGIN_M):
    return (RADAR_ZONE_X_WEST - margin <= x <= RADAR_ZONE_X_EAST + margin
            and RADAR_ZONE_Y_SOUTH - margin <= y <= RADAR_ZONE_Y_NORTH + margin)


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
        # City pedestrians spawn and despawn anywhere on their edge, so any
        # edge touching the radar zone is out of the pool.
        if e.allows("pedestrian") and not _edge_touches_zone(e):
            ped.append(eid)
    return drivable, ped


def _edge_touches_zone(edge):
    if edge.getID() in RADAR_ZONE_EDGES:
        return True
    return any(in_radar_zone_carla(*sumo_to_carla_xy(x, y))
               for x, y in edge.getShape())


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
        for suffix, edges, dpos, apos in STRETCH_PED_CORRIDORS:
            lines.append(
                f'    <personFlow id="ped_stretch_{suffix}" begin="0" '
                f'end="{duration}" perHour="{stretch_ph:.1f}" departPos="{dpos:.1f}">\n'
                f'        <walk edges="{edges}" arrivalPos="{apos:.1f}"/>\n'
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


def check_spawn_exclusion(net_file: str, route_content: str,
                          margin: float = SPAWN_EXCLUSION_MARGIN_M):
    """Return a list of human-readable violations: every vehicle, flow, person
    or personFlow in ``route_content`` whose spawn or despawn point falls inside
    the radar zone. Empty list = clean.

    Spawn = first edge at departPos, despawn = last edge at arrivalPos (default
    0 / end). A non-numeric departPos/arrivalPos (random, free, ...) can land
    anywhere on the edge, so the whole edge is checked.
    """
    import xml.etree.ElementTree as ET
    if sumolib is None:
        raise ImportError("sumolib is required for check_spawn_exclusion")
    net = sumolib.net.readNet(net_file)
    root = ET.fromstring(route_content)

    def edge_points(eid, pos, at_end):
        try:
            e = net.getEdge(eid)
        except KeyError:
            return []
        shape, length = e.getShape(), e.getLength()
        if pos is None:
            off = length if at_end else 0.0
        else:
            try:
                off = float(pos)
                if off < 0:
                    off += length
            except ValueError:
                return [sumolib.geomhelper.positionAtShapeOffset(shape, length * k / 10.0)
                        for k in range(11)]
        off = max(0.0, min(off, length))
        return [sumolib.geomhelper.positionAtShapeOffset(shape, off)]

    def bad(eid, pos, at_end):
        for x, y in edge_points(eid, pos, at_end):
            cx, cy = sumo_to_carla_xy(x, y)
            if in_radar_zone_carla(cx, cy, margin):
                return (round(cx, 1), round(cy, 1))
        return None

    routes, dists = {}, {}
    for r in root.iter("route"):
        if r.get("id") and r.get("edges"):
            routes[r.get("id")] = r.get("edges").split()
    for d in root.iter("routeDistribution"):
        dists[d.get("id")] = [x.get("refId") for x in d.findall("route")]

    out = []
    for el in list(root):
        tag, ident = el.tag, el.get("id")
        if tag in ("vehicle", "flow", "trip"):
            cands = []
            if el.get("route"):
                rid = el.get("route")
                cands = [routes.get(x, []) for x in dists.get(rid, [rid])]
            elif el.find("route") is not None:
                cands = [el.find("route").get("edges", "").split()]
            elif el.get("from"):
                cands = [[el.get("from"), el.get("to") or el.get("from")]]
            for edges in cands:
                if not edges:
                    continue
                p = bad(edges[0], el.get("departPos"), False)
                if p:
                    out.append(f"{tag} {ident}: spawns at CARLA {p} on edge {edges[0]}")
                p = bad(edges[-1], el.get("arrivalPos"), True)
                if p:
                    out.append(f"{tag} {ident}: despawns at CARLA {p} on edge {edges[-1]}")
        elif tag in ("person", "personFlow"):
            stages = [c for c in el if c.tag in ("walk", "personTrip", "ride")]
            if not stages:
                continue
            first, last = stages[0], stages[-1]
            f_edge = (first.get("edges") or first.get("from") or "").split()[:1]
            l_edge = (last.get("edges") or last.get("to") or "").split()[-1:]
            if f_edge:
                p = bad(f_edge[0], el.get("departPos"), False)
                if p:
                    out.append(f"{tag} {ident}: spawns at CARLA {p} on edge {f_edge[0]}")
            if l_edge:
                p = bad(l_edge[0], last.get("arrivalPos"), True)
                if p:
                    out.append(f"{tag} {ident}: despawns at CARLA {p} on edge {l_edge[0]}")
    return out
