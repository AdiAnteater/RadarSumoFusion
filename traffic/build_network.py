"""
build_network.py
================
ONE-TIME network rebuild. Replaces build_ped_network.py (delete that file).

Run once from an activated venv:

    python build_network.py

Output: network/Town10HD_Opt_fixed.net.xml
Every run (runner.py / gui_launcher.py) uses this net from now on -- it is NOT
optional and NOT pedestrian-only.


WHY THIS EXISTS
---------------
Town10HD_Opt.net.xml came out of netconvert with a defect. Each edge carries a
lane type string such as

    sidewalk|shoulder|shoulder|driving|driving|shoulder|median|shoulder

so lane index 0 is the SIDEWALK. netconvert gave every shoulder and median lane
disallow="all", but it left all 46 sidewalk lanes with NO permission attribute
at all, and with full road speed (13.89 m/s) and width 6.00 m. With no
restriction, "allow" defaults to every vClass, so SUMO treats the sidewalk as an
ordinary traffic lane.

Two consequences, both observed:

1. CARS DRIVE ON THE SIDEWALK AND GET STUCK.
   departLane="random" chooses uniformly among the lanes the vType is allowed
   on -- {0, 3, 4} -- so roughly a third of traffic starts on the sidewalk.
   departLane="first" chooses the LOWEST allowed index, which is 0, so every
   bus, truck and slow-lane flow started there too. 20 of the 46 sidewalk lanes
   (including 20, 18, -19, -21) have no outgoing connection at any real
   junction, so those vehicles reach the junction, find no connection, get no
   traffic-light link, and stop dead. That is the "stuck on 18_0/19_0/21_0 with
   no stop light" symptom -- and 20_0 sits ~8 m off the carriageway (CARLA
   y=5.4 versus 13.3/16.8 for the real lanes 20_3/20_4), which is why it looked
   like the same road from a distance.

2. PEDESTRIANS COULD NOT EXIST.
   netconvert's --crossings.guess only fires on lanes it RECOGNISES as
   sidewalks, i.e. lanes permitting pedestrian and not much else. Lanes that
   permit everything are not sidewalks to netconvert, so the old
   build_ped_network.py produced a net with zero crossings and zero
   walkingareas. Persons then had no walkable junction connectivity, every
   person route failed, and city mode's ignore-route-errors discarded them
   silently -- "no pedestrians visible at all".
   (The old script's "crossings: 0" check was also looking for <crossing> and
   <walkingarea> ELEMENTS, which do not exist in a .net.xml. They are written as
   <edge function="crossing"> and <edge function="walkingarea">. This script
   counts them correctly.)


WHAT THIS SCRIPT DOES
---------------------
Step 1  Read network/Town10HD_Opt.net.xml, and for every non-internal edge set
        the lanes whose position in the edge's type string is "sidewalk" to
        allow="pedestrian" (dropping any disallow) and speed 2.78 m/s. Nothing
        else is touched.
Step 2  Re-run netconvert on the patched net with --crossings.guess and
        --walkingareas so pedestrians get real junction connectivity.
Step 3  Verify the result and refuse to write a net that would break the
        coordinate transform.

Verified on netconvert 1.27 against this exact net:
  - all 46 normal edge ids preserved (20, -20, 21, 20_3, ... unchanged)
  - netOffset 109.34,135.96 preserved  -> the CARLA transform is untouched
  - convBoundary 0.00,0.00,214.07,199.44 preserved
  - lane LATERAL positions identical (20_3 stays at sumo y=122.71)
  - 37 crossings and 46 walkingareas created
  - traffic lights keep their vehicle link states verbatim; pedestrian states
    are appended. Junction 189 keeps its 90 s cycle exactly. Junction 719 gains
    an exclusive pedestrian phase (90 s -> 100 s cycle).
  - edge 20 length 52.77 m -> 55.13 m. netconvert reclaims a little junction
    area when it builds walkingareas. The road does not move; only the
    edge/junction boundary does. The bus stop at 20_3 startPos 20 endPos 35 is
    still comfortably inside. Report the stretch as ~55 m in the methods
    section.

CROSSINGS ONLY ON CARLA ZEBRAS (Nov 2026)
-----------------------------------------
--crossings.guess put a crossing on every junction arm: 37 of them, but CARLA's
Town10 has only 16 zebra crossings. The extra 21 had pedestrians walking across
open road with no zebra (four of them straight over the boulevard at 189 and
841), and cars stopped mid-stretch to yield to them. The 16 that did match a
zebra sat 2-5 m outward of it and 4-7 m short, so a car leaving a junction
waited for the walker right on top of the rendered zebra.

The build now runs in two netconvert passes:
  2a  --crossings.guess, only to learn which road edges each junction arm has.
  2b  --crossings.guess false plus an explicit crossing file: one crossing per
      CARLA zebra, on the arm whose guessed crossing is nearest (centroid within
      ZEBRA_MATCH_MAX_M, axis within ZEBRA_MATCH_MAX_DEG), with the crossing's
      shape set to the zebra centreline and its width to the zebra width.
Zebras are read straight from network/Town10HD_Opt.xodr (object
type="crosswalk"; the plan view is lines and arcs only), in SUMO coords via
sumo = (odr_x + 109.34, odr_y + 135.96). Edges, lanes and the netOffset are
unchanged; only crossings, walkingareas and pedestrian signal states move.

NOTE ON ARTEFACTS: the CARLA server renders its own traffic lights on its own
cycle. SUMO's TLS controls the SUMO vehicles that get mirrored in. The two were
never phase-locked, and this rebuild does not change that. If red-light
compliance matters for the labelled data, that is a separate job.
"""

import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE     = os.path.dirname(os.path.abspath(__file__))
NET_DIR  = os.path.join(HERE, "network")
IN_NET   = os.path.join(NET_DIR, "Town10HD_Opt.net.xml")
PATCHED  = os.path.join(NET_DIR, "_Town10HD_Opt_sidewalks.net.xml")   # intermediate
OUT_NET  = os.path.join(NET_DIR, "Town10HD_Opt_fixed.net.xml")
XODR     = os.path.join(NET_DIR, "Town10HD_Opt.xodr")
GUESSED  = os.path.join(NET_DIR, "_Town10HD_Opt_guessed.net.xml")    # intermediate
CROSS_XML = os.path.join(NET_DIR, "_zebra_crossings.con.xml")        # intermediate

# SUMO = OpenDRIVE + netOffset (CARLA = OpenDRIVE with y flipped).
NET_OFFSET = (109.34, 135.96)
# A guessed crossing belongs to a zebra when their centres are this close and
# their axes this parallel. Measured: true pairs are 1.9-5.3 m / <1 deg apart,
# the nearest wrong pair is 7.8 m.
ZEBRA_MATCH_MAX_M = 6.0
ZEBRA_MATCH_MAX_DEG = 10.0

EXPECTED_OFFSET = "109.34,135.96"

# The lane type token, taken from the edge's own type string, that identifies a
# sidewalk. Do not guess by index -- edges on this net have either 5 or 8 lanes.
SIDEWALK_TOKEN = "sidewalk"

# SUMO's usual sidewalk speed. Pedestrians move at their vType maxSpeed, so this
# is cosmetic, but it stops the lane from reading as a 50 km/h road.
SIDEWALK_SPEED = "2.78"


def find_netconvert():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sumo_env
    return sumo_env.sumo_binary("netconvert")


def patch_sidewalks(in_path, out_path):
    """Set every sidewalk lane to allow="pedestrian". Returns the count."""
    tree = ET.parse(in_path)
    root = tree.getroot()

    patched = 0
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        types = (edge.get("type") or "").split("|")
        for lane in edge.findall("lane"):
            idx = int(lane.get("index"))
            token = types[idx] if idx < len(types) else ""
            if token != SIDEWALK_TOKEN:
                continue
            lane.set("allow", "pedestrian")
            lane.attrib.pop("disallow", None)
            lane.set("speed", SIDEWALK_SPEED)
            patched += 1

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return patched


def _odr_ref_point(road, s):
    """(x, y, hdg) of a road's reference line at s. Town10 uses only line and
    arc geometries."""
    geoms = road.find("planView").findall("geometry")
    g = [g for g in geoms if float(g.get("s")) <= s + 1e-9][-1]
    ds = s - float(g.get("s"))
    x, y, h = float(g.get("x")), float(g.get("y")), float(g.get("hdg"))
    kind = g[0]
    k = float(kind.get("curvature", 0.0)) if kind.tag == "arc" else 0.0
    if kind.tag not in ("line", "arc"):
        raise ValueError(f"unsupported OpenDRIVE geometry {kind.tag}")
    if abs(k) < 1e-12:
        return x + ds * math.cos(h), y + ds * math.sin(h), h
    h2 = h + k * ds
    return (x + (math.sin(h2) - math.sin(h)) / k,
            y - (math.cos(h2) - math.cos(h)) / k, h2)


def load_zebras(xodr_path):
    """CARLA zebra crossings as dicts in SUMO coords:
    {id, a, b (centreline end points), width}."""
    root = ET.parse(xodr_path).getroot()
    out = []
    for road in root.findall("road"):
        for o in road.iter("object"):
            if o.get("type") != "crosswalk":
                continue
            s, t, hdg = float(o.get("s")), float(o.get("t")), float(o.get("hdg"))
            rx, ry, rh = _odr_ref_point(road, s)
            ox, oy = rx - t * math.sin(rh), ry + t * math.cos(rh)
            ah = rh + hdg
            pts = []
            for cl in o.find("outline").findall("cornerLocal"):
                u, v = float(cl.get("u")), float(cl.get("v"))
                pts.append((ox + u * math.cos(ah) - v * math.sin(ah) + NET_OFFSET[0],
                            oy + u * math.sin(ah) + v * math.cos(ah) + NET_OFFSET[1]))
            if len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-6:
                pts = pts[:-1]
            if len(pts) != 4:
                continue
            sides = sorted(((pts[i], pts[(i + 1) % 4]) for i in range(4)),
                           key=lambda ab: math.dist(*ab))
            mid = lambda ab: ((ab[0][0] + ab[1][0]) / 2.0, (ab[0][1] + ab[1][1]) / 2.0)
            short1, short2 = sides[0], sides[1]
            width = (math.dist(*short1) + math.dist(*short2)) / 2.0
            out.append({"id": o.get("id"), "a": mid(short1), "b": mid(short2),
                        "width": width})
    return out


def _guessed_crossings(net_path):
    """[{id, node, edges, shape}] for every crossing in a guessed net."""
    out = []
    for e in ET.parse(net_path).getroot().findall("edge"):
        if e.get("function") != "crossing":
            continue
        lane = e.find("lane")
        shape = [tuple(map(float, p.split(","))) for p in lane.get("shape").split()]
        out.append({"id": e.get("id"), "node": e.get("id")[1:].rsplit("_c", 1)[0],
                    "edges": e.get("crossingEdges"), "shape": shape})
    return out


def match_zebras(zebras, guessed):
    """One guessed crossing (arm) per zebra. Returns [(zebra, crossing)]."""
    def centre(pts):
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def axis_deg(p, q):
        return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))

    pairs = []
    for z in zebras:
        zc = centre([z["a"], z["b"]])
        best = None
        for c in guessed:
            d = math.dist(zc, centre(c["shape"]))
            diff = abs((axis_deg(z["a"], z["b"]) - axis_deg(c["shape"][0], c["shape"][-1])
                        + 90.0) % 180.0 - 90.0)
            if d <= ZEBRA_MATCH_MAX_M and diff <= ZEBRA_MATCH_MAX_DEG:
                if best is None or d < best[0]:
                    best = (d, c)
        if best is not None:
            pairs.append((z, best[1]))
    return pairs


def write_crossing_file(pairs, out_path):
    """Explicit <crossing> elements: zebra centreline, oriented like the guessed
    crossing so netconvert attaches the same walking areas to each end."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<connections>"]
    for z, c in pairs:
        a, b = z["a"], z["b"]
        if math.dist(a, c["shape"][0]) > math.dist(b, c["shape"][0]):
            a, b = b, a
        shape = f"{a[0]:.2f},{a[1]:.2f} {b[0]:.2f},{b[1]:.2f}"
        lines.append(f'    <crossing node="{c["node"]}" edges="{c["edges"]}" '
                     f'width="{z["width"]:.2f}" shape="{shape}"/>'
                     f'  <!-- zebra {z["id"]} (was {c["id"]}) -->')
    lines.append("</connections>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_netconvert(cmd):
    print("          " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"ERROR: netconvert failed (code {result.returncode}). "
                 f"Is SUMO_HOME set with netconvert in its bin/?")


def verify(out_path, in_path):
    """Refuse to accept a net that would break routing or the transform."""
    src = ET.parse(in_path).getroot()
    dst = ET.parse(out_path).getroot()

    problems = []

    loc = dst.find("location")
    offset = loc.get("netOffset") if loc is not None else None
    if offset != EXPECTED_OFFSET:
        problems.append(f"netOffset is {offset}, expected {EXPECTED_OFFSET} -- "
                        f"the CARLA transform would be wrong")

    src_ids = {e.get("id") for e in src.findall("edge") if e.get("function") != "internal"}
    dst_ids = {e.get("id") for e in dst.findall("edge") if not e.get("function")}
    missing = src_ids - dst_ids
    if missing:
        problems.append(f"{len(missing)} normal edge ids vanished: {sorted(missing)[:8]}")

    crossings = sum(1 for e in dst.findall("edge") if e.get("function") == "crossing")
    walkareas = sum(1 for e in dst.findall("edge") if e.get("function") == "walkingarea")
    if crossings == 0:
        problems.append("no crossings were built -- pedestrians will not be able "
                        "to cross roads")
    if walkareas == 0:
        problems.append("no walkingareas were built -- pedestrians will not be "
                        "able to pass junctions")

    # The whole point: no lane may be both a sidewalk and drivable.
    leaks = []
    for edge in dst.findall("edge"):
        if edge.get("function"):
            continue
        types = (edge.get("type") or "").split("|")
        for lane in edge.findall("lane"):
            idx = int(lane.get("index"))
            token = types[idx] if idx < len(types) else ""
            if token != SIDEWALK_TOKEN:
                continue
            if lane.get("allow") != "pedestrian":
                leaks.append(lane.get("id"))
    if leaks:
        problems.append(f"{len(leaks)} sidewalk lanes still drivable: {leaks[:8]}")

    return problems, crossings, walkareas, len(dst_ids)


def main():
    if not os.path.isfile(IN_NET):
        sys.exit(f"ERROR: input net not found: {IN_NET}")

    print("Step 1/3  Restricting sidewalk lanes to pedestrians ...")
    n = patch_sidewalks(IN_NET, PATCHED)
    print(f"          patched {n} sidewalk lanes -> allow=\"pedestrian\"")
    if n == 0:
        sys.exit("ERROR: no sidewalk lanes found. Is this the right net file? "
                 "Expected edge type strings like "
                 "'sidewalk|shoulder|shoulder|driving|driving'.")

    print("\nStep 2/3  Running netconvert to build crossings and walkingareas ...")
    netconvert = find_netconvert()
    common = [
        "--walkingareas", "true",
        # Do NOT add --geometry.remove: it collapses the intermediate nodes this
        # net depends on. Keep unregulated nodes explicit for the same reason.
        "--keep-nodes-unregulated.explicit", "true",
    ]
    print("  2a  guess crossings (only to learn which edges each arm has)")
    run_netconvert([
        netconvert, "--sumo-net-file", PATCHED,
        "--crossings.guess", "true",
        # Roads here are exactly 13.89 m/s; the default threshold is also 13.89
        # and the comparison is strict, so nudge it up or no crossing is guessed.
        "--crossings.guess.speed-threshold", "13.9",
        "--default.crossing-width", "3.0",
    ] + common + ["-o", GUESSED])
    if not os.path.isfile(XODR):
        sys.exit(f"ERROR: OpenDRIVE file not found: {XODR}")
    zebras = load_zebras(XODR)
    guessed = _guessed_crossings(GUESSED)
    pairs = match_zebras(zebras, guessed)
    print(f"          CARLA zebras: {len(zebras)}, guessed crossings: {len(guessed)}, "
          f"kept (on a zebra): {len(pairs)}")
    unmatched = sorted({z["id"] for z in zebras} - {z["id"] for z, _ in pairs})
    if unmatched:
        print(f"          WARNING: zebras with no junction arm: {unmatched}")
    write_crossing_file(pairs, CROSS_XML)
    print("  2b  build with crossings on the CARLA zebras only")
    run_netconvert([
        netconvert, "--sumo-net-file", PATCHED,
        "--crossings.guess", "false",
        "-x", CROSS_XML,
    ] + common + ["-o", OUT_NET])

    print("\nStep 3/3  Verifying ...")
    problems, crossings, walkareas, n_edges = verify(OUT_NET, IN_NET)
    print(f"          normal edges  : {n_edges} (expect 46, ids unchanged)")
    print(f"          netOffset     : {EXPECTED_OFFSET} preserved")
    print(f"          crossings     : {crossings}")
    print(f"          walkingareas  : {walkareas}")
    print(f"          sidewalk lanes: pedestrian-only, cars locked out")

    for tmp in (PATCHED, GUESSED, CROSS_XML):
        try:
            os.remove(tmp)
        except OSError:
            pass

    if problems:
        print("\nFAILED -- the rebuilt net is not safe to use:")
        for p in problems:
            print("  * " + p)
        sys.exit(1)

    print(f"\nWrote {OUT_NET}")
    print("Done. runner.py and gui_launcher.py will pick this up automatically.")
    print("You can delete build_ped_network.py and Town10HD_Opt_ped.net.xml.")


if __name__ == "__main__":
    main()
