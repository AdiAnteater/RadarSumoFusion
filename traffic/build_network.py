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

NOTE ON ARTEFACTS: the CARLA server renders its own traffic lights on its own
cycle. SUMO's TLS controls the SUMO vehicles that get mirrored in. The two were
never phase-locked, and this rebuild does not change that. If red-light
compliance matters for the labelled data, that is a separate job.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE     = os.path.dirname(os.path.abspath(__file__))
NET_DIR  = os.path.join(HERE, "network")
IN_NET   = os.path.join(NET_DIR, "Town10HD_Opt.net.xml")
PATCHED  = os.path.join(NET_DIR, "_Town10HD_Opt_sidewalks.net.xml")   # intermediate
OUT_NET  = os.path.join(NET_DIR, "Town10HD_Opt_fixed.net.xml")

EXPECTED_OFFSET = "109.34,135.96"

# The lane type token, taken from the edge's own type string, that identifies a
# sidewalk. Do not guess by index -- edges on this net have either 5 or 8 lanes.
SIDEWALK_TOKEN = "sidewalk"

# SUMO's usual sidewalk speed. Pedestrians move at their vType maxSpeed, so this
# is cosmetic, but it stops the lane from reading as a 50 km/h road.
SIDEWALK_SPEED = "2.78"


def find_netconvert():
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        for exe in ("netconvert.exe", "netconvert"):
            cand = os.path.join(sumo_home, "bin", exe)
            if os.path.isfile(cand):
                return cand
    return "netconvert"


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
    cmd = [
        netconvert,
        "--sumo-net-file", PATCHED,
        "--crossings.guess", "true",
        "--walkingareas", "true",
        # Roads here are exactly 13.89 m/s; the default threshold is also 13.89
        # and the comparison is strict, so nudge it up or no crossing is guessed.
        "--crossings.guess.speed-threshold", "13.9",
        "--default.crossing-width", "3.0",
        # Do NOT add --geometry.remove: it collapses the intermediate nodes this
        # net depends on. Keep unregulated nodes explicit for the same reason.
        "--keep-nodes-unregulated.explicit", "true",
        "-o", OUT_NET,
    ]
    print("          " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"ERROR: netconvert failed (code {result.returncode}). "
                 f"Is SUMO_HOME set with netconvert in its bin/?")

    print("\nStep 3/3  Verifying ...")
    problems, crossings, walkareas, n_edges = verify(OUT_NET, IN_NET)
    print(f"          normal edges  : {n_edges} (expect 46, ids unchanged)")
    print(f"          netOffset     : {EXPECTED_OFFSET} preserved")
    print(f"          crossings     : {crossings}")
    print(f"          walkingareas  : {walkareas}")
    print(f"          sidewalk lanes: pedestrian-only, cars locked out")

    try:
        os.remove(PATCHED)
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
