"""Detect car/truck -> pedestrian collisions in a capture's actor_frames.jsonl.

No collision sensor was logged, so we infer hits geometrically: per frame, test
every vehicle's oriented footprint (OBB) against every pedestrian. CARLA walkers
here don't ragdoll (pitch/roll stay 0), so the usable signal is footprint overlap.

We distinguish two regimes:
  - "run-through": the pedestrian's *center* falls strictly inside the vehicle
    rectangle. A pedestrian merely standing flush against a parked car keeps its
    center OUTSIDE the rectangle, so this is a strong drive-over signal.
  - "graze": OBBs touch (within ped radius) but the ped center is outside.

Consecutive frames for the same (vehicle, ped) pair are grouped into one event.
For each event we report duration, peak penetration, and how far the vehicle
moved while the ped center stayed inside it (a moving vehicle => true run-over).

Usage: python tools/detect_ped_collisions.py CAPTURE_DIR [--label NAME]
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("capture_dir", type=Path)
ap.add_argument("--label", default=None)
ap.add_argument("--gap", type=int, default=5,
                help="Max tick gap to still consider an event continuous (default 5).")
args = ap.parse_args()
LABEL = args.label or args.capture_dir.name
JSONL = args.capture_dir / "actor_frames.jsonl"


def ped_inside_vehicle(vx, vy, vyaw_deg, ex, ey, px, py, prad):
    """Return (touch, center_inside, penetration). Transform ped center into the
    vehicle's local frame and compare against half-extents."""
    yaw = math.radians(vyaw_deg)
    cs, sn = math.cos(yaw), math.sin(yaw)
    dx, dy = px - vx, py - vy
    # world -> vehicle local (rotate by -yaw)
    lx = dx * cs + dy * sn
    ly = -dx * sn + dy * cs
    alx, aly = abs(lx), abs(ly)
    touch = (alx <= ex + prad) and (aly <= ey + prad)
    center_inside = (alx <= ex) and (aly <= ey)
    # depth ped center has crossed inside the vehicle rectangle (>=0 only if inside)
    pen = min(ex - alx, ey - aly) if center_inside else 0.0
    return touch, center_inside, pen


# pass 1: collect raw per-frame hits ------------------------------------------
# active[(vid,pid)] = ongoing event dict; closed = finished events
active = {}
closed = []
veh_meta = {}          # vid -> type_id / class_label (last seen)
frames_seen = 0
last_frame = None

with JSONL.open() as f:
    for line in f:
        rec = json.loads(line)
        fr = int(rec["frame"])
        frames_seen += 1
        vehs, peds = [], []
        for a in rec["actors"]:
            loc = a["location"]; rot = a.get("rotation") or {}
            ext = (a.get("bbox") or {}).get("extent") or {}
            if a.get("kind") == "vehicle":
                vehs.append((a["id"], loc["x"], loc["y"], rot.get("yaw", 0.0),
                             float(ext.get("x", 2.0)), float(ext.get("y", 1.0))))
                veh_meta[a["id"]] = (a.get("class_label", "?"), a.get("type_id", "?"))
            elif a.get("kind") == "pedestrian":
                peds.append((a["id"], loc["x"], loc["y"],
                             float(ext.get("x", 0.2))))
        hits_this_frame = set()
        for vid, vx, vy, vyaw, ex, ey in vehs:
            vdiag = math.hypot(ex, ey) + 1.0
            for pid, px, py, prad in peds:
                if abs(px - vx) > vdiag + prad or abs(py - vy) > vdiag + prad:
                    continue  # coarse reject
                touch, inside, pen = ped_inside_vehicle(vx, vy, vyaw, ex, ey, px, py, prad)
                if not touch:
                    continue
                key = (vid, pid)
                hits_this_frame.add(key)
                ev = active.get(key)
                if ev is None or fr - ev["last_fr"] > args.gap:
                    if ev is not None:
                        closed.append(ev)
                    ev = active[key] = {
                        "vid": vid, "pid": pid,
                        "first_fr": fr, "last_fr": fr,
                        "n_frames": 0, "n_inside": 0,
                        "max_pen": 0.0,
                        "vstart": (vx, vy), "vend": (vx, vy),
                    }
                ev["last_fr"] = fr
                ev["n_frames"] += 1
                if inside:
                    ev["n_inside"] += 1
                ev["max_pen"] = max(ev["max_pen"], pen)
                ev["vend"] = (vx, vy)
        # close events whose pair didn't appear this frame and exceeded the gap
        for key in list(active.keys()):
            if key not in hits_this_frame and fr - active[key]["last_fr"] > args.gap:
                closed.append(active.pop(key))
        last_frame = fr

closed.extend(active.values())

# rank: real run-overs first (have center-inside frames), by penetration*span
def veh_moved(ev):
    (x0, y0), (x1, y1) = ev["vstart"], ev["vend"]
    return math.hypot(x1 - x0, y1 - y0)

run_overs = [e for e in closed if e["n_inside"] > 0]
grazes    = [e for e in closed if e["n_inside"] == 0]
run_overs.sort(key=lambda e: (e["max_pen"], e["n_inside"]), reverse=True)

print(f"\n===== {LABEL} =====")
print(f"frames scanned: {frames_seen}  (tick {0 if last_frame is None else 'range ends '}{last_frame})")
print(f"events: {len(run_overs)} run-through (ped center inside vehicle), "
      f"{len(grazes)} graze-only (footprint touch, center outside)\n")

if run_overs:
    print("RUN-THROUGH EVENTS (vehicle footprint covered a pedestrian's center):")
    print(f"{'veh_id':>7} {'class':>5} {'ped_id':>7} {'frames':>14} {'#frm':>5} "
          f"{'#inside':>7} {'max_pen_m':>9} {'veh_moved_m':>11}")
    for e in run_overs:
        cls, _ = veh_meta.get(e["vid"], ("?", "?"))
        span = f"{e['first_fr']}-{e['last_fr']}"
        print(f"{e['vid']:>7} {cls:>5} {e['pid']:>7} {span:>14} {e['n_frames']:>5} "
              f"{e['n_inside']:>7} {e['max_pen']:>9.3f} {veh_moved(e):>11.2f}")
else:
    print("No run-through events: no vehicle footprint ever covered a pedestrian center.")

# graze summary (contact but not run over) — likely peds brushing past stopped cars
if grazes:
    gd = defaultdict(int)
    for e in grazes:
        gd[veh_meta.get(e["vid"], ("?",""))[0]] += 1
    print(f"\ngraze-only events by vehicle class: {dict(gd)} "
          f"(footprint touched but center stayed outside — typically a ped passing "
          f"flush by a vehicle, not a hit)")
