import csv
from collections import defaultdict
from pathlib import Path

path = Path(
    r"c:\Users\Colin\Downloads\CARLA_Latest\CARLA_Latest\Scripts\DatasetCreation\Data\sensor_capture_20260524_161300\radar_data.csv"
)
rows = list(csv.DictReader(path.open(encoding="utf-8")))
print("total_rows", len(rows))

by_sensor = defaultdict(list)
by_frame_sensor = defaultdict(int)
for r in rows:
    by_sensor[r["sensor_label"]].append(r)
    by_frame_sensor[(r["sensor_label"], r["frame"])] += 1

print("\n=== Per sensor ===")
for s in sorted(by_sensor, key=lambda x: int(x[1:])):
    rs = by_sensor[s]
    frames = {r["frame"] for r in rs}
    msgs = len(frames)
    dets = len(rs)
    per_msg = [by_frame_sensor[(s, f)] for f in frames]
    matched = sum(1 for r in rs if r.get("matched_actor_id", "").strip())
    veh = sum(1 for r in rs if r.get("matched_vehicle_id", "").strip())
    cand = sum(1 for r in rs if r.get("had_actor_candidates", "") == "1")
    v_nonzero = sum(1 for r in rs if abs(float(r["velocity_mps"])) > 0.01)
    depths = [float(r["depth_m"]) for r in rs]
    loc = (
        float(rs[0]["sensor_world_x_m"]),
        float(rs[0]["sensor_world_y_m"]),
        float(rs[0]["sensor_yaw_deg"]),
    )
    avg_msg = dets / max(msgs, 1)
    print(
        f"{s}: msgs={msgs} dets={dets} avg={avg_msg:.1f}/msg range={min(per_msg)}-{max(per_msg)} "
        f"matched={matched} veh={veh} cand={cand} |v|>0={v_nonzero} "
        f"depth=[{min(depths):.1f},{max(depths):.1f}] yaw={loc[2]:.1f}"
    )

ts = [float(r["timestamp"]) for r in rows]
frames = [int(r["frame"]) for r in rows]
dur = max(ts) - min(ts)
unique_msgs = len({(r["sensor_label"], r["frame"]) for r in rows})
print(f"\n=== Time ===")
print(f"sim duration: {dur:.1f}s  ({min(ts):.1f} -> {max(ts):.1f})")
print(f"unique radar messages captured: {unique_msgs}")
print(f"captured msg rate: {unique_msgs / dur:.3f}/s total  ({unique_msgs / dur / 8:.3f}/s per radar)")
print(f"row rate: {len(rows) / dur:.2f} detections/s")
print(f"expected at 20Hz tick x 8 radars: ~{20*8*dur:.0f} messages if all processed")

print("\n=== Velocity ===")
bins = {"0": 0, "0-0.5": 0, "0.5-2": 0, "2-5": 0, "5+": 0}
for r in rows:
    v = abs(float(r["velocity_mps"]))
    if v < 0.01:
        bins["0"] += 1
    elif v < 0.5:
        bins["0-0.5"] += 1
    elif v < 2:
        bins["0.5-2"] += 1
    elif v < 5:
        bins["2-5"] += 1
    else:
        bins["5+"] += 1
print(bins)

print("\n=== Depth bands ===")
bands = {"0-5": 0, "5-15": 0, "15-25": 0, "25-35": 0, "35+": 0}
for r in rows:
    d = float(r["depth_m"])
    if d <= 5:
        bands["0-5"] += 1
    elif d <= 15:
        bands["5-15"] += 1
    elif d <= 25:
        bands["15-25"] += 1
    elif d <= 35:
        bands["25-35"] += 1
    else:
        bands["35+"] += 1
print(bands)

print("\n=== South row R8->R2 corridor ===")
for s in ["R2", "R4", "R6", "R8"]:
    rs = by_sensor.get(s, [])
    msgs = len({r["frame"] for r in rs})
    veh = sum(1 for r in rs if r.get("matched_vehicle_id", "").strip())
    print(f"{s}: {len(rs)} dets, {msgs} msgs, {veh} vehicle matches")

matched_rows = [r for r in rows if r.get("matched_vehicle_id", "").strip()]
print(f"\n=== Vehicle matches ({len(matched_rows)}) ===")
for r in matched_rows[:20]:
    print(
        f"  {r['sensor_label']} frame={r['frame']} depth={float(r['depth_m']):.1f}m "
        f"v={float(r['velocity_mps']):.2f} veh={r['matched_vehicle_id']}"
    )
