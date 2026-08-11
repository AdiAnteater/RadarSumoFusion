"""Render a side-by-side video: camera frame (left) + combined BEV radar (right).

For each (sampled) camera frame:
  - left half  : camera PNG
  - right half : top-down BEV showing all 8 radar positions, their FOV wedges,
                 every matched detection from that frame coloured by class
                 (vehicle / pedestrian), unmatched clutter as small grey dots,
                 and ground-truth actor footprints (rectangles) for sanity.

Run:
  python tools/render_bev_video.py CAPTURE_DIR OUT_MP4 [N_FRAMES] [-w WINDOW] [-s {radar,camera}]

By default the cadence follows the radar (one frame per radar tick, subsampled to
N_FRAMES) with the camera image held between its sparser updates, so radar frames
that don't line up with the camera rate are still shown. Use -s camera for the
legacy one-frame-per-camera-tick behaviour.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from PIL import Image
import numpy as np

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a side-by-side camera + combined-BEV radar video from a capture dir.")
    p.add_argument("capture_dir", type=Path,
                   help="Capture directory (contains camera_data.csv, radar_data_labeled.csv, ...).")
    p.add_argument("out_mp4", type=Path, help="Output .mp4 path.")
    p.add_argument("n_frames", type=int, nargs="?", default=200,
                   help="Max camera frames to render, evenly subsampled (default 200).")
    p.add_argument("-w", "--window", type=int, default=2,
                   help="Accumulate radar returns over +/-WINDOW world ticks around each "
                        "rendered frame to counter single-tick radar sparsity (CARLA "
                        "re-randomises ~150 rays/sweep each tick, so any one tick misses an "
                        "in-range actor ~50%% of the time). 0 = legacy single-tick view. "
                        "Default 2 (~0.1 s, ~1 m motion smear).")
    p.add_argument("-s", "--source", choices=("radar", "camera"), default="radar",
                   help="What drives the video cadence. 'radar' (default): one frame per "
                        "radar tick, evenly subsampled to N_FRAMES, with the camera image "
                        "held between its (much sparser) updates — so radar frames that "
                        "don't line up with the camera rate are still shown. 'camera': "
                        "legacy one-frame-per-camera-tick behaviour.")
    p.add_argument("--keep-frames", action="store_true",
                   help="Keep the intermediate per-frame PNG directory. By default it is "
                        "deleted after a successful encode (it is only scratch for ffmpeg).")
    p.add_argument("--start-on-vehicle", action="store_true",
                   help="Trim to the vehicle-active span: start LEAD_TICKS before the first radar "
                        "tick that sees a vehicle (so you catch the first car driving in) and end "
                        "TRAIL_TICKS after the last, dropping the empty warm-up/cooldown. "
                        "Only affects -s radar.")
    p.add_argument("--vehicle-only", action="store_true",
                   help="Within the (optionally trimmed) span, keep only ticks that contain a "
                        "matched vehicle return — maximally vehicle-dense (no approach frames).")
    p.add_argument("--lead-ticks", type=int, default=90,
                   help="Ticks of lead-in before the first vehicle for --start-on-vehicle (default 90).")
    p.add_argument("--trail-ticks", type=int, default=60,
                   help="Ticks of trailing margin after the last vehicle for --start-on-vehicle (default 60).")
    p.add_argument("--around-frame", type=int, default=None,
                   help="Center the render on this radar frame id: keep only ticks within "
                        "+/- (--span-ticks)/2 of it. Use to focus on the most vehicle-dense "
                        "frame. Overrides --start-on-vehicle/--vehicle-only windowing.")
    p.add_argument("--span-ticks", type=int, default=400,
                   help="Total width (in radar ticks) of the --around-frame window (default 400).")
    p.add_argument("--fps", type=float, default=15.0,
                   help="Output video frame rate (default 15). Lower it to slow playback / "
                        "stretch a fixed number of frames into a longer clip for observation.")
    return p.parse_args()


_args = _parse_args()
CAPTURE_DIR = _args.capture_dir
OUT_MP4 = _args.out_mp4
MAX_FRAMES = _args.n_frames
WINDOW = max(0, _args.window)
SOURCE = _args.source
KEEP_FRAMES = _args.keep_frames
START_ON_VEHICLE = _args.start_on_vehicle
VEHICLE_ONLY = _args.vehicle_only
LEAD_TICKS = max(0, _args.lead_ticks)
TRAIL_TICKS = max(0, _args.trail_ticks)
AROUND_FRAME = _args.around_frame
SPAN_TICKS = max(1, _args.span_ticks)
FPS = max(0.1, _args.fps)
RADAR_RANGE_M = 35.0
RADAR_HFOV_DEG = 120.0

CAMERA_CSV = CAPTURE_DIR / "camera_data.csv"
RADAR_CSV = CAPTURE_DIR / "radar_data_labeled.csv"
ACTOR_JSONL = CAPTURE_DIR / "actor_frames.jsonl"
RADAR_EXTRINSICS = CAPTURE_DIR / "radar_extrinsics.csv"

CLASS_COLOR = {"vehicle": "#3aaaff", "pedestrian": "#ff5b5b"}
CLUTTER_COLOR = "#555555"
RADAR_PALETTE = [
    "#ff7070", "#70d870", "#70a8ff", "#ffc060",
    "#d878d8", "#70d8d8", "#ffaa50", "#a890ff",
]

print(f"[1/6] Reading camera frames from {CAMERA_CSV.name}...", flush=True)
camera_rows = []  # (frame, image_path), sorted by frame
with CAMERA_CSV.open(newline="") as f:
    for row in csv.DictReader(f):
        camera_rows.append((int(row["frame"]), row["image_path"]))
camera_rows.sort()
cam_frame_ids = [fr for fr, _ in camera_rows]
print(f"      {len(camera_rows)} camera frames on disk", flush=True)


def held_camera(tick):
    """Most-recent camera frame at or before ``tick`` (held between camera updates).
    Falls back to the earliest frame for ticks before the first camera frame."""
    if not camera_rows:
        return (None, None)
    j = bisect.bisect_right(cam_frame_ids, tick) - 1
    return camera_rows[max(j, 0)]


# Build the render plan: list of (radar_tick_for_BEV, camera_frame_id, camera_path).
if SOURCE == "radar":
    print(f"[2/6] Enumerating radar ticks in {RADAR_CSV.name}...", flush=True)
    radar_ticks = set()
    vehicle_ticks = set()  # ticks with >=1 matched vehicle return
    with RADAR_CSV.open() as f:
        hdr = f.readline().rstrip("\n").split(",")
        fi = hdr.index("frame")
        ki = hdr.index("matched_actor_kind")
        maxcol = max(fi, ki)
        for line in f:
            parts = line.split(",", maxcol + 1)
            try:
                fr = int(parts[fi])
            except (ValueError, IndexError):
                continue
            radar_ticks.add(fr)
            if len(parts) > ki and parts[ki].strip() == "vehicle":
                vehicle_ticks.add(fr)
    radar_ticks = sorted(radar_ticks)
    # Focus on a fixed window centered on a specific frame (e.g. the most vehicle-dense
    # one). Takes precedence over the vehicle-span windowing below.
    if AROUND_FRAME is not None:
        lo, hi = AROUND_FRAME - SPAN_TICKS // 2, AROUND_FRAME + SPAN_TICKS // 2
        kept = [t for t in radar_ticks if lo <= t <= hi]
        print(f"      --around-frame {AROUND_FRAME}: window [{lo},{hi}] keeps "
              f"{len(kept)}/{len(radar_ticks)} ticks", flush=True)
        radar_ticks = kept or radar_ticks
    # Optionally focus on the vehicle-active span: skip the empty warm-up (cars still
    # driving in) and cooldown, and/or keep only ticks that contain a vehicle.
    elif (START_ON_VEHICLE or VEHICLE_ONLY) and vehicle_ticks:
        fv, lv = min(vehicle_ticks), max(vehicle_ticks)
        if START_ON_VEHICLE:
            lo, hi = fv - LEAD_TICKS, lv + TRAIL_TICKS
            kept = [t for t in radar_ticks if lo <= t <= hi]
            print(f"      --start-on-vehicle: first vehicle tick={fv}, last={lv}; window "
                  f"[{lo},{hi}] keeps {len(kept)}/{len(radar_ticks)} ticks "
                  f"(dropped {len(radar_ticks) - len(kept)} empty warm-up/cooldown)", flush=True)
            radar_ticks = kept
        if VEHICLE_ONLY:
            before = len(radar_ticks)
            radar_ticks = [t for t in radar_ticks if t in vehicle_ticks]
            print(f"      --vehicle-only: kept {len(radar_ticks)}/{before} ticks with a vehicle",
                  flush=True)
    elif START_ON_VEHICLE or VEHICLE_ONLY:
        print("      (no matched-vehicle ticks found; rendering the full span)", flush=True)
    # Even subsample to MAX_FRAMES so length/render time stay bounded while the
    # cadence follows the radar — not the much sparser camera. Every radar tick is
    # a candidate, so radar frames that don't align with the camera rate are shown.
    if len(radar_ticks) > MAX_FRAMES:
        stride = len(radar_ticks) / MAX_FRAMES
        selected = [radar_ticks[int(i * stride)] for i in range(MAX_FRAMES)]
    else:
        selected = radar_ticks
    render_plan = [(t, *held_camera(t)) for t in selected]
    every = max(1, round(len(radar_ticks) / max(len(render_plan), 1)))
    print(f"      {len(radar_ticks)} radar ticks -> {len(render_plan)} rendered "
          f"(~every {every}th tick; camera held between its updates)", flush=True)
else:  # legacy: one frame per camera tick, nearest radar tick drawn
    print("[2/6] Camera-driven cadence (legacy)...", flush=True)
    if len(camera_rows) > MAX_FRAMES:
        step = len(camera_rows) // MAX_FRAMES
        cam_sel = camera_rows[::step][:MAX_FRAMES]
    else:
        cam_sel = camera_rows
    render_plan = [(fr, fr, path) for fr, path in cam_sel]
    print(f"      {len(render_plan)} camera frames selected", flush=True)

plan_ticks = [t for t, _, _ in render_plan]
# Load every tick we might draw: the BEV tick ±WINDOW (accumulation), plus ±2 slop
# so a camera-driven tick can still reach its nearest radar tick.
NEAR_SLOP = WINDOW + 2
ANY_NEAR = set()
for t in plan_ticks:
    for d in range(-NEAR_SLOP, NEAR_SLOP + 1):
        ANY_NEAR.add(t + d)

print(f"[3/6] Loading radar detections for the selected ticks from {RADAR_CSV.name}...", flush=True)
det_by_frame = defaultdict(list)  # frame -> list of (sensor_label, x_world, y_world, matched_kind)
sensor_positions = {}  # sensor_label -> (x, y, z, yaw_deg)
with RADAR_CSV.open(newline="") as f:
    for row in csv.DictReader(f):
        try:
            fr = int(row["frame"])
        except ValueError:
            continue
        if fr not in ANY_NEAR:
            continue
        s = row["sensor_label"].strip()
        sx = float(row["sensor_world_x_m"]); sy = float(row["sensor_world_y_m"]); sz = float(row["sensor_world_z_m"])
        sy_yaw = float(row["sensor_yaw_deg"])
        if s not in sensor_positions:
            sensor_positions[s] = (sx, sy, sz, sy_yaw)
        depth = float(row["depth_m"]); az = float(row["azimuth_rad"]); alt = float(row["altitude_rad"])
        # Convert (depth, az, alt) in sensor frame -> world XY
        lx = depth * math.cos(alt) * math.cos(az)
        ly = depth * math.cos(alt) * math.sin(az)
        yaw_rad = math.radians(sy_yaw)
        cs, sn = math.cos(yaw_rad), math.sin(yaw_rad)
        wx = sx + lx * cs - ly * sn
        wy = sy + lx * sn + ly * cs
        kind = row["matched_actor_kind"].strip() or "clutter"
        det_by_frame[fr].append((s, wx, wy, kind))
print(f"      detections loaded for {len(det_by_frame)} ticks", flush=True)

print(f"[4/6] Loading actor ground-truth footprints from {ACTOR_JSONL.name}...", flush=True)
actors_by_frame = defaultdict(list)  # frame -> list of (kind, x, y, yaw_deg, ext_x, ext_y)
with ACTOR_JSONL.open() as f:
    for line in f:
        rec = json.loads(line)
        fr = int(rec["frame"])
        if fr not in ANY_NEAR:
            continue
        for a in rec["actors"]:
            loc = a["location"]; rot = a.get("rotation") or {"yaw": 0}
            bbox = a.get("bbox") or {}
            ext = bbox.get("extent") or {}
            actors_by_frame[fr].append((
                a["kind"], float(loc["x"]), float(loc["y"]),
                float(rot.get("yaw", 0.0)),
                float(ext.get("x", 0.5)), float(ext.get("y", 0.5)),
            ))
print(f"      actor frames loaded for {len(actors_by_frame)} ticks", flush=True)

# Compute BEV bounds from all sensors + a generous margin
xs = [v[0] for v in sensor_positions.values()]
ys = [v[1] for v in sensor_positions.values()]
margin = 12.0
bev_x0, bev_x1 = min(xs) - margin, max(xs) + margin
bev_y0, bev_y1 = min(ys) - margin, max(ys) + margin
print(f"[5/6] BEV bounds: x=[{bev_x0:.1f},{bev_x1:.1f}] y=[{bev_y0:.1f},{bev_y1:.1f}]", flush=True)

def fov_wedge(sx, sy, yaw_deg, range_m=RADAR_RANGE_M, fov_deg=RADAR_HFOV_DEG, steps=24):
    yaw_rad = math.radians(yaw_deg)
    half = math.radians(fov_deg) / 2.0
    pts = [(sx, sy)]
    for i in range(steps + 1):
        a = yaw_rad - half + (2 * half) * (i / steps)
        pts.append((sx + range_m * math.cos(a), sy + range_m * math.sin(a)))
    return pts

def find_nearest_frame(target_fr, candidate_set):
    if target_fr in candidate_set:
        return target_fr
    for d in (1, -1, 2, -2):
        if target_fr + d in candidate_set:
            return target_fr + d
    return None

print(f"[6/6] Rendering {len(render_plan)} frames "
      f"({SOURCE}-driven, accumulation window ±{WINDOW} ticks)...", flush=True)
tmp_dir = OUT_MP4.parent / (OUT_MP4.stem + "_frames")
tmp_dir.mkdir(parents=True, exist_ok=True)
# Stable sensor color map
sensor_color = {s: RADAR_PALETTE[i % len(RADAR_PALETTE)] for i, s in enumerate(sorted(sensor_positions))}

# Decode each held camera image at most once: plan ticks are sorted, so the held
# camera frame only advances monotonically.
_cam_cache = {"frame": None, "img": None}


def load_camera(cam_frame, cam_path):
    if cam_path is None:
        return np.zeros((600, 800, 3), dtype=np.uint8)
    if _cam_cache["frame"] != cam_frame:
        try:
            _cam_cache["img"] = np.array(Image.open(cam_path).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            print(f"      cam read failed for frame {cam_frame} ({e}); using black")
            _cam_cache["img"] = np.zeros((600, 800, 3), dtype=np.uint8)
        _cam_cache["frame"] = cam_frame
    return _cam_cache["img"]


for idx, (fr, cam_frame, cam_path) in enumerate(render_plan):
    if idx % 25 == 0:
        print(f"      frame {idx}/{len(render_plan)} (radar tick {fr}, cam {cam_frame})", flush=True)
    cam_img = load_camera(cam_frame, cam_path)
    cam_h, cam_w = cam_img.shape[:2]

    # BEV radar tick: in radar mode `fr` IS a radar tick; in camera mode snap to the
    # nearest radar tick. Actor footprints always use the nearest actor tick.
    if SOURCE == "radar":
        rd_fr = fr if fr in det_by_frame else (find_nearest_frame(fr, det_by_frame.keys()) or fr)
    else:
        rd_fr = find_nearest_frame(fr, det_by_frame.keys()) or fr
    ac_fr = find_nearest_frame(fr, actors_by_frame.keys()) or fr

    # Accumulate radar returns over ±WINDOW ticks around the matched radar tick so
    # an in-range actor that a single sweep happened to miss still shows up. Footprints
    # (actors) stay on the single nearest tick to avoid smearing ground truth.
    dets = [d for k in range(rd_fr - WINDOW, rd_fr + WINDOW + 1)
            for d in det_by_frame.get(k, [])]
    actors = actors_by_frame.get(ac_fr, [])

    # Figure: 16:6 aspect, camera on left at native ratio, BEV on right
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(14, 5.25),
                                          gridspec_kw={"width_ratios": [800.0/600.0, (bev_x1-bev_x0)/(bev_y1-bev_y0)]})
    fig.patch.set_facecolor("#101015")

    # ---- Left: camera ----
    ax_cam.imshow(cam_img)
    ax_cam.set_xticks([]); ax_cam.set_yticks([])
    ax_cam.set_title(f"Camera C10  |  cam tick {cam_frame}", color="white", fontsize=10)
    for spine in ax_cam.spines.values():
        spine.set_edgecolor("#444")

    # ---- Right: BEV ----
    ax_bev.set_facecolor("#181820")
    ax_bev.set_xlim(bev_x0, bev_x1)
    ax_bev.set_ylim(bev_y1, bev_y0)  # invert y so +y is down (CARLA top-down)
    ax_bev.set_aspect("equal")
    win_lbl = f" ±{WINDOW}t" if WINDOW else ""
    ax_bev.set_title(f"BEV — all 8 radars + matched dets  |  tick {rd_fr}{win_lbl}",
                     color="white", fontsize=10)
    ax_bev.tick_params(colors="#777", labelsize=8)
    for spine in ax_bev.spines.values():
        spine.set_edgecolor("#444")

    # FOV wedges per radar
    for s, (sx, sy, sz, syaw) in sensor_positions.items():
        pts = fov_wedge(sx, sy, syaw)
        poly = Polygon(pts, closed=True, facecolor=sensor_color[s], alpha=0.08,
                       edgecolor=sensor_color[s], linewidth=0.8)
        ax_bev.add_patch(poly)

    # Actor ground-truth footprints (oriented rectangles)
    for kind, ax_, ay_, ayaw, ex, ey in actors:
        yaw = math.radians(ayaw)
        cs, sn = math.cos(yaw), math.sin(yaw)
        corners_local = [(+ex, +ey), (+ex, -ey), (-ex, -ey), (-ex, +ey)]
        corners_world = [(ax_ + lx*cs - ly*sn, ay_ + lx*sn + ly*cs) for lx, ly in corners_local]
        color = CLASS_COLOR.get(kind, "#888")
        ax_bev.add_patch(Polygon(corners_world, closed=True, facecolor=color,
                                 alpha=0.25, edgecolor=color, linewidth=0.8))

    # Detections — unmatched first (background)
    clutter_x = [d[1] for d in dets if d[3] == "clutter"]
    clutter_y = [d[2] for d in dets if d[3] == "clutter"]
    if clutter_x:
        ax_bev.scatter(clutter_x, clutter_y, s=4, c=CLUTTER_COLOR, alpha=0.5, linewidths=0)
    # Matched on top, coloured per class
    for cls in ("vehicle", "pedestrian"):
        xs = [d[1] for d in dets if d[3] == cls]
        ys = [d[2] for d in dets if d[3] == cls]
        if xs:
            ax_bev.scatter(xs, ys, s=18, c=CLASS_COLOR[cls], alpha=0.95, edgecolors="white", linewidths=0.3, label=cls)

    # Sensor markers (triangles)
    for s, (sx, sy, sz, syaw) in sensor_positions.items():
        yaw = math.radians(syaw)
        tri = [
            (sx + 1.2*math.cos(yaw), sy + 1.2*math.sin(yaw)),
            (sx + 0.6*math.cos(yaw + 2.6), sy + 0.6*math.sin(yaw + 2.6)),
            (sx + 0.6*math.cos(yaw - 2.6), sy + 0.6*math.sin(yaw - 2.6)),
        ]
        ax_bev.add_patch(Polygon(tri, closed=True, facecolor=sensor_color[s], edgecolor="white", linewidth=0.4))
        ax_bev.annotate(s, (sx, sy), xytext=(4, 4), textcoords="offset points",
                        color=sensor_color[s], fontsize=8, weight="bold")

    n_m = sum(1 for d in dets if d[3] in CLASS_COLOR)
    n_c = sum(1 for d in dets if d[3] == "clutter")
    ax_bev.text(0.02, 0.97, f"matched={n_m}  clutter={n_c}  actors={len(actors)}",
                transform=ax_bev.transAxes, color="white", fontsize=8,
                verticalalignment="top",
                bbox=dict(facecolor="#000", alpha=0.4, edgecolor="none", pad=2))

    plt.tight_layout()
    out_png = tmp_dir / f"frame_{idx:05d}.png"
    fig.savefig(out_png, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)

print(f"      done. scratch frames in {tmp_dir}", flush=True)
print(f"[ffmpeg] encoding {OUT_MP4}...", flush=True)
subprocess.check_call([
    "ffmpeg", "-y", "-loglevel", "error",
    "-framerate", str(FPS),
    "-i", str(tmp_dir / "frame_%05d.png"),
    # matplotlib frame size depends on per-capture BEV bounds and can be odd;
    # libx264 + yuv420p needs even dimensions, so round down to the nearest even.
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
    str(OUT_MP4),
])
print(f"Wrote {OUT_MP4}", flush=True)

# The per-frame PNGs are only scratch for ffmpeg; drop them unless asked to keep.
# Only runs on a successful encode (check_call raises otherwise, leaving them for debug).
if not KEEP_FRAMES:
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Removed scratch frames {tmp_dir}", flush=True)
