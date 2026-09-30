"""Render one camera + BEV video per campaign segment, in a single CSV pass.

The labeled radar file is tens of GB, so this streams it once and encodes each
scenario as the frames go by. Playback subsamples the 20 Hz capture.

Run:
  python tools/render_segment_bev_videos.py CAPTURE_DIR [--stride 5] [--fps 10]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image

import imageio_ffmpeg

RADAR_RANGE_M = 35.0
RADAR_HFOV_DEG = 120.0
CLASS_COLOR = {"vehicle": "#3aaaff", "pedestrian": "#ff5b5b"}
CLUTTER_COLOR = "#555555"
RADAR_PALETTE = [
    "#ff7070", "#70d870", "#70a8ff", "#ffc060",
    "#d878d8", "#70d8d8", "#ffaa50", "#a890ff",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One BEV video per campaign segment.")
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--stride", type=int, default=5,
                   help="Render every Nth radar tick (default 5 = 0.25 s at 20 Hz).")
    p.add_argument("--window", type=int, default=2,
                   help="Accumulate radar returns over +/- this many ticks (default 2).")
    p.add_argument("--fps", type=float, default=10.0, help="Output frame rate (default 10).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory for the mp4 files (default: CAPTURE_DIR/analysis).")
    return p.parse_args()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def fov_wedge(sx, sy, yaw_deg, range_m=RADAR_RANGE_M, fov_deg=RADAR_HFOV_DEG, steps=24):
    yaw_rad = math.radians(yaw_deg)
    half = math.radians(fov_deg) / 2.0
    pts = [(sx, sy)]
    for i in range(steps + 1):
        a = yaw_rad - half + (2 * half) * (i / steps)
        pts.append((sx + range_m * math.cos(a), sy + range_m * math.sin(a)))
    return pts


def main() -> None:
    args = _parse_args()
    capture = args.capture_dir
    stride = max(1, args.stride)
    window = max(0, args.window)
    fps = max(0.1, args.fps)
    out_dir = args.out_dir or (capture / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = json.loads((capture / "segments.json").read_text(encoding="utf-8"))["segments"]
    jobs = []
    selected: list[tuple[int, int]] = []
    needed: set[int] = set()
    for si, seg in enumerate(segments):
        start, end = int(seg["start_frame"]), int(seg["end_frame"])
        frames = list(range(start, end + 1, stride))
        if frames[-1] != end:
            frames.append(end)
        name = f"S{int(seg['scenario']):02d} {seg['scenario_name']}"
        out = out_dir / f"bev_s{int(seg['scenario']):02d}_{slug(seg['scenario_name'])}.mp4"
        jobs.append({
            "name": name,
            "start": start,
            "end": end,
            "rate_hz": float(seg.get("rate_hz") or 20.0),
            "out": out,
            "n": len(frames),
        })
        for fr in frames:
            selected.append((fr, si))
            for d in range(-window, window + 1):
                t = fr + d
                if start <= t <= end:
                    needed.add(t)
    selected.sort()
    print(f"{len(jobs)} segments, {len(selected)} frames, stride {stride}, "
          f"window ±{window}, {fps:g} fps", flush=True)
    for job in jobs:
        print(f"  {job['name']} -> {job['out'].name} ({job['n']} frames)", flush=True)

    sensors = []
    with (capture / "radar_extrinsics.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sensors.append((
                row["sensor_label"].strip(),
                float(row["x"]), float(row["y"]), float(row["yaw"]),
            ))
    sensor_color = {s[0]: RADAR_PALETTE[i % len(RADAR_PALETTE)] for i, s in enumerate(sensors)}
    xs = [s[1] for s in sensors]
    ys = [s[2] for s in sensors]
    margin = 12.0
    bev_x0, bev_x1 = min(xs) - margin, max(xs) + margin
    bev_y0, bev_y1 = min(ys) - margin, max(ys) + margin

    cam_by_frame: dict[int, str] = {}
    want_cam = {fr for fr, _ in selected}
    with (capture / "camera_data.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fr = int(row["frame"])
            if fr in want_cam:
                cam_by_frame[fr] = row["image_path"]
    print(f"camera paths for {len(cam_by_frame)}/{len(want_cam)} rendered frames", flush=True)

    actors_by_frame: dict[int, list] = defaultdict(list)
    with (capture / "actor_frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            fr = int(line[line.find('"frame":') + 8:].split(",", 1)[0])
            if fr not in want_cam:
                continue
            rec = json.loads(line)
            for a in rec["actors"]:
                loc = a["location"]
                rot = a.get("rotation") or {}
                ext = (a.get("bbox") or {}).get("extent") or {}
                ax_, ay_ = float(loc["x"]), float(loc["y"])
                if not (bev_x0 - 5 <= ax_ <= bev_x1 + 5 and bev_y0 - 5 <= ay_ <= bev_y1 + 5):
                    continue
                actors_by_frame[fr].append((
                    a["kind"], ax_, ay_, float(rot.get("yaw", 0.0)),
                    float(ext.get("x", 0.5)), float(ext.get("y", 0.5)),
                ))
    print(f"actor footprints for {len(actors_by_frame)} frames", flush=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fig, (ax_cam, ax_bev) = plt.subplots(
        1, 2, figsize=(14, 5.25),
        gridspec_kw={"width_ratios": [800.0 / 600.0, (bev_x1 - bev_x0) / (bev_y1 - bev_y0)]},
    )
    fig.patch.set_facecolor("#101015")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.08, wspace=0.12)
    blank = np.zeros((600, 800, 3), dtype=np.uint8)
    cam_cache: dict = {"frame": None, "img": None}

    proc: subprocess.Popen | None = None
    open_si: int | None = None
    rendered = 0
    t0 = time.perf_counter()

    def close_encoder() -> None:
        nonlocal proc, open_si
        if proc is None:
            return
        assert proc.stdin is not None
        proc.stdin.close()
        code = proc.wait()
        err_fh = getattr(proc, "_err_fh", None)
        if err_fh is not None:
            err_fh.close()
        out = jobs[open_si]["out"]
        err_path = getattr(proc, "_err_path", None)
        err = err_path.read_text(encoding="utf-8", errors="replace").strip() if err_path else ""
        if err_path:
            err_path.unlink(missing_ok=True)
        if code != 0:
            raise RuntimeError(f"ffmpeg failed for {out} (exit {code}): {err}")
        print(f"wrote {out}", flush=True)
        proc = None
        open_si = None

    def open_encoder(si: int) -> None:
        nonlocal proc, open_si
        if open_si == si:
            return
        close_encoder()
        out = jobs[si]["out"]
        err_path = out.with_suffix(".ffmpeg.log")
        err_fh = err_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "image2pipe", "-vcodec", "png", "-framerate", str(fps),
                "-i", "-",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(out),
            ],
            stdin=subprocess.PIPE,
            stderr=err_fh,
        )
        proc._err_path = err_path  # type: ignore[attr-defined]
        proc._err_fh = err_fh  # type: ignore[attr-defined]
        open_si = si

    def load_camera(fr: int) -> np.ndarray:
        path = cam_by_frame.get(fr)
        if not path:
            return blank
        if cam_cache["frame"] != fr:
            try:
                cam_cache["img"] = np.array(Image.open(path).convert("RGB"))
            except OSError as exc:
                print(f"camera read failed for frame {fr} ({exc})", flush=True)
                cam_cache["img"] = blank
            cam_cache["frame"] = fr
        return cam_cache["img"]

    def render(fr: int, si: int, dets: list[tuple[float, float, str]]) -> None:
        nonlocal rendered
        job = jobs[si]
        open_encoder(si)
        sim_s = (fr - job["start"]) / job["rate_hz"]
        cam = load_camera(fr)
        actors = actors_by_frame.get(fr, [])

        ax_cam.clear()
        ax_bev.clear()
        ax_cam.imshow(cam)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        ax_cam.set_title(f"{job['name']}   |   camera C10   |   t={sim_s:.1f}s",
                         color="white", fontsize=10)
        for spine in ax_cam.spines.values():
            spine.set_edgecolor("#444")

        ax_bev.set_facecolor("#181820")
        ax_bev.set_xlim(bev_x0, bev_x1)
        ax_bev.set_ylim(bev_y1, bev_y0)
        ax_bev.set_aspect("equal")
        ax_bev.set_title(f"BEV  |  tick {fr}  |  ±{window} ticks", color="white", fontsize=10)
        ax_bev.tick_params(colors="#777", labelsize=8)
        for spine in ax_bev.spines.values():
            spine.set_edgecolor("#444")

        for label, sx, sy, syaw in sensors:
            ax_bev.add_patch(Polygon(
                fov_wedge(sx, sy, syaw), closed=True,
                facecolor=sensor_color[label], alpha=0.08,
                edgecolor=sensor_color[label], linewidth=0.8,
            ))
        for kind, ax_, ay_, ayaw, ex, ey in actors:
            yaw = math.radians(ayaw)
            cs, sn = math.cos(yaw), math.sin(yaw)
            corners = [(+ex, +ey), (+ex, -ey), (-ex, -ey), (-ex, +ey)]
            world = [(ax_ + lx * cs - ly * sn, ay_ + lx * sn + ly * cs) for lx, ly in corners]
            color = CLASS_COLOR.get(kind, "#888888")
            ax_bev.add_patch(Polygon(world, closed=True, facecolor=color, alpha=0.25,
                                     edgecolor=color, linewidth=0.8))

        clutter_x = [d[0] for d in dets if d[2] not in CLASS_COLOR]
        clutter_y = [d[1] for d in dets if d[2] not in CLASS_COLOR]
        if clutter_x:
            ax_bev.scatter(clutter_x, clutter_y, s=4, c=CLUTTER_COLOR, alpha=0.5, linewidths=0)
        for cls in ("vehicle", "pedestrian"):
            px = [d[0] for d in dets if d[2] == cls]
            py = [d[1] for d in dets if d[2] == cls]
            if px:
                ax_bev.scatter(px, py, s=18, c=CLASS_COLOR[cls], alpha=0.95,
                               edgecolors="white", linewidths=0.3)
        for label, sx, sy, syaw in sensors:
            yaw = math.radians(syaw)
            tri = [
                (sx + 1.2 * math.cos(yaw), sy + 1.2 * math.sin(yaw)),
                (sx + 0.6 * math.cos(yaw + 2.6), sy + 0.6 * math.sin(yaw + 2.6)),
                (sx + 0.6 * math.cos(yaw - 2.6), sy + 0.6 * math.sin(yaw - 2.6)),
            ]
            ax_bev.add_patch(Polygon(tri, closed=True, facecolor=sensor_color[label],
                                     edgecolor="white", linewidth=0.4))
            ax_bev.annotate(label, (sx, sy), xytext=(4, 4), textcoords="offset points",
                            color=sensor_color[label], fontsize=8, weight="bold")
        n_m = sum(1 for d in dets if d[2] in CLASS_COLOR)
        ax_bev.text(
            0.02, 0.97,
            f"matched={n_m}  clutter={len(dets) - n_m}  actors={len(actors)}",
            transform=ax_bev.transAxes, color="white", fontsize=8, verticalalignment="top",
            bbox=dict(facecolor="#000000", alpha=0.4, edgecolor="none", pad=2),
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        proc.stdin.write(buf.getvalue())
        rendered += 1
        if rendered == 1 or rendered % 25 == 0:
            elapsed = time.perf_counter() - t0
            rate = rendered / elapsed
            left = (len(selected) - rendered) / rate if rate else 0
            print(f"frame {rendered}/{len(selected)}  {job['name']} t={sim_s:.1f}s  "
                  f"{rate:.2f} fps  eta {left/60:.1f} min", flush=True)

    buf_pts: dict[int, list[tuple[float, float, str]]] = {}
    next_i = 0
    rows = 0
    csv_path = capture / "radar_data_labeled.csv"
    print(f"streaming {csv_path.name}...", flush=True)
    with csv_path.open(encoding="utf-8") as f:
        f.readline()
        for line in f:
            rows += 1
            parts = line.split(",", 30)
            try:
                fr = int(parts[2])
            except (IndexError, ValueError):
                continue
            while next_i < len(selected) and selected[next_i][0] + window < fr:
                sel, si = selected[next_i]
                dets = [p for k in range(sel - window, sel + window + 1) for p in buf_pts.get(k, ())]
                render(sel, si, dets)
                next_i += 1
                if next_i < len(selected):
                    cutoff = selected[next_i][0] - window
                else:
                    cutoff = sel + window + 1
                for k in [k for k in buf_pts if k < cutoff]:
                    del buf_pts[k]
            if fr not in needed:
                continue
            try:
                kind = parts[16].strip() or "clutter"
                buf_pts.setdefault(fr, []).append((float(parts[28]), float(parts[29]), kind))
            except (IndexError, ValueError):
                continue
            if rows % 2_000_000 == 0:
                print(f"  scanned {rows/1e6:.1f}M rows, rendered {rendered}", flush=True)

    while next_i < len(selected):
        sel, si = selected[next_i]
        dets = [p for k in range(sel - window, sel + window + 1) for p in buf_pts.get(k, ())]
        render(sel, si, dets)
        next_i += 1
    close_encoder()
    plt.close(fig)
    print(f"done in {(time.perf_counter() - t0)/60:.1f} min, {rows} rows", flush=True)


if __name__ == "__main__":
    main()
