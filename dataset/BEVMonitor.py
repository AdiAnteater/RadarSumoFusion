"""Lightweight 2D bird's-eye-view monitor for the dataset pipeline.

Connects to the same CARLA server as the capture pipeline and renders a small
clipped top-down view of the sensor corridor with:
  - Static lane center lines (one-time backdrop).
  - Radar sensors as triangles + translucent FOV wedges.
  - Camera sensors as small squares.
  - Vehicles and pedestrians as oriented rectangles (color by type).
  - Live radar detections as dots (optional, off by default to stay cheap).

No CARLA rendering is used — all drawing is in pygame on a CPU surface, so this
works on a headless GPU server even when CARLA is launched with -RenderOffScreen.

Output modes (mutually exclusive):
  --display     pygame window (needs X / VNC / xpra forwarding).
  --snapshots   PNG every N frames into the active capture folder (default
                when no $DISPLAY).
  --video       single MP4 in the capture folder (needs imageio-ffmpeg).

Run independently of Start.py, e.g.:
    python BEVMonitor.py --snapshots --rate 5
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import carla
import numpy as np
import pygame

DATASET_RADAR_ROLE_PREFIX = "dataset_radar_"
DATASET_CAMERA_ROLE_PREFIX = "dataset_camera_"

DEFAULT_RADAR_RANGE_M = 35.0
DEFAULT_RADAR_FOV_DEG = 120.0

COLORS = {
    "bg": (18, 18, 22),
    "road": (60, 60, 70),
    "lane": (110, 110, 120),
    "vehicle": (90, 200, 255),
    "bike": (255, 180, 90),
    "ped": (255, 110, 110),
    "radar": (255, 235, 80),
    "radar_fov": (255, 235, 80, 36),  # fallback when role_name is missing
    "camera": (180, 255, 180),
    "radar_dot": (255, 120, 50),
    "hud": (230, 230, 230),
    "frame": (40, 40, 50),
}

# Per-radar palette: solid RGB for the triangle/label, same hue with low alpha
# for the FOV wedge. Cycled by sorted role_name (so R1→R4 are stable across runs).
RADAR_PALETTE = [
    ((255, 110, 110), (255, 110, 110, 60)),   # red
    ((110, 220, 110), (110, 220, 110, 60)),   # green
    ((110, 180, 255), (110, 180, 255, 60)),   # blue
    ((255, 200,  90), (255, 200,  90, 60)),   # amber
    ((220, 110, 220), (220, 110, 220, 60)),   # magenta
    ((110, 230, 220), (110, 230, 220, 60)),   # teal
]


def radar_short_name(radar) -> str:
    role = radar.attributes.get("role_name", "")
    return role.split("_")[-1] if role else f"id{radar.id}"


def radar_color_pair(radar, palette_index: int):
    """Return (solid_rgb, wedge_rgba) for this radar, stable across frames."""
    return RADAR_PALETTE[palette_index % len(RADAR_PALETTE)]

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--display", action="store_true",
                      help="Open a pygame window (needs $DISPLAY).")
    mode.add_argument("--snapshots", action="store_true",
                      help="Write PNG snapshots into the capture folder.")
    mode.add_argument("--video", action="store_true",
                      help="Write a single MP4 into the capture folder.")
    p.add_argument("--rate", type=float, default=10.0,
                   help="Refresh rate in Hz (default 10).")
    p.add_argument("--snapshot-every", type=int, default=5,
                   help="In --snapshots mode, save every Nth rendered frame.")
    p.add_argument("--margin", type=float, default=12.0,
                   help="Meters of padding around the radar bounding box.")
    p.add_argument("--width", type=int, default=900, help="Pixel width.")
    p.add_argument("--height", type=int, default=540, help="Pixel height.")
    p.add_argument("--no-roads", action="store_true",
                   help="Skip the static lane backdrop.")
    p.add_argument("--no-radar-points", action="store_true",
                   help="Do not subscribe to radar detections.")
    p.add_argument("--out", type=str, default="",
                   help="Output directory (overrides .last_dataset_capture_dir).")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--duration", type=float, default=0.0,
                   help="Stop after N seconds (0 = run until Ctrl+C).")
    return p.parse_args()


def resolve_output_dir(explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    pointer = SCRIPT_DIR / ".last_dataset_capture_dir"
    if pointer.is_file():
        raw = pointer.read_text(encoding="utf-8").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (SCRIPT_DIR / p).resolve()
            if p.is_dir():
                return p
    fallback = SCRIPT_DIR / f"bev_monitor_{time.strftime('%Y%m%d_%H%M%S')}"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def discover_sensors(world: carla.World):
    radars, cameras = [], []
    for actor in world.get_actors():
        if not actor.type_id.startswith("sensor."):
            continue
        role = actor.attributes.get("role_name", "")
        if role.startswith(DATASET_RADAR_ROLE_PREFIX):
            radars.append(actor)
        elif role.startswith(DATASET_CAMERA_ROLE_PREFIX):
            cameras.append(actor)
    return radars, cameras


def compute_view_bounds(radars, margin: float):
    if not radars:
        return (-50.0, -100.0, 50.0, 0.0)
    xs = [a.get_transform().location.x for a in radars]
    ys = [a.get_transform().location.y for a in radars]
    return (min(xs) - margin, min(ys) - margin,
            max(xs) + margin, max(ys) + margin)


class WorldToScreen:
    """Affine map from world XY (meters) to screen XY (pixels), Y inverted."""

    def __init__(self, bounds, width: int, height: int):
        x0, y0, x1, y1 = bounds
        wx, wy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        # Preserve aspect ratio: pick the tighter scale, then center.
        scale = min(width / wx, height / wy) * 0.96
        self.scale = scale
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.cx = (x0 + x1) / 2.0
        self.cy = (y0 + y1) / 2.0
        self.width, self.height = width, height

    def __call__(self, wx: float, wy: float):
        sx = self.width / 2.0 + (wx - self.cx) * self.scale
        # Invert Y so +y world points downward on screen (standard CARLA top-down).
        sy = self.height / 2.0 + (wy - self.cy) * self.scale
        return (int(sx), int(sy))

    def vec(self, dx: float, dy: float):
        return (dx * self.scale, dy * self.scale)


def actor_color(actor: carla.Actor) -> tuple:
    tid = actor.type_id
    if tid.startswith("walker."):
        return COLORS["ped"]
    if "motorcycle" in tid or "bike" in tid or "bh" in tid:
        return COLORS["bike"]
    return COLORS["vehicle"]


def oriented_box_corners(transform: carla.Transform, extent: carla.Vector3D):
    """4 world-XY corners of an actor's footprint (counter-clockwise)."""
    ex, ey = extent.x, extent.y
    local = [(ex, ey), (ex, -ey), (-ex, -ey), (-ex, ey)]
    yaw = math.radians(transform.rotation.yaw)
    cs, sn = math.cos(yaw), math.sin(yaw)
    cx, cy = transform.location.x, transform.location.y
    return [(cx + lx * cs - ly * sn, cy + lx * sn + ly * cs) for (lx, ly) in local]


def fov_wedge_points(sensor: carla.Actor, range_m: float, fov_deg: float, steps: int = 24):
    tr = sensor.get_transform()
    cx, cy = tr.location.x, tr.location.y
    yaw = math.radians(tr.rotation.yaw)
    half = math.radians(fov_deg) / 2.0
    pts = [(cx, cy)]
    for i in range(steps + 1):
        a = yaw - half + (2 * half) * (i / steps)
        pts.append((cx + range_m * math.cos(a), cy + range_m * math.sin(a)))
    return pts


def sensor_range_fov(sensor: carla.Actor) -> tuple[float, float]:
    attrs = sensor.attributes
    try:
        rng = float(attrs.get("range", DEFAULT_RADAR_RANGE_M))
    except ValueError:
        rng = DEFAULT_RADAR_RANGE_M
    try:
        fov = float(attrs.get("horizontal_fov", DEFAULT_RADAR_FOV_DEG))
    except ValueError:
        fov = DEFAULT_RADAR_FOV_DEG
    return rng, fov


class RadarPointBuffer:
    """Holds the most recent radar detection cloud (in world XY) per sensor."""

    def __init__(self, radars):
        self._lock = threading.Lock()
        self._points: dict[int, np.ndarray] = {}
        self._radars = radars

    def update_from_measurement(self, sensor_id: int, sensor: carla.Actor, meas):
        # Each detection: depth (m), azimuth (rad, CCW from sensor +X), altitude (rad).
        tr = sensor.get_transform()
        yaw = math.radians(tr.rotation.yaw)
        cs, sn = math.cos(yaw), math.sin(yaw)
        cx, cy = tr.location.x, tr.location.y
        pts = []
        for d in meas:
            # CARLA radar azimuth: positive = right of sensor forward; ignore altitude for BEV.
            r = d.depth * math.cos(d.altitude)
            az = d.azimuth
            lx = r * math.cos(az)
            ly = r * math.sin(az)
            wx = cx + lx * cs - ly * sn
            wy = cy + lx * sn + ly * cs
            pts.append((wx, wy))
        with self._lock:
            self._points[sensor_id] = np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)

    def snapshot(self) -> dict[int, np.ndarray]:
        with self._lock:
            return dict(self._points)


def build_lane_backdrop(world: carla.World, bounds, step: float = 2.0):
    """Pre-rasterize lane center polylines within the view box."""
    x0, y0, x1, y1 = bounds
    waypoints = world.get_map().generate_waypoints(step)
    by_road: dict[tuple, list] = {}
    for w in waypoints:
        loc = w.transform.location
        if not (x0 <= loc.x <= x1 and y0 <= loc.y <= y1):
            continue
        key = (w.road_id, w.lane_id)
        by_road.setdefault(key, []).append((loc.x, loc.y, w.s))
    # Sort each lane's waypoints by s so we can connect them cleanly.
    polylines = []
    for pts in by_road.values():
        if len(pts) < 2:
            continue
        pts.sort(key=lambda t: t[2])
        polylines.append([(x, y) for (x, y, _) in pts])
    return polylines


def draw_frame(surface, ctx, *, hud_text: str):
    surface.fill(COLORS["bg"])

    w2s = ctx["w2s"]
    width, height = surface.get_width(), surface.get_height()

    # Lane backdrop (static).
    for poly in ctx["lane_polylines"]:
        if len(poly) < 2:
            continue
        pygame.draw.lines(
            surface,
            COLORS["lane"],
            False,
            [w2s(x, y) for (x, y) in poly],
            1,
        )

    # Frame border.
    pygame.draw.rect(surface, COLORS["frame"],
                     pygame.Rect(0, 0, width, height), 2)

    world = ctx["world"]
    radars = ctx["radars"]
    cameras = ctx["cameras"]

    # Stable color assignment: sort radars by role_name so R1 always gets palette[0].
    radars_sorted = sorted(radars, key=radar_short_name)
    radar_colors = {r.id: radar_color_pair(r, i) for i, r in enumerate(radars_sorted)}

    # Sensor FOV wedges first (translucent fill + solid outline).
    wedge_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    for r in radars:
        rng, fov = sensor_range_fov(r)
        pts = [w2s(x, y) for (x, y) in fov_wedge_points(r, rng, fov)]
        solid_rgb, wedge_rgba = radar_colors[r.id]
        pygame.draw.polygon(wedge_surface, wedge_rgba, pts)
        # Crisp outline so FOV extent is unambiguous.
        pygame.draw.polygon(wedge_surface, (*solid_rgb, 200), pts, 2)
    surface.blit(wedge_surface, (0, 0))

    # Vehicles + pedestrians.
    try:
        actors = world.get_actors()
    except RuntimeError:
        actors = []

    actor_count = 0
    for a in actors:
        tid = a.type_id
        if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
            continue
        try:
            tr = a.get_transform()
            bbox = a.bounding_box
        except RuntimeError:
            continue
        corners = oriented_box_corners(tr, bbox.extent)
        screen_pts = [w2s(x, y) for (x, y) in corners]
        pygame.draw.polygon(surface, actor_color(a), screen_pts)
        # Heading tick.
        loc = tr.location
        yaw = math.radians(tr.rotation.yaw)
        nose = (loc.x + math.cos(yaw) * (bbox.extent.x + 0.5),
                loc.y + math.sin(yaw) * (bbox.extent.x + 0.5))
        pygame.draw.line(surface, COLORS["hud"], w2s(loc.x, loc.y), w2s(*nose), 1)
        actor_count += 1

    # Radar detection dots.
    if ctx["radar_points"] is not None:
        for cloud in ctx["radar_points"].snapshot().values():
            for (wx, wy) in cloud:
                px, py = w2s(float(wx), float(wy))
                if 0 <= px < width and 0 <= py < height:
                    surface.set_at((px, py), COLORS["radar_dot"])

    # Sensors on top.
    font = ctx["font"]
    for r in radars:
        tr = r.get_transform()
        c = w2s(tr.location.x, tr.location.y)
        yaw = math.radians(tr.rotation.yaw)
        tip = (c[0] + int(math.cos(yaw) * 10), c[1] + int(math.sin(yaw) * 10))
        left = (c[0] + int(math.cos(yaw + 2.5) * 6),
                c[1] + int(math.sin(yaw + 2.5) * 6))
        right = (c[0] + int(math.cos(yaw - 2.5) * 6),
                 c[1] + int(math.sin(yaw - 2.5) * 6))
        solid_rgb, _ = radar_colors[r.id]
        pygame.draw.polygon(surface, solid_rgb, [tip, left, right])
        short = radar_short_name(r)
        rng, fov = sensor_range_fov(r)
        label = font.render(
            f"{short} yaw={tr.rotation.yaw:+.0f}° fov={fov:.0f}° rng={rng:.0f}m",
            True, solid_rgb,
        )
        surface.blit(label, (c[0] + 8, c[1] - 14))
    for cam in cameras:
        tr = cam.get_transform()
        c = w2s(tr.location.x, tr.location.y)
        pygame.draw.rect(surface, COLORS["camera"],
                         pygame.Rect(c[0] - 3, c[1] - 3, 7, 7))
        role = cam.attributes.get("role_name", "")
        short = role.split("_")[-1] if role else f"id{cam.id}"
        label = font.render(short, True, COLORS["camera"])
        surface.blit(label, (c[0] + 6, c[1] + 6))

    # HUD.
    hud_lines = [
        hud_text,
        f"actors={actor_count}  radars={len(radars)}  cameras={len(cameras)}",
    ]
    for i, line in enumerate(hud_lines):
        img = font.render(line, True, COLORS["hud"])
        surface.blit(img, (8, 6 + i * 14))


def make_video_writer(path: Path, size, fps):
    try:
        import imageio_ffmpeg  # noqa: F401
        import imageio.v2 as imageio
    except ImportError as e:
        raise SystemExit(
            "--video requires `pip install imageio imageio-ffmpeg`"
        ) from e
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=6,
        macro_block_size=None, format="FFMPEG",
    )
    return writer


def run() -> int:
    args = parse_args()

    # Decide output mode. Auto-pick snapshots when no display is available.
    has_display = bool(os.environ.get("DISPLAY")) or sys.platform == "win32"
    if not (args.display or args.snapshots or args.video):
        if has_display:
            args.display = True
        else:
            args.snapshots = True

    out_dir = resolve_output_dir(args.out)
    print(f"[BEV] Output directory: {out_dir}")

    if args.snapshots:
        snap_dir = out_dir / "bev_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BEV] Snapshots: {snap_dir} (every {args.snapshot_every} frames)")
    else:
        snap_dir = None

    # Pygame init. Use dummy video driver when running headless and not displaying.
    if not args.display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # We don't need audio — silence ALSA spam on servers without a sound card.
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_caption("CARLA BEV Monitor")
    if args.display:
        screen = pygame.display.set_mode((args.width, args.height))
    else:
        screen = pygame.Surface((args.width, args.height))

    font = pygame.font.SysFont("monospace", 12)

    # Connect to CARLA.
    print(f"[BEV] Connecting to CARLA at {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Wait briefly for tagged sensors to appear.
    radars, cameras = [], []
    deadline = time.time() + 30.0
    while time.time() < deadline:
        radars, cameras = discover_sensors(world)
        if radars:
            break
        time.sleep(0.5)
    if not radars:
        print(
            "[BEV] No dataset_radar_* sensors found yet. "
            "Run RadarCameraSetup{4,8,12,14}.py first."
        )
    print(f"[BEV] Found {len(radars)} radars, {len(cameras)} cameras.")

    bounds = compute_view_bounds(radars, args.margin)
    print(f"[BEV] View bounds (world XY): x=[{bounds[0]:.1f},{bounds[2]:.1f}] "
          f"y=[{bounds[1]:.1f},{bounds[3]:.1f}]")
    w2s = WorldToScreen(bounds, args.width, args.height)

    lane_polylines = []
    if not args.no_roads:
        try:
            lane_polylines = build_lane_backdrop(world, bounds)
            print(f"[BEV] Backdrop: {len(lane_polylines)} lane polylines.")
        except RuntimeError as e:
            print(f"[BEV] Lane backdrop unavailable: {e}")

    radar_buffer = None
    radar_listeners = []
    if not args.no_radar_points and radars:
        radar_buffer = RadarPointBuffer(radars)
        for r in radars:
            sid = r.id

            def _make_cb(sensor=r, sensor_id=sid):
                def _cb(meas):
                    try:
                        radar_buffer.update_from_measurement(sensor_id, sensor, meas)
                    except Exception:  # noqa: BLE001
                        pass
                return _cb
            try:
                r.listen(_make_cb())
                radar_listeners.append(r)
            except RuntimeError as e:
                print(f"[BEV] Could not subscribe to radar {r.id}: {e}")

    stop_flag = {"v": False}

    def _on_sigint(_sig, _frm):
        stop_flag["v"] = True
    signal.signal(signal.SIGINT, _on_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_sigint)

    video_writer = None
    if args.video:
        video_path = out_dir / "bev.mp4"
        video_writer = make_video_writer(video_path, (args.width, args.height), args.rate)
        print(f"[BEV] Video: {video_path}")

    ctx = {
        "w2s": w2s,
        "world": world,
        "radars": radars,
        "cameras": cameras,
        "lane_polylines": lane_polylines,
        "radar_points": radar_buffer,
        "font": font,
    }

    period = 1.0 / max(args.rate, 0.1)
    started = time.time()
    last_fps_check = started
    frames = 0
    snapshot_idx = 0

    try:
        while not stop_flag["v"]:
            t0 = time.time()

            if args.display:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        stop_flag["v"] = True

            now = time.time()
            if now - last_fps_check >= 1.0:
                fps = frames / (now - last_fps_check)
                last_fps_check = now
                frames = 0
            else:
                fps = -1.0

            hud = f"t={now - started:6.1f}s  fps={fps:4.1f}" if fps >= 0 else \
                  f"t={now - started:6.1f}s"
            draw_frame(screen, ctx, hud_text=hud)

            if args.display:
                pygame.display.flip()

            if args.snapshots and (snapshot_idx % max(args.snapshot_every, 1) == 0):
                fname = snap_dir / f"bev_{snapshot_idx:06d}.png"
                pygame.image.save(screen, str(fname))
            if args.video and video_writer is not None:
                rgb = pygame.surfarray.array3d(screen)
                # pygame stores (w,h,3); imageio wants (h,w,3).
                video_writer.append_data(np.transpose(rgb, (1, 0, 2)))

            snapshot_idx += 1
            frames += 1

            if args.duration > 0 and (now - started) >= args.duration:
                break

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        for r in radar_listeners:
            try:
                r.stop()
            except RuntimeError:
                pass
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:  # noqa: BLE001
                pass
        pygame.quit()
        print("[BEV] Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
