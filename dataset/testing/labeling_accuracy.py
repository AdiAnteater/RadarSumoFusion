"""
Pure helpers for tools/analyze_labeling_accuracy.py (no CARLA server needed).

- live gate constants, read from CaptureRadarCameraData.py source via ast (the
  module itself imports carla and spins up capture machinery)
- vectorised OBB geometry that mirrors CaptureRadarCameraData._obb_margin_from_local
  and actor_snapshot_in_sensor_fov
- background-corrected precision / recall sweep over the nearest-margin histogram
- Tier-A static taxonomy and an optional lane-zone raster from the map's .xodr
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np

CAPTURE_MODULE = Path(__file__).resolve().parents[1] / "capture" / "CaptureRadarCameraData.py"

GATE_CONSTANT_NAMES = (
    "RADAR_HIT_MATCH_MAX_MARGIN_M",
    "RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M",
    "BBOX_MATCH_EXTENT_INFLATION_M",
    "RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M",
    "GROUND_REJECT_MARGIN_M",
    "GROUND_REJECT_CLEARANCE_M",
    "RADAR_MAX_RANGE_M",
    "RADAR_HORIZONTAL_FOV_DEG",
    "ROAD_SURFACE_MAX_Z_M",
    "STRUCTURE_MIN_Z_M",
)


def load_gate_constants(path: Path = CAPTURE_MODULE) -> dict[str, float]:
    """Module-level numeric constants from the capture script (literal assignments only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in GATE_CONSTANT_NAMES:
            try:
                found[target.id] = float(ast.literal_eval(node.value))
            except ValueError:
                pass
    missing = [n for n in GATE_CONSTANT_NAMES if n not in found]
    if missing:
        raise RuntimeError(f"constants not found as literals in {path.name}: {missing}")
    return found


# ---------------------------------------------------------------- actor kinds

KINDS = ("car", "bicycle", "pedestrian", "truck")


def actor_kind(kind: str, type_id: str, class_label: str = "") -> str:
    """car / bicycle / pedestrian / truck. type_id is the reliable key for bicycles."""
    type_id = (type_id or "").lower()
    if (kind or "").lower() in ("pedestrian", "walker") or type_id.startswith("walker."):
        return "pedestrian"
    if "crossbike" in type_id or (class_label or "").lower() == "bicycle":
        return "bicycle"
    if "carlacola" in type_id or (class_label or "").lower() == "truck":
        return "truck"
    return "car"


# ---------------------------------------------------------------- geometry

def carla_rotation_matrix(pitch_deg, yaw_deg, roll_deg) -> np.ndarray:
    """Local->world rotation, identical to LibCarla geom::Transform::TransformPoint."""
    p, y, r = (math.radians(float(v)) for v in (pitch_deg, yaw_deg, roll_deg))
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ])


def actor_frame_arrays(actors: list[dict]) -> dict[str, np.ndarray]:
    """Per-frame actor arrays: OBB world centre, world->box rotation, extents, ids, kinds, origin xy."""
    ids, kinds, centers, rinv, ext, origin = [], [], [], [], [], []
    for a in actors:
        bbox = a.get("bbox")
        loc, rot = a.get("location"), a.get("rotation")
        if not bbox or not loc or not rot:
            continue
        r_actor = carla_rotation_matrix(rot["pitch"], rot["yaw"], rot["roll"])
        bl = bbox["location"]
        c = np.array([loc["x"], loc["y"], loc["z"]]) + r_actor @ np.array([bl["x"], bl["y"], bl["z"]])
        r_inv = r_actor.T
        br = bbox.get("rotation")
        if br:
            r_inv = carla_rotation_matrix(br["pitch"], br["yaw"], br["roll"]).T @ r_inv
        ids.append(int(a["id"]))
        kinds.append(actor_kind(a.get("kind", ""), a.get("type_id", ""), a.get("class_label", "")))
        centers.append(c)
        rinv.append(r_inv)
        e = bbox["extent"]
        ext.append([e["x"], e["y"], e["z"]])
        origin.append([loc["x"], loc["y"]])
    return {
        "ids": np.array(ids, dtype=np.int64),
        "kinds": np.array(kinds, dtype=object),
        "centers": np.array(centers, dtype=float).reshape(-1, 3),
        "rinv": np.array(rinv, dtype=float).reshape(-1, 3, 3),
        "extents": np.array(ext, dtype=float).reshape(-1, 3),
        "origin_xy": np.array(origin, dtype=float).reshape(-1, 2),
    }


def obb_margins(points: np.ndarray, fa: dict, inflation: float, ground_clear: float,
                ground_margin: float) -> np.ndarray:
    """(N, A) labeler margins: 0 inside, ground_margin for below-ground-plane hits,
    else distance to the inflated box. Mirrors _obb_margin_from_local."""
    d = points[:, None, :] - fa["centers"][None, :, :]
    local = np.einsum("ajk,nak->naj", fa["rinv"], d)
    ax = np.abs(local)
    e = fa["extents"][None, :, :]
    inside_true = np.all(ax <= e, axis=2)
    ground = local[:, :, 2] < -e[:, :, 2] + ground_clear
    out = np.maximum(ax - (e + inflation), 0.0)
    margin = np.sqrt(np.sum(out * out, axis=2))
    margin = np.where(ground, ground_margin, margin)
    return np.where(inside_true, 0.0, margin)


def in_sensor_fov(sensor_xy: np.ndarray, sensor_yaw_deg: np.ndarray, target_xy: np.ndarray,
                  max_range_m: float, hfov_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised actor_snapshot_in_sensor_fov: planar range and |bearing - yaw| <= hfov/2.
    Returns (in_fov, planar_range, in_angle)."""
    dx = target_xy[..., 0] - sensor_xy[..., 0]
    dy = target_xy[..., 1] - sensor_xy[..., 1]
    rng = np.hypot(dx, dy)
    bearing = np.degrees(np.arctan2(dy, dx))
    delta = np.abs((bearing - sensor_yaw_deg + 180.0) % 360.0 - 180.0)
    in_angle = delta <= hfov_deg * 0.5
    return in_angle & (rng <= max_range_m), rng, in_angle


# ---------------------------------------------------------------- background fit / sweep

FIT_VARIANTS = (
    ("linear 0.25-2.0 m", 1, 0.25, 2.0),
    ("linear 0.50-2.0 m", 1, 0.50, 2.0),
    ("quadratic 0.25-2.0 m", 2, 0.25, 2.0),
    ("quadratic 0.50-2.0 m", 2, 0.50, 2.0),
)
PRIMARY_FIT = FIT_VARIANTS[0][0]


def background_sweep(edges: np.ndarray, hist: np.ndarray) -> dict:
    """Fit the clutter ramp per FIT_VARIANTS, extrapolate to 0, derive on-body counts,
    precision(T) and recall(T) at each upper bin edge T."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    cum_all = np.cumsum(hist).astype(float)
    variants = {}
    for name, deg, lo, hi in FIT_VARIANTS:
        m = (centers >= lo) & (centers <= hi)
        coef = np.polyfit(centers[m], hist[m].astype(float), deg)
        bg = np.clip(np.polyval(coef, centers), 0.0, None)
        resid = hist[m] - np.polyval(coef, centers[m])
        ss_tot = float(np.sum((hist[m] - hist[m].mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        onbody = np.clip(hist - bg, 0.0, None)
        cum_on = np.cumsum(onbody)
        total_on = cum_on[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(cum_all > 0, cum_on / cum_all, np.nan)
            recall = cum_on / total_on if total_on > 0 else np.full_like(cum_on, np.nan)
        slope_at_mid = float(np.polyval(np.polyder(coef), 0.5 * (lo + hi)))
        variants[name] = {
            "coef": coef, "background": bg, "onbody": onbody, "precision": precision,
            "recall": recall, "total_onbody": float(total_on), "r2": r2,
            "bg_at_zero": float(np.polyval(coef, 0.0)), "slope_mid": slope_at_mid,
        }
    stack_p = np.vstack([v["precision"] for v in variants.values()])
    stack_r = np.vstack([v["recall"] for v in variants.values()])
    return {
        "edges": edges, "centers": centers, "hist": hist, "cum_all": cum_all,
        "variants": variants, "primary": variants[PRIMARY_FIT],
        "precision_lo": np.nanmin(stack_p, axis=0), "precision_hi": np.nanmax(stack_p, axis=0),
        "recall_lo": np.nanmin(stack_r, axis=0), "recall_hi": np.nanmax(stack_r, axis=0),
    }


def fit_warnings(sweep: dict, t_idx: int, spread_tol: float = 0.02) -> list[str]:
    """Human-readable instability flags for the plot annotation."""
    warns = []
    for name, v in sweep["variants"].items():
        if v["slope_mid"] <= 0:
            warns.append(f"{name}: ramp slope <= 0 (not a rising clutter ramp)")
        if v["bg_at_zero"] < 0:
            warns.append(f"{name}: ramp extrapolates below 0 at T=0 (clipped)")
        if not (v["r2"] >= 0.5):
            warns.append(f"{name}: poor fit R^2={v['r2']:.2f}")
    p_spread = sweep["precision_hi"][t_idx] - sweep["precision_lo"][t_idx]
    r_spread = sweep["recall_hi"][t_idx] - sweep["recall_lo"][t_idx]
    if p_spread > spread_tol or r_spread > spread_tol:
        warns.append(
            f"fit-sensitive at gate: precision spread {100 * p_spread:.1f} pp, "
            f"recall spread {100 * r_spread:.1f} pp across fits"
        )
    return warns


# ---------------------------------------------------------------- static taxonomy (Tier A)

STATIC_CATEGORIES = (
    "actor: car",
    "actor: bicycle",
    "actor: pedestrian",
    "actor: truck",
    "near-actor ground",
    "near-actor off-body",
    "ground",
    "low static (0.25-1.0 m)",
    "mid static (1.0-4.5 m)",
    "high static (>=4.5 m)",
)
GROUND_CATEGORY_IDS = (4, 6)
KIND_TO_CATEGORY = {"car": 0, "bicycle": 1, "pedestrian": 2, "truck": 3}


def static_category(matched_kind_code: np.ndarray, had_candidates: np.ndarray, z: np.ndarray,
                    road_max_z: float, structure_min_z: float) -> np.ndarray:
    """Category index per row. matched_kind_code: -1 unmatched, else KIND_TO_CATEGORY value."""
    cat = np.select(
        [
            matched_kind_code >= 0,
            had_candidates & (z <= road_max_z),
            had_candidates,
            z <= road_max_z,
            z < 1.0,
            z < structure_min_z,
        ],
        [matched_kind_code, 4, 5, 6, 7, 8],
        default=9,
    )
    return cat.astype(np.int8)


# ---------------------------------------------------------------- lane zones from .xodr

ZONES = ("carriageway", "sidewalk", "off-network")
_CARRIAGEWAY_LANE_TYPES = {"Driving", "Shoulder", "Median", "Parking", "Bidirectional",
                           "Biking", "Stop", "Border", "Restricted"}


class ZoneRaster:
    """Lane-type raster built offline from the map .xodr via carla.Map (no server).

    get_waypoint(project_to_road=False) leaves 0.3-1.7 m unmapped seams between
    adjacent lanes on Town10HD, so unmapped cells within ``seam_fill_m`` of a mapped
    cell inherit its zone; farther cells are off-network.
    """

    def __init__(self, xodr_path: Path, map_name: str, bounds: tuple[float, float, float, float],
                 res_m: float = 0.25, seam_fill_m: float = 1.0):
        import carla  # client lib only; carla.Map parses OpenDRIVE locally

        self.x0, self.x1, self.y0, self.y1 = bounds
        self.res = res_m
        cmap = carla.Map(map_name, Path(xodr_path).read_text(encoding="utf-8"))
        xs = np.arange(self.x0, self.x1, res_m) + res_m / 2
        ys = np.arange(self.y0, self.y1, res_m) + res_m / 2
        grid = np.full((len(xs), len(ys)), -1, dtype=np.int8)
        any_lane = carla.LaneType.Any
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                wp = cmap.get_waypoint(carla.Location(float(x), float(y), 0.0),
                                       project_to_road=False, lane_type=any_lane)
                if wp is None:
                    continue
                lt = str(wp.lane_type)
                grid[i, j] = 1 if lt == "Sidewalk" else (0 if lt in _CARRIAGEWAY_LANE_TYPES else 2)
        self.unmapped_frac_before_fill = float(np.mean(grid < 0))
        for _ in range(int(round(seam_fill_m / res_m))):
            holes = grid < 0
            if not holes.any():
                break
            padded = np.pad(grid, 1, constant_values=-1)
            fill = np.full_like(grid, -1)
            # carriageway wins ties over sidewalk (curb seams sit next to both)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = padded[1 + di:1 + di + grid.shape[0], 1 + dj:1 + dj + grid.shape[1]]
                    take = (nb >= 0) & ((fill < 0) | (nb < fill))
                    fill = np.where(take, nb, fill)
            grid = np.where(holes, fill, grid)
        grid[grid < 0] = 2
        self.grid = grid

    def lookup(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        i = np.floor((x - self.x0) / self.res).astype(np.int64)
        j = np.floor((y - self.y0) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.grid.shape[0]) & (j >= 0) & (j < self.grid.shape[1])
        out = np.full(x.shape, 2, dtype=np.int8)
        out[ok] = self.grid[i[ok], j[ok]]
        return out


def load_radar_extrinsics(capture_dir: Path) -> list[dict]:
    """radar_extrinsics.json (fallback .csv), sorted by sensor_label R1..Rn."""
    jp = capture_dir / "radar_extrinsics.json"
    if jp.is_file():
        rows = json.loads(jp.read_text(encoding="utf-8"))
    else:
        import csv
        with (capture_dir / "radar_extrinsics.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    radars = [{"label": r["sensor_label"], "x": float(r["x"]), "y": float(r["y"]),
               "z": float(r["z"]), "yaw": float(r["yaw"])} for r in rows]
    return sorted(radars, key=lambda r: int(r["label"].lstrip("Rr") or 0))
