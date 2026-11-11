"""Shared readers for capture folders, used by scenario_insights.py and
covisibility.py.

A capture folder (dataset/Data/sensor_capture_*) holds:
    radar_data_labeled.csv   one row per radar return, matched actor + scenario
    actor_frames.jsonl       every actor's pose and bbox, every frame
    segments.json            scenario windows (start_frame .. end_frame)
    run_meta.json            capture settings (dataset_env holds DATASET_* overrides)

No CARLA import: everything here runs offline on a laptop or the server.
"""

import glob
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    raise SystemExit("pandas is required: pip install pandas")


LABELED_CSV = "radar_data_labeled.csv"

# Object classes, same grouping as pointnet/build_frames.py.
CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
_CLASS_OF = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "van": "vehicle",
    "vehicle": "vehicle",
    "pedestrian": "pedestrian", "walker": "pedestrian",
    "bicycle": "cyclist", "motorcycle": "cyclist", "bike": "cyclist", "cyclist": "cyclist",
}

# Capture defaults (dataset/capture/CaptureRadarCameraData.py). run_meta.json
# only records env overrides, so older captures taken with other defaults need
# the values passed on the command line.
DEFAULT_RANGE_M = 35.0
DEFAULT_HFOV_DEG = 120.0
DEFAULT_VFOV_DEG = 30.0
DEFAULT_RATE_HZ = 20.0


def object_class(cls, kind=""):
    """Map a matched_actor_class / class_label (+ kind) to vehicle /
    pedestrian / cyclist, or None for anything else."""
    c = str(cls or "").strip().lower()
    if c in _CLASS_OF:
        return _CLASS_OF[c]
    k = str(kind or "").strip().lower()
    if k in ("walker", "pedestrian"):
        return "pedestrian"
    if k == "vehicle":
        return "vehicle"
    return None


def expand_captures(paths):
    """Globs -> capture folders that contain radar_data_labeled.csv."""
    out = []
    for p in paths:
        for q in sorted(glob.glob(p)) or [p]:
            d = Path(q)
            if d.is_file() and d.name == LABELED_CSV:
                d = d.parent
            if (d / LABELED_CSV).is_file():
                out.append(d)
            else:
                print(f"[skip] {d}: no {LABELED_CSV}")
    return out


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class Segments:
    """Scenario windows of one capture. lookup(frame) -> segment dict or None."""

    def __init__(self, cap):
        data = load_json(Path(cap) / "segments.json", {}) or {}
        self.segs = sorted(data.get("segments", []), key=lambda s: s["start_frame"])
        self._starts = np.array([s["start_frame"] for s in self.segs], np.int64)

    def __bool__(self):
        return bool(self.segs)

    def lookup(self, frame):
        if not self.segs:
            return None
        i = int(np.searchsorted(self._starts, frame, side="right")) - 1
        if i < 0:
            return None
        s = self.segs[i]
        return s if s["start_frame"] <= frame <= s["end_frame"] else None

    def mask(self, frames, stride=1):
        """Vectorized: True where a frame falls inside a segment (and on the
        stride). No segments file -> every frame (on the stride) is kept."""
        frames = np.asarray(frames, np.int64)
        keep = np.ones(len(frames), bool) if stride <= 1 else (frames % stride == 0)
        if not self.segs:
            return keep
        ends = np.array([s["end_frame"] for s in self.segs], np.int64)
        i = np.searchsorted(self._starts, frames, side="right") - 1
        inside = (i >= 0) & (frames <= ends[np.clip(i, 0, None)])
        return keep & inside

    def rate_hz(self):
        for s in self.segs:
            if s.get("rate_hz"):
                return float(s["rate_hz"])
        return DEFAULT_RATE_HZ


def scenario_key(seg, by="scenario"):
    """Group key for a segment: 's03 Heavy Demand' (by scenario) or with the
    segment id appended (by segment)."""
    if seg is None:
        return None
    sid = int(seg.get("scenario", -1))
    name = seg.get("scenario_name", "") or ""
    key = f"s{sid:02d} {name}".strip()
    if by == "segment":
        key += f" #{seg.get('segment_id', 0)}"
    return key


def run_env(cap):
    meta = load_json(Path(cap) / "run_meta.json", {}) or {}
    return meta.get("dataset_env", {}) or {}


def radar_params(cap, range_m=None, hfov_deg=None, vfov_deg=None):
    """(range, hfov, vfov) for a capture: CLI value > run_meta env > default."""
    env = run_env(cap)

    def pick(cli, key, default):
        if cli is not None:
            return float(cli)
        try:
            return float(env.get(key, "") or default)
        except ValueError:
            return default
    return (pick(range_m, "DATASET_RADAR_RANGE_M", DEFAULT_RANGE_M),
            pick(hfov_deg, "DATASET_RADAR_HORIZONTAL_FOV_DEG", DEFAULT_HFOV_DEG),
            pick(vfov_deg, "DATASET_RADAR_VERTICAL_FOV_DEG", DEFAULT_VFOV_DEG))


def csv_columns(cap):
    with open(Path(cap) / LABELED_CSV, "r", encoding="utf-8", errors="replace") as f:
        return f.readline().strip().split(",")


_STR_COLS = ("sensor_label", "matched_actor_class", "matched_actor_kind",
             "matched_actor_id", "visible")


def iter_labeled(cap, cols, chunksize=1_000_000, max_rows=0):
    """Chunks of radar_data_labeled.csv with the requested columns that exist.

    Uses pyarrow's streaming CSV reader when installed (several times faster
    on multi-GB files: pip install pyarrow), otherwise pandas in chunks. Both
    yield pandas DataFrames with the id / label columns as strings.
    """
    have = set(csv_columns(cap))
    use = [c for c in cols if c in have]
    str_cols = {c: str for c in _STR_COLS if c in use}
    seen = 0
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError:
        pa = None
    if pa is not None and not env_flag("CAPTURE_IO_NO_PYARROW"):
        reader = pacsv.open_csv(
            str(Path(cap) / LABELED_CSV),
            read_options=pacsv.ReadOptions(block_size=64 << 20),
            convert_options=pacsv.ConvertOptions(
                include_columns=use,
                column_types={c: pa.string() for c in str_cols},
                strings_can_be_null=True))
        for batch in reader:
            ch = batch.to_pandas()
            if max_rows:
                if seen >= max_rows:
                    return
                ch = ch.iloc[:max_rows - seen]
            seen += len(ch)
            yield ch
        return
    for ch in pd.read_csv(Path(cap) / LABELED_CSV, usecols=use, chunksize=chunksize,
                          dtype=str_cols, low_memory=False):
        if max_rows:
            if seen >= max_rows:
                return
            ch = ch.iloc[:max_rows - seen]
        seen += len(ch)
        yield ch


def radar_poses(cap, max_rows=3_000_000):
    """{label: dict(x, y, z, yaw, pitch, roll)} from the sensor columns of the
    labeled CSV (the pose each radar actually had in that capture)."""
    cols = ["sensor_label", "sensor_world_x_m", "sensor_world_y_m", "sensor_world_z_m",
            "sensor_yaw_deg", "sensor_pitch_deg", "sensor_roll_deg"]
    expected = None
    meta = load_json(Path(cap) / "run_meta.json", {}) or {}
    if meta.get("radar_count"):
        expected = int(meta["radar_count"])
    poses = {}
    for ch in iter_labeled(cap, cols, chunksize=500_000, max_rows=max_rows):
        first = ch.drop_duplicates("sensor_label")
        for _, r in first.iterrows():
            lab = str(r["sensor_label"])
            if lab in poses or not lab.startswith("R"):
                continue
            poses[lab] = dict(x=float(r["sensor_world_x_m"]), y=float(r["sensor_world_y_m"]),
                              z=float(r["sensor_world_z_m"]), yaw=float(r["sensor_yaw_deg"]),
                              pitch=float(r["sensor_pitch_deg"]),
                              roll=float(r["sensor_roll_deg"]))
        if expected and len(poses) >= expected:
            break
    return dict(sorted(poses.items(), key=lambda kv: _natkey(kv[0])))


def _natkey(s):
    digits = "".join(ch for ch in s if ch.isdigit())
    return (s.rstrip("0123456789"), int(digits) if digits else 0)


def rig_zone(poses, margin=0.0):
    """(xmin, xmax, ymin, ymax): bounding box of the radar mounts. Objects
    inside it are 'on the monitored stretch' for that capture's rig."""
    xs = [p["x"] for p in poses.values()]
    ys = [p["y"] for p in poses.values()]
    return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def in_zone(zone, x, y):
    return zone[0] <= x <= zone[1] and zone[2] <= y <= zone[3]


def iter_actor_frames(cap, frame_ok=None):
    """(frame, [actor dicts]) from actor_frames.jsonl. frame_ok(frame) -> bool
    skips a line before its actors are parsed (the file is large)."""
    path = Path(cap) / "actor_frames.jsonl"
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if frame_ok is not None:
                # cheap frame peek: lines start with {"frame":12345,
                head = line[:40]
                i = head.find('"frame":')
                if i >= 0:
                    j = i + 8
                    k = j
                    while k < len(head) and (head[k].isdigit() or head[k] == " "):
                        k += 1
                    try:
                        if not frame_ok(int(head[j:k])):
                            continue
                    except ValueError:
                        pass
            try:
                d = json.loads(line)
            except ValueError:
                continue
            yield int(d["frame"]), d.get("actors", [])


def actor_class(a):
    return object_class(a.get("class_label"), a.get("kind") or a.get("type_id", "").split(".")[0])


def jensen_shannon(p, q):
    """Jensen-Shannon DISTANCE (sqrt of the base-2 divergence), in [0, 1]."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def cache_path(cap, tool, **sig):
    """<capture>/_analysis_cache/<tool>_<settings>.pkl"""
    tag = "_".join(f"{k}{v}" for k, v in sorted(sig.items()) if v not in (None, "", 0, False))
    d = Path(cap) / "_analysis_cache"
    return d / f"{tool}{'_' + tag if tag else ''}.pkl"


def load_cache(path, cap):
    """Cached per-capture result, or None if missing / older than the CSV."""
    import pickle
    try:
        if path.stat().st_mtime < (Path(cap) / LABELED_CSV).stat().st_mtime:
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
        return None


def save_cache(path, obj):
    import pickle
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
    except OSError as e:
        print(f"  note: cache not written ({e})")


def write_csv(path, rows, header):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def try_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # noqa: BLE001
        return None


def out_dir(arg, captures, name):
    d = Path(arg) if arg else (captures[0] / name if len(captures) == 1 else Path(name))
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
