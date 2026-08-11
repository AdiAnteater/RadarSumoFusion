"""
Derive the radar hit->actor match-margin threshold from the UNCENSORED
nearest-OBB-margin distribution of a capture.

The labeled CSV's `nearest_actor_bbox_margin_m` is overwritten with the accepted
margin on matched rows (evaluate_radar_detection_label), so it is censored at the
current 1.5/2.0 m cutoff and cannot reveal a valley. Here we recompute the true
nearest margin per detection straight from radar_data.csv + actor_frames.jsonl,
reusing the exact geometry/candidate-gating the labeler uses, with NO matching and
NO overwrite. Candidates are still gated to the 7 m candidate pre-filter, so the
signal is right-censored at 7 m only (irrelevant for a sub-3 m cutoff).

Usage:
  python tools/derive_match_threshold_uncensored.py <capture_dir> [frame_stride]

Standalone ad-hoc analyzer. The production equivalent now lives in
testing/RadarLabelingTestReport.derive_margin_threshold (rendered as a panel in
every capture's radar_labeling_summary.png, or applied via DATASET_RADAR_AUTO_MARGIN=1).
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dc)
_dc.bootstrap(str(_root / "capture" / "LabelRadarCapture.py"))

import carla  # noqa: E402
from capture.actor_frame_log import ActorFrameLogger  # noqa: E402
from capture.CaptureRadarCameraData import (  # noqa: E402
    RADAR_HORIZONTAL_FOV_DEG,
    RADAR_MAX_RANGE_M,
    actor_snapshots_for_radar_detection,
    nearest_actor_bbox_margin_m,
    radar_detection_world_location,
)
from capture.LabelRadarCapture import _detection_from_row, _transform_from_row  # noqa: E402

capture_dir = Path(sys.argv[1])
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1

BIN = 0.05
MAXM = 7.0
edges = np.arange(0.0, MAXM + BIN, BIN)
centers = 0.5 * (edges[:-1] + edges[1:])
n_bins = len(centers)

print(f"loading actor frames from {capture_dir.name} ...", flush=True)
actors_by_frame = ActorFrameLogger.load_by_frame(
    capture_dir / ActorFrameLogger.ACTOR_FRAMES_FILENAME
)
print(f"  {len(actors_by_frame):,} frames", flush=True)

hist = np.zeros(n_bins, dtype=np.int64)
hist_veh = np.zeros(n_bins, dtype=np.int64)  # nearest candidate is a vehicle
n_rows = 0
n_proc = 0
n_cand = 0

radar_csv = capture_dir / "radar_data.csv"
with radar_csv.open(newline="", encoding="utf-8") as rf:
    reader = csv.DictReader(rf)
    for row in reader:
        n_rows += 1
        frame_id = int(row["frame"])
        if stride > 1 and (frame_id % stride) != 0:
            continue
        actors = actors_by_frame.get(frame_id)
        if not actors:
            continue
        n_proc += 1
        sensor_tf = _transform_from_row(row)
        det = _detection_from_row(row)
        cands = actor_snapshots_for_radar_detection(
            sensor_tf, det, actors, None,
            max_range_m=RADAR_MAX_RANGE_M,
            horizontal_fov_deg=RADAR_HORIZONTAL_FOV_DEG,
        )
        if not cands:
            continue
        hit = radar_detection_world_location(sensor_tf, det)
        m = nearest_actor_bbox_margin_m(hit, cands, None)
        if m is None:
            continue
        n_cand += 1
        b = min(int(m / BIN), n_bins - 1)
        hist[b] += 1
        # kind of the nearest candidate (for veh-only view)
        kinds = {a.get("kind") for a in cands}
        if "vehicle" in kinds:
            hist_veh[b] += 1
        if n_cand % 50000 == 0:
            print(f"  ...{n_cand:,} candidate samples", flush=True)

tot = int(hist.sum())
print(f"\nrows in csv         : {n_rows:,}")
print(f"rows processed      : {n_proc:,}  (frame_stride={stride})")
print(f"candidate samples   : {tot:,}")

n_zero = int(hist[0])
print(f"  margin==0 bin (inside inflated OBB): {n_zero:,}  ({100*n_zero/max(tot,1):.1f}%)")

cum = np.cumsum(hist)
def pct(p):
    idx = int(np.searchsorted(cum, p / 100.0 * tot))
    return centers[min(idx, n_bins - 1)]
print("\npercentiles (uncensored nearest margin):")
for p in (50, 75, 90, 95, 99):
    print(f"  p{p:<2d} = {pct(p):.2f} m")

# --- on-body spike vs clutter ramp ---
# In a ray-cast sim a genuine vehicle hit lands ON the OBB -> margin ~ 0 (absorbed
# by the 0.75 m inflation). So the on-body population is the SPIKE near 0; the
# clutter that merely shares the beam ramps UP with margin (no far lobe, no valley).
# Treat everything past the spike as clutter; report precision = spike / accepted.
SPIKE_EDGE = 0.15
spike = int(hist[centers < SPIKE_EDGE].sum())
print(f"\non-body spike (<{SPIKE_EDGE:.2f} m): {spike:,}  ({100*spike/max(tot,1):.1f}% of candidate returns)")

# trough: minimum-density bin in [0.1, 1.0) m -> the only natural gap before ramp.
win = (centers >= 0.10) & (centers < 1.0)
idxs = np.where(win)[0]
trough_i = idxs[int(np.argmin(hist[idxs]))]
print(f"trough (min density 0.1-1.0 m): {centers[trough_i]:.2f} m  ({hist[trough_i]} / bin)")

print("\nthreshold sweep (spike=on-body, rest=clutter):")
print("   T(m)  accepted   onbody  clutter  est_precision  +clutter/+0.25m")
prev_acc = spike
for T in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
    k = int(np.searchsorted(centers, T, side="right"))
    accepted = int(hist[:k].sum())
    clutter = accepted - spike
    prec = spike / max(accepted, 1)
    dclutter = accepted - prev_acc
    print(f"  {T:4.2f}  {accepted:8,d}  {spike:7,d}  {clutter:7,d}  {100*prec:11.1f}%  {dclutter:13,d}")
    prev_acc = accepted

print("\nlow-margin detail (per 0.05 m bin):")
for i in range(0, int(2.0 / BIN)):
    bar = "#" * int(70 * hist[i] / max(hist.max(), 1))
    tag = "  <-- spike" if i == 0 else ("  <-- trough" if i == trough_i else "")
    print(f" {centers[i]:4.2f}  {int(hist[i]):8,d}  {bar}{tag}")
