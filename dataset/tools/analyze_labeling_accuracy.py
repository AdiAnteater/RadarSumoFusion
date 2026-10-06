"""
Read-only radar labeling accuracy report for one labeled capture.

Replaces the misleading "precision vs match threshold" QA curve (fixed spike
numerator / accepted count, which must fall as T widens) with:

  1. background-corrected precision / recall vs match threshold T, from the
     nearest-OBB-margin histogram of candidate returns with a fitted clutter ramp
  2. actor-frame recall: (frame, radar, actor) triples with the actor in the
     radar's FOV; found = >=1 return labeled with that actor, illuminated = >=1
     return inside that actor's labeler box (margin 0, ground rejection applied)
  3. Tier-A static taxonomy of every return (CSV only), with an optional
     carriageway / sidewalk / off-network split from the map .xodr

Streams radar_data_labeled.csv in chunks; nothing is re-labeled or re-captured.

Usage:
  python tools/analyze_labeling_accuracy.py <capture_dir> [--chunksize N] [--max-rows N] [--no-zones]

Writes into <capture_dir>/radar_labeling_qa/:
  threshold_sweep.csv, labeling_accuracy.png, actor_frame_recall.csv,
  static_breakdown.csv, static_breakdown.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from testing.labeling_accuracy import (  # noqa: E402
    GROUND_CATEGORY_IDS,
    KIND_TO_CATEGORY,
    PRIMARY_FIT,
    STATIC_CATEGORIES,
    ZONES,
    ZoneRaster,
    actor_frame_arrays,
    actor_kind,
    background_sweep,
    fit_warnings,
    in_sensor_fov,
    load_gate_constants,
    load_radar_extrinsics,
    obb_margins,
    static_category,
)

LABELED_CSV = "radar_data_labeled.csv"
QA_DIR = "radar_labeling_qa"
XODR_DEFAULT = Path(__file__).resolve().parents[4] / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive"

USECOLS = [
    "sensor_label", "frame", "matched_actor_id", "matched_actor_type_id",
    "had_actor_candidates", "nearest_actor_bbox_margin_m",
    "matched_actor_bbox_margin_m", "hit_world_x_m", "hit_world_y_m", "hit_world_z_m",
]
DTYPES = {
    "sensor_label": "category", "frame": np.int32, "matched_actor_id": np.float64,
    "matched_actor_type_id": str, "had_actor_candidates": np.int8,
    "nearest_actor_bbox_margin_m": np.float64, "matched_actor_bbox_margin_m": np.float64,
    "hit_world_x_m": np.float64, "hit_world_y_m": np.float64, "hit_world_z_m": np.float64,
}

BIN_W = 0.025
T_MAX = 2.0
SPIKE_EDGE_M = 0.15
RANGE_BINS = (0.0, 10.0, 20.0, 30.0, 35.0, np.inf)
RANGE_LABELS = ("0-10 m", "10-20 m", "20-30 m", "30-35 m", ">35 m")
ACTOR_KEY = 100_000
RIG_PREFILTER_M = 95.0  # radar-to-rig-centre (~33 m) + effective ray reach (~55 m) + slack


def triple_key(frame, radar_idx, actor_id):
    return (np.asarray(frame, np.int64) * 16 + np.asarray(radar_idx, np.int64)) * ACTOR_KEY + np.asarray(actor_id, np.int64)


def load_segment_frames(capture_dir: Path) -> tuple[int, int]:
    segs = json.loads((capture_dir / "segments.json").read_text(encoding="utf-8"))["segments"]
    return min(int(s["start_frame"]) for s in segs), max(int(s["end_frame"]) for s in segs)


def load_actor_frames(path: Path, rig_center: np.ndarray) -> dict[int, dict]:
    frames = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fa = actor_frame_arrays(d["actors"])
            near = np.hypot(*(fa["origin_xy"] - rig_center).T) <= RIG_PREFILTER_M
            frames[int(d["frame"])] = {k: v[near] for k, v in fa.items()}
    return frames


def build_denominator(actor_frames, radars, seg_lo, seg_hi, max_range, hfov):
    """Every (frame, radar, actor) in the recorded segment with FOV flags and planar range."""
    fr, aid, kind, oxy = [], [], [], []
    for f, fa in actor_frames.items():
        if seg_lo <= f <= seg_hi and len(fa["ids"]):
            fr.append(np.full(len(fa["ids"]), f, np.int64))
            aid.append(fa["ids"])
            kind.append(fa["kinds"])
            oxy.append(fa["origin_xy"])
    fr, aid, kind, oxy = map(np.concatenate, (fr, aid, kind, oxy))
    n_r = len(radars)
    sxy = np.array([[r["x"], r["y"]] for r in radars])
    syaw = np.array([r["yaw"] for r in radars])
    in_fov, rng, in_angle = in_sensor_fov(sxy[None, :, :], syaw[None, :], oxy[:, None, :], max_range, hfov)
    ridx = np.broadcast_to(np.arange(n_r)[None, :], in_fov.shape)
    return pd.DataFrame({
        "key": triple_key(np.repeat(fr, n_r), ridx.ravel(), np.repeat(aid, n_r)),
        "frame": np.repeat(fr, n_r),
        "radar": np.array([r["label"] for r in radars])[ridx.ravel()],
        "actor_id": np.repeat(aid, n_r),
        "kind": np.repeat(kind, n_r),
        "range_m": rng.ravel(),
        "in_angle": in_angle.ravel(),
        "in_fov": in_fov.ravel(),
    })


class Accumulator:
    def __init__(self, radar_labels, gc, zone_raster):
        self.radar_idx = {lab: i for i, lab in enumerate(radar_labels)}
        self.n_r = len(radar_labels)
        self.gc = gc
        self.zones = zone_raster
        self.edges = np.round(np.arange(0.0, T_MAX + BIN_W / 2, BIN_W), 6)
        self.hist = np.zeros(len(self.edges) - 1, np.int64)
        self.n = dict(rows=0, rows_outside_segment=0, with_candidates=0, ground_reject=0,
                      cand_gt_tmax=0, cand_nan=0, spike_lt_015=0, matched=0,
                      matched_lt_015=0, matched_no_actor_frame=0, check_pairs=0, check_agree=0,
                      failed_match=0, failed_inside_box=0, rows_no_actor_frame=0)
        self.static = np.zeros((len(STATIC_CATEGORIES), self.n_r), np.int64)
        self.static_zone = np.zeros((len(STATIC_CATEGORIES), len(ZONES), self.n_r), np.int64)
        self.found_keys: list[np.ndarray] = []
        self.illum_keys: list[np.ndarray] = []
        self.kind_code_cache: dict[str, int] = {}

    def _kind_codes(self, type_ids: pd.Series) -> np.ndarray:
        for t in type_ids.dropna().unique():
            if t not in self.kind_code_cache:
                self.kind_code_cache[t] = KIND_TO_CATEGORY[actor_kind("", t)]
        return type_ids.map(self.kind_code_cache).fillna(-1).to_numpy().astype(np.int8)

    def add(self, df: pd.DataFrame, actor_frames: dict, seg_lo: int, seg_hi: int):
        gc = self.gc
        n = len(df)
        self.n["rows"] += n
        frame = df["frame"].to_numpy()
        self.n["rows_outside_segment"] += int(np.sum((frame < seg_lo) | (frame > seg_hi)))
        radar = df["sensor_label"].map(self.radar_idx).astype(np.int64).to_numpy()
        had = df["had_actor_candidates"].to_numpy() == 1
        margin = df["nearest_actor_bbox_margin_m"].to_numpy()
        mid = df["matched_actor_id"].to_numpy()
        matched = ~np.isnan(mid)
        hx, hy, hz = (df[c].to_numpy() for c in ("hit_world_x_m", "hit_world_y_m", "hit_world_z_m"))

        # 1. margin histogram of candidate returns
        cm = margin[had]
        self.n["with_candidates"] += int(had.sum())
        self.n["cand_nan"] += int(np.isnan(cm).sum())
        gr = np.abs(cm - gc["GROUND_REJECT_MARGIN_M"]) < 1e-6
        self.n["ground_reject"] += int(gr.sum())
        inrange = (cm >= 0) & (cm <= T_MAX) & ~gr
        self.n["cand_gt_tmax"] += int(np.sum((cm > T_MAX) & ~gr))
        bi = np.clip(np.searchsorted(self.edges, cm[inrange], side="left") - 1, 0, len(self.hist) - 1)
        self.hist += np.bincount(bi, minlength=len(self.hist))
        self.n["spike_lt_015"] += int(np.sum(cm < SPIKE_EDGE_M))
        self.n["matched"] += int(matched.sum())
        self.n["matched_lt_015"] += int(np.sum(matched & (margin < SPIKE_EDGE_M)))
        self.n["failed_match"] += int(np.sum(had & ~matched))

        # 4. static taxonomy
        kind_code = np.where(matched, self._kind_codes(df["matched_actor_type_id"]), -1)
        cat = static_category(kind_code, had, hz, gc["ROAD_SURFACE_MAX_Z_M"], gc["STRUCTURE_MIN_Z_M"])
        np.add.at(self.static, (cat, radar), 1)
        if self.zones is not None:
            g = np.isin(cat, GROUND_CATEGORY_IDS)
            z = self.zones.lookup(hx[g], hy[g])
            np.add.at(self.static_zone, (cat[g], z, radar[g]), 1)

        # 3. found triples
        if matched.any():
            self.found_keys.append(np.unique(triple_key(frame[matched], radar[matched], mid[matched].astype(np.int64))))

        # 3. illuminated triples + geometry cross-check against the labeler's own margins
        order = np.argsort(frame, kind="stable")
        uf, starts = np.unique(frame[order], return_index=True)
        ends = np.append(starts[1:], n)
        pts_all = np.column_stack([hx, hy, hz])
        mmargin = df["matched_actor_bbox_margin_m"].to_numpy()
        keys = []
        for f, s, e in zip(uf, starts, ends):
            rows = order[s:e]
            fa = actor_frames.get(int(f))
            if fa is None:
                self.n["rows_no_actor_frame"] += len(rows)
            if fa is None or not len(fa["ids"]):
                self.n["matched_no_actor_frame"] += int(matched[rows].sum())
                continue
            m = obb_margins(pts_all[rows], fa, gc["BBOX_MATCH_EXTENT_INFLATION_M"],
                            gc["GROUND_REJECT_CLEARANCE_M"], gc["GROUND_REJECT_MARGIN_M"])
            inside = m <= 0.0
            ri, ai = np.nonzero(inside)
            if len(ri):
                keys.append(triple_key(f, radar[rows[ri]], fa["ids"][ai]))
            fm = had[rows] & ~matched[rows]
            self.n["failed_inside_box"] += int(np.sum(inside[fm].any(axis=1)))
            mr = np.nonzero(matched[rows])[0]
            if len(mr):
                col = {a: j for j, a in enumerate(fa["ids"])}
                js = np.array([col.get(int(a), -1) for a in mid[rows[mr]]])
                ok = js >= 0
                self.n["matched_no_actor_frame"] += int(np.sum(~ok))
                mine = m[mr[ok], js[ok]]
                theirs = mmargin[rows[mr[ok]]]
                self.n["check_pairs"] += int(ok.sum())
                self.n["check_agree"] += int(np.sum(np.abs(mine - theirs) <= 2e-3))
        if keys:
            self.illum_keys.append(np.unique(np.concatenate(keys)))


def recall_table(den: pd.DataFrame, found: np.ndarray, illum: np.ndarray) -> pd.DataFrame:
    den = den.copy()
    den["found"] = np.isin(den["key"].to_numpy(), found)
    den["illum"] = np.isin(den["key"].to_numpy(), illum)
    den["range_bin"] = pd.cut(den["range_m"], RANGE_BINS, labels=RANGE_LABELS, right=False)

    def summarise(g: pd.DataFrame, group_type: str, group: str) -> dict:
        fov = g[g["in_fov"]]
        n_fov = len(fov)
        n_ill = int(fov["illum"].sum())
        n_fi = int((fov["found"] & fov["illum"]).sum())
        n_found = int(fov["found"].sum())
        if group_type == "range" and group == ">35 m":
            ang = g[g["in_angle"]]
            n_ill = int(ang["illum"].sum())
            n_fi = int((ang["found"] & ang["illum"]).sum())
            n_found = int(ang["found"].sum())
        return {
            "group_type": group_type, "group": group, "in_fov": n_fov,
            "illuminated": n_ill, "found": n_found, "found_and_illuminated": n_fi,
            "found_not_illuminated": n_found - n_fi,
            "labeler_recall": n_fi / n_ill if n_ill else np.nan,
            "system_recall": int(fov["found"].sum()) / n_fov if n_fov else np.nan,
        }

    out = [summarise(den, "overall", "all (in FOV)")]
    for lab in RANGE_LABELS:
        g = den[den["range_bin"] == lab]
        if len(g) and (g["in_fov"].any() or (g["in_angle"] & (g["illum"] | g["found"])).any()):
            out.append(summarise(g, "range", lab))
    for lab in sorted(den["radar"].unique(), key=lambda s: int(s[1:])):
        out.append(summarise(den[den["radar"] == lab], "radar", lab))
    for k in ("car", "bicycle", "pedestrian", "truck"):
        g = den[den["kind"] == k]
        if len(g):
            out.append(summarise(g, "kind", k))
    return pd.DataFrame(out)


def write_threshold_sweep(path: Path, sweep: dict, spike: int):
    p = sweep["primary"]
    edges_hi = sweep["edges"][1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        old = np.where(edges_hi >= SPIKE_EDGE_M - 1e-9, spike / sweep["cum_all"], np.nan)
    cols = {
        "T_m": edges_hi,
        "bin_count": sweep["hist"],
        "candidates_le_T": sweep["cum_all"].astype(np.int64),
        "background_bin": p["background"],
        "onbody_bin": p["onbody"],
        "onbody_le_T": np.cumsum(p["onbody"]),
        "precision_bc": p["precision"],
        "precision_bc_lo": sweep["precision_lo"],
        "precision_bc_hi": sweep["precision_hi"],
        "recall_bc": p["recall"],
        "recall_bc_lo": sweep["recall_lo"],
        "recall_bc_hi": sweep["recall_hi"],
        "precision_old_fixed_numerator_lower_bound": old,
    }
    for name, v in sweep["variants"].items():
        tag = name.replace(" ", "_").replace("-", "_").replace(".", "p")
        cols[f"precision_{tag}"] = v["precision"]
        cols[f"recall_{tag}"] = v["recall"]
    pd.DataFrame(cols).to_csv(path, index=False, float_format="%.6f")
    return old


def pct(x):
    return f"{100 * x:.2f}%" if np.isfinite(x) else "n/a"


def plot_accuracy(path, sweep, old, head, warns, rec, gc, n, seg):
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
    T = sweep["edges"][1:]
    p = sweep["primary"]

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lines = [
        "Radar labeling accuracy (read-only analysis)",
        "",
        f"Precision @ {head['gate']:.2f} m, background-corrected: {pct(head['precision_bc'])}",
        f"      fit band {pct(head['precision_bc_lo'])} - {pct(head['precision_bc_hi'])}",
        f"Precision @ {head['gate']:.2f} m, LOWER BOUND: {pct(head['precision_lb'])}",
        "      (margin<0.15 m & accepted) / accepted",
        f"Recall @ {head['gate']:.2f} m, background-corrected: {pct(head['recall_bc'])}",
        f"      fit band {pct(head['recall_bc_lo'])} - {pct(head['recall_bc_hi'])}",
        f"Labeler recall (found / illuminated): {pct(head['labeler_recall'])}",
        f"System recall (found / in FOV): {pct(head['system_recall'])}",
        "",
        f"Gate: primary {gc['RADAR_HIT_MATCH_MAX_MARGIN_M']} m, single-cand "
        f"{gc['RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M']} m,",
        f"  OBB inflation {gc['BBOX_MATCH_EXTENT_INFLATION_M']} m, candidate pre-filter "
        f"{gc['RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M']} m",
        f"FOV {gc['RADAR_HORIZONTAL_FOV_DEG']:.0f} deg / {gc['RADAR_MAX_RANGE_M']:.0f} m, "
        f"frames {seg[0]}-{seg[1]}",
        f"Ground-reject placeholder ({gc['GROUND_REJECT_MARGIN_M']:.0f} m) excluded: {n['ground_reject']:,}",
        f"Background fit: {PRIMARY_FIT} (band = 4 fits)",
    ]
    if head["precision_bc"] < head["precision_lb"]:
        lines += ["", "Note: corrected precision < 'lower bound' because the",
                  "extrapolated ramp assigns some <0.15 m spike returns to",
                  "clutter; the bound assumes the spike is 100% on-body."]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    if warns:
        ax.text(0.0, 0.02, "FIT WARNINGS:\n" + "\n".join(warns), va="bottom", fontsize=8,
                color="firebrick", wrap=True)

    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(T, 100 * sweep["precision_lo"], 100 * sweep["precision_hi"], color="C0", alpha=0.2)
    ax.fill_between(T, 100 * sweep["recall_lo"], 100 * sweep["recall_hi"], color="C1", alpha=0.2)
    ax.plot(T, 100 * p["precision"], "C0-", label="precision (background-corrected)")
    ax.plot(T, 100 * p["recall"], "C1-", label="recall (background-corrected)")
    ax.plot(T, 100 * old, "C0--", label="old fixed-numerator curve (lower bound)")
    ax.axvline(head["gate"], color="k", lw=0.8, ls=":")
    ax.set_xlabel("match threshold T (m)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.set_title("Precision and recall vs match threshold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    c = sweep["centers"]
    ax.step(c, sweep["hist"], where="mid", color="0.3", lw=1, label=f"candidate returns ({BIN_W} m bins)")
    ax.plot(c, p["background"], "r-", lw=1.2, label=f"clutter ramp ({PRIMARY_FIT})")
    for name, v in sweep["variants"].items():
        if name != PRIMARY_FIT:
            ax.plot(c, v["background"], "r:", lw=0.7)
    ax.fill_between(c, p["background"], np.maximum(sweep["hist"], p["background"]), step="mid",
                    color="C2", alpha=0.35, label="estimated on-body")
    ax.axvline(head["gate"], color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("nearest-actor OBB margin (m)")
    ax.set_ylabel("returns per bin")
    ax.set_title("Margin histogram (candidates, ground-reject excluded)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    def recall_bars(ax, sub, title):
        x = np.arange(len(sub))
        ax.bar(x - 0.2, 100 * sub["labeler_recall"], 0.4, label="labeler (found / illuminated)")
        ax.bar(x + 0.2, 100 * sub["system_recall"], 0.4, label="system (found / in FOV)")
        ax.set_xticks(x, [f"{g}\n{i:,}" for g, i in zip(sub["group"], sub["in_fov"])], fontsize=8)
        ax.set_xlabel("group / in-FOV actor-frames", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="lower left")

    recall_bars(fig.add_subplot(gs[1, 0]), rec[rec.group_type == "range"],
                "Actor-frame recall by range (>35 m: angular FOV only)")
    recall_bars(fig.add_subplot(gs[1, 1]), rec[rec.group_type == "radar"], "Actor-frame recall by radar")
    recall_bars(fig.add_subplot(gs[1, 2]), rec[rec.group_type == "kind"], "Actor-frame recall by actor kind")

    fig.text(0.01, 0.005,
             f"Gate diagnostics (not accuracy): point-weighted match rate = matched / with-candidates = "
             f"{n['matched']:,} / {n['with_candidates']:,} = {pct(n['matched'] / max(n['with_candidates'], 1))}. "
             f"Illuminated = return inside the labeler box (true OBB, or OBB+{gc['BBOX_MATCH_EXTENT_INFLATION_M']} m "
             f"above the ground-reject plane). Geometry check vs CSV matched margins: "
             f"{pct(n['check_agree'] / max(n['check_pairs'], 1))} agree within 2 mm.",
             fontsize=8, color="0.3")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_static(path_csv, path_png, acc, radar_labels, zone_note):
    static, sz = acc.static, acc.static_zone
    totals = static.sum(axis=0)
    grand = int(totals.sum())
    rows = []
    for ci, cname in enumerate(STATIC_CATEGORIES):
        cnt = int(static[ci].sum())
        if cnt == 0 and cname == "actor: truck":
            continue
        rows.append({"category": cname, "zone": "all", "radar": "all", "count": cnt, "percent": 100 * cnt / grand})
        if acc.zones is not None and ci in GROUND_CATEGORY_IDS:
            for zi, zname in enumerate(ZONES):
                zc = int(sz[ci, zi].sum())
                rows.append({"category": cname, "zone": zname, "radar": "all", "count": zc, "percent": 100 * zc / grand})
        for ri, rl in enumerate(radar_labels):
            rows.append({"category": cname, "zone": "all", "radar": rl, "count": int(static[ci, ri]),
                         "percent": 100 * static[ci, ri] / max(totals[ri], 1)})
            if acc.zones is not None and ci in GROUND_CATEGORY_IDS:
                for zi, zname in enumerate(ZONES):
                    rows.append({"category": cname, "zone": zname, "radar": rl, "count": int(sz[ci, zi, ri]),
                                 "percent": 100 * sz[ci, zi, ri] / max(totals[ri], 1)})
    pd.DataFrame(rows).to_csv(path_csv, index=False, float_format="%.4f")

    keep = [i for i, c in enumerate(STATIC_CATEGORIES) if not (c == "actor: truck" and static[i].sum() == 0)]
    names = [STATIC_CATEGORIES[i] for i in keep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(keep))[::-1]
    zone_colors = {"carriageway": "0.35", "sidewalk": "tab:orange", "off-network": "tab:green"}
    for yi, ci in zip(y, keep):
        cnt = static[ci].sum()
        if acc.zones is not None and ci in GROUND_CATEGORY_IDS:
            left = 0
            for zi, zname in enumerate(ZONES):
                zc = sz[ci, zi].sum()
                ax1.barh(yi, zc, left=left, color=zone_colors[zname],
                         label=zname if ci == GROUND_CATEGORY_IDS[1] else None)
                left += zc
        else:
            ax1.barh(yi, cnt, color="tab:blue" if ci < 4 else ("tab:red" if ci == 5 else "tab:purple"))
        ax1.text(max(cnt, 1) * 1.15, yi, f"{cnt:,}  ({100 * cnt / grand:.2f}%)", va="center", fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlim(1, grand * 30)
    ax1.set_yticks(y, names)
    ax1.set_xlabel("returns (log scale)")
    ax1.set_title(f"Static taxonomy, all radars ({grand:,} returns)")
    if acc.zones is not None:
        ax1.legend(title="ground zone", fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3, axis="x")

    pmat = 100 * static[keep] / np.maximum(totals[None, :], 1)
    im = ax2.imshow(np.log10(np.maximum(pmat, 1e-3)), aspect="auto", cmap="viridis")
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            ax2.text(j, i, f"{pmat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if pmat[i, j] < 5 else "black")
    ax2.set_xticks(range(len(radar_labels)), radar_labels)
    ax2.set_yticks(range(len(keep)), names)
    ax2.set_title("Percent of each radar's returns")
    fig.colorbar(im, ax=ax2, label="log10(%)")
    fig.text(0.01, 0.005, zone_note, fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: stop after N rows")
    ap.add_argument("--no-zones", action="store_true", help="skip the .xodr carriageway/sidewalk split")
    ap.add_argument("--xodr", type=Path, default=None)
    args = ap.parse_args()

    cap = args.capture_dir.resolve()
    out_dir = cap / QA_DIR
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    gc = load_gate_constants()
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        params = json.loads(summary_path.read_text(encoding="utf-8")).get("parameters", {})
        for key, const in (("hit_match_max_margin_m", "RADAR_HIT_MATCH_MAX_MARGIN_M"),
                           ("single_candidate_max_margin_m", "RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M"),
                           ("bbox_extent_inflation_m", "BBOX_MATCH_EXTENT_INFLATION_M"),
                           ("candidate_max_range_m", "RADAR_MAX_RANGE_M"),
                           ("candidate_horizontal_fov_deg", "RADAR_HORIZONTAL_FOV_DEG")):
            if key in params and abs(float(params[key]) - gc[const]) > 1e-9:
                print(f"WARNING: capture labeled with {key}={params[key]} but source has "
                      f"{const}={gc[const]}; using the capture's value", flush=True)
                gc[const] = float(params[key])
    print("Gate constants:", {k: gc[k] for k in sorted(gc)}, flush=True)

    seg_lo, seg_hi = load_segment_frames(cap)
    radars = load_radar_extrinsics(cap)
    radar_labels = [r["label"] for r in radars]
    rig_center = np.array([np.mean([r["x"] for r in radars]), np.mean([r["y"] for r in radars])])

    print("Loading actor_frames.jsonl ...", flush=True)
    actor_frames = load_actor_frames(cap / "actor_frames.jsonl", rig_center)
    den = build_denominator(actor_frames, radars, seg_lo, seg_hi,
                            gc["RADAR_MAX_RANGE_M"], gc["RADAR_HORIZONTAL_FOV_DEG"])
    print(f"  {len(actor_frames):,} frames, {int(den['in_fov'].sum()):,} in-FOV actor-frame triples "
          f"({time.time() - t0:.1f}s)", flush=True)

    zone_raster, zone_note = None, ""
    if not args.no_zones:
        run_meta = json.loads((cap / "run_meta.json").read_text(encoding="utf-8"))
        map_name = str(run_meta.get("map", "")).split("/")[-1] or "Town10HD_Opt"
        xodr = args.xodr or (XODR_DEFAULT / f"{map_name}.xodr")
        try:
            print(f"Rasterising lane zones from {xodr} ...", flush=True)
            pad = 60.0
            xs, ys = [r["x"] for r in radars], [r["y"] for r in radars]
            zone_raster = ZoneRaster(xodr, map_name, (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad))
            zone_note = (f"Ground zones: lane types from {xodr.name} via offline carla.Map, 0.25 m raster, "
                         f"rig +/-60 m, lane seams <= 1 m filled from the adjacent lane "
                         f"({100 * zone_raster.unmapped_frac_before_fill:.0f}% of cells off any lane before filling); "
                         f"carriageway = driving/shoulder/median/parking; outside raster = off-network.")
        except Exception as exc:  # noqa: BLE001
            zone_raster = None
            zone_note = f"Zone split skipped: {type(exc).__name__}: {exc}"
        print(f"  {zone_note} ({time.time() - t0:.1f}s)", flush=True)
    else:
        zone_note = "Zone split skipped (--no-zones)."

    acc = Accumulator(radar_labels, gc, zone_raster)
    csv_path = cap / LABELED_CSV
    size = csv_path.stat().st_size
    print(f"Streaming {csv_path.name} ({size / 1e9:.2f} GB) ...", flush=True)
    dtypes = dict(DTYPES, sensor_label=pd.CategoricalDtype(radar_labels))
    reader = pd.read_csv(csv_path, usecols=USECOLS, dtype=dtypes, chunksize=args.chunksize,
                         nrows=args.max_rows, engine="c")
    for chunk in reader:
        acc.add(chunk, actor_frames, seg_lo, seg_hi)
        el = time.time() - t0
        print(f"  {acc.n['rows']:,} rows | matched {acc.n['matched']:,} | {el:.0f}s", flush=True)

    n = acc.n
    found = np.unique(np.concatenate(acc.found_keys)) if acc.found_keys else np.array([], np.int64)
    illum = np.unique(np.concatenate(acc.illum_keys)) if acc.illum_keys else np.array([], np.int64)

    sweep = background_sweep(acc.edges, acc.hist)
    gate = gc["RADAR_HIT_MATCH_MAX_MARGIN_M"]
    t_idx = int(np.argmin(np.abs(acc.edges[1:] - gate)))
    old = write_threshold_sweep(out_dir / "threshold_sweep.csv", sweep, n["spike_lt_015"])
    warns = fit_warnings(sweep, t_idx)

    rec = recall_table(den, found, illum)
    rec.to_csv(out_dir / "actor_frame_recall.csv", index=False, float_format="%.6f")
    overall = rec[rec.group_type == "overall"].iloc[0]

    head = {
        "gate": gate,
        "precision_bc": float(sweep["primary"]["precision"][t_idx]),
        "precision_bc_lo": float(sweep["precision_lo"][t_idx]),
        "precision_bc_hi": float(sweep["precision_hi"][t_idx]),
        "precision_lb": n["matched_lt_015"] / max(n["matched"], 1),
        "recall_bc": float(sweep["primary"]["recall"][t_idx]),
        "recall_bc_lo": float(sweep["recall_lo"][t_idx]),
        "recall_bc_hi": float(sweep["recall_hi"][t_idx]),
        "labeler_recall": float(overall["labeler_recall"]),
        "system_recall": float(overall["system_recall"]),
    }
    plot_accuracy(out_dir / "labeling_accuracy.png", sweep, old, head, warns, rec, gc, n, (seg_lo, seg_hi))
    write_static(out_dir / "static_breakdown.csv", out_dir / "static_breakdown.png", acc, radar_labels, zone_note)

    bad = [k for k in ("precision_bc", "precision_lb", "recall_bc", "labeler_recall", "system_recall")
           if not (np.isfinite(head[k]) and 0.0 <= head[k] <= 1.0)]
    in_den = np.isin(found, den["key"].to_numpy())
    print("\n" + "=" * 64)
    print(f"Rows {n['rows']:,} (outside segment {n['rows_outside_segment']:,}); with candidates "
          f"{n['with_candidates']:,}; matched {n['matched']:,}; failed match {n['failed_match']:,}")
    print(f"Ground-reject placeholder rows excluded from sweep: {n['ground_reject']:,}; "
          f"candidates >{T_MAX} m: {n['cand_gt_tmax']:,}; NaN margin: {n['cand_nan']:,}")
    print(f"Precision @ {gate} m background-corrected: {pct(head['precision_bc'])} "
          f"[{pct(head['precision_bc_lo'])} - {pct(head['precision_bc_hi'])}]")
    print(f"Precision @ {gate} m lower bound (<0.15 m & accepted / accepted): {pct(head['precision_lb'])}")
    print(f"Recall    @ {gate} m background-corrected: {pct(head['recall_bc'])} "
          f"[{pct(head['recall_bc_lo'])} - {pct(head['recall_bc_hi'])}]")
    print(f"Labeler recall (found & illuminated / illuminated, in FOV): {pct(head['labeler_recall'])} "
          f"({int(overall['found_and_illuminated']):,} / {int(overall['illuminated']):,})")
    print(f"System recall (found / in FOV): {pct(head['system_recall'])} "
          f"({int(overall['found']):,} / {int(overall['in_fov']):,})")
    print(f"Found triples total {len(found):,} (outside segment/FOV table: {int((~in_den).sum()):,}); "
          f"illuminated total {len(illum):,}")
    print(f"Failed-match rows inside some actor's labeler box: {n['failed_inside_box']:,}")
    print(f"Geometry check: {n['check_agree']:,} / {n['check_pairs']:,} matched rows reproduce the CSV margin "
          f"within 2 mm; matched rows with no actor snapshot: {n['matched_no_actor_frame']:,}")
    print(f"Point-weighted match rate (gate diagnostic only): {pct(n['matched'] / max(n['with_candidates'], 1))}")
    for w in warns:
        print("FIT WARNING:", w)
    if bad:
        print("HEADLINE OUT OF RANGE:", bad)
    print(f"Wrote to {out_dir}: threshold_sweep.csv, labeling_accuracy.png, actor_frame_recall.csv, "
          f"static_breakdown.csv, static_breakdown.png ({time.time() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
