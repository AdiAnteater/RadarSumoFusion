"""Scenario insights: points per object class, and how different the scenarios are.

Part 1, points
  points_per_scan.csv     returns per radar scan (raw and visible == 1), per scenario.
                          Real roadside radars report ~30-50 per scan; ours ~500+
                          at the default 15000 points/s.
  points_per_object.csv   per scenario x class:
                            - returns per object per radar scan (scans with >= 1 return)
                            - returns per object per fused frame (all radars pooled)
                            - % of objects on the stretch with >= 1 return that frame

Part 2, diversity between scenarios
  Every scenario is summarised by fixed-bin histograms of 8 features:
      class_mix            vehicle / pedestrian / cyclist share of object-frames
      speed_mps            object speed (from consecutive actor_frames poses)
      heading_deg          object yaw, 30 deg bins
      lateral_y_m          where across the road objects are (lane / sidewalk use)
      objects_per_frame    how crowded the stretch is
      points_per_object    fused returns per object per frame
      return_range_m       range of object returns from their radar
      return_speed_mps     |radial velocity| of object returns
  Two scenarios are compared per feature with the Jensen-Shannon distance
  (0 = identical histograms, 1 = no overlap). The overall distance between two
  scenarios is the mean over the features; the DIVERSITY SCORE of the set is the
  mean pairwise overall distance. Per scenario, "distinctiveness" is its mean
  distance to all others (low = close to some other scenario, i.e. redundant).
  diversity_matrix.csv, diversity_by_feature.csv, diversity_heatmap.png

  insights_report.md      everything above on one page

Usage (from dataset/):
  python -m tools.scenario_insights Data/sensor_capture_20260929_210312
  python -m tools.scenario_insights "Data/sensor_capture_2026092*" --out Data/insights_sep
  python -m tools.scenario_insights CAP --by segment       # one entry per segment
"""

import argparse
import collections
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import capture_io as io  # noqa: E402


FEATURES = {
    "class_mix": None,                                     # categorical, 3 bins
    "speed_mps": np.r_[np.arange(0.0, 20.01, 1.0), np.inf],
    "heading_deg": np.arange(-180.0, 180.01, 30.0),
    "lateral_y_m": np.r_[-np.inf, np.arange(-6.0, 48.01, 2.0), np.inf],
    "objects_per_frame": np.r_[np.arange(0.0, 40.01, 2.0), np.inf],
    "points_per_object": np.array([0, 1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 130, 200, 300, 500, np.inf], float),
    "return_range_m": np.r_[np.arange(0.0, 50.01, 2.5), np.inf],
    "return_speed_mps": np.r_[np.arange(0.0, 20.01, 1.0), np.inf],
}


def _hist(values, bins):
    if len(values) == 0:
        return np.zeros(len(bins) - 1, np.int64)
    return np.histogram(np.clip(values, bins[0] if np.isfinite(bins[0]) else -1e9, 1e12), bins=bins)[0]


def qstats(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return [0, "", "", "", ""]
    return [int(a.size), round(float(a.mean()), 1), float(np.median(a)),
            float(np.percentile(a, 10)), float(np.percentile(a, 90))]


class Group:
    def __init__(self):
        self.scan_raw, self.scan_vis = [], []
        self.obj_scan = collections.defaultdict(list)      # class -> returns per object-scan
        self.obj_fused = collections.defaultdict(list)     # class -> fused returns (seen objects)
        self.obj_total = collections.Counter()             # class -> object-frames on stretch
        self.obj_seen = collections.Counter()              # class -> with >= 1 return
        self.h = {k: np.zeros(3 if v is None else len(v) - 1, np.int64) for k, v in FEATURES.items()}

    def merge(self, other):
        """Add another Group (same scenario from another capture) into this one."""
        self.scan_raw += other.scan_raw
        self.scan_vis += other.scan_vis
        for c, v in other.obj_scan.items():
            self.obj_scan[c] += v
        for c, v in other.obj_fused.items():
            self.obj_fused[c] += v
        self.obj_total.update(other.obj_total)
        self.obj_seen.update(other.obj_seen)
        for k, v in other.h.items():
            self.h[k] = self.h[k] + v
        return self

    def to_state(self):
        return {"scan_raw": self.scan_raw, "scan_vis": self.scan_vis,
                "obj_scan": dict(self.obj_scan), "obj_fused": dict(self.obj_fused),
                "obj_total": dict(self.obj_total), "obj_seen": dict(self.obj_seen), "h": self.h}

    @classmethod
    def from_state(cls, st):
        g = cls()
        g.scan_raw, g.scan_vis = list(st["scan_raw"]), list(st["scan_vis"])
        g.obj_scan.update(st["obj_scan"])
        g.obj_fused.update(st["obj_fused"])
        g.obj_total.update(st["obj_total"])
        g.obj_seen.update(st["obj_seen"])
        for k, v in st["h"].items():
            if k in g.h and len(g.h[k]) == len(v):
                g.h[k] = np.asarray(v, np.int64)
        return g


def analyse_capture(cap, args, groups):
    poses = io.radar_poses(cap)
    if not poses:
        print(f"[skip] {cap}: no radar poses")
        return
    zone = io.rig_zone(poses, args.zone_margin)
    segs = io.Segments(cap)
    rate = segs.rate_hz()
    keys = [io.scenario_key(s, args.by) for s in segs.segs] if segs else []

    def key_of_frames(frames):
        if not segs:
            return np.array(["all"] * len(frames), object)
        i = np.searchsorted(segs._starts, frames, side="right") - 1
        return np.array(keys, object)[np.clip(i, 0, None)]

    cols = ["frame", "sensor_label", "matched_actor_id", "matched_actor_class",
            "matched_actor_kind", "visible", "depth_m", "velocity_mps"]
    have = set(io.csv_columns(cap))
    has_vis = "visible" in have
    if not has_vis:
        print(f"  note: {cap.name} has no 'visible' column; 'visible' = all returns")
    scans = {}                                  # (frame, sensor) -> [raw, vis]
    objscan = {}                                # (frame, sensor, actor) -> vis count
    cls_of = {}
    print(f"[{cap.name}] reading {io.LABELED_CSV} ...")
    for ch in io.iter_labeled(cap, cols, max_rows=args.max_rows):
        fr = ch["frame"].to_numpy(np.int64)
        ch = ch[segs.mask(fr, args.frame_stride)]
        if ch.empty:
            continue
        vis = (ch["visible"].astype(str).str.strip().isin(["1", "1.0", "True", "true"]).to_numpy()
               if has_vis else np.ones(len(ch), bool))
        ch = ch.assign(_vis=vis.astype(np.int64))
        g = ch.groupby(["frame", "sensor_label"])["_vis"].agg(["size", "sum"])
        for (f, s), (n, v) in zip(g.index, g.to_numpy()):
            cur = scans.get((f, s))
            if cur is None:
                scans[(f, s)] = [int(n), int(v)]
            else:
                cur[0] += int(n)
                cur[1] += int(v)
        m = ch["matched_actor_id"].notna() & (ch["matched_actor_id"].astype(str).str.len() > 0)
        ob = ch[m & (ch["_vis"] == 1)]
        if ob.empty:
            continue
        aid = ob["matched_actor_id"].astype(float).astype(np.int64)
        ob = ob.assign(_aid=aid.to_numpy())
        for a, c, k in zip(ob["_aid"].to_numpy(), ob["matched_actor_class"].astype(str),
                           ob.get("matched_actor_kind", ob["matched_actor_class"]).astype(str)):
            if a not in cls_of:
                cls_of[a] = io.object_class(c, k)
        go = ob.groupby(["frame", "sensor_label", "_aid"]).size()
        for (f, s, a), n in go.items():
            objscan[(f, s, a)] = objscan.get((f, s, a), 0) + int(n)
        # return features, per group
        gk = key_of_frames(ob["frame"].to_numpy(np.int64))
        depth = ob["depth_m"].astype(float).to_numpy()
        vel = np.abs(ob["velocity_mps"].astype(float).to_numpy())
        for key in np.unique(gk):
            sel = gk == key
            G = groups.setdefault(key, Group())
            G.h["return_range_m"] += _hist(depth[sel], FEATURES["return_range_m"])
            G.h["return_speed_mps"] += _hist(vel[sel], FEATURES["return_speed_mps"])

    # per-scan stats
    for (f, s), (n, v) in scans.items():
        key = key_of_frames(np.array([f]))[0]
        G = groups.setdefault(key, Group())
        G.scan_raw.append(n)
        G.scan_vis.append(v)
    fused = collections.Counter()
    for (f, s, a), n in objscan.items():
        key = key_of_frames(np.array([f]))[0]
        c = cls_of.get(a)
        if c:
            groups.setdefault(key, Group()).obj_scan[c].append(n)
        fused[(f, a)] += n

    # actor pass: kinematics, density, seen rate
    frames_with_scans = {f for f, _s in scans}
    prev = {}
    print(f"[{cap.name}] reading actor_frames.jsonl ...")
    for f, actors in io.iter_actor_frames(cap, frame_ok=lambda fr: fr in frames_with_scans):
        key = key_of_frames(np.array([f]))[0]
        G = groups.setdefault(key, Group())
        n_here = 0
        for a in actors:
            c = io.actor_class(a)
            if c is None:
                continue
            x, y = a["location"]["x"], a["location"]["y"]
            aid = int(a["id"])
            p = prev.get(aid)
            prev[aid] = (f, x, y)
            if not io.in_zone(zone, x, y):
                continue
            n_here += 1
            G.obj_total[c] += 1
            G.h["class_mix"][io.CLASS_NAMES.index(c)] += 1
            k = fused.get((f, aid), 0)
            G.h["points_per_object"] += _hist([k], FEATURES["points_per_object"])
            if k:
                G.obj_seen[c] += 1
                G.obj_fused[c].append(k)
            G.h["heading_deg"] += _hist([((a.get("rotation", {}).get("yaw", 0.0) + 180) % 360) - 180],
                                        FEATURES["heading_deg"])
            G.h["lateral_y_m"] += _hist([y], FEATURES["lateral_y_m"])
            if p and 0 < f - p[0] <= 5 * max(1, args.frame_stride):
                v = math.hypot(x - p[1], y - p[2]) * rate / (f - p[0])
                G.h["speed_mps"] += _hist([v], FEATURES["speed_mps"])
        G.h["objects_per_frame"] += _hist([n_here], FEATURES["objects_per_frame"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="capture folders (globs ok)")
    ap.add_argument("--out", default="", help="output folder")
    ap.add_argument("--by", choices=("scenario", "segment"), default="scenario")
    ap.add_argument("--zone-margin", type=float, default=0.0)
    ap.add_argument("--frame-stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--max-rows", type=int, default=0, help="stop after N CSV rows (quick test)")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-read every capture (default: reuse <capture>/_analysis_cache)")
    ap.add_argument("--cache-only", action="store_true",
                    help="only build the per-capture cache, no report (one capture per call)")
    args = ap.parse_args()

    caps = io.expand_captures(args.captures)
    if not caps:
        sys.exit("no capture with radar_data_labeled.csv found")
    groups = {}
    for cap in caps:
        cp = io.cache_path(cap, "insights", b=args.by, z=args.zone_margin,
                           f=args.frame_stride, m=args.max_rows)
        st = None if args.no_cache else io.load_cache(cp, cap)
        if st is None:
            mine = {}
            analyse_capture(cap, args, mine)
            st = {k: g.to_state() for k, g in mine.items()}
            io.save_cache(cp, st)
        else:
            print(f"[{cap.name}] cached: {cp}")
        for k, gs in st.items():
            g = Group.from_state(gs)
            if k in groups:
                groups[k].merge(g)
            else:
                groups[k] = g
    if args.cache_only:
        print("cache built")
        return
    groups = {k: v for k, v in groups.items() if v.scan_raw}
    if not groups:
        sys.exit("nothing read")
    out = io.out_dir(args.out, caps, "insights")
    names = sorted(groups)

    # ---- points tables (plus an ALL row)
    allg = Group()
    for G in groups.values():
        allg.scan_raw += G.scan_raw
        allg.scan_vis += G.scan_vis
        for c in io.CLASS_NAMES:
            allg.obj_scan[c] += G.obj_scan[c]
            allg.obj_fused[c] += G.obj_fused[c]
            allg.obj_total[c] += G.obj_total[c]
            allg.obj_seen[c] += G.obj_seen[c]
    table = [("ALL", allg)] + ([(n, groups[n]) for n in names] if names != ["all"] else [])
    io.write_csv(out / "points_per_scan.csv",
                 [[n] + qstats(G.scan_raw) + qstats(G.scan_vis)[1:] for n, G in table],
                 ["group", "scans", "raw_mean", "raw_median", "raw_p10", "raw_p90",
                  "visible_mean", "visible_median", "visible_p10", "visible_p90"])
    rows = []
    for n, G in table:
        for c in io.CLASS_NAMES:
            if not G.obj_total[c] and not G.obj_scan[c]:
                continue
            seen = 100.0 * G.obj_seen[c] / G.obj_total[c] if G.obj_total[c] else float("nan")
            rows.append([n, c] + qstats(G.obj_scan[c]) + qstats(G.obj_fused[c])
                        + [G.obj_total[c], round(seen, 1)])
    io.write_csv(out / "points_per_object.csv", rows,
                 ["group", "class",
                  "object_scans", "per_scan_mean", "per_scan_median", "per_scan_p10", "per_scan_p90",
                  "object_frames_seen", "fused_mean", "fused_median", "fused_p10", "fused_p90",
                  "object_frames_on_stretch", "seen_pct"])

    # ---- diversity
    feats = list(FEATURES)
    D = {f: np.full((len(names), len(names)), np.nan) for f in feats}
    by_feat = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            for f in feats:
                d = 0.0 if i == j else io.jensen_shannon(groups[a].h[f], groups[b].h[f])
                D[f][i, j] = d
                if i < j:
                    by_feat.append([f, a, b, round(d, 4)])
    with np.errstate(invalid="ignore"):
        overall = np.nanmean(np.stack([D[f] for f in feats]), axis=0)
    io.write_csv(out / "diversity_by_feature.csv", by_feat, ["feature", "scenario_a", "scenario_b", "js_distance"])
    io.write_csv(out / "diversity_matrix.csv",
                 [[a] + [round(float(x), 4) for x in overall[i]] for i, a in enumerate(names)],
                 ["scenario"] + names)
    iu = np.triu_indices(len(names), 1)
    score = float(np.nanmean(overall[iu])) if len(names) > 1 else float("nan")
    distinct = {a: float(np.nanmean(np.delete(overall[i], i))) if len(names) > 1 else float("nan")
                for i, a in enumerate(names)}
    feat_spread = {f: float(np.nanmean(D[f][iu])) if len(names) > 1 else float("nan") for f in feats}

    # ---- report
    md = [f"# Scenario insights\n\nCaptures: {', '.join(c.name for c in caps)}\n",
          "## Returns per radar scan\n",
          "| group | scans | raw mean | raw median | visible mean | visible median |",
          "|---|---|---|---|---|---|"]
    for n, G in table:
        r, v = qstats(G.scan_raw), qstats(G.scan_vis)
        md.append(f"| {n} | {r[0]} | {r[1]} | {r[2]:.0f} | {v[1]} | {v[2]:.0f} |")
    md += ["\n## Returns per object (visible returns)\n",
           "| group | class | per radar scan: mean / median | per fused frame: mean / median | seen on stretch |",
           "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[3]} / {r[4]} | {r[8]} / {r[9]} | {r[13]}% of {r[12]} |")
    if len(names) > 1:
        md += [f"\n## Diversity\n\nDiversity score (mean pairwise Jensen-Shannon distance, 0-1): **{score:.3f}**\n",
               "Feature spread (mean pairwise distance per feature; higher = scenarios differ more on it):\n",
               "| feature | spread |", "|---|---|"]
        for f in sorted(feats, key=lambda f: -np.nan_to_num(feat_spread[f])):
            md.append(f"| {f} | {feat_spread[f]:.3f} |")
        md += ["\nDistinctiveness (mean distance to the other scenarios; lowest = most redundant):\n",
               "| scenario | distinctiveness | closest scenario | distance |", "|---|---|---|---|"]
        for i, a in sorted(enumerate(names), key=lambda t: distinct[t[1]]):
            row = overall[i].copy()
            row[i] = np.inf
            j = int(np.nanargmin(row))
            md.append(f"| {a} | {distinct[a]:.3f} | {names[j]} | {row[j]:.3f} |")
    (out / "insights_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    plt = io.try_pyplot()
    if plt is not None:
        if len(names) > 1:
            fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(names), 0.8 + 0.6 * len(names)))
            im = ax.imshow(overall, cmap="Blues", vmin=0, vmax=max(0.3, float(np.nanmax(overall))))
            ax.set_xticks(range(len(names)), [n.split(" ")[0] for n in names], rotation=45, ha="right")
            ax.set_yticks(range(len(names)), names)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{overall[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if overall[i, j] > 0.6 * np.nanmax(overall) else "black")
            ax.set_title(f"Scenario distance (JS, mean of {len(feats)} features), score {score:.3f}")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(out / "diversity_heatmap.png", dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        data = [allg.obj_fused[c] for c in io.CLASS_NAMES if allg.obj_fused[c]]
        labs = [c for c in io.CLASS_NAMES if allg.obj_fused[c]]
        if data:
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(labs) + 1), labs)
            ax.set_ylabel("returns per object per fused frame")
            ax.set_title("Points per object (visible returns, all radars)")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(out / "points_per_object.png", dpi=150)
        plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
