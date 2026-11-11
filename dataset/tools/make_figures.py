"""Figures from the scenario_insights and covisibility outputs.

Reads the CSVs the two tools write and renders a figure set (PNG, 150 dpi):

  fig1_points_per_scan.png        returns per radar scan per scenario, raw vs
                                  visible, with the 30-50 band of real radars
  fig2_points_per_object.png      returns per object per fused frame, per class
  fig3_seen_rate.png              % of objects on the stretch with >= 1 return
  fig4_diversity_matrix.png       pairwise scenario distance (Jensen-Shannon)
  fig5_feature_spread.png         which features separate the scenarios most
  fig6_distinctiveness.png        per scenario: mean distance to the others
  fig7_covisibility.png           objects seen by exactly k radars, measured vs
                                  predicted, one panel per class
  fig8_covis_gap_by_scenario.png  mean radars per object, measured vs predicted
  fig9_realization_by_radar.png   P(returns | object in cone) per radar and class

Usage (from dataset/):
  python -m tools.make_figures --insights Data/analysis/insights \\
      --covis Data/analysis/covisibility --out Data/analysis/figures
Either input may be omitted; only its figures are skipped.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import capture_io as io  # noqa: E402

# Reference palette (dataviz skill, light mode). Classes use categorical slots
# 1-3, the only slots validated for all pairs. "Predicted" is a neutral grey so
# the measured series carries the colour.
CLASS_COLOR = {"vehicle": "#2a78d6", "pedestrian": "#eb6834", "cyclist": "#1baf7a"}
MEASURED = "#2a78d6"
PREDICTED = "#b9b8b1"
RAW = "#d7d6cf"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e8e7e2"
SURFACE = "#fcfcfb"
REAL_BAND = (30, 50)


def read_csv(path):
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "font.size": 9,
    })


def short(name):
    """'s03 Heavy Demand' -> 's03 Heavy Demand' (kept), trims long names."""
    return name if len(name) <= 26 else name[:25] + "."


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"  {out / name}")


# ---------------------------------------------------------------- insights
def insights_figures(plt, d, out):
    scan = [r for r in read_csv(d / "points_per_scan.csv") if r["group"] != "ALL"]
    obj = read_csv(d / "points_per_object.csv")
    groups = [r["group"] for r in scan]

    if scan:
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(scan) + 1.4))
        y = np.arange(len(scan))[::-1]
        raw = [fnum(r["raw_mean"]) for r in scan]
        vis = [fnum(r["visible_mean"]) for r in scan]
        ax.axvspan(*REAL_BAND, color="#1baf7a", alpha=0.18, lw=0,
                   label=f"real roadside radars ({REAL_BAND[0]}-{REAL_BAND[1]})")
        ax.barh(y, raw, height=0.62, color=RAW, label="all returns")
        ax.barh(y, vis, height=0.36, color=MEASURED, label="visible (after FMCW filter)")
        for yi, v in zip(y, vis):
            ax.text(v + 6, yi, f"{v:.0f}", va="center", fontsize=7, color=TEXT)
        ax.set_yticks(y, [short(g) for g in groups])
        ax.set_xlabel("returns per radar scan (mean)")
        ax.set_xlim(0, max(raw + vis) * 1.12)
        ax.set_ylim(-0.7, y.max() + 0.7)
        ax.grid(axis="y", visible=False)
        ax.set_title("Returns per radar scan, by scenario", pad=22)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
        save(fig, out, "fig1_points_per_scan.png")
        plt.close(fig)

    per = [r for r in obj if r["group"] != "ALL"]
    classes = [c for c in io.CLASS_NAMES if any(r["class"] == c for r in per)]
    if per and classes:
        fig, axes = plt.subplots(1, len(classes), figsize=(3.4 * len(classes), 0.42 * len(groups) + 1.6),
                                 sharey=True, squeeze=False)
        y = np.arange(len(groups))[::-1]
        for ax, c in zip(axes[0], classes):
            vals = []
            for g in groups:
                r = next((r for r in per if r["group"] == g and r["class"] == c), None)
                vals.append(fnum(r["fused_mean"]) if r else np.nan)
            ax.barh(y, np.nan_to_num(vals), height=0.6, color=CLASS_COLOR[c])
            for yi, v in zip(y, vals):
                if v == v:
                    ax.text(v, yi, f" {v:.0f}", va="center", fontsize=7, color=TEXT)
            ax.set_title(c, color=TEXT, fontsize=10)
            ax.set_xlim(0, np.nanmax(vals) * 1.25 if np.isfinite(np.nanmax(vals)) else 1)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("returns / object / fused frame")
        axes[0][0].set_yticks(y, [short(g) for g in groups])
        fig.suptitle("Returns per object (visible, all radars pooled)", fontsize=11, fontweight="bold",
                     color=TEXT, x=0.02, ha="left")
        save(fig, out, "fig2_points_per_object.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(groups) + 1.4))
        offs = {c: (i - (len(classes) - 1) / 2) * 0.22 for i, c in enumerate(classes)}
        for c in classes:
            xs, ys = [], []
            for yi, g in zip(y, groups):
                r = next((r for r in per if r["group"] == g and r["class"] == c), None)
                if r and fnum(r["seen_pct"]) == fnum(r["seen_pct"]):
                    xs.append(fnum(r["seen_pct"]))
                    ys.append(yi + offs[c])
            ax.scatter(xs, ys, s=36, color=CLASS_COLOR[c], label=c, zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
        ax.set_yticks(y, [short(g) for g in groups])
        ax.set_xlim(0, 104)
        ax.set_xlabel("% of object-frames on the stretch with >= 1 return")
        ax.grid(axis="y", visible=False)
        ax.set_title("How often each class is seen at all", pad=22)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(classes), borderaxespad=0.2)
        save(fig, out, "fig3_seen_rate.png")
        plt.close(fig)

    mat = read_csv(d / "diversity_matrix.csv")
    if len(mat) > 1:
        names = [r["scenario"] for r in mat]
        M = np.array([[fnum(r[n]) for n in names] for r in mat])
        fig, ax = plt.subplots(figsize=(0.62 * len(names) + 3.2, 0.55 * len(names) + 1.6))
        vmax = max(0.3, float(np.nanmax(M)))
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(names)), [n.split(" ")[0] for n in names])
        ax.set_yticks(range(len(names)), [short(n) for n in names])
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.6 * vmax else TEXT)
        iu = np.triu_indices(len(names), 1)
        ax.set_title(f"Scenario distance (diversity score {np.nanmean(M[iu]):.3f})")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("Jensen-Shannon distance (0 = same, 1 = no overlap)", fontsize=8)
        cb.outline.set_visible(False)
        save(fig, out, "fig4_diversity_matrix.png")
        plt.close(fig)

        rows = read_csv(d / "diversity_by_feature.csv")
        feats = sorted({r["feature"] for r in rows})
        spread = {f: np.nanmean([fnum(r["js_distance"]) for r in rows if r["feature"] == f]) for f in feats}
        order = sorted(feats, key=lambda f: spread[f])
        fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(order) + 1.3))
        ax.barh(range(len(order)), [spread[f] for f in order], height=0.6, color=MEASURED)
        for i, f in enumerate(order):
            ax.text(spread[f] + 0.005, i, f"{spread[f]:.2f}", va="center", fontsize=7, color=TEXT)
        ax.set_yticks(range(len(order)), [f.replace("_", " ") for f in order])
        ax.set_xlabel("mean pairwise Jensen-Shannon distance")
        ax.set_xlim(0, max(0.1, max(spread.values()) * 1.18))
        ax.grid(axis="y", visible=False)
        ax.set_title("What separates the scenarios")
        save(fig, out, "fig5_feature_spread.png")
        plt.close(fig)

        dist = {}
        for i, a in enumerate(names):
            others = np.delete(M[i], i)
            j = int(np.nanargmin(np.where(np.arange(len(names)) == i, np.inf, M[i])))
            dist[a] = (float(np.nanmean(others)), names[j], float(M[i, j]))
        order = sorted(names, key=lambda a: dist[a][0])
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.3))
        ax.barh(range(len(order)), [dist[a][0] for a in order], height=0.6, color=MEASURED)
        for i, a in enumerate(order):
            m, near, dn = dist[a]
            ax.text(m + 0.005, i, f"{m:.2f}   closest: {near.split(' ')[0]} ({dn:.2f})",
                    va="center", fontsize=7, color=TEXT2)
        ax.set_yticks(range(len(order)), [short(a) for a in order])
        ax.set_xlim(0, max(dist[a][0] for a in order) * 1.7)
        ax.set_xlabel("mean distance to the other scenarios (low = redundant)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Distinctiveness of each scenario")
        save(fig, out, "fig6_distinctiveness.png")
        plt.close(fig)


# ---------------------------------------------------------------- covisibility
def covis_figures(plt, d, out):
    comp = read_csv(d / "covis_compare.csv")
    classes = [c for c in io.CLASS_NAMES if any(r["class"] == c for r in comp)]
    summ = read_csv(d / "covis_summary.csv")
    nobj = {(r["class"], r["source"]): int(r["object_frames"]) for r in summ if r["group"] == "ALL"}
    if comp and classes:
        fig, axes = plt.subplots(1, len(classes), figsize=(3.9 * len(classes), 3.3), squeeze=False)
        for ax, c in zip(axes[0], classes):
            rows = [r for r in comp if r["class"] == c]
            ks = np.array([int(r["k"]) for r in rows])
            pm = np.array([fnum(r["measured_exactly_pct"]) for r in rows])
            pp = np.array([fnum(r["predicted_exactly_pct"]) for r in rows])
            keep = (pm > 0.05) | (pp > 0.05) | (ks <= 1)
            last = ks[keep].max() if keep.any() else ks.max()
            sel = ks <= last
            w = 0.38
            ax.bar(ks[sel] - w / 2 - 0.01, pp[sel], w, color=PREDICTED, label="predicted (FOV overlap)")
            ax.bar(ks[sel] + w / 2 + 0.01, pm[sel], w, color=CLASS_COLOR[c], label="measured (returns)")
            ax.set_xticks(ks[sel])
            ax.set_xlabel("radars seeing the object")
            mk = float((ks * pm).sum() / 100)
            pk = float((ks * pp).sum() / 100)
            ax.set_title(f"{c}  ({nobj.get((c, 'measured'), 0):,} object-frames)\n"
                         f"mean radars: measured {mk:.2f}, predicted {pk:.2f}", fontsize=9)
            ax.grid(axis="x", visible=False)
        axes[0][0].set_ylabel("% of object-frames")
        from matplotlib.patches import Patch
        fig.legend([Patch(color=PREDICTED)] + [Patch(color=CLASS_COLOR[c]) for c in classes],
                   ["predicted (FOV overlap)"] + [f"measured: {c}" for c in classes],
                   loc="upper left", ncol=1 + len(classes), fontsize=7, bbox_to_anchor=(0.01, 0.92))
        fig.suptitle("Covisibility: radars that see each object, measured vs FOV-predicted",
                     fontsize=11, fontweight="bold", color=TEXT, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(out / "fig7_covisibility.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
        print(f"  {out / 'fig7_covisibility.png'}")
        plt.close(fig)

    groups = sorted({r["group"] for r in summ if r["group"] != "ALL"})
    if len(groups) > 1 and classes:
        def mean_k(g, c, s):
            r = next((r for r in summ if r["group"] == g and r["class"] == c and r["source"] == s), None)
            return fnum(r["mean_k"]) if r else np.nan
        fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes), 0.42 * len(groups) + 1.7),
                                 sharey=True, squeeze=False)
        y = np.arange(len(groups))[::-1]
        for ax, c in zip(axes[0], classes):
            pm = [mean_k(g, c, "measured") for g in groups]
            pp = [mean_k(g, c, "predicted") for g in groups]
            for yi, a, b in zip(y, pm, pp):
                if a == a and b == b:
                    ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.scatter(pp, y, s=34, color=PREDICTED, label="predicted", zorder=2)
            ax.scatter(pm, y, s=34, color=CLASS_COLOR[c], label="measured", zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
            ax.set_title(c, fontsize=10)
            ax.set_xlabel("mean radars per object")
            ax.grid(axis="y", visible=False)
            ax.set_xlim(left=0)
        axes[0][0].set_yticks(y, [short(g) for g in groups])
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], ls="", marker="o", color=PREDICTED)]
                   + [Line2D([], [], ls="", marker="o", color=CLASS_COLOR[c]) for c in classes],
                   ["predicted"] + [f"measured: {c}" for c in classes],
                   loc="upper left", ncol=1 + len(classes), fontsize=7, bbox_to_anchor=(0.01, 0.93))
        fig.suptitle("Occlusion gap by scenario (grey line = what occlusion and missed rays take away)",
                     fontsize=11, fontweight="bold", color=TEXT, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(out / "fig8_covis_gap_by_scenario.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
        print(f"  {out / 'fig8_covis_gap_by_scenario.png'}")
        plt.close(fig)

    per = read_csv(d / "covis_per_radar.csv")
    if per:
        radars = sorted({r["radar"] for r in per}, key=io._natkey)
        cls = [c for c in io.CLASS_NAMES if any(r["class"] == c for r in per)]
        M = np.full((len(cls), len(radars)), np.nan)
        for r in per:
            if r["class"] in cls:
                M[cls.index(r["class"]), radars.index(r["radar"])] = fnum(r["realization_pct"])
        fig, ax = plt.subplots(figsize=(0.7 * len(radars) + 3.4, 0.55 * len(cls) + 1.8))
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(radars)), radars)
        ax.set_yticks(range(len(cls)), cls)
        ax.grid(False)
        for i in range(len(cls)):
            for j in range(len(radars)):
                if M[i, j] == M[i, j]:
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 60 else TEXT)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("% of in-cone object-frames with returns", fontsize=8)
        cb.outline.set_visible(False)
        ax.set_title("Realization per radar: P(returns | object in its cone)")
        save(fig, out, "fig9_realization_by_radar.png")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insights", default="", help="scenario_insights output folder")
    ap.add_argument("--covis", default="", help="covisibility output folder")
    ap.add_argument("--out", required=True, help="figure folder")
    args = ap.parse_args()
    plt = io.try_pyplot()
    if plt is None:
        sys.exit("matplotlib is required: pip install matplotlib")
    style(plt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out}")
    if args.insights:
        insights_figures(plt, Path(args.insights), out)
    if args.covis:
        covis_figures(plt, Path(args.covis), out)


if __name__ == "__main__":
    main()
