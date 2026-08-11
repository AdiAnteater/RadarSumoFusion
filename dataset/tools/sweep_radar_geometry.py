"""
Radar geometry parameter sweep — no CARLA required.

Sweeps (height H, pitch, y_offset) and scores each configuration against:
  - Coverage of all 4 lanes (truck, car, pedestrian targets)
  - Coverage of both verges
  - Coverage of far footpath
  - Occlusion-free view over nearest-lane trucks to all farther zones
  - Near-lane truck top must not be clipped above the FOV upper limit

Road cross-section (left-radar perspective, Y measured from left radar):
  y_offset shifts the radar further into the footpath (negative Y direction).
  With y_offset=0, radar is at the inner footpath/verge boundary.

  Zone            | Y_near  | Y_far   | Notes
  ----------------+---------+---------+------
  Left footpath   |  -6.0   |   0.0   | behind radar if y_offset=0; adjusts with y_offset
  Left verge      |   0.0   |   2.63  |
  Lane 1 (near)   |   2.63  |   6.13  |
  Lane 2          |   6.13  |   9.63  |
  Lane 3          |   9.63  |  13.13  |
  Lane 4 (far)    |  13.13  |  16.63  |
  Right verge     |  16.63  |  19.26  |
  Right footpath  |  19.26  |  25.26  |

Usage:
  python sweep_radar_geometry.py                # print top-20 configs
  python sweep_radar_geometry.py --csv out.csv  # also write CSV
  python sweep_radar_geometry.py --plot         # scatter plot (requires matplotlib)
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, fields
from typing import List, Optional

# ---------------------------------------------------------------------------
# Road geometry constants (metres)
# ---------------------------------------------------------------------------
FOOTPATH_WIDTH = 6.0
VERGE_WIDTH    = 2.63
LANE_WIDTH     = 3.5
NUM_LANES      = 4
ROAD_WIDTH     = NUM_LANES * LANE_WIDTH       # 14.0 m

# From left radar (y_offset=0) inward:
Y_VERGE_NEAR  = 0.0
Y_VERGE_FAR   = VERGE_WIDTH                                     # 2.63
Y_L1_NEAR     = VERGE_WIDTH                                     # 2.63
Y_L1_FAR      = VERGE_WIDTH + LANE_WIDTH                        # 6.13
Y_L2_NEAR     = Y_L1_FAR                                        # 6.13
Y_L2_FAR      = Y_L2_NEAR + LANE_WIDTH                         # 9.63
Y_L3_NEAR     = Y_L2_FAR                                        # 9.63
Y_L3_FAR      = Y_L3_NEAR + LANE_WIDTH                         # 13.13
Y_L4_NEAR     = Y_L3_FAR                                        # 13.13
Y_L4_FAR      = Y_L4_NEAR + LANE_WIDTH                         # 16.63
Y_RV_NEAR     = Y_L4_FAR                                        # 16.63
Y_RV_FAR      = Y_RV_NEAR + VERGE_WIDTH                        # 19.26
Y_RFP_NEAR    = Y_RV_FAR                                        # 19.26
Y_RFP_FAR     = Y_RFP_NEAR + FOOTPATH_WIDTH                    # 25.26
Y_RIGHT_RADAR = Y_RFP_FAR                                       # 25.26

# Target heights (from CARLA actor catalog)
H_TRUCK = 3.5    # european_hgv full height
H_CAR   = 2.1    # tallest car (Nissan Patrol)
H_PED   = 2.0    # adult pedestrian

# Radar sensor limits
RADAR_RANGE_M    = 35.0
H_FOV_DEG        = 120.0   # horizontal — fixed per user request
V_FOV_DEG        = 30.0    # vertical   — fixed per user request
V_HALF           = V_FOV_DEG / 2.0   # 15°

# ---------------------------------------------------------------------------
# Sweep ranges
# ---------------------------------------------------------------------------
HEIGHT_RANGE  = [round(h * 0.5, 1) for h in range(8, 25)]    # 4.0 – 12.0 m
PITCH_RANGE   = list(range(5, 31))                             # 5° – 30°

# y_offset = distance the radar is placed INTO the footpath from its inner boundary.
#   y_offset = 0.0  → radar at inner footpath edge  (footpath/verge boundary)
#   y_offset = 6.0  → radar at outer footpath edge  (kerb / street boundary)
# Range is strictly limited to [0, FOOTPATH_WIDTH] — the radar stays on footpath,
# never in the verge (y_offset < 0) or beyond the outer footpath edge (y_offset > 6).
YOFFSET_RANGE = [round(o * 0.5, 1) for o in range(0, int(FOOTPATH_WIDTH / 0.5) + 1)]

# ---------------------------------------------------------------------------
# Score weights (module-level so MAX_SCORE is computed once)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "lane1_truck": 10, "lane1_car": 8,
    "lane2_truck": 10, "lane2_car": 8,
    "lane3_truck": 9,  "lane4_truck": 8,
    "right_verge": 5,  "right_footpath": 4,
    "left_verge":  5,  "near_footpath": 3,
    "occ_l1_to_l2": 6, "occ_l1_to_l3": 5, "occ_l1_to_l4": 4,
    "occ_l2_to_l3": 3, "occ_l2_to_rv":  3,
}
MAX_SCORE = sum(SCORE_WEIGHTS.values())  # 91


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def depression_deg(H: float, h_target: float, Y: float) -> float:
    """
    Depression angle (degrees, positive = looking down) from a radar at height H
    to a point at height h_target and lateral distance Y.
    Y must be > 0.
    """
    return math.degrees(math.atan2(H - h_target, Y))


def in_fov(depr: float, pitch: float) -> bool:
    """True if depression angle `depr` falls inside [pitch-15, pitch+15]."""
    return (pitch - V_HALF) <= depr <= (pitch + V_HALF)


def zone_covered(H: float, pitch: float, Y_near: float, Y_far: float,
                 h_target: float) -> bool:
    """
    True if the radar can hit the top of a target of height h_target
    anywhere within the lateral band [Y_near, Y_far].

    We check two critical points:
      - nearest edge (Y_near): highest depression — must not exceed FOV lower limit
      - farthest edge (Y_far): lowest depression  — must not fall below FOV upper limit
    Both checks are against the TARGET TOP (most demanding point).
    We also need the slant range to stay within RADAR_RANGE_M.
    """
    if Y_near <= 0.0 or Y_far <= 0.0:
        return False

    depr_near = depression_deg(H, h_target, Y_near)   # steepest angle
    depr_far  = depression_deg(H, h_target, Y_far)    # shallowest angle

    lower_limit = pitch - V_HALF
    upper_limit = pitch + V_HALF

    # FOV must overlap the range [depr_far, depr_near]
    fov_overlaps = lower_limit <= depr_near and upper_limit >= depr_far

    # Farthest point slant range (worst case)
    slant = math.hypot(Y_far, H - h_target)
    in_range = slant <= RADAR_RANGE_M

    return fov_overlaps and in_range


def occlusion_free(H: float, h_blocker: float, Y_blocker: float,
                   Y_target: float) -> bool:
    """
    True if a radar at height H can see over a blocker (top at h_blocker,
    centre at Y_blocker) to reach a target at Y_target.

    H_min = (h_blocker * Y_target - h_target_ground * Y_blocker) / (Y_target - Y_blocker)
    With h_target_ground = 0 (ground plane):
    H_min = h_blocker * Y_target / (Y_target - Y_blocker)
    """
    if Y_target <= Y_blocker:
        return True   # target is closer — blocker not relevant
    H_min = h_blocker * Y_target / (Y_target - Y_blocker)
    return H >= H_min


def near_footpath_coverage(H: float, pitch: float, y_offset: float) -> bool:
    """
    Left (near) footpath: Y_near = -(FOOTPATH_WIDTH - y_offset) ... Y_far = -y_offset
    relative to radar. Since we measure from the radar outward (positive Y = into road),
    the footpath is at negative Y — use absolute values and check via upper FOV limit.

    The nearest footpath edge from the radar is at |y_offset| (right behind it if 0),
    farthest is at FOOTPATH_WIDTH - y_offset away on the other side.

    Simplification: treat near footpath as a single band behind the radar.
    We check both edges with the pedestrian target height.
    """
    # Distance from radar to far footpath edge (behind radar)
    Y_fp_far = FOOTPATH_WIDTH - y_offset   # e.g. 6.0 m behind if y_offset=0
    if Y_fp_far <= 0:
        return True  # radar is at far edge, footpath in front of it

    # To look backward the radar must cover negative-Y, which requires a
    # 120-degree horizontal FOV to wrap behind. Vertical geometry:
    # depression to pedestrian top behind radar = arctan((H - H_PED) / Y_fp_far)
    # (same formula, direction doesn't matter for vertical depression)
    depr_fp = depression_deg(H, H_PED, Y_fp_far)
    slant = math.hypot(Y_fp_far, H - H_PED)
    return in_fov(depr_fp, pitch) and slant <= RADAR_RANGE_M


# ---------------------------------------------------------------------------
# Per-configuration evaluation
# ---------------------------------------------------------------------------

@dataclass
class Config:
    H:        float
    pitch:    float
    y_offset: float

    # Coverage flags
    left_verge:      bool = False
    lane1_truck:     bool = False
    lane1_car:       bool = False
    lane2_truck:     bool = False
    lane2_car:       bool = False
    lane3_truck:     bool = False
    lane4_truck:     bool = False
    right_verge:     bool = False
    right_footpath:  bool = False
    near_footpath:   bool = False

    # Occlusion-free flags
    occ_l1_to_l2:    bool = False
    occ_l1_to_l3:    bool = False
    occ_l1_to_l4:    bool = False
    occ_l2_to_l3:    bool = False
    occ_l2_to_rv:    bool = False

    score:    float = 0.0
    max_range_m: float = 0.0

    def coverage_count(self) -> int:
        flags = [
            self.lane1_truck, self.lane1_car,
            self.lane2_truck, self.lane2_car,
            self.lane3_truck, self.lane4_truck,
            self.right_verge, self.right_footpath,
            self.left_verge,  self.near_footpath,
        ]
        return sum(flags)


def evaluate(H: float, pitch: float, y_offset: float) -> Config:
    cfg = Config(H=H, pitch=pitch, y_offset=y_offset)

    # Lateral distances FROM the radar (positive = into road)
    # Radar sits at -y_offset relative to verge boundary
    # so every fixed zone boundary shifts by +y_offset
    shift = y_offset

    yv_n  = Y_VERGE_NEAR  + shift
    yv_f  = Y_VERGE_FAR   + shift
    y1_n  = Y_L1_NEAR     + shift
    y1_f  = Y_L1_FAR      + shift
    y2_n  = Y_L2_NEAR     + shift
    y2_f  = Y_L2_FAR      + shift
    y3_n  = Y_L3_NEAR     + shift
    y3_f  = Y_L3_FAR      + shift
    y4_n  = Y_L4_NEAR     + shift
    y4_f  = Y_L4_FAR      + shift
    yrv_n = Y_RV_NEAR     + shift
    yrv_f = Y_RV_FAR      + shift
    yfp_n = Y_RFP_NEAR    + shift
    yfp_f = Y_RFP_FAR     + shift

    cfg.left_verge     = zone_covered(H, pitch, yv_n,  yv_f,  H_PED)
    cfg.lane1_truck    = zone_covered(H, pitch, y1_n,  y1_f,  H_TRUCK)
    cfg.lane1_car      = zone_covered(H, pitch, y1_n,  y1_f,  H_CAR)
    cfg.lane2_truck    = zone_covered(H, pitch, y2_n,  y2_f,  H_TRUCK)
    cfg.lane2_car      = zone_covered(H, pitch, y2_n,  y2_f,  H_CAR)
    cfg.lane3_truck    = zone_covered(H, pitch, y3_n,  y3_f,  H_TRUCK)
    cfg.lane4_truck    = zone_covered(H, pitch, y4_n,  y4_f,  H_TRUCK)
    cfg.right_verge    = zone_covered(H, pitch, yrv_n, yrv_f, H_PED)
    cfg.right_footpath = zone_covered(H, pitch, yfp_n, yfp_f, H_PED)
    cfg.near_footpath  = near_footpath_coverage(H, pitch, y_offset)

    # Occlusion: can radar see over L1 truck to reach L2 / L3 / L4 / right-verge?
    L1_blocker_Y = (y1_n + y1_f) / 2.0  # L1 centre
    cfg.occ_l1_to_l2 = occlusion_free(H, H_TRUCK, L1_blocker_Y, y2_f)
    cfg.occ_l1_to_l3 = occlusion_free(H, H_TRUCK, L1_blocker_Y, y3_f)
    cfg.occ_l1_to_l4 = occlusion_free(H, H_TRUCK, L1_blocker_Y, y4_f)
    L2_blocker_Y = (y2_n + y2_f) / 2.0
    cfg.occ_l2_to_l3 = occlusion_free(H, H_TRUCK, L2_blocker_Y, y3_f)
    cfg.occ_l2_to_rv = occlusion_free(H, H_TRUCK, L2_blocker_Y, yrv_f)

    cfg.score = sum(SCORE_WEIGHTS[f] for f in SCORE_WEIGHTS if getattr(cfg, f))

    # Maximum useful range: farthest zone covered
    for y_edge, label in [
        (yfp_f,  "right_footpath"),
        (yrv_f,  "right_verge"),
        (y4_f,   "lane4"),
        (y3_f,   "lane3"),
        (y2_f,   "lane2"),
        (y1_f,   "lane1"),
    ]:
        depr = depression_deg(H, 0.0, y_edge)
        slant = math.hypot(y_edge, H)
        if in_fov(depr, pitch) and slant <= RADAR_RANGE_M:
            cfg.max_range_m = round(slant, 2)
            break

    return cfg


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> List[Config]:
    results = []
    for H in HEIGHT_RANGE:
        for pitch in PITCH_RANGE:
            for y_offset in YOFFSET_RANGE:
                results.append(evaluate(H, pitch, y_offset))
    results.sort(key=lambda c: (-c.score, c.H, c.pitch))
    return results


def print_table(configs: List[Config], top_n: int = 20) -> None:
    cols = [
        ("H(m)", 5),  ("pitch", 5), ("yoff", 4),
        ("score", 5),
        ("L1T", 3), ("L1C", 3), ("L2T", 3), ("L2C", 3),
        ("L3T", 3), ("L4T", 3),
        ("RV",  2), ("RFP", 3), ("LV", 2), ("NFP", 3),
        ("o12", 3), ("o13", 3), ("o14", 3),
    ]
    header = "  ".join(f"{h:>{w}}" for h, w in cols)
    print(header)
    print("-" * len(header))

    def row(c: Config):
        def b(v): return "Y" if v else "."
        vals = [
            f"{c.H:>5.1f}", f"{c.pitch:>5}", f"{c.y_offset:>4.1f}",
            f"{c.score:>5.0f}",
            b(c.lane1_truck), b(c.lane1_car), b(c.lane2_truck), b(c.lane2_car),
            b(c.lane3_truck), b(c.lane4_truck),
            b(c.right_verge), b(c.right_footpath), b(c.left_verge), b(c.near_footpath),
            b(c.occ_l1_to_l2), b(c.occ_l1_to_l3), b(c.occ_l1_to_l4),
        ]
        return "  ".join(f"{v:>{w}}" for v, (_, w) in zip(vals, cols))

    for c in configs[:top_n]:
        print(row(c))


def write_csv(configs: List[Config], path: str) -> None:
    fnames = [f.name for f in fields(Config)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        for c in configs:
            w.writerow({f: getattr(c, f) for f in fnames})
    print(f"Wrote {len(configs)} rows to {path}")


# ---------------------------------------------------------------------------
# Per-config FOV visualization
# ---------------------------------------------------------------------------

def _make_fov_polygon(x_radar: float, direction: int, H: float,
                      pitch_deg: float, radar_range: float, n_steps: int = 80):
    """
    Fan-shaped polygon vertices for one radar's FOV wedge.
    direction: +1 = looking right (left radar), -1 = looking left (right radar).
    Returns (xs, zs) as plain lists.
    """
    import numpy as np
    angles = np.linspace(pitch_deg - V_HALF, pitch_deg + V_HALF, n_steps)
    xs = [x_radar]
    zs = [H]
    for a_deg in angles:
        a = math.radians(a_deg)
        sin_a = math.sin(a)
        t_gnd = (H / sin_a) if sin_a > 1e-4 else radar_range * 50
        t = min(t_gnd, radar_range)
        xs.append(x_radar + direction * t * math.cos(a))
        zs.append(max(0.0, H - t * sin_a))
    xs.append(x_radar)
    zs.append(H)
    return xs, zs


def visualize_config(cfg: Config, rank: int = 1, save_path: Optional[str] = None) -> None:
    """
    Side-elevation FOV coverage plot for a single radar configuration.

    Left radar (blue)  at x = -y_offset, looking right (+x).
    Right radar (orange) at x = Y_RFP_NEAR + y_offset, looking left (-x).
    x = 0 is the left footpath / verge inner boundary.

    Layout to avoid all text overlap:
    - Zone labels go in a coloured strip BELOW the ground line (y < 0).
    - Target silhouettes have all text INSIDE the rectangle.
    - Radar labels sit on the OUTER side of each pole (away from road).
    - Reference height labels are on the LEFT margin only.
    - FOV info and occlusion info use axes-fraction coords (transAxes).
    - Title uses ax.set_title (auto-placed, never overlaps content).
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if save_path else matplotlib.get_backend())
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — pip install matplotlib")
        return

    shift = cfg.y_offset
    H     = cfg.H
    pitch = cfg.pitch

    x_lr = -shift                 # left radar  (looking right, +x)
    x_rr = Y_RFP_NEAR + shift     # right radar (looking left,  -x)

    # ── Zone definitions ─────────────────────────────────────────────────────
    # (xl, xr, label, bg, left_flag, right_flag, h_target)
    ZONES = [
        (-FOOTPATH_WIDTH, 0,        "L. Footpath", "#d5f0d5", "near_footpath",  "right_footpath", H_PED  ),
        (Y_VERGE_NEAR, Y_VERGE_FAR, "L. Verge",    "#e8d5f0", "left_verge",     "right_verge",    H_PED  ),
        (Y_L1_NEAR, Y_L1_FAR,       "Lane 1",      "#cce5ff", "lane1_truck",    "lane4_truck",    H_TRUCK),
        (Y_L2_NEAR, Y_L2_FAR,       "Lane 2",      "#ccf0e4", "lane2_truck",    "lane3_truck",    H_TRUCK),
        (Y_L3_NEAR, Y_L3_FAR,       "Lane 3",      "#fffacc", "lane3_truck",    "lane2_truck",    H_TRUCK),
        (Y_L4_NEAR, Y_L4_FAR,       "Lane 4",      "#ffe5cc", "lane4_truck",    "lane1_truck",    H_TRUCK),
        (Y_RV_NEAR,  Y_RV_FAR,      "R. Verge",    "#e8d5f0", "right_verge",    "left_verge",     H_PED  ),
        (Y_RFP_NEAR, Y_RFP_FAR,     "R. Footpath", "#d5f0d5", "right_footpath", "near_footpath",  H_PED  ),
    ]

    # ── Figure: 3-row gridspec ────────────────────────────────────────────────
    # Row 0 (tall) : elevation cross-section  |  coverage table
    # Row 1 (thin) : occlusion + params info bar  (spans both columns)
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[5, 1],
        height_ratios=[10, 1],
        hspace=0.10, wspace=0.04,
    )
    ax     = fig.add_subplot(gs[0, 0])   # main elevation view
    ax_tbl = fig.add_subplot(gs[0, 1])   # coverage table
    ax_bar = fig.add_subplot(gs[1, :])   # info bar
    ax.set_facecolor("#f7f9fc")
    ax_tbl.axis("off")
    ax_bar.axis("off")

    # Axis limits: extra space below ground for zone-label strip
    x_min   = -FOOTPATH_WIDTH - 0.5
    x_max   = Y_RFP_FAR + 0.5
    STRIP_H = 1.4          # height of zone-label strip below ground
    z_max   = H + 3.5      # headroom above radar

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-STRIP_H, z_max)

    # ── Zone-label strip (below ground line, y < 0) ───────────────────────────
    for (xl, xr, label, bg, lf, rf, ht) in ZONES:
        ax.add_patch(mpatches.Rectangle(
            (xl, -STRIP_H), xr - xl, STRIP_H,
            facecolor=bg, edgecolor="#aaaaaa", linewidth=0.6, alpha=0.85, zorder=2,
        ))
        ax.text((xl + xr) / 2, -STRIP_H / 2, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#222222", zorder=3)

    # ── Zone backgrounds (above ground) ──────────────────────────────────────
    for (xl, xr, label, bg, lf, rf, ht) in ZONES:
        ax.add_patch(mpatches.Rectangle(
            (xl, 0), xr - xl, z_max,
            facecolor=bg, edgecolor="#cccccc", linewidth=0.5, alpha=0.40, zorder=1,
        ))
        ax.axvline(xl, color="#aaaaaa", lw=0.7, ls="--", alpha=0.6, zorder=2)
    ax.axvline(Y_RFP_FAR, color="#aaaaaa", lw=0.7, ls="--", alpha=0.6, zorder=2)

    # ── Ground line ───────────────────────────────────────────────────────────
    ax.axhline(0, color="#5d4037", linewidth=2.5, zorder=6)

    # ── Target silhouettes (all text inside the rectangle) ────────────────────
    for (xl, xr, label, bg, lf, rf, ht) in ZONES:
        lcov = getattr(cfg, lf, False)
        rcov = getattr(cfg, rf, False)
        ok   = lcov or rcov
        ec   = "#1a7a40" if ok else "#c0392b"
        fc   = "#b2f2ce" if ok else "#ffc0c0"
        cx   = (xl + xr) / 2
        w    = (xr - xl) * 0.52

        ax.add_patch(mpatches.Rectangle(
            (cx - w / 2, 0.05), w, ht - 0.10,
            linewidth=2.0, edgecolor=ec, facecolor=fc, alpha=0.70, zorder=4,
        ))
        # Height at top-inside
        ax.text(cx, ht - 0.15, f"{ht}m",
                ha="center", va="top", fontsize=7.0,
                color="#333333", fontweight="bold", zorder=5, clip_on=True)
        # Coverage badge centred vertically
        who = ("L+R" if (lcov and rcov) else
               ("L"  if lcov else ("R" if rcov else "✗")))
        ax.text(cx, ht / 2, who,
                ha="center", va="center", fontsize=9.0,
                color=ec, fontweight="bold", zorder=5, clip_on=True)

    # ── FOV wedges ────────────────────────────────────────────────────────────
    left_xs,  left_zs  = _make_fov_polygon(x_lr, +1, H, pitch, RADAR_RANGE_M)
    right_xs, right_zs = _make_fov_polygon(x_rr, -1, H, pitch, RADAR_RANGE_M)

    ax.fill(left_xs,  left_zs,  color="#1a6ebd", alpha=0.18, zorder=3, label="Left radar FOV")
    ax.fill(right_xs, right_zs, color="#c0470a", alpha=0.18, zorder=3, label="Right radar FOV")
    for xs, zs, col in [(left_xs, left_zs, "#1a6ebd"), (right_xs, right_zs, "#c0470a")]:
        ax.plot([xs[0], xs[1]],   [zs[0], zs[1]],   color=col, lw=2.0, zorder=4)
        ax.plot([xs[0], xs[-2]], [zs[0], zs[-2]], color=col, lw=2.0, zorder=4)

    # ── Radar poles and labels ─────────────────────────────────────────────────
    # Labels go on the OUTER side of each pole (away from road) to avoid FOV overlap.
    for x_r, col, side, ha_side, x_lbl in [
        (x_lr, "#1a6ebd", "Left",  "right", x_lr - 0.25),
        (x_rr, "#c0470a", "Right", "left",  x_rr + 0.25),
    ]:
        ax.plot([x_r, x_r], [0, H], color=col, lw=4.0, zorder=5, solid_capstyle="round")
        ax.scatter([x_r], [H], s=200, color=col, zorder=7, marker="D",
                   edgecolors="white", linewidths=2.0)
        ax.text(x_lbl, H + 0.2,
                f"{side} Radar\nH = {H:.1f} m\n{pitch}° tilt",
                ha=ha_side, va="bottom", fontsize=9.0, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, alpha=0.95),
                zorder=8)

    # ── Reference height lines (labels on LEFT margin, clear of silhouettes) ──
    for ht, lbl, col in [
        (H_TRUCK, f"Truck  {H_TRUCK}m", "#8b1a1a"),
        (H_CAR,   f"Car  {H_CAR}m",    "#0a3d6b"),
        (H_PED,   f"Ped  {H_PED}m",    "#145a32"),
    ]:
        ax.axhline(ht, color=col, lw=1.0, ls=":", alpha=0.55, zorder=2)
        ax.text(x_min + 0.1, ht + 0.07, lbl,
                ha="left", va="bottom", fontsize=7.5, color=col, zorder=3)

    ax.axhline(H, color="gray", lw=0.7, ls="-.", alpha=0.28, zorder=2)

    # ── FOV window box (top-left corner, using axes fraction) ─────────────────
    ax.text(0.005, 0.99,
            f"V-FOV = {V_FOV_DEG:.0f}°  |  [{pitch - V_HALF:.0f}° – {pitch + V_HALF:.0f}°] depression\n"
            f"H-FOV = {H_FOV_DEG:.0f}°  |  Range = {RADAR_RANGE_M:.0f} m",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888", alpha=0.90),
            zorder=9)

    # ── Legend, grid, axes labels ─────────────────────────────────────────────
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    ax.grid(axis="y", color="gray", alpha=0.15, lw=0.5, ls="--")
    ax.set_xlabel("Lateral distance (m)  —  x = 0 at left footpath / verge boundary",
                  fontsize=10)
    ax.set_ylabel("Height above ground (m)", fontsize=10)
    # Suppress negative y-tick labels (they live in the label strip)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0])

    # ── Coverage table ────────────────────────────────────────────────────────
    tbl_rows = []
    for (xl, xr, label, bg, lf, rf, ht) in ZONES:
        lcov = getattr(cfg, lf, False)
        rcov = getattr(cfg, rf, False)
        tbl_rows.append([
            label.replace(". ", ".\n"),
            "✓" if lcov else "–",
            "✓" if rcov else "–",
            "✓" if (lcov or rcov) else "✗",
        ])

    tbl = ax_tbl.table(
        cellText=tbl_rows,
        colLabels=["Zone", "Left", "Right", "Net"],
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    tbl.scale(1.0, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            txt = cell.get_text().get_text()
            if col == 3:
                ok_cell = txt == "✓"
                cell.set_facecolor("#c8efcc" if ok_cell else "#f5c6c6")
                cell.set_text_props(
                    color="#145a32" if ok_cell else "#922b21",
                    fontweight="bold", size=11,
                )
            elif txt == "✓":
                cell.set_facecolor("#e8f8ee")
                cell.set_text_props(color="#1a7a40")
            else:
                cell.set_facecolor("#fdf5f5")
                cell.set_text_props(color="#bbbbbb")

    ax_tbl.set_title("Zone Coverage\n(Left / Right / Net)",
                     fontsize=9.5, pad=8, fontweight="bold")

    # ── Info bar (occlusion + target heights) ─────────────────────────────────
    def occ(flag): return "✓" if getattr(cfg, flag) else "✗"
    info = (
        f"Occlusion-free over L1 truck ({H_TRUCK}m):   "
        f"L1→L2 {occ('occ_l1_to_l2')}   L1→L3 {occ('occ_l1_to_l3')}   L1→L4 {occ('occ_l1_to_l4')}"
        f"     |     "
        f"L2→L3 {occ('occ_l2_to_l3')}   L2→R.Verge {occ('occ_l2_to_rv')}"
        f"     |     "
        f"Targets:  Truck {H_TRUCK}m  ·  Car {H_CAR}m  ·  Ped {H_PED}m"
        f"     |     "
        f"Footpath depth = {shift:.1f} m  (0 = inner edge, 6 = outer edge)"
    )
    ax_bar.text(0.5, 0.5, info, ha="center", va="center",
                fontsize=9.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.45", fc="#eef2f7", ec="#c0c8d0", alpha=0.95),
                transform=ax_bar.transAxes)

    # ── Title ─────────────────────────────────────────────────────────────────
    # ── Title (on main axes – never overlaps figure content) ─────────────────
    pct = cfg.score / MAX_SCORE * 100
    ax.set_title(
        f"Rank #{rank}  ·  H = {H:.1f} m  ·  Pitch = {pitch}°  ·  "
        f"Footpath depth = {shift:.1f} m  ·  "
        f"Score = {cfg.score:.0f} / {MAX_SCORE}  ({pct:.0f}%)  ·  "
        f"max_range = {cfg.max_range_m} m",
        fontsize=11, fontweight="bold", pad=12,
    )

    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08, wspace=0.06)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_scores(configs: List[Config]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    heights  = [c.H for c in configs]
    pitches  = [c.pitch for c in configs]
    scores   = [c.score for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = axes[0].scatter(heights, pitches, c=scores, cmap="viridis", s=18, alpha=0.7)
    axes[0].set_xlabel("Radar height H (m)")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].set_title("Score vs H and pitch (all y_offsets)")
    plt.colorbar(sc1, ax=axes[0], label="score")

    # Best y_offset per (H, pitch)
    best = {}
    for c in configs:
        key = (c.H, c.pitch)
        if key not in best or c.score > best[key].score:
            best[key] = c
    bh = [k[0] for k in best]; bp = [k[1] for k in best]
    bs = [best[k].score for k in best]; by = [best[k].y_offset for k in best]
    sc2 = axes[1].scatter(bh, bp, c=by, cmap="plasma", s=25, alpha=0.8)
    axes[1].set_xlabel("Radar height H (m)")
    axes[1].set_ylabel("Pitch (deg)")
    axes[1].set_title("Best y_offset (m into footpath) per (H, pitch)")
    plt.colorbar(sc2, ax=axes[1], label="y_offset (m)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Detailed summary for a single config
# ---------------------------------------------------------------------------

def print_detail(cfg: Config) -> None:
    shift = cfg.y_offset
    print(f"\n{'='*60}")
    print(f"  H={cfg.H}m  pitch={cfg.pitch}deg  y_offset={cfg.y_offset}m")
    print(f"  Score={cfg.score:.0f}  max_range={cfg.max_range_m}m")
    print(f"{'='*60}")

    zones = [
        ("Left footpath (behind radar)", -(FOOTPATH_WIDTH - shift), 0.0,  H_PED,   "near_footpath"),
        ("Left verge",    Y_VERGE_NEAR+shift, Y_VERGE_FAR+shift,  H_PED,   "left_verge"),
        ("Lane 1 truck",  Y_L1_NEAR+shift,   Y_L1_FAR+shift,    H_TRUCK,  "lane1_truck"),
        ("Lane 1 car",    Y_L1_NEAR+shift,   Y_L1_FAR+shift,    H_CAR,    "lane1_car"),
        ("Lane 2 truck",  Y_L2_NEAR+shift,   Y_L2_FAR+shift,    H_TRUCK,  "lane2_truck"),
        ("Lane 2 car",    Y_L2_NEAR+shift,   Y_L2_FAR+shift,    H_CAR,    "lane2_car"),
        ("Lane 3 truck",  Y_L3_NEAR+shift,   Y_L3_FAR+shift,    H_TRUCK,  "lane3_truck"),
        ("Lane 4 truck",  Y_L4_NEAR+shift,   Y_L4_FAR+shift,    H_TRUCK,  "lane4_truck"),
        ("Right verge",   Y_RV_NEAR+shift,   Y_RV_FAR+shift,    H_PED,    "right_verge"),
        ("Right footpath",Y_RFP_NEAR+shift,  Y_RFP_FAR+shift,   H_PED,    "right_footpath"),
    ]

    for label, yn, yf, ht, flag in zones:
        covered = getattr(cfg, flag)
        if yn <= 0:
            depr_n = depression_deg(cfg.H, ht, abs(yn)) if abs(yn) > 0.01 else 999
            depr_f_str = "n/a (behind)"
        else:
            depr_n = depression_deg(cfg.H, ht, yn)
            depr_f = depression_deg(cfg.H, ht, yf)
            depr_f_str = f"{depr_f:5.1f}°"
        fov_lo = cfg.pitch - V_HALF
        fov_hi = cfg.pitch + V_HALF
        mark = "OK" if covered else "--"
        print(f"  [{mark}] {label:<25} depr_near={depr_n:5.1f}°  depr_far={depr_f_str}  fov=[{fov_lo:.0f},{fov_hi:.0f}]°")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sweep radar geometry parameters.")
    parser.add_argument("--csv",      metavar="FILE", help="Write all results to CSV file")
    parser.add_argument("--plot",     action="store_true", help="Show score scatter plots")
    parser.add_argument("--top",      type=int, default=20, help="Number of top configs to print (default 20)")
    parser.add_argument("--detail",   action="store_true", help="Print detailed breakdown for top-5 configs")
    parser.add_argument("--viz",      metavar="N", type=int, nargs="?", const=5,
                        help="Show FOV elevation plots for top N configs (default 5); requires matplotlib")
    parser.add_argument("--save-viz", metavar="DIR",
                        help="Save FOV plots as PNG files to DIR instead of showing interactively")
    args = parser.parse_args()

    print(f"Sweeping {len(HEIGHT_RANGE)} heights x {len(PITCH_RANGE)} pitches "
          f"x {len(YOFFSET_RANGE)} y_offsets = "
          f"{len(HEIGHT_RANGE)*len(PITCH_RANGE)*len(YOFFSET_RANGE)} configurations...")
    print(f"V_FOV={V_FOV_DEG}°  H_FOV={H_FOV_DEG}°  range={RADAR_RANGE_M}m\n")

    results = run_sweep()

    print(f"\nTop {args.top} configurations (Y=covered, .=not covered):\n")
    print("  L1T/L1C = Lane1 Truck/Car  L2T/L2C = Lane2 Truck/Car")
    print("  L3T/L4T = Lane3/4 Truck  RV = Right Verge  RFP = Right Footpath")
    print("  LV = Left Verge  NFP = Near Footpath  o12/o13/o14 = Occlusion-free L1->L2/L3/L4\n")
    print_table(results, top_n=args.top)

    if args.detail:
        for c in results[:5]:
            print_detail(c)

    # Perfect-score configs
    max_score = results[0].score
    perfect = [c for c in results if c.score == max_score]
    print(f"\nTotal configs with max score ({max_score:.0f}): {len(perfect)}")
    if perfect:
        print("\nLowest-height perfect config:")
        best_low_h = min(perfect, key=lambda c: (c.H, c.pitch))
        print_detail(best_low_h)

        print("\nLowest-pitch perfect config (gentlest tilt):")
        best_low_p = min(perfect, key=lambda c: (c.pitch, c.H))
        print_detail(best_low_p)

    if args.csv:
        write_csv(results, args.csv)

    if args.plot:
        plot_scores(results)

    n_viz = args.viz
    save_dir = args.save_viz
    if n_viz is not None or save_dir is not None:
        n_viz = n_viz or 5
        print(f"\nGenerating FOV visualizations for top {n_viz} configs...")
        for i, c in enumerate(results[:n_viz], 1):
            path = None
            if save_dir:
                fname = f"rank{i:02d}_H{c.H:.1f}_p{c.pitch}_y{c.y_offset:.1f}.png"
                path = os.path.join(save_dir, fname)
            visualize_config(c, rank=i, save_path=path)


if __name__ == "__main__":
    main()
