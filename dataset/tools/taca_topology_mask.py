import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict


@dataclass
class Radar:
    radar_id: str
    x: float
    y: float
    phi_deg: float


def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def build_bev_grid(x_min, x_max, y_min, y_max, resolution):
    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def compute_radar_maps(radar, X, Y, R_max, fov_half_angle_deg, theta_3db_deg):
    phi = np.deg2rad(radar.phi_deg)
    theta_max = np.deg2rad(fov_half_angle_deg)
    theta_3db = np.deg2rad(theta_3db_deg)

    sigma_theta = theta_3db / np.sqrt(2 * np.log(2))

    dx = X - radar.x
    dy = Y - radar.y

    r = np.sqrt(dx**2 + dy**2)
    alpha = np.arctan2(dy, dx)
    delta_theta = wrap_to_pi(alpha - phi)

    H = ((r <= R_max) & (np.abs(delta_theta) <= theta_max)).astype(float)

    range_decay_power = 4.0
    beta = 10.0

    base_range = (R_max / (r + R_max)) ** range_decay_power
    w_range = H * np.log1p(beta * base_range) / np.log1p(beta)

    w_angle = np.exp(-(delta_theta**2) / (2 * sigma_theta**2))

    w = H * w_range * w_angle

    return H, w


def compute_overlap_matrices(
    radars,
    X,
    Y,
    R_max,
    fov_half_angle_deg,
    theta_3db_deg,
    eps=1e-8,
):
    M = len(radars)

    gamma = 0.5

    # -------------------------
    # BEV maps: used for coverage and plotting
    # -------------------------
    hard_maps = []
    reliability_maps = []

    for radar in radars:
        H, w = compute_radar_maps(
            radar=radar,
            X=X,
            Y=Y,
            R_max=R_max,
            fov_half_angle_deg=fov_half_angle_deg,
            theta_3db_deg=theta_3db_deg,
        )
        hard_maps.append(H)
        reliability_maps.append(w)

    hard_maps = np.array(hard_maps)
    reliability_maps = np.array(reliability_maps)

    bev_area = X.size

    # -------------------------
    # Intrinsic grid: not clipped by task BEV
    # -------------------------
    xs_radar = np.array([r.x for r in radars])
    ys_radar = np.array([r.y for r in radars])

    x_min_intr = np.min(xs_radar) - R_max
    x_max_intr = np.max(xs_radar) + R_max
    y_min_intr = np.min(ys_radar) - R_max
    y_max_intr = np.max(ys_radar) + R_max

    if X.shape[1] > 1:
        resolution = X[0, 1] - X[0, 0]
    else:
        resolution = 1.0

    X_intr, Y_intr = build_bev_grid(
        x_min=x_min_intr,
        x_max=x_max_intr,
        y_min=y_min_intr,
        y_max=y_max_intr,
        resolution=resolution,
    )

    intrinsic_hard_maps = []
    intrinsic_reliability_maps = []

    for radar in radars:
        H_intr, w_intr = compute_radar_maps(
            radar=radar,
            X=X_intr,
            Y=Y_intr,
            R_max=R_max,
            fov_half_angle_deg=fov_half_angle_deg,
            theta_3db_deg=theta_3db_deg,
        )
        intrinsic_hard_maps.append(H_intr)
        intrinsic_reliability_maps.append(w_intr)

    intrinsic_hard_maps = np.array(intrinsic_hard_maps)
    intrinsic_reliability_maps = np.array(intrinsic_reliability_maps)

    # -------------------------
    # Final score:
    # S_ij = Q_intrinsic_ij * (C_bev_ij)^gamma
    # gamma = 0.5
    # -------------------------
    S = np.zeros((M, M), dtype=float)
    Q_intrinsic_matrix = np.zeros((M, M), dtype=float)
    C_bev_matrix = np.zeros((M, M), dtype=float)

    for i in range(M):
        for j in range(M):
            if i == j:
                S[i, j] = 1.0
                Q_intrinsic_matrix[i, j] = 1.0
                C_bev_matrix[i, j] = 1.0
                continue

            # Intrinsic quality, computed outside the task BEV crop
            Hi_intr = intrinsic_hard_maps[i]
            Hj_intr = intrinsic_hard_maps[j]

            wi_intr = intrinsic_reliability_maps[i]
            wj_intr = intrinsic_reliability_maps[j]

            shared_hard_intr = Hi_intr * Hj_intr
            shared_soft_intr = shared_hard_intr * np.sqrt(wi_intr * wj_intr)

            intrinsic_shared_area = np.sum(shared_hard_intr)

            if intrinsic_shared_area == 0:
                Q_intrinsic = 0.0
            else:
                Q_intrinsic = np.sum(shared_soft_intr) / (intrinsic_shared_area + eps)

            # BEV coverage, computed only inside the task BEV crop
            Hi_bev = hard_maps[i]
            Hj_bev = hard_maps[j]

            shared_hard_bev = Hi_bev * Hj_bev
            bev_shared_area = np.sum(shared_hard_bev)

            C_bev = bev_shared_area / (bev_area + eps)

            S_final = Q_intrinsic * (C_bev ** gamma)

            S[i, j] = S_final
            Q_intrinsic_matrix[i, j] = Q_intrinsic
            C_bev_matrix[i, j] = C_bev

    # No initial thresholding
    S_tilde = S.copy()

    maps = {
        "hard_maps": hard_maps,
        "reliability_maps": reliability_maps,
        "Q_intrinsic_matrix": Q_intrinsic_matrix,
        "C_bev_matrix": C_bev_matrix,
        "intrinsic_hard_maps": intrinsic_hard_maps,
        "intrinsic_reliability_maps": intrinsic_reliability_maps,
    }

    return S, S_tilde, maps


def mst_topology_ordering(
    S_tilde: np.ndarray,
    radar_ids: List[str],
) -> Tuple[List[int], List[str]]:
    M = S_tilde.shape[0]

    edges = []
    for i in range(M):
        for j in range(i + 1, M):
            w = max(S_tilde[i, j], S_tilde[j, i])
            if w > 0:
                edges.append((w, i, j))

    edges.sort(reverse=True)

    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    mst_adj = defaultdict(list)

    for w, i, j in edges:
        if union(i, j):
            mst_adj[i].append((j, w))
            mst_adj[j].append((i, w))

    for i in range(M):
        mst_adj[i] = sorted(mst_adj[i], key=lambda x: x[1], reverse=True)

    # -------------------------
    # New: choose DFS start as an endpoint of the MST diameter
    # -------------------------
    def farthest_node(start):
        stack = [(start, -1, 0)]
        farthest = (start, 0)

        while stack:
            u, parent_node, depth = stack.pop()

            if depth > farthest[1]:
                farthest = (u, depth)

            for v, _ in mst_adj[u]:
                if v != parent_node:
                    stack.append((v, u, depth + 1))

        return farthest[0]

    if len(edges) > 0:
        a = farthest_node(0)
        b = farthest_node(a)

        # Either a or b is valid. Pick the weaker-affinity endpoint
        # so traversal starts from a peripheral radar.
        start = min(a, b, key=lambda i: np.sum(S_tilde[i]))
    else:
        start = int(np.argmax(np.sum(S_tilde, axis=1)))

    ordering = []
    seen = set()

    def dfs(u):
        seen.add(u)
        ordering.append(u)

        for v, _ in mst_adj[u]:
            if v not in seen:
                dfs(v)

    dfs(start)

    while len(ordering) < M:
        remaining = [i for i in range(M) if i not in seen]
        next_start = min(remaining, key=lambda i: np.sum(S_tilde[i]))
        dfs(next_start)

    ordered_ids = [radar_ids[i] for i in ordering]

    return ordering, ordered_ids


def rowwise_softmax_over_neighbors(scores, beta=1.0, eps=1e-12):
    valid = scores > 0

    probs = np.zeros_like(scores, dtype=float)

    if np.sum(valid) == 0:
        return probs

    # KEY CHANGE: use log(scores) instead of scores
    logits = beta * np.log(scores[valid] + eps)

    # Numerical stability
    logits = logits - np.max(logits)

    exp_logits = np.exp(logits)
    probs[valid] = exp_logits / (np.sum(exp_logits) + eps)

    return probs


def select_implicit_K(
    S_tilde: np.ndarray,
    ordering: List[int],
    target_mass: float = 0.95,
    softmax_beta: float = 1.0,
    include_self: bool = True,
) -> Tuple[int, Dict[str, object]]:
    M = len(ordering)

    S_ordered = S_tilde[np.ix_(ordering, ordering)]
    topo_probs = np.zeros((M, M), dtype=float)

    for m in range(M):
        scores = S_ordered[m].copy()

        if not include_self:
            scores[m] = 0.0

        topo_probs[m] = rowwise_softmax_over_neighbors(
            scores=scores,
            beta=softmax_beta,
        )

    max_K = M - 1
    captured_mass_by_K = []

    for K in range(max_K + 1):
        row_masses = []

        for m in range(M):
            mass = 0.0

            for n in range(M):
                if abs(m - n) <= K:
                    mass += topo_probs[m, n]

            row_masses.append(mass)

        row_masses = np.array(row_masses)

        captured_mass_by_K.append(
            {
                "K": K,
                "mean_mass": float(np.mean(row_masses)),
                "min_mass": float(np.min(row_masses)),
                "median_mass": float(np.median(row_masses)),
                "row_masses": row_masses,
            }
        )

    selected_K = max_K

    for item in captured_mass_by_K:
        if item["mean_mass"] >= target_mass:
            selected_K = item["K"]
            break

    info = {
        "S_ordered": S_ordered,
        "topology_softmax_probs": topo_probs,
        "captured_mass_by_K": captured_mass_by_K,
        "target_mass": target_mass,
        "softmax_beta": softmax_beta,
    }

    return selected_K, info


def build_windowed_attention_mask(
    S_tilde: np.ndarray,
    ordering: List[int],
    K: int,
    include_self: bool = True,
) -> np.ndarray:
    M = len(ordering)
    mask = np.zeros((M, M), dtype=int)

    for m in range(M):
        for n in range(M):
            i = ordering[m]
            j = ordering[n]

            if not include_self and i == j:
                continue

            if abs(m - n) <= K and S_tilde[i, j] > 0:
                mask[m, n] = 1

    return mask


def compute_windowed_taca_mask(
    radars: List[Radar],
    bev_limits: Tuple[float, float, float, float],
    resolution: float,
    R_max: float,
    fov_half_angle_deg: float,
    theta_3db_deg: float,
    target_mass: float = 0.95,
    softmax_beta: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, object]:

    x_min, x_max, y_min, y_max = bev_limits

    X, Y = build_bev_grid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=resolution,
    )

    S, S_tilde, maps = compute_overlap_matrices(
        radars=radars,
        X=X,
        Y=Y,
        R_max=R_max,
        fov_half_angle_deg=fov_half_angle_deg,
        theta_3db_deg=theta_3db_deg,
        eps=eps,
    )

    radar_ids = [r.radar_id for r in radars]

    ordering, ordered_ids = mst_topology_ordering(
        S_tilde=S_tilde,
        radar_ids=radar_ids,
    )

    selected_K, K_info = select_implicit_K(
        S_tilde=S_tilde,
        ordering=ordering,
        target_mass=target_mass,
        softmax_beta=softmax_beta,
        include_self=False,
    )

    attention_mask = build_windowed_attention_mask(
        S_tilde=S_tilde,
        ordering=ordering,
        K=selected_K,
        include_self=True,
    )

    return {
        "X": X,
        "Y": Y,
        "ordered_radar_indices": ordering,
        "ordered_radar_ids": ordered_ids,
        "soft_overlap_matrix": S,
        "thresholded_overlap_matrix": S_tilde,
        "selected_K": selected_K,
        "K_selection_info": K_info,
        "windowed_attention_mask": attention_mask,
        "hard_maps": maps["hard_maps"],
        "reliability_maps": maps["reliability_maps"],
        "Q_intrinsic_matrix": maps["Q_intrinsic_matrix"],
        "C_bev_matrix": maps["C_bev_matrix"],
        "intrinsic_hard_maps": maps["intrinsic_hard_maps"],
        "intrinsic_reliability_maps": maps["intrinsic_reliability_maps"],
    }


def plot_radar_poses(ax, radars, arrow_len=8.0):
    for radar in radars:
        phi = np.deg2rad(radar.phi_deg)
        dx = arrow_len * np.cos(phi)
        dy = arrow_len * np.sin(phi)

        ax.scatter(radar.x, radar.y, marker="x", s=80)
        ax.arrow(
            radar.x,
            radar.y,
            dx,
            dy,
            head_width=2.0,
            length_includes_head=True,
        )
        ax.text(radar.x + 1.0, radar.y + 1.0, radar.radar_id)


def plot_hard_fovs_with_overlaps(result, radars, bev_limits):
    hard_maps = result["hard_maps"]
    hard_overlap_count = np.sum(hard_maps, axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        hard_overlap_count,
        origin="lower",
        extent=[bev_limits[0], bev_limits[1], bev_limits[2], bev_limits[3]],
        aspect="equal",
    )

    plt.colorbar(im, ax=ax, label="Number of hard FoVs covering cell")
    plot_radar_poses(ax, radars)

    ax.set_title("Hard FoVs with Hard Overlaps")
    ax.set_xlabel("X BEV")
    ax.set_ylabel("Y BEV")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_soft_fov_overlaps(result, radars, bev_limits):
    hard_maps = result["hard_maps"]
    reliability_maps = result["reliability_maps"]

    M = hard_maps.shape[0]
    soft_overlap_map = np.zeros_like(hard_maps[0], dtype=float)

    for i in range(M):
        for j in range(i + 1, M):
            pair_soft_overlap = (
                hard_maps[i]
                * hard_maps[j]
                * np.sqrt(reliability_maps[i] * reliability_maps[j])
            )
            soft_overlap_map += pair_soft_overlap

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        soft_overlap_map,
        origin="lower",
        extent=[bev_limits[0], bev_limits[1], bev_limits[2], bev_limits[3]],
        aspect="equal",
    )

    plt.colorbar(im, ax=ax, label="Summed soft overlap strength")
    plot_radar_poses(ax, radars)

    ax.set_title("Soft FoV Overlaps inside BEV")
    ax.set_xlabel("X BEV")
    ax.set_ylabel("Y BEV")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_final_mask_based_fovs(result, radars, bev_limits, use_soft=True):
    hard_maps = result["hard_maps"]
    reliability_maps = result["reliability_maps"]
    ordering = result["ordered_radar_indices"]
    ordered_ids = result["ordered_radar_ids"]
    mask = result["windowed_attention_mask"]

    M = len(ordering)
    final_overlap_map = np.zeros_like(hard_maps[0], dtype=float)

    for m in range(M):
        for n in range(M):
            if m == n:
                continue

            if mask[m, n] == 0:
                continue

            i = ordering[m]
            j = ordering[n]

            if use_soft:
                pair_map = hard_maps[i] * hard_maps[j] * np.sqrt(
                    reliability_maps[i] * reliability_maps[j]
                )
            else:
                pair_map = hard_maps[i] * hard_maps[j]

            final_overlap_map += pair_map

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        final_overlap_map,
        origin="lower",
        extent=[bev_limits[0], bev_limits[1], bev_limits[2], bev_limits[3]],
        aspect="equal",
    )

    label = "Final retained soft overlap strength" if use_soft else "Final retained hard overlap count"
    plt.colorbar(im, ax=ax, label=label)

    plot_radar_poses(ax, radars)

    for m in range(M):
        for n in range(m + 1, M):
            if mask[m, n] == 1 or mask[n, m] == 1:
                i = ordering[m]
                j = ordering[n]

                xi, yi = radars[i].x, radars[i].y
                xj, yj = radars[j].x, radars[j].y

                ax.plot([xi, xj], [yi, yj], linewidth=2)

    ax.set_title("Final Mask-Based FoV Overlaps")
    ax.set_xlabel("X BEV")
    ax.set_ylabel("Y BEV")
    ax.grid(True, alpha=0.3)

    text = "Ordered IDs: " + " → ".join(ordered_ids)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def plot_attention_mask(result):
    mask = result["windowed_attention_mask"]
    ordered_ids = result["ordered_radar_ids"]

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(mask, origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="Attention allowed")

    ax.set_xticks(np.arange(len(ordered_ids)))
    ax.set_yticks(np.arange(len(ordered_ids)))
    ax.set_xticklabels(ordered_ids)
    ax.set_yticklabels(ordered_ids)

    ax.set_xlabel("Key / Value radar")
    ax.set_ylabel("Query radar")
    ax.set_title("Final Windowed Attention Mask")

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            ax.text(j, i, str(mask[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def print_K_selection_summary(result):
    print("\nAutomatic K Selection Summary:")
    print(f"Target probability mass: {result['K_selection_info']['target_mass']}")
    print(f"Softmax beta: {result['K_selection_info']['softmax_beta']}")
    print(f"Selected K: {result['selected_K']}")

    print("\nCaptured mass by K:")
    for item in result["K_selection_info"]["captured_mass_by_K"]:
        print(
            f"K={item['K']:>2} | "
            f"mean={item['mean_mass']:.4f} | "
            f"median={item['median_mass']:.4f} | "
            f"min={item['min_mass']:.4f}"
        )


if __name__ == "__main__":

#CARLA config
    # inter_radar_dist = 25
    # radars = [
    #     Radar("R1", x=-50, y=-50, phi_deg=-135),
    #     Radar("R2", x=-50, y=-75, phi_deg=135),
    #     Radar("R3", x=-50+inter_radar_dist, y=-50, phi_deg=-135),
    #     Radar("R4", x=-50+inter_radar_dist, y=-75, phi_deg=135),

    #     Radar("R5", x=-50+3*inter_radar_dist, y=-50, phi_deg=-135),
    #     Radar("R6", x=-50+3*inter_radar_dist, y=-75, phi_deg=135),
    #     Radar("R7", x=-50+2*inter_radar_dist, y=-50, phi_deg=-135),
    #     Radar("R8", x=-50+2*inter_radar_dist, y=-75, phi_deg=135),

    # ]

    # bev_limits = (-80, 80.0, -90.0, -40.0)

    # #RadarScenes configuration
    radars = [
        Radar("S1", x=3.85925, y=-0.69908, phi_deg=-25),  # right inner
        Radar("S2", x=3.66101, y= 0.87680, phi_deg= 85),  # left outer
        Radar("S3", x=3.66198, y=-0.87376, phi_deg=-85),  # right outer
        Radar("S4", x=3.86152, y= 0.69770, phi_deg= 25),  # left inner
    ]


    bev_limits = (-64.0, 64.0, -64.0, 64.0)

    result = compute_windowed_taca_mask(
        radars=radars,
        bev_limits=bev_limits,
        resolution=0.5,
        R_max=60,
        fov_half_angle_deg=60.0,
        theta_3db_deg=15.0,
        target_mass=0.9,
        softmax_beta=1.0,
    )

#NuScenes configuration

#     radars = [
#     Radar("RADAR_FRONT",       x= 3.412, y= 0.000, phi_deg=   0.20),
#     Radar("RADAR_FRONT_LEFT",  x= 2.422, y= 0.800, phi_deg=  88.36),
#     Radar("RADAR_FRONT_RIGHT", x= 2.422, y=-0.800, phi_deg= -90.98),
#     Radar("RADAR_BACK_LEFT",   x=-0.562, y= 0.628, phi_deg= 174.41),
#     Radar("RADAR_BACK_RIGHT",  x=-0.562, y=-0.618, phi_deg=-176.11),
# ]

    # bev_limits = (-80.0, 85.0, -120.0, -10.0)
#     bev_limits = (-54.0, 54.0, -54.0, 54.0)

#     result = compute_windowed_taca_mask(
#         radars=radars,
#         bev_limits=bev_limits,
#         resolution=1.0,
#         R_max=70,
#         fov_half_angle_deg=60.0,
#         theta_3db_deg=15,
#         target_mass=0.9,
#         softmax_beta=1.0,
#     )

    print("Ordered Radar Indices:")
    print(result["ordered_radar_indices"])

    print("\nOrdered Radar IDs:")
    print(result["ordered_radar_ids"])

    print("\nIntrinsic Quality Matrix Q_intrinsic:")
    print(np.round(result["Q_intrinsic_matrix"], 4))

    print("\nBEV Coverage Matrix C_bev:")
    print(np.round(result["C_bev_matrix"], 4))

    print("\nFinal Score Matrix S = Q_intrinsic * sqrt(C_bev):")
    print(np.round(result["soft_overlap_matrix"], 4))

    print("\nNo-threshold Overlap Matrix S_tilde:")
    print(np.round(result["thresholded_overlap_matrix"], 4))

    print("\nOrdered Final Score Matrix:")
    print(np.round(result["K_selection_info"]["S_ordered"], 4))

    print("\nRow-wise Softmax Probabilities over Ordered Neighbors:")
    print(np.round(result["K_selection_info"]["topology_softmax_probs"], 4))

    print_K_selection_summary(result)

    mask = result["windowed_attention_mask"]
    ordered_ids = result["ordered_radar_ids"]

    print("\nTACA Attention Mask (rows = Query, cols = Key/Value):\n")

    header = "      " + " ".join([f"{rid:>5}" for rid in ordered_ids])
    print(header)

    for i, rid in enumerate(ordered_ids):
        row_vals = " ".join([f"{mask[i, j]:>5}" for j in range(len(ordered_ids))])
        print(f"{rid:>5} {row_vals}")

    plot_hard_fovs_with_overlaps(result, radars, bev_limits)
    plot_soft_fov_overlaps(result, radars, bev_limits)
    plot_final_mask_based_fovs(result, radars, bev_limits, use_soft=True)
    plot_attention_mask(result)
