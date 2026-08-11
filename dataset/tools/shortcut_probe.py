#!/usr/bin/env python
'''
shortcut_probe.py — measure absolute-position (and other single-feature) class
leakage in a CARLA radar capture, reconstructing the original /tmp probe that
found pos_xy ~= 0.96 foreground mIoU on sensor_capture_20260605_182322.

Method (per the dataset-shortcut-leakage finding):
  * 2-class foreground task: pedestrian vs vehicle (car+truck).
  * kNN classifier per feature group; metric = macro IoU over the 2 classes.
  * Actor-level (group) split so points of one actor never straddle train/test
    -> kills the trivial "memorise the track" leak; what survives is genuine
    feature->class predictability. Actor ids are namespaced by capture so they
    never collide across the multi-capture campaign.
  * 5 seeds (5 different group splits); report mean +/- std.

Feature groups (all z-scored on train stats before kNN):
  pos_xy        [x, y]                      GLOBAL, sensor-invariant (early-fusion coords)
  range         [depth]                     LOCAL (per-sensor)
  azimuth       [azimuth]                   LOCAL (per-sensor)
  range_azimuth [depth, azimuth]            LOCAL polar (per-sensor) -- NOTE: pooled across
                                            sensors WITHOUT sensor id, mirroring early fusion
  rcs           [rcs_dBsm]                  legit radar feature
  vel           [velocity_mps]              legit radar feature
  rcs_vel       [rcs_dBsm, velocity_mps]
  local4        [depth, azimuth, vel, rcs]  the "all-local" sensor-frame vector
  all           [x, y, depth, azimuth, rcs, vel]

World (x,y) is computed with the SAME spherical->world transform the model's
CarlaRadarDataset uses, so pos_xy is exactly the coordinate the model sees.

Usage:
  python shortcut_probe.py --tag old  --csv /path/a/radar_data_labeled.csv
  python shortcut_probe.py --tag new  --csv /scratch/.../cap1/radar_data_labeled.csv \
                                             /scratch/.../cap2/radar_data_labeled.csv ...
Run with an env that has scikit-learn, e.g.
  /home/tamoghnd/miniconda3/envs/radarscenes_segmentation/bin/python
'''
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score

USECOLS = ['depth_m', 'azimuth_rad', 'altitude_rad', 'velocity_mps',
           'sensor_world_x_m', 'sensor_world_y_m', 'sensor_yaw_deg',
           'matched_actor_class', 'matched_actor_id', 'rcs_dBsm']

PED = 0
VEH = 1

# matched_actor_class string -> ClassificationLabel id (0-5), matching
# carla_dataset.CARLA_CLASS_TO_LABEL + labels.label_to_clabel exactly.
# unmatched / unknown -> STATIC(5).
CLABEL_CAR, CLABEL_PED, CLABEL_PEDGRP, CLABEL_2W, CLABEL_LV, CLABEL_STATIC = 0, 1, 2, 3, 4, 5
CLABEL_NAMES = {0: 'car', 1: 'ped', 2: 'ped_group', 3: 'two_wheeler',
                4: 'large_veh', 5: 'static'}
CLASS_TO_CLABEL = {
    'car': CLABEL_CAR, 'van': CLABEL_CAR,
    'truck': CLABEL_LV, 'bus': CLABEL_LV, 'trailer': CLABEL_LV,
    'motorcycle': CLABEL_2W, 'motorbike': CLABEL_2W,
    'bicycle': CLABEL_2W, 'rider': CLABEL_2W,
    'pedestrian': CLABEL_PED,
}


def spherical_to_world(depth, az, alt, sx, sy, yaw_deg):
    '''Exact copy of carla_dataset._spherical_to_world (BEV, yaw only).'''
    ca = np.cos(alt)
    lx = depth * ca * np.cos(az)
    ly = depth * ca * np.sin(az)
    yaw = np.radians(yaw_deg)
    cy, syaw = np.cos(yaw), np.sin(yaw)
    wx = sx + lx * cy - ly * syaw
    wy = sy + lx * syaw + ly * cy
    return wx, wy


def load_capture(path, cap_idx, multiclass=False, static_cap=60000):
    df = pd.read_csv(path, usecols=USECOLS)
    cls = df['matched_actor_class'].astype(str).str.lower().fillna('')
    clab = np.full(len(df), CLABEL_STATIC, dtype=np.int64)
    for name, c in CLASS_TO_CLABEL.items():
        clab[(cls == name).to_numpy()] = c
    matched = clab != CLABEL_STATIC                 # mapped to a foreground class

    if multiclass:
        y = clab
        keep = matched.copy()                       # keep all foreground
        stat_idx = np.where(~matched)[0]            # subsample static to bound memory
        if static_cap and len(stat_idx) > static_cap:
            rng = np.random.default_rng(1000 + cap_idx)
            stat_idx = rng.choice(stat_idx, static_cap, replace=False)
        keep[stat_idx] = True
    else:                                           # 2-class ped vs vehicle, fg only
        y = np.full(len(df), -1, dtype=np.int64)
        y[clab == CLABEL_PED] = PED
        y[(clab == CLABEL_CAR) | (clab == CLABEL_LV)] = VEH
        keep = y >= 0
    df = df[keep].reset_index(drop=True)
    y = y[keep]
    matched_k = matched[keep]

    depth = df['depth_m'].to_numpy(np.float64)
    az = df['azimuth_rad'].to_numpy(np.float64)
    alt = df['altitude_rad'].to_numpy(np.float64)
    wx, wy = spherical_to_world(depth, az, alt,
                                df['sensor_world_x_m'].to_numpy(np.float64),
                                df['sensor_world_y_m'].to_numpy(np.float64),
                                df['sensor_yaw_deg'].to_numpy(np.float64))
    rcs = df['rcs_dBsm'].to_numpy(np.float64)
    rcs = np.nan_to_num(rcs, nan=np.nanmedian(rcs))
    vel = np.nan_to_num(df['velocity_mps'].to_numpy(np.float64), nan=0.0)

    feats = {'x': wx, 'y': wy, 'depth': depth, 'azimuth': az,
             'rcs': rcs, 'vel': vel}
    # integer groups namespaced by capture: dynamic actors grouped (no track
    # straddle); static returns get unique negative ids (split per-point, which
    # mirrors the real same-site capture-level eval where clutter repeats).
    base = cap_idx * 100_000_000
    aid = np.nan_to_num(df['matched_actor_id'].to_numpy(np.float64), nan=0).astype(np.int64)
    groups = np.where(matched_k, base + aid,
                      -(base + np.arange(len(df)) + 1)).astype(np.int64)
    return feats, y, groups


GROUPS = {
    'pos_xy':        ['x', 'y'],
    'x_only':        ['x'],
    'y_only':        ['y'],
    'range':         ['depth'],
    'azimuth':       ['azimuth'],
    'range_azimuth': ['depth', 'azimuth'],
    'rcs':           ['rcs'],
    'vel':           ['vel'],
    'rcs_vel':       ['rcs', 'vel'],
    'local4':        ['depth', 'azimuth', 'vel', 'rcs'],
    'all':           ['x', 'y', 'depth', 'azimuth', 'rcs', 'vel'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--csv', nargs='+', required=True)
    ap.add_argument('--cap', type=int, default=80000,
                    help='max foreground points used (subsampled, fixed seed)')
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--test-size', type=float, default=0.3)
    ap.add_argument('--multiclass', action='store_true',
                    help='full ClassificationLabel task (car/large_veh/ped/'
                         'two_wheeler/static) instead of 2-class ped-vs-vehicle')
    ap.add_argument('--per-class-cap', type=int, default=30000,
                    help='multiclass: balanced cap per class')
    args = ap.parse_args()

    all_feats = {k: [] for k in ['x', 'y', 'depth', 'azimuth', 'rcs', 'vel']}
    ys, gs = [], []
    for ci, path in enumerate(args.csv):
        f, y, g = load_capture(path, ci, multiclass=args.multiclass)
        for k in all_feats:
            all_feats[k].append(f[k])
        ys.append(y); gs.append(g)
        if args.multiclass:
            comp = ' '.join(f'{CLABEL_NAMES[c]} {int((y==c).sum()):,}'
                            for c in sorted(set(y.tolist())))
            print(f'  loaded {path.split("/")[-2]}: {len(y):,} pts ({comp})',
                  flush=True)
        else:
            print(f'  loaded {path.split("/")[-2]}: {len(y):,} fg pts '
                  f'(ped {int((y==PED).sum()):,} / veh {int((y==VEH).sum()):,})',
                  flush=True)
    feats = {k: np.concatenate(v) for k, v in all_feats.items()}
    y = np.concatenate(ys)
    groups = np.concatenate(gs)

    rng = np.random.default_rng(0)
    if args.multiclass:
        present = sorted(set(y.tolist()))
        # balanced per-class subsample (5 seeds vary the SPLIT, not the sample)
        idx = np.concatenate([
            rng.choice(np.where(y == c)[0],
                       min((y == c).sum(), args.per_class_cap), replace=False)
            for c in present])
        labels = present
        names = [CLABEL_NAMES[c] for c in present]
    else:
        n = len(y)
        idx = (rng.choice(n, args.cap, replace=False) if n > args.cap
               else np.arange(n))
        labels = [PED, VEH]
        names = ['ped', 'veh']
    feats = {k: v[idx] for k, v in feats.items()}
    y = y[idx]; groups = groups[idx]
    n = len(y)
    bal = ' '.join(f'{CLABEL_NAMES[c] if args.multiclass else names[i]}'
                   f'={(y==c).mean():.2f}'
                   for i, c in enumerate(labels))
    print(f'\n[{args.tag}] {n:,} pts, {len(set(groups.tolist())):,} groups, '
          f'k={args.k}, {args.seeds} seeds | balance: {bal}\n', flush=True)

    # majority-class floor: predict the single most-common class -> macro IoU
    maj = max(labels, key=lambda c: (y == c).sum())
    floor = jaccard_score(y, np.full_like(y, maj), average='macro',
                          labels=labels, zero_division=0)

    hdr = 'per-class IoU [' + ' '.join(names) + ']'
    print(f'{"feature group":<14} {"macro mIoU":<13} {hdr}')
    print('-' * (29 + len(hdr)))
    for name, cols in GROUPS.items():
        X = np.column_stack([feats[c] for c in cols])
        macro, perclass = [], []
        for s in range(args.seeds):
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                    random_state=s)
            tr, te = next(gss.split(X, y, groups))
            sc = StandardScaler().fit(X[tr])
            knn = KNeighborsClassifier(n_neighbors=args.k, n_jobs=-1)
            knn.fit(sc.transform(X[tr]), y[tr])
            pred = knn.predict(sc.transform(X[te]))
            macro.append(jaccard_score(y[te], pred, average='macro',
                                       labels=labels, zero_division=0))
            perclass.append(jaccard_score(y[te], pred, average=None,
                                          labels=labels, zero_division=0))
        macro = np.array(macro); perclass = np.array(perclass).mean(0)
        pc = ' '.join(f'{v:.2f}' for v in perclass)
        print(f'{name:<14} {macro.mean():.3f}+/-{macro.std():.3f}  [{pc}]')
    print('-' * (29 + len(hdr)))
    print(f'{"majority floor":<14} {floor:.3f}        '
          f'(predict-all-{CLABEL_NAMES[maj] if args.multiclass else names[labels.index(maj)]})')


if __name__ == '__main__':
    main()
