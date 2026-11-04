# PointNet with early fusion

Per-point segmentation of the fused radar cloud: background, vehicle, pedestrian, cyclist.

**Early fusion:** for every CARLA frame, the returns of all radars are pooled into one point
cloud in world coordinates, with no sensor id. PointNet then labels every point.

**Why PointNet for the denser data:** it is a shared per-point MLP plus one max-pool. Cost is
linear in the number of points, and every frame is subsampled to a fixed size for training.
A fused frame has about 4,000 points (8 radars), so a 4,096-point PointNet batch covers the
whole frame.

## 1. Build the fused frames (once per capture, no GPU)

From the repo root:

```bash
python pointnet/build_frames.py dataset/Data/sensor_capture_20260927_125157 dataset/Data/sensor_capture_20260929_012142 dataset/Data/sensor_capture_20260929_072526 dataset/Data/sensor_capture_20260929_210312
```

This reads `radar_data_labeled.csv` (about 2 min per 5 GB) and writes
`<capture>/pointnet_frames/` (about 0.5 GB per capture). The four campaign captures are
already built.

Per point: world x, y, z, radial velocity, RCS, and the unit line of sight from its radar.
The default `--mode measured` keeps only returns with `visible == 1` and rebuilds positions
from the noisy depth and azimuth, so the model sees what a real radar would report.
`--mode clean` gives every return at its exact hit position.

## 2. Install PyTorch with CUDA

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Train

```bash
python pointnet/train.py --data "dataset/Data/sensor_capture_*/pointnet_frames"
```

The GPU is used automatically, with mixed precision. Defaults: 40 epochs, batch 16,
4,096 points per frame, full-width PointNet (about 0.9M parameters). Each epoch prints
validation IoU per class. The best epoch is picked by foreground mIoU, the mean over the
vehicle, pedestrian and cyclist IoU.

Output in `runs/pointnet_<time>/`: `best.pt`, `last.pt`, `log.csv` (per epoch),
`results.md` / `results.json` (val and test IoU, precision, recall, confusion matrix).

Useful flags:

| Flag | Default | Use |
|---|---|---|
| `--num-points` | 4096 | points per training frame (random subset) |
| `--batch-size` | 16 | raise to 32-64 on a big GPU |
| `--width` | 1.0 | channel multiplier; 0.5 is 4x cheaper |
| `--frames-per-epoch` | all | cap epoch length |
| `--train-stride` / `--eval-stride` | 1 / 2 | skip near-duplicate frames |
| `--aug` | light | `strong` = random 360 deg rotation, mirror, 10 m shift (see below) |
| `--feature-tnet` | off | PointNet's 64x64 feature alignment |
| `--workers` | 4 | data loader processes |

Evaluate a saved model: `python pointnet/train.py --data ... --eval-only --checkpoint runs/pointnet_X/best.pt`

## Split: whole scenarios are held out

Default: val = scenarios 4 and 10, test = scenario 9, train = the rest
(`--val-scenarios`, `--test-scenarios`).

Every campaign so far used seed 42. So the same scenario in two captures is near-identical
traffic, and a random-frame or per-capture split would put the same vehicles in train and
test. Scenarios 4, 9 and 10 appear in only one capture each.

**Remaining leak:** ambient pedestrians, cyclists and city cars are also generated from seed 42,
so they repeat across scenarios. Pedestrian and cyclist scores are therefore optimistic. Fix:
give every campaign row its own `seed` and re-capture.

## Position shortcut

With `--aug light`, the model sees absolute world coordinates. On a fixed infrastructure rig
this is legitimate prior knowledge (cars are on the road, people on the sidewalk). It does not
transfer to a new site. `--aug strong` randomly rotates, mirrors and shifts each frame, which
removes that cue. Train both and report the gap.

## Baseline (CPU test run, 2026-10-14)

Reduced budget to check the pipeline: every 3rd frame of the four campaign captures
(5,513 frames), `--width 0.5`, 20 epochs of 1,200 frames, `--eval-stride 3`. Val = scenarios
4 and 10 (179 frames), test = scenario 9 (87 frames). A GPU run on all frames should do better.

| Run | Split | fg mIoU | IoU vehicle | IoU pedestrian | IoU cyclist |
|---|---|---|---|---|---|
| `--aug light` | val | 0.547 | 0.860 | 0.686 | 0.095 |
| `--aug light` | test | 0.545 | 0.868 | 0.720 | 0.048 |
| `--aug strong` | val | 0.459 | 0.790 | 0.588 | 0.000 |
| `--aug strong` | test | 0.426 | 0.795 | 0.484 | 0.000 |

Background IoU is 0.995+ in every run. Vehicle precision / recall 0.93 / 0.92, pedestrian
0.71 / 0.95 (light, val).

- **Absolute position is worth about 0.1 fg mIoU.** Most of the signal is Doppler, RCS and shape.
- **Cyclists fail.** There are only about 5k training points (0.03%), and most false positives
  are vehicle points: the class also holds motorcycles, which drive like cars. Next steps:
  more cyclists in the campaigns, or a model with local neighbourhood features
  (PointNet++, DGCNN). Plain PointNet only sees each point plus one global max-pool.
- **BatchNorm:** without the re-estimation step, eval-mode scores collapse (vehicle IoU 0.05
  on training frames). Keep it on, or use `--norm gn`.
