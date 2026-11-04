# Diverse capture campaign v1

Two campaign files for `fusion_gui.py` / `run_fusion.py`. Together they cover all 11 scenarios across every traffic setting a row exposes. Both support both goals: PointNet segmentation (fg mIoU, no class left behind) and scenario/event-taxonomy coverage for the paper.

| File | Rows | Recorded | Warm-up | Total | Role |
|---|---|---|---|---|---|
| `diverse_core_v1.json` | 67 | 9,105 s | 1,340 s | 2.90 h | Minimum viable set. Run this first. |
| `diverse_ext_v1.json` | 45 | 6,150 s | 900 s | 1.96 h | Targeted top-up. Run only if core falls short. |
| Both | 112 | 15,255 s | 2,240 s | 4.86 h | Inside the 5 h cap. |

Totals are recorded time plus the 20 s warm-up per row. Per-row cleanup sweep and runner start are on top of that and not counted.

## How to run

```
python run_fusion.py --campaign campaigns/diverse_core_v1.json
python run_fusion.py --campaign campaigns/diverse_ext_v1.json
```

Or load either file in `fusion_gui.py`. Each file is one capture. `train.py` already indexes multiple capture dirs, so the two captures combine at training time.

I split it in two instead of one 5 h file for two reasons. A crash on a 5 h capture costs the whole session. Also, the extension only pays off if the core shows a gap, so it should stay optional.

## What varies and why

Every row field varies. Rows are Latin-hypercube sampled per scenario, so each scenario sees its full range of every axis without needing a full grid.

| Field | Range | Notes |
|---|---|---|
| `density` | Per-scenario range (below) | Kept inside what the entry edges absorb. The validation report showed insertion backlog at 60 for S05, S10 and S11. |
| `duration` | 90 to 180 s, 15 s steps | Your 1.5 to 3 min window. |
| `direction` | BOTH / WB / EB, about 60/25/15 | Shifts the Doppler sign balance per radar. Fixed to BOTH where it has no effect. |
| `ambient_vehicles` | 10 to 60 (0 in two edge rows) | Capped at 40 for S03, S10 and S11 to avoid gridlock. |
| `pedestrians` | 10 to 90 core, 50 to 100 ext | |
| `bicycles` | 20 to 95 core, 50 to 100 ext | Deliberately high. See class balance. |
| `seed` | Unique per row | Encodes the source. `1SSii` = core, `2SSii` = ext, `290xx` = edge cases. S = scenario, i = row. |
| `signals` | green about 2/3, static about 1/3 | Static adds red-light queues and stopped vehicles on the stretch. |

### Per-scenario density ranges

| Scenario | Density | Why |
|---|---|---|
| S01 Free Flow | 5 to 35 | Free flow should stay sparse. |
| S02 Moderate Demand | 20 to 55 | Runner scales demand by 1.5. |
| S03 Heavy Demand | 25 to 55 | Runner scales demand by 2.0. 55 already gives about 1,030 veh/h/dir, right at capacity. |
| S04 Stop and Go | 40 to 75 | The shockwave needs a queue to propagate into. |
| S05 Mixed Vehicles | 30 to 60 | Backlog showed at 60. |
| S06 Rush Hour | 40 to 80 | The 80/20 split is baked into the route file. Direction stays BOTH. |
| S07 Aggressive LC | 50 to 85 | Lane changes need gaps that are contested. |
| S08 Bottleneck | 40 to 75 | |
| S09 Overtaking | 40 to 70 | |
| S10 Multi-Lane Overtake | fixed 50 | The route file has no `DENSITY_*` placeholders, so density does nothing. Ambient, VRUs, seed and signals still vary. |
| S11 Occlusion | fixed 50 | Same as S10. |

## Class balance

PointNet's classes are background, vehicle, pedestrian and cyclist. Cyclist holds both bicycles and motorcycles. The current README run puts cyclists at 0.03% of points and IoU under 0.1, so "no class suffers" depends almost entirely on cyclists.

The plan pushes in three ways:

1. Bicycle sliders average 60 in core and 84 in the extension, against 10 in past campaigns. At 100 the ambient generator spawns 120 bikes/h map-wide.
2. S05 is the only motorcycle source. It gets 7 core rows instead of 6, plus 2 extension rows with bicycles at 80 to 100.
3. Two extension edge rows run with near-empty roads and VRUs only, so the cyclists and walkers are not hidden behind cars. One also works as a false-positive check on a mostly background scene.

Vehicles need no help. Pedestrians sit in the middle and get raised in the extension.

## Splits

`train.py --split scenario` holds out whole scenarios: val = S04 and S10, test = S09. Any held-out scenario with few frames gives a noisy mIoU. Core gives every scenario about 800 recorded seconds. The extension adds 2 extra rows each to S04, S09 and S10, so each reaches about 1,500 s.

If you move to a different scenario split later, nothing here needs to change. Every scenario already sees the full range of every axis.

## Check after the core capture

Look at `radar_labeling_qa/segment_summary.md` and the `class_counts` in PointNet's `meta.json`.

- If cyclist points sit under about 1% of foreground, run the extension.
- If any `segment_summary` row shows a large insertion backlog, cut that scenario's density top end before the extension.
- If S04 or S11 show no seed or pair activity in the run log, check that the controller fix made it to the server copy.

## Scope notes

- The scenario 4 and 11 controller fixes are already in `RadarSumoFusion2/traffic/runner.py`, so I made no code changes.
- Weather, sensor pose and car-following model are untouched, as agreed.
- Pedestrian walkingarea geometry is unchanged. Some frames will show walkers standing in the road.

## Appendix: every row

### Core (diverse_core_v1.json)

| # | Scn | Dens | Dur | Dir | Amb | Ped | Bike | Seed | Sig | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S01 | 26 | 90 | BOTH | 40 | 50 | 40 | 10100 | green | S01 LHS | seg+taxonomy |
| 2 | S02 | 32 | 180 | BOTH | 40 | 10 | 70 | 10200 | static | S02 LHS | seg+taxonomy |
| 3 | S03 | 34 | 120 | BOTH | 40 | 25 | 70 | 10300 | green | S03 LHS | seg+taxonomy |
| 4 | S04 | 70 | 165 | BOTH | 55 | 20 | 90 | 10400 | green | S04 LHS | seg+taxonomy |
| 5 | S05 | 51 | 90 | BOTH | 55 | 55 | 70 | 10500 | static | S05 LHS | seg+taxonomy |
| 6 | S06 | 50 | 165 | BOTH | 55 | 80 | 85 | 10600 | green | S06 LHS | seg+taxonomy |
| 7 | S07 | 62 | 165 | BOTH | 50 | 75 | 35 | 10700 | static | S07 LHS | seg+taxonomy |
| 8 | S08 | 50 | 90 | BOTH | 25 | 65 | 85 | 10800 | static | S08 LHS | seg+taxonomy |
| 9 | S09 | 42 | 90 | BOTH | 20 | 90 | 90 | 10900 | green | S09 LHS | seg+taxonomy |
| 10 | S10 | 50 | 180 | BOTH | 25 | 90 | 65 | 11000 | green | S10 LHS | seg+taxonomy |
| 11 | S11 | 50 | 165 | BOTH | 25 | 40 | 25 | 11100 | green | S11 LHS | seg+taxonomy |
| 12 | S01 | 31 | 180 | WB | 50 | 25 | 80 | 10101 | static | S01 LHS | seg+taxonomy |
| 13 | S02 | 23 | 135 | WB | 20 | 25 | 50 | 10201 | static | S02 LHS | seg+taxonomy |
| 14 | S03 | 35 | 165 | WB | 15 | 45 | 60 | 10301 | green | S03 LHS | seg+taxonomy |
| 15 | S04 | 58 | 90 | WB | 25 | 30 | 70 | 10401 | green | S04 LHS | seg+taxonomy |
| 16 | S05 | 55 | 150 | WB | 45 | 20 | 95 | 10501 | green | S05 LHS | seg+taxonomy |
| 17 | S06 | 64 | 120 | BOTH | 30 | 60 | 95 | 10601 | green | S06 LHS | seg+taxonomy |
| 18 | S07 | 85 | 120 | WB | 40 | 50 | 50 | 10701 | green | S07 LHS | seg+taxonomy |
| 19 | S08 | 74 | 165 | WB | 60 | 25 | 25 | 10801 | green | S08 LHS | seg+taxonomy |
| 20 | S09 | 58 | 165 | WB | 55 | 25 | 85 | 10901 | green | S09 LHS | seg+taxonomy |
| 21 | S10 | 50 | 135 | BOTH | 20 | 65 | 60 | 11001 | green | S10 LHS | seg+taxonomy |
| 22 | S11 | 50 | 135 | BOTH | 25 | 65 | 35 | 11101 | green | S11 LHS | seg+taxonomy |
| 23 | S01 | 11 | 150 | EB | 55 | 80 | 30 | 10102 | green | S01 LHS | seg+taxonomy |
| 24 | S02 | 30 | 135 | EB | 30 | 80 | 25 | 10202 | green | S02 LHS | seg+taxonomy |
| 25 | S03 | 45 | 120 | EB | 20 | 90 | 75 | 10302 | static | S03 LHS | seg+taxonomy |
| 26 | S04 | 68 | 135 | EB | 15 | 60 | 40 | 10402 | static | S04 LHS | seg+taxonomy |
| 27 | S05 | 42 | 165 | EB | 40 | 85 | 50 | 10502 | green | S05 LHS | seg+taxonomy |
| 28 | S06 | 57 | 120 | BOTH | 25 | 15 | 50 | 10602 | static | S06 LHS | seg+taxonomy |
| 29 | S07 | 52 | 135 | EB | 15 | 60 | 70 | 10702 | green | S07 LHS | seg+taxonomy |
| 30 | S08 | 60 | 165 | EB | 45 | 85 | 40 | 10802 | green | S08 LHS | seg+taxonomy |
| 31 | S09 | 49 | 120 | EB | 35 | 20 | 45 | 10902 | static | S09 LHS | seg+taxonomy |
| 32 | S10 | 50 | 90 | BOTH | 35 | 45 | 95 | 11002 | green | S10 LHS | seg+taxonomy |
| 33 | S11 | 50 | 165 | BOTH | 35 | 60 | 55 | 11102 | static | S11 LHS | seg+taxonomy |
| 34 | S01 | 9 | 135 | BOTH | 30 | 10 | 60 | 10103 | green | S01 LHS | seg+taxonomy |
| 35 | S02 | 50 | 120 | BOTH | 55 | 60 | 40 | 10203 | green | S02 LHS | seg+taxonomy |
| 36 | S03 | 28 | 150 | BOTH | 10 | 70 | 35 | 10303 | static | S03 LHS | seg+taxonomy |
| 37 | S04 | 51 | 120 | BOTH | 40 | 45 | 45 | 10403 | static | S04 LHS | seg+taxonomy |
| 38 | S05 | 47 | 135 | BOTH | 20 | 40 | 30 | 10503 | green | S05 LHS | seg+taxonomy |
| 39 | S06 | 69 | 150 | BOTH | 15 | 25 | 25 | 10603 | static | S06 LHS | seg+taxonomy |
| 40 | S07 | 75 | 165 | BOTH | 55 | 30 | 90 | 10703 | green | S07 LHS | seg+taxonomy |
| 41 | S08 | 41 | 150 | BOTH | 20 | 20 | 70 | 10803 | green | S08 LHS | seg+taxonomy |
| 42 | S09 | 66 | 135 | BOTH | 40 | 50 | 30 | 10903 | green | S09 LHS | seg+taxonomy |
| 43 | S10 | 50 | 150 | BOTH | 10 | 70 | 85 | 11003 | static | S10 LHS | seg+taxonomy |
| 44 | S11 | 50 | 135 | BOTH | 20 | 85 | 75 | 11103 | static | S11 LHS | seg+taxonomy |
| 45 | S01 | 17 | 150 | BOTH | 25 | 45 | 60 | 10104 | static | S01 LHS | seg+taxonomy |
| 46 | S02 | 47 | 90 | BOTH | 50 | 50 | 85 | 10204 | green | S02 LHS | seg+taxonomy |
| 47 | S03 | 50 | 90 | BOTH | 30 | 60 | 95 | 10304 | green | S03 LHS | seg+taxonomy |
| 48 | S04 | 44 | 180 | BOTH | 50 | 70 | 30 | 10404 | green | S04 LHS | seg+taxonomy |
| 49 | S05 | 60 | 120 | BOTH | 30 | 35 | 35 | 10504 | green | S05 LHS | seg+taxonomy |
| 50 | S06 | 76 | 105 | BOTH | 45 | 70 | 65 | 10604 | green | S06 LHS | seg+taxonomy |
| 51 | S07 | 58 | 120 | BOTH | 20 | 15 | 75 | 10704 | green | S07 LHS | seg+taxonomy |
| 52 | S08 | 65 | 120 | BOTH | 30 | 45 | 45 | 10804 | static | S08 LHS | seg+taxonomy |
| 53 | S09 | 65 | 135 | BOTH | 15 | 70 | 70 | 10904 | green | S09 LHS | seg+taxonomy |
| 54 | S10 | 50 | 120 | BOTH | 20 | 10 | 30 | 11004 | green | S10 LHS | seg+taxonomy |
| 55 | S11 | 50 | 120 | BOTH | 15 | 15 | 65 | 11104 | green | S11 LHS | seg+taxonomy |
| 56 | S01 | 21 | 120 | WB | 10 | 70 | 95 | 10105 | green | S01 LHS | seg+taxonomy |
| 57 | S02 | 42 | 165 | WB | 15 | 70 | 95 | 10205 | green | S02 LHS | seg+taxonomy |
| 58 | S03 | 53 | 150 | WB | 35 | 15 | 20 | 10305 | green | S03 LHS | seg+taxonomy |
| 59 | S04 | 54 | 135 | WB | 30 | 85 | 75 | 10405 | green | S04 LHS | seg+taxonomy |
| 60 | S05 | 31 | 105 | WB | 15 | 65 | 65 | 10505 | green | S05 LHS | seg+taxonomy |
| 61 | S06 | 46 | 180 | BOTH | 40 | 40 | 40 | 10605 | green | S06 LHS | seg+taxonomy |
| 62 | S07 | 72 | 105 | WB | 25 | 65 | 30 | 10705 | static | S07 LHS | seg+taxonomy |
| 63 | S08 | 55 | 135 | WB | 50 | 55 | 90 | 10805 | green | S08 LHS | seg+taxonomy |
| 64 | S09 | 53 | 165 | WB | 45 | 45 | 60 | 10905 | static | S09 LHS | seg+taxonomy |
| 65 | S10 | 50 | 135 | BOTH | 35 | 30 | 40 | 11005 | static | S10 LHS | seg+taxonomy |
| 66 | S11 | 50 | 90 | BOTH | 30 | 35 | 95 | 11105 | green | S11 LHS | seg+taxonomy |
| 67 | S05 | 34 | 165 | EB | 35 | 75 | 80 | 10506 | static | S05 LHS | seg+taxonomy |

### Extension (diverse_ext_v1.json)

| # | Scn | Dens | Dur | Dir | Amb | Ped | Bike | Seed | Sig | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S01 | 27 | 90 | BOTH | 35 | 80 | 100 | 20100 | green | S01 VRU boost | seg: cyclist+ped |
| 2 | S02 | 46 | 90 | BOTH | 20 | 70 | 95 | 20200 | static | S02 VRU boost | seg: cyclist+ped |
| 3 | S03 | 37 | 150 | BOTH | 35 | 65 | 80 | 20300 | static | S03 VRU boost | seg: cyclist+ped |
| 4 | S04 | 64 | 135 | BOTH | 30 | 85 | 95 | 20400 | static | S04 VRU boost | seg: cyclist+ped |
| 5 | S05 | 53 | 180 | BOTH | 25 | 65 | 75 | 20500 | green | S05 VRU boost | seg: cyclist+ped |
| 6 | S06 | 46 | 165 | BOTH | 25 | 70 | 90 | 20600 | green | S06 VRU boost | seg: cyclist+ped |
| 7 | S07 | 53 | 135 | BOTH | 35 | 70 | 90 | 20700 | green | S07 VRU boost | seg: cyclist+ped |
| 8 | S08 | 64 | 135 | BOTH | 20 | 85 | 85 | 20800 | green | S08 VRU boost | seg: cyclist+ped |
| 9 | S09 | 62 | 135 | BOTH | 25 | 85 | 75 | 20900 | green | S09 VRU boost | seg: cyclist+ped |
| 10 | S10 | 50 | 165 | BOTH | 30 | 95 | 80 | 21000 | green | S10 VRU boost | seg: cyclist+ped |
| 11 | S11 | 50 | 165 | BOTH | 35 | 55 | 75 | 21100 | green | S11 VRU boost | seg: cyclist+ped |
| 12 | S01 | 7 | 165 | WB | 20 | 65 | 85 | 20101 | static | S01 VRU boost | seg: cyclist+ped |
| 13 | S02 | 33 | 135 | WB | 15 | 60 | 75 | 20201 | green | S02 VRU boost | seg: cyclist+ped |
| 14 | S03 | 29 | 150 | WB | 10 | 70 | 75 | 20301 | green | S03 VRU boost | seg: cyclist+ped |
| 15 | S04 | 46 | 165 | WB | 25 | 60 | 75 | 20401 | green | S04 VRU boost | seg: cyclist+ped |
| 16 | S05 | 43 | 90 | WB | 15 | 70 | 100 | 20501 | green | S05 VRU boost | seg: cyclist+ped |
| 17 | S06 | 74 | 120 | BOTH | 10 | 55 | 80 | 20601 | static | S06 VRU boost | seg: cyclist+ped |
| 18 | S07 | 64 | 180 | WB | 10 | 55 | 75 | 20701 | static | S07 VRU boost | seg: cyclist+ped |
| 19 | S08 | 44 | 105 | WB | 10 | 60 | 95 | 20801 | static | S08 VRU boost | seg: cyclist+ped |
| 20 | S09 | 48 | 165 | WB | 10 | 60 | 90 | 20901 | static | S09 VRU boost | seg: cyclist+ped |
| 21 | S10 | 50 | 150 | BOTH | 15 | 55 | 100 | 21001 | static | S10 VRU boost | seg: cyclist+ped |
| 22 | S11 | 50 | 150 | BOTH | 5 | 70 | 95 | 21101 | static | S11 VRU boost | seg: cyclist+ped |
| 23 | S01 | 16 | 135 | EB | 10 | 90 | 70 | 20102 | green | S01 VRU boost | seg: cyclist+ped |
| 24 | S02 | 29 | 180 | EB | 35 | 95 | 90 | 20202 | green | S02 VRU boost | seg: cyclist+ped |
| 25 | S03 | 52 | 90 | EB | 25 | 95 | 90 | 20302 | green | S03 VRU boost | seg: cyclist+ped |
| 26 | S04 | 62 | 120 | EB | 15 | 80 | 80 | 20402 | green | S04 VRU boost | seg: cyclist+ped |
| 27 | S05 | 32 | 120 | EB | 35 | 95 | 85 | 20502 | static | S05 VRU boost | seg: cyclist+ped |
| 28 | S06 | 60 | 135 | BOTH | 30 | 95 | 100 | 20602 | green | S06 VRU boost | seg: cyclist+ped |
| 29 | S07 | 78 | 105 | EB | 25 | 85 | 85 | 20702 | green | S07 VRU boost | seg: cyclist+ped |
| 30 | S08 | 60 | 165 | EB | 30 | 80 | 80 | 20802 | green | S08 VRU boost | seg: cyclist+ped |
| 31 | S09 | 60 | 105 | EB | 30 | 100 | 90 | 20902 | green | S09 VRU boost | seg: cyclist+ped |
| 32 | S10 | 50 | 90 | BOTH | 25 | 75 | 85 | 21002 | green | S10 VRU boost | seg: cyclist+ped |
| 33 | S11 | 50 | 105 | BOTH | 20 | 85 | 85 | 21102 | green | S11 VRU boost | seg: cyclist+ped |
| 34 | S01 | 1 | 150 | BOTH | 0 | 100 | 100 | 29001 | green | edge: near-empty road, VRUs only | seg: cyclist+ped, false-positive check |
| 35 | S03 | 55 | 150 | BOTH | 60 | 60 | 60 | 29003 | static | edge: saturation, dense occlusion | seg+taxonomy |
| 36 | S04 | 70 | 90 | BOTH | 25 | 90 | 50 | 20450 | static | S04 held-out depth | seg: eval stability |
| 37 | S05 | 51 | 165 | BOTH | 45 | 65 | 95 | 20560 | green | S05 motorcycle depth | seg: cyclist |
| 38 | S08 | 75 | 180 | WB | 40 | 80 | 80 | 29004 | static | edge: work zone queue with VRUs | seg+taxonomy |
| 39 | S09 | 59 | 180 | BOTH | 45 | 30 | 100 | 20950 | static | S09 held-out depth | seg: eval stability |
| 40 | S10 | 50 | 165 | BOTH | 20 | 70 | 55 | 21050 | static | S10 held-out depth | seg: eval stability |
| 41 | S01 | 5 | 120 | EB | 0 | 0 | 100 | 29002 | static | edge: cyclists alone on EB | seg: cyclist |
| 42 | S04 | 48 | 165 | WB | 45 | 15 | 90 | 20451 | green | S04 held-out depth | seg: eval stability |
| 43 | S05 | 40 | 90 | WB | 20 | 45 | 90 | 20561 | green | S05 motorcycle depth | seg: cyclist |
| 44 | S09 | 44 | 105 | WB | 15 | 75 | 60 | 20951 | green | S09 held-out depth | seg: eval stability |
| 45 | S10 | 50 | 120 | BOTH | 40 | 20 | 85 | 21051 | green | S10 held-out depth | seg: eval stability |
