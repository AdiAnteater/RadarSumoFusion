Scenario validation (density=60, direction=BOTH, duration=240s, signals=green, ambient=30, pedestrians=20, bicycles=10, seed=42)

| scenario | name | ok | crossed_stretch | stretch_occupancy_mean | stretch_occupancy_max | stretch_mean_speed_mps | stretch_stopped_fraction | stretch_lane_changes | stretch_passes | pedestrians_seen | bicycles_seen | teleports | collisions | insertion_backlog_at_end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Free Flow Traffic | True | 57 | 1.64 | 5 | 7.96 | 0.291 | 0 | 8 | 12 | 4 | 0 | 0 | 0 |
| 2 | Moderate Demand | True | 58 | 1.41 | 6 | 9.49 | 0.204 | 0 | 6 | 12 | 4 | 0 | 0 | 0 |
| 3 | Heavy Demand | True | 59 | 1.59 | 7 | 8.52 | 0.154 | 0 | 3 | 12 | 4 | 0 | 0 | 0 |
| 4 | Stop and Go Traffic | True | 60 | 1.45 | 6 | 9.4 | 0.16 | 0 | 10 | 12 | 4 | 0 | 0 | 0 |
| 5 | Mixed Vehicles | True | 53 | 1.37 | 8 | 8.86 | 0.115 | 0 | 10 | 12 | 4 | 0 | 0 | 7 |
| 6 | Directional Rush Hour | True | 29 | 1.01 | 4 | 6.6 | 0.362 | 0 | 0 | 12 | 4 | 0 | 0 | 0 |
| 7 | Aggressive Lane Changing | True | 56 | 2.14 | 8 | 6.01 | 0.093 | 0 | 3 | 12 | 4 | 0 | 0 | 0 |
| 8 | Bottleneck / Work Zone | True | 56 | 1.99 | 7 | 6.41 | 0.397 | 8 | 17 | 12 | 4 | 0 | 0 | 1 |
| 9 | Overtaking | True | 57 | 3.43 | 9 | 3.76 | 0.195 | 0 | 10 | 12 | 4 | 0 | 0 | 0 |
| 10 | Simultaneous Multi-Lane Overtake | True | 78 | 5.34 | 12 | 3.23 | 0.209 | 0 | 46 | 12 | 4 | 0 | 0 | 22 |
| 11 | Occlusion | True | 50 | 2.04 | 6 | 5.62 | 0.314 | 0 | 20 | 12 | 4 | 0 | 0 | 17 |

- S01 Free Flow Traffic: {'sumo_warnings': {}}
- S02 Moderate Demand: {'sumo_warnings': {}}
- S03 Heavy Demand: {'sumo_warnings': {}}
- S04 Stop and Go Traffic: {'sumo_warnings': {}, 'shockwave_seeds_entered': 4}
- S05 Mixed Vehicles: {'sumo_warnings': {}, 'buses_dwelled_at_stop': 1}
    - note: 7 vehicles still waiting to be inserted at the end (demand above what the entry edges can absorb)
- S06 Directional Rush Hour: {'sumo_warnings': {}}
- S07 Aggressive Lane Changing: {'sumo_warnings': {}}
- S08 Bottleneck / Work Zone: {'sumo_warnings': {'person': 1}, 'blocker_parked': True}
    - note: 1 vehicles still waiting to be inserted at the end (demand above what the entry edges can absorb)
- S09 Overtaking: {'sumo_warnings': {}}
- S10 Simultaneous Multi-Lane Overtake: {'sumo_warnings': {}}
    - note: 22 vehicles still waiting to be inserted at the end (demand above what the entry edges can absorb)
- S11 Occlusion: {'sumo_warnings': {}, 'occlusion_pairs_locked': 1}
    - note: 17 vehicles still waiting to be inserted at the end (demand above what the entry edges can absorb)
