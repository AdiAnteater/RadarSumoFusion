"""Pedestrian sidewalk lock and crossing blend, without CARLA.

    python traffic/tests/test_pedestrian_path.py
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TRAFFIC = os.path.dirname(HERE)
if TRAFFIC not in sys.path:
    sys.path.insert(0, TRAFFIC)

import pedestrian_path as ped  # noqa: E402
import validate_scenarios as val  # noqa: E402


class SidewalkLockTests(unittest.TestCase):
    def test_rejects_a_jump_to_the_other_sidewalk(self):
        raw = (0.0, 0.0)
        locked = (0.2, 0.0)
        other_side = (12.0, 0.0)
        chosen = ped.choose_sidewalk_point(raw, other_side, locked, snap_max=3.0, continuity_m=2.0)
        self.assertEqual(chosen, locked)

    def test_follows_the_same_sidewalk_as_it_walks(self):
        raw = (1.0, 0.0)
        locked = (0.2, 0.0)
        nearest = (1.1, 0.1)
        chosen = ped.choose_sidewalk_point(raw, nearest, locked, snap_max=3.0, continuity_m=2.0)
        self.assertEqual(chosen, nearest)

    def test_blend_starts_at_the_kerb_and_reaches_the_crossing(self):
        sidewalk = (0.0, 0.0)
        near = ped.blend_crossing((1.0, 0.0), sidewalk, blend_m=4.0)
        far = ped.blend_crossing((4.0, 0.0), sidewalk, blend_m=4.0)
        self.assertAlmostEqual(near[0], 0.25)
        self.assertEqual(far, (4.0, 0.0))

    def test_crossing_lane_ids(self):
        self.assertTrue(ped.is_crossing_lane(":189_c0_0"))
        self.assertTrue(ped.is_crossing_lane(":189_w1_0"))
        self.assertFalse(ped.is_crossing_lane("20_0"))


class PedestrianRouteTests(unittest.TestCase):
    def test_most_walking_time_is_not_inside_junctions(self):
        args = val.argparse.Namespace(
            density=40,
            duration=120,
            direction="BOTH",
            ambient_vehicles=0,
            pedestrians=20,
            bicycles=0,
            seed=42,
            step_length=0.1,
            stretch_signals="green",
        )
        log_path = tempfile.mktemp(suffix="_ped.log")
        res = val.run_one(1, args, log_path)
        self.assertGreater(res["pedestrians_seen"], 0, msg=res["problems"])
        self.assertGreater(res["ped_max_on_stretch"], 0)
        self.assertLess(
            res["ped_crossing_fraction"], 0.45,
            msg=f"crossing fraction {res['ped_crossing_fraction']}",
        )


if __name__ == "__main__":
    unittest.main()
