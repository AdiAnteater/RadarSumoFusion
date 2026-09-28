"""Pedestrian sidewalk lock and walking step, without CARLA.

    python traffic/tests/test_pedestrian_path.py
"""

import math
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
    def test_rejects_another_lane_even_a_metre_away(self):
        raw = (0.0, 0.0)
        locked = (0.2, 0.0)
        other_side = (0.2, 1.0)
        chosen = ped.choose_sidewalk_point(
            raw, other_side, locked, snap_max=3.0,
            nearest_lane=(7, 1), locked_lane=(7, 0))
        self.assertEqual(chosen, locked)

    def test_follows_the_same_sidewalk_as_it_walks(self):
        raw = (1.0, 0.0)
        locked = (0.2, 0.0)
        nearest = (1.1, 0.1)
        chosen = ped.choose_sidewalk_point(
            raw, nearest, locked, snap_max=3.0,
            nearest_lane=(7, 0), locked_lane=(7, 0))
        self.assertEqual(chosen, nearest)

    def test_lookup_follows_the_locked_pavement_plus_the_step(self):
        # SUMO is a metre into the road. The step is along the pavement.
        query = ped.sidewalk_query_point(
            raw=(1.0, 1.0), locked=(0.0, 0.0), prev_raw=(0.0, 1.0))
        self.assertEqual(query, (1.0, 0.0))

    def test_rendered_step_is_capped_at_the_walk(self):
        stepped = ped.step_toward((0.0, 0.0), (10.0, 0.0), sumo_moved=0.14)
        moved = math.hypot(stepped[0], stepped[1])
        self.assertAlmostEqual(moved, 0.14 + ped.CORRECTION_M)

    def test_a_stopped_walker_does_not_flicker_at_the_kerb(self):
        state = {
            "locked": (0.0, 0.0),
            "locked_lane": (7, 0),
            "rendered": (0.0, 0.0),
            "yaw": 0.0,
            "prev_raw": (0.2, 0.5),
            "crossing_ticks": 0,
        }
        # Same SUMO sample. The nearest pavement is the other side of the street.
        x, y, yaw, state = ped.next_pose(
            (0.2, 0.5), 180.0, "20_0", (0.0, 1.0), (7, 1), state, snap_max=3.0)
        self.assertEqual((x, y), (0.0, 0.0))
        self.assertEqual(yaw, 0.0)
        self.assertEqual(state["locked"], (0.0, 0.0))
        # The lane id flips to the walking area. They are still waiting.
        x, y, yaw, state = ped.next_pose(
            (0.2, 0.5), 180.0, ":189_w0_0", None, None, state, snap_max=3.0)
        self.assertEqual((x, y), (0.0, 0.0))
        self.assertEqual(yaw, 0.0)
        self.assertEqual(state["locked"], (0.0, 0.0))

    def test_a_real_crossing_is_approached_by_walking(self):
        state = {
            "locked": (0.0, 0.0),
            "locked_lane": (7, 0),
            "rendered": (0.0, 0.0),
            "yaw": 0.0,
            "prev_raw": (0.0, 2.0),
            "crossing_ticks": ped.CROSSING_COMMIT_TICKS - 1,
        }
        raw = (0.12, 2.0)
        x, y, yaw, state = ped.next_pose(
            raw, 180.0, ":189_c0_0", None, None, state, snap_max=3.0)
        moved = math.hypot(x, y)
        self.assertAlmostEqual(moved, 0.12 + ped.CORRECTION_M)
        self.assertGreater(math.hypot(x - raw[0], y - raw[1]), 1.0)
        self.assertNotAlmostEqual(yaw, 180.0, places=3)
        self.assertEqual(state["crossing_ticks"], ped.CROSSING_COMMIT_TICKS)
        # Still within snap of the pavement, so the lock is kept.
        self.assertEqual(state["locked"], (0.0, 0.0))

    def test_lock_releases_only_after_a_real_crossing_and_distance(self):
        far = {
            "locked": (0.0, 0.0),
            "locked_lane": (7, 0),
            "rendered": (0.0, 0.0),
            "yaw": 0.0,
            "prev_raw": (0.0, 4.0),
            "crossing_ticks": 0,
        }
        _, _, _, early = ped.next_pose(
            (0.1, 4.0), 0.0, ":189_c0_0", None, None, far, snap_max=3.0)
        self.assertEqual(early["locked"], (0.0, 0.0))
        far["crossing_ticks"] = ped.CROSSING_COMMIT_TICKS - 1
        _, _, _, released = ped.next_pose(
            (0.1, 4.0), 0.0, ":189_c0_0", None, None, far, snap_max=3.0)
        self.assertIsNone(released["locked"])

    def test_first_tick_uses_the_target(self):
        x, y, yaw, state = ped.next_pose(
            (2.0, 0.5), 90.0, "20_0", (2.0, 0.0), (7, 0), None, snap_max=3.0)
        self.assertEqual((x, y), (2.0, 0.0))
        self.assertEqual(yaw, 90.0)
        self.assertEqual(state["locked"], (2.0, 0.0))
        self.assertEqual(state["locked_lane"], (7, 0))

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
