"""30 second headless run of every scenario.

Checks that moderate and heavy demand insert more vehicles than free flow,
and that the scenarios are not the same run. Stretch behaviour needs the
travel time from the map edge, so a second pass at 90 s compares the
monitored stretch. No CARLA.

    python traffic/tests/test_scenario_difference.py
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TRAFFIC = os.path.dirname(HERE)
if TRAFFIC not in sys.path:
    sys.path.insert(0, TRAFFIC)

import runner  # noqa: E402
import validate_scenarios as val  # noqa: E402


def run_all(duration):
    out = {}
    for sid in sorted(runner.SCENARIOS):
        args = val.argparse.Namespace(
            density=60,
            duration=duration,
            direction="BOTH",
            ambient_vehicles=0,
            pedestrians=0,
            bicycles=0,
            seed=42,
            step_length=0.1,
            stretch_signals="green",
        )
        log_path = tempfile.mktemp(suffix=f"_s{sid}.log")
        out[sid] = val.run_one(sid, args, log_path)
    return out


def signature(res):
    return (
        res["departed"],
        res["crossed_stretch"],
        res["stretch_occupancy_mean"],
        res["stretch_mean_speed_mps"],
        res["stretch_stopped_fraction"],
        res["stretch_lane_changes"],
    )


def _print_table(title, results):
    print(f"\n{title}")
    print(f"{'id':>4} {'departed':>8} {'crossed':>8} {'occ':>6} {'mps':>6} {'stopped':>8} {'lc':>4}")
    for sid, res in results.items():
        print(f"{sid:4d} {res['departed']:8d} {res['crossed_stretch']:8d} "
              f"{res['stretch_occupancy_mean']:6.2f} {res['stretch_mean_speed_mps']:6.2f} "
              f"{res['stretch_stopped_fraction']:8.3f} {res['stretch_lane_changes']:4d}")


class ScenarioDifferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.short = run_all(30)
        cls.stretch = run_all(90)
        _print_table("30s insertion", cls.short)
        _print_table("90s stretch", cls.stretch)

    def test_30s_densities_differ(self):
        free = self.short[1]["departed"]
        moderate = self.short[2]["departed"]
        heavy = self.short[3]["departed"]
        self.assertGreater(moderate, free, msg=f"moderate {moderate} free {free}")
        self.assertGreater(heavy, moderate, msg=f"heavy {heavy} moderate {moderate}")

    def test_90s_stretch_signatures_are_not_all_the_same(self):
        sigs = {sid: signature(res) for sid, res in self.stretch.items()}
        # Each scenario's stretch signature differs from free flow, except free
        # flow itself. Same-slider scenarios separate by stops, lane changes,
        # or how many vehicles get through.
        free = sigs[1]
        same = [sid for sid, sig in sigs.items() if sid != 1 and sig == free]
        self.assertEqual(same, [], msg=f"same as free flow: {same} {free}")
        self.assertGreater(self.stretch[4]["stretch_stopped_fraction"],
                           self.stretch[1]["stretch_stopped_fraction"])
        self.assertGreater(self.stretch[7]["stretch_lane_changes"], 0)
        self.assertGreater(self.stretch[3]["stretch_occupancy_mean"],
                           self.stretch[1]["stretch_occupancy_mean"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
