"""Stop-and-go must hold a seed on the stretch and stop more than free flow.

Headless SUMO, no CARLA. Run from the repo root:

    python traffic/tests/test_stop_and_go.py
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TRAFFIC = os.path.dirname(HERE)
if TRAFFIC not in sys.path:
    sys.path.insert(0, TRAFFIC)

import validate_scenarios as val  # noqa: E402


def run(scenario_id, duration, density=60):
    args = val.argparse.Namespace(
        density=density,
        duration=duration,
        direction="BOTH",
        ambient_vehicles=0,
        pedestrians=0,
        bicycles=0,
        seed=42,
        step_length=0.1,
        stretch_signals="green",
    )
    log_path = tempfile.mktemp(suffix=f"_s{scenario_id}.log")
    return val.run_one(scenario_id, args, log_path)


class StopAndGoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.free = run(1, duration=100)
        cls.stop = run(4, duration=100)

    def test_seed_enters_and_is_held(self):
        self.assertGreater(self.stop["shockwave_seeds_entered"], 0)
        self.assertEqual(self.stop["teleports"], 0)
        self.assertTrue(self.stop["ok"], self.stop["problems"])

    def test_stops_more_than_free_flow(self):
        self.assertGreater(
            self.stop["stretch_stopped_fraction"],
            self.free["stretch_stopped_fraction"],
            msg=f"stop-and-go {self.stop['stretch_stopped_fraction']} "
                f"vs free flow {self.free['stretch_stopped_fraction']}",
        )


if __name__ == "__main__":
    unittest.main()
