"""Scenarios 7, 9 and 10 must change lanes on the monitored stretch.

    python traffic/tests/test_lane_change.py
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


def run(scenario_id, duration=120, density=60):
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


class LaneChangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {sid: run(sid) for sid in (7, 9, 10)}

    def test_each_overtake_scenario_changes_lanes(self):
        for sid, res in self.results.items():
            self.assertEqual(res["teleports"], 0, msg=f"scenario {sid}: {res['notes']}")
            self.assertGreater(
                res["stretch_lane_changes"], 0,
                msg=f"scenario {sid} lane changes={res['stretch_lane_changes']} "
                    f"passes={res['stretch_passes']}",
            )


if __name__ == "__main__":
    unittest.main()
