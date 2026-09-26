"""Moderate and heavy demand must be heavier than free flow at the same slider.

    python traffic/tests/test_demand.py
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


def run(scenario_id, duration=90, density=60):
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


class DemandScaleTests(unittest.TestCase):
    def test_slider_scales(self):
        base = runner.density_to_vph(60)
        self.assertEqual(runner.scenario_vph(1, base), base)
        self.assertEqual(runner.scenario_vph(2, base), int(round(base * 1.5)))
        self.assertEqual(runner.scenario_vph(3, base), int(round(base * 2.0)))

    def test_heavy_inserts_more_than_free_flow(self):
        free = run(1)
        heavy = run(3)
        self.assertGreater(heavy["departed"], free["departed"])
        self.assertGreater(
            heavy["stretch_occupancy_mean"],
            free["stretch_occupancy_mean"],
        )


if __name__ == "__main__":
    unittest.main()
