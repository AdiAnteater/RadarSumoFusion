"""Every named occlusion pair must travel side by side on the stretch.

    python traffic/tests/test_occlusion.py
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


def run():
    args = val.argparse.Namespace(
        density=60,
        duration=180,
        direction="BOTH",
        ambient_vehicles=0,
        pedestrians=0,
        bicycles=0,
        seed=42,
        step_length=0.1,
        stretch_signals="green",
    )
    log_path = tempfile.mktemp(suffix="_s11.log")
    return val.run_one(11, args, log_path)


class OcclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run()

    def test_every_pair_locks(self):
        self.assertGreater(self.res["occlusion_pairs_together"], 0)
        self.assertEqual(
            self.res["occlusion_pairs_locked"],
            self.res["occlusion_pairs_together"],
            msg=self.res["problems"],
        )
        self.assertEqual(self.res["teleports"], 0, msg=self.res["notes"])
        self.assertEqual(self.res["insertion_backlog_at_end"], 0)
        self.assertTrue(self.res["ok"], self.res["problems"])


if __name__ == "__main__":
    unittest.main()
