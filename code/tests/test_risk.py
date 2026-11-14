from __future__ import annotations

import math
import unittest

from psr_planner.models import Pose, TargetBelief
from psr_planner.risk import calibrate_candidate, clopper_pearson_lower


class ClopperPearsonTests(unittest.TestCase):
    def test_all_successes_has_closed_form(self) -> None:
        trials = 30
        confidence = 0.95
        expected = (1.0 - confidence) ** (1.0 / trials)
        actual = clopper_pearson_lower(trials, trials, confidence)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_zero_successes_returns_zero(self) -> None:
        self.assertEqual(clopper_pearson_lower(0, 20, 0.95), 0.0)

    def test_stable_candidate_is_accepted(self) -> None:
        target = TargetBelief("target", 0.0, 0.0, 0.02, 0.02)
        pose = Pose(-0.75, 0.0, 0.0, "candidate")
        result = calibrate_candidate(
            pose,
            target,
            trials=300,
            risk_epsilon=0.10,
            confidence=0.95,
            seed=7,
            base_position_sigma=0.02,
            base_yaw_sigma=math.radians(1.0),
        )
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.lower_confidence, 0.90)


if __name__ == "__main__":
    unittest.main()
