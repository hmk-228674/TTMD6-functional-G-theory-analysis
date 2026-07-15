from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_action_specific_reml.py"
SPEC = importlib.util.spec_from_file_location("ttmd6_reml", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
REML = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REML)


class CoreMathTests(unittest.TestCase):
    def test_trapezoid_weights(self) -> None:
        observed = REML.integration_weights(5)
        expected = np.array([0.125, 0.25, 0.25, 0.25, 0.125])
        np.testing.assert_allclose(observed, expected)
        self.assertAlmostEqual(float(observed.sum()), 1.0)

    def test_integrated_relative_reliability(self) -> None:
        observed = REML.integrated_relative_reliability(
            np.array([2.0]), np.array([8.0]), 4
        )
        np.testing.assert_allclose(observed, np.array([0.5]))

    def test_required_trials(self) -> None:
        observed = REML.required_trials(2.0, 8.0, 0.80)
        self.assertEqual(int(observed), 16)
        self.assertTrue(np.isinf(REML.required_trials(0.0, 8.0, 0.80)))

    def test_pointwise_curve_bounds(self) -> None:
        between = np.array([0.0, 1.0, 2.0])
        within = np.array([1.0, 2.0, 3.0])
        curve = REML.reliability_curve(between, within, 10)
        self.assertTrue(np.all(curve >= 0.0))
        self.assertTrue(np.all(curve <= 1.0))
        self.assertEqual(float(curve[0]), 0.0)

    def test_canonical_action_labels(self) -> None:
        self.assertEqual(REML.ACTION_LABEL[2], "forehand drive")
        self.assertEqual(REML.ACTION_LABEL[5], "backhand drive")
        self.assertEqual(len(REML.ACTION_LABEL), 6)


if __name__ == "__main__":
    unittest.main()
