from __future__ import annotations

import unittest

import numpy as np

from g1_race_vision.controller import ControllerCommand
from g1_race_vision.depth_safety import DepthSafetyConfig, DepthSafetyGate


class DepthSafetyGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = DepthSafetyGate()
        self.command = ControllerCommand(0.2, 0.0, 0.08, "TRACK", True)

    @staticmethod
    def depth(distance_m: float) -> np.ndarray:
        return np.full((100, 160), distance_m, dtype=np.float32)

    def test_missing_depth_fails_stopped(self) -> None:
        result = self.gate.evaluate(None, 0.0)
        actual = self.gate.apply(self.command, result)

        self.assertEqual(result.reason, "DEPTH_MISSING")
        self.assertFalse(result.motion_allowed)
        self.assertEqual((actual.vx, actual.vy, actual.wz), (0.0, 0.0, 0.0))

    def test_stale_depth_fails_stopped(self) -> None:
        result = self.gate.evaluate(self.depth(2.0), 0.201)

        self.assertEqual(result.reason, "DEPTH_STALE")
        self.assertFalse(result.motion_allowed)

    def test_invalid_depth_fraction_fails_stopped(self) -> None:
        depth = np.zeros((100, 160), dtype=np.float32)
        result = self.gate.evaluate(depth, 0.01)

        self.assertEqual(result.reason, "DEPTH_INSUFFICIENT")
        self.assertFalse(result.motion_allowed)

    def test_close_obstacle_stops_all_axes(self) -> None:
        result = self.gate.evaluate(self.depth(0.70), 0.01)
        actual = self.gate.apply(self.command, result)

        self.assertEqual(result.reason, "OBSTACLE_STOP")
        self.assertEqual((actual.vx, actual.vy, actual.wz), (0.0, 0.0, 0.0))

    def test_slow_zone_scales_only_forward_speed(self) -> None:
        result = self.gate.evaluate(self.depth(1.15), 0.01)
        actual = self.gate.apply(self.command, result)

        self.assertEqual(result.reason, "OBSTACLE_SLOW")
        self.assertAlmostEqual(result.speed_scale, 0.5, places=5)
        self.assertAlmostEqual(actual.vx, 0.1, places=5)
        self.assertEqual(actual.wz, self.command.wz)

    def test_clear_corridor_preserves_command(self) -> None:
        result = self.gate.evaluate(self.depth(2.0), 0.01)
        actual = self.gate.apply(self.command, result)

        self.assertEqual(result.reason, "CLEAR")
        self.assertTrue(result.motion_allowed)
        self.assertEqual(actual, self.command)

    def test_sparse_bad_pixels_do_not_trigger_false_stop(self) -> None:
        depth = self.depth(2.0)
        depth[30:35, 60:65] = 0.3
        result = self.gate.evaluate(depth, 0.01)

        self.assertEqual(result.reason, "CLEAR")

    def test_config_rejects_inverted_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            DepthSafetyConfig(stop_distance_m=1.5, slow_distance_m=0.8)


if __name__ == "__main__":
    unittest.main()
