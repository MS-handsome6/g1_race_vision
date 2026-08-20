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


class SprintObstacleBypassTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = DepthSafetyGate(
            DepthSafetyConfig(allow_lane_bypass=True)
        )
        self.command = ControllerCommand(0.60, 0.0, 0.05, "TRACK", True)

    @staticmethod
    def corridor(
        *, centre_m: float, left_m: float, right_m: float
    ) -> np.ndarray:
        depth = np.full((100, 160), right_m, dtype=np.float32)
        depth[:, :56] = left_m
        depth[:, 56:104] = centre_m
        return depth

    def confirmed(self, depth: np.ndarray):
        result = None
        for _ in range(3):
            result = self.gate.evaluate(depth, 0.01)
        return result

    def test_close_obstacle_uses_clear_left_lane_without_hard_stop(self) -> None:
        depth = self.corridor(centre_m=0.70, left_m=4.0, right_m=1.0)
        result = self.confirmed(depth)
        actual = self.gate.apply(self.command, result)

        self.assertEqual(result.reason, "OBSTACLE_BYPASS_LEFT")
        self.assertTrue(result.motion_allowed)
        self.assertGreater(actual.vx, 0.0)
        self.assertGreater(actual.wz, 0.0)
        self.assertFalse(actual.hard_stop)

    def test_close_obstacle_uses_clear_right_lane(self) -> None:
        depth = self.corridor(centre_m=0.70, left_m=1.0, right_m=4.0)
        result = self.confirmed(depth)

        self.assertEqual(result.reason, "OBSTACLE_BYPASS_RIGHT")
        self.assertLess(result.avoidance_yaw_rps, 0.0)

    def test_occupied_side_lanes_soft_yield_instead_of_emergency_stop(self) -> None:
        depth = self.corridor(centre_m=0.70, left_m=1.0, right_m=1.0)
        result = self.gate.evaluate(depth, 0.01)
        actual = self.gate.apply(self.command, result)

        self.assertEqual(result.reason, "OBSTACLE_YIELD")
        self.assertTrue(result.motion_allowed)
        self.assertEqual(actual.vx, 0.0)
        self.assertFalse(actual.hard_stop)

    def test_selected_side_does_not_flip_with_clearance_noise(self) -> None:
        first = self.confirmed(
            self.corridor(centre_m=0.70, left_m=4.0, right_m=3.0)
        )
        second = self.gate.evaluate(
            self.corridor(centre_m=0.70, left_m=3.0, right_m=5.0),
            0.01,
        )

        self.assertEqual(first.reason, "OBSTACLE_BYPASS_LEFT")
        self.assertEqual(second.reason, "OBSTACLE_BYPASS_LEFT")

    def test_clear_corridor_is_confirmed_before_bypass_releases(self) -> None:
        self.confirmed(
            self.corridor(centre_m=0.70, left_m=4.0, right_m=1.0)
        )
        clear = self.corridor(centre_m=4.0, left_m=4.0, right_m=4.0)
        for _ in range(5):
            result = self.gate.evaluate(clear, 0.01)
            self.assertEqual(result.reason, "OBSTACLE_BYPASS_LEFT")

        result = self.gate.evaluate(clear, 0.01)
        self.assertEqual(result.reason, "CLEAR")
        self.assertFalse(self.gate.avoidance_active)

    def test_num7_override_keeps_legacy_hard_stop(self) -> None:
        depth = self.corridor(centre_m=0.70, left_m=4.0, right_m=4.0)
        result = self.gate.evaluate(depth, 0.01, allow_bypass=False)

        self.assertEqual(result.reason, "OBSTACLE_STOP")
        self.assertFalse(result.motion_allowed)

    def test_side_clearance_requires_multiple_frames(self) -> None:
        depth = self.corridor(centre_m=0.70, left_m=4.0, right_m=1.0)

        first = self.gate.evaluate(depth, 0.01)
        second = self.gate.evaluate(depth, 0.01)

        self.assertEqual(first.reason, "OBSTACLE_PASS_CHECK")
        self.assertEqual(second.reason, "OBSTACLE_PASS_CHECK")
        self.assertFalse(first.avoidance_yaw_rps)
        self.assertFalse(second.avoidance_yaw_rps)

    def test_sensor_failure_still_hard_stops_sprint(self) -> None:
        result = self.gate.evaluate(None, 0.0)
        actual = self.gate.apply(self.command, result)

        self.assertEqual(result.reason, "DEPTH_MISSING")
        self.assertTrue(actual.hard_stop)


if __name__ == "__main__":
    unittest.main()
