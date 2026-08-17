from __future__ import annotations

import unittest

from g1_race_vision.controller import LaneFollowerConfig, LaneFollowerController
from g1_race_vision.line_detector import DetectionResult


class LaneFollowerControllerTest(unittest.TestCase):
    def test_center_corridor_ignores_visual_sway_and_runs_straight(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                max_forward_accel_mps2=20.0,
                max_yaw_accel_rps2=20.0,
            )
        )
        commands = []
        for index in range(60):
            commands.append(
                controller.update(
                    DetectionResult(
                        valid=True,
                        lateral_error=0.12 if index % 2 else -0.12,
                        heading_error_rad=0.20 if index % 2 else -0.20,
                        confidence=1.0,
                    ),
                    now=0.05 * index,
                    heading_hold_error_rad=0.0,
                )
            )

        self.assertLess(max(abs(command.wz) for command in commands), 1e-9)
        self.assertAlmostEqual(
            commands[-1].vx,
            controller.config.cruise_speed_mps,
        )

    def test_correction_hysteresis_does_not_reverse_before_recentering(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                error_filter_alpha=1.0,
                max_lateral_error_rate_per_s=100.0,
                max_yaw_accel_rps2=100.0,
            )
        )
        right = controller.update(
            DetectionResult(valid=True, lateral_error=0.30, confidence=1.0),
            now=0.0,
            heading_hold_error_rad=0.0,
        )
        crossed = controller.update(
            DetectionResult(valid=True, lateral_error=-0.10, confidence=1.0),
            now=0.05,
            heading_hold_error_rad=0.0,
        )
        controller.update(
            DetectionResult(valid=True, lateral_error=0.0, confidence=1.0),
            now=0.10,
            heading_hold_error_rad=0.0,
        )
        left = controller.update(
            DetectionResult(valid=True, lateral_error=-0.30, confidence=1.0),
            now=0.15,
            heading_hold_error_rad=0.0,
        )

        self.assertLess(right.wz, 0.0)
        self.assertLessEqual(crossed.wz, 0.0)
        self.assertGreater(left.wz, 0.0)

    def test_sustained_outward_drift_enters_before_wide_static_gate(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                max_forward_accel_mps2=20.0,
                max_yaw_accel_rps2=20.0,
            )
        )
        commands = []
        for index in range(50):
            lateral = min(0.24, 0.10 + 0.004 * index)
            commands.append(
                controller.update(
                    DetectionResult(
                        valid=True,
                        lateral_error=lateral,
                        confidence=1.0,
                    ),
                    now=0.05 * index,
                    heading_hold_error_rad=0.0,
                )
            )

        first_correction = next(
            index
            for index, command in enumerate(commands)
            if command.wz < 0.0
        )
        self.assertLess(
            0.10 + 0.004 * first_correction,
            controller.config.correction_enter_lateral_error,
        )

    def test_alternating_outward_rate_does_not_confirm_drift(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                max_forward_accel_mps2=20.0,
                max_yaw_accel_rps2=20.0,
            )
        )
        commands = []
        for index in range(80):
            commands.append(
                controller.update(
                    DetectionResult(
                        valid=True,
                        lateral_error=0.20 if index % 2 else -0.20,
                        confidence=1.0,
                    ),
                    now=0.05 * index,
                    heading_hold_error_rad=0.0,
                )
            )

        self.assertLess(max(abs(command.wz) for command in commands), 1e-9)

    def test_persistent_camera_center_offset_is_corrected(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                max_forward_accel_mps2=20.0,
                max_yaw_accel_rps2=20.0,
            )
        )
        commands = []
        for index in range(40):
            commands.append(
                controller.update(
                    DetectionResult(
                        valid=True,
                        lateral_error=0.085,
                        confidence=1.0,
                    ),
                    now=0.05 * index,
                    heading_hold_error_rad=0.0,
                )
            )

        self.assertLess(commands[-1].wz, 0.0)
        first_correction = next(
            index
            for index, command in enumerate(commands)
            if command.wz < 0.0
        )
        self.assertGreaterEqual(
            0.05 * first_correction,
            controller.config.center_alignment_confirm_s - 0.15,
        )

    def test_lane_to_right_commands_right_turn(self):
        controller = LaneFollowerController()
        result = DetectionResult(
            valid=True,
            lateral_error=0.35,
            heading_error_rad=0.05,
            confidence=0.9,
        )
        for index in range(8):
            command = controller.update(result, now=0.05 * index)
        self.assertGreater(command.vx, 0.0)
        self.assertLess(command.wz, 0.0)

    def test_lane_to_left_commands_left_turn(self):
        controller = LaneFollowerController()
        result = DetectionResult(
            valid=True,
            lateral_error=-0.35,
            heading_error_rad=-0.05,
            confidence=0.9,
        )
        for index in range(8):
            command = controller.update(result, now=0.05 * index)
        self.assertGreater(command.wz, 0.0)

    def test_no_initial_line_starts_straight_under_imu_hold(self):
        controller = LaneFollowerController()
        first = controller.update(
            DetectionResult(valid=False),
            now=0.0,
            heading_hold_error_rad=0.1,
        )

        self.assertEqual(first.state, "STRAIGHT_IMU_NO_LINE")
        self.assertGreater(first.vx, 0.0)
        self.assertLess(first.wz, 0.0)
        self.assertFalse(first.perception_valid)

    def test_long_line_loss_continues_straight_under_imu_hold(self):
        controller = LaneFollowerController()
        valid = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            running = controller.update(valid, now=0.1 * index)
        lost = controller.update(
            DetectionResult(valid=False),
            now=5.0,
            heading_hold_error_rad=0.1,
        )
        self.assertEqual(lost.state, "STRAIGHT_IMU_LINE_LOST")
        self.assertAlmostEqual(lost.vx, running.vx)
        self.assertLess(lost.wz, 0.0)

    def test_forward_speed_ramps_up(self):
        controller = LaneFollowerController()
        valid = DetectionResult(valid=True, confidence=1.0)
        first = controller.update(valid, now=0.0)
        second = controller.update(valid, now=0.1)
        self.assertGreater(first.vx, 0.0)
        self.assertLess(first.vx, second.vx)
        self.assertLess(second.vx, controller.config.cruise_speed_mps)

    def test_brief_dropout_holds_speed_and_does_not_stop(self):
        controller = LaneFollowerController()
        valid = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            command = controller.update(valid, now=0.1 * index)

        lost = controller.update(DetectionResult(valid=False), now=1.95)

        self.assertEqual(lost.state, "STRAIGHT_IMU_LINE_LOST")
        self.assertAlmostEqual(lost.vx, command.vx)
        self.assertGreater(lost.vx, controller.config.minimum_tracking_speed_mps)

    def test_sustained_dropout_keeps_cruise_speed(self):
        controller = LaneFollowerController()
        valid = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            command = controller.update(valid, now=0.1 * index)

        speeds = []
        for index in range(1, 31):
            lost = controller.update(
                DetectionResult(valid=False), now=1.9 + 0.1 * index
            )
            speeds.append(lost.vx)

        self.assertTrue(all(speed > 0.0 for speed in speeds))
        self.assertTrue(
            all(after >= before - 1e-9 for before, after in zip(speeds, speeds[1:]))
        )
        self.assertEqual(lost.state, "STRAIGHT_IMU_LINE_LOST")
        self.assertAlmostEqual(lost.vx, controller.config.cruise_speed_mps)

    def test_finish_approach_keeps_lane_correction_active(self):
        controller = LaneFollowerController()
        turning = DetectionResult(
            valid=True,
            lateral_error=0.4,
            heading_error_rad=0.2,
            confidence=1.0,
        )
        for index in range(20):
            command = controller.update(turning, now=0.1 * index)

        finish = controller.follow_lane_through_finish(
            turning,
            now=2.0,
            heading_hold_error_rad=0.0,
        )

        self.assertEqual(finish.state, "FINISH_APPROACH_LANE_FOLLOW")
        self.assertGreater(finish.vx, 0.0)
        self.assertTrue(finish.perception_valid)
        self.assertLess(finish.wz, 0.0)

    def test_post_finish_braking_starts_after_line_and_reaches_zero(self):
        controller = LaneFollowerController()
        valid = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            running = controller.update(valid, now=0.1 * index)

        speeds = []
        for index in range(1, 20):
            braking = controller.brake_after_finish(now=1.9 + 0.1 * index)
            speeds.append(braking.vx)
            if braking.state == "POST_FINISH_STOP":
                break

        self.assertLess(speeds[0], running.vx)
        self.assertTrue(
            all(after <= before for before, after in zip(speeds, speeds[1:]))
        )
        self.assertEqual(braking.state, "POST_FINISH_STOP")
        self.assertEqual(braking.vx, 0.0)

    def test_post_finish_braking_keeps_visual_lane_correction(self):
        controller = LaneFollowerController()
        centered = DetectionResult(valid=True, confidence=1.0)
        controller.update(centered, now=0.0)
        lane_center_to_right = DetectionResult(
            valid=True,
            lateral_error=0.35,
            heading_error_rad=0.05,
            confidence=1.0,
        )

        for index in range(1, 9):
            controller.update(
                lane_center_to_right,
                now=0.05 * index,
                heading_hold_error_rad=0.0,
            )

        braking = controller.brake_after_finish(
            result=lane_center_to_right,
            now=0.45,
            heading_hold_error_rad=0.0,
        )

        self.assertTrue(braking.perception_valid)
        self.assertLess(braking.wz, 0.0)

    def test_dropout_uses_imu_heading_hold_without_visual_inference(self):
        controller = LaneFollowerController()
        valid = DetectionResult(valid=True, confidence=1.0)
        controller.update(valid, now=0.0, heading_hold_error_rad=0.0)

        lost = controller.update(
            DetectionResult(valid=False),
            now=0.1,
            heading_hold_error_rad=0.2,
        )

        self.assertFalse(lost.perception_valid)
        self.assertLess(lost.wz, 0.0)
        self.assertGreater(lost.vx, 0.0)

    def test_heading_outlier_is_rejected_against_imu(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                cruise_speed_mps=5.1,
                max_forward_accel_mps2=20.0,
                heading_kp=1.5,
            )
        )
        commands = []
        for index in range(20):
            noisy_heading = 0.70 if index % 2 else -0.70
            commands.append(
                controller.update(
                    DetectionResult(
                        valid=True,
                        heading_error_rad=noisy_heading,
                        confidence=0.9,
                    ),
                    now=0.05 * index,
                    heading_hold_error_rad=0.0,
                )
            )

        self.assertLess(max(abs(command.wz) for command in commands), 0.05)
        self.assertGreater(commands[-1].vx, 4.0)

    def test_lane_lock_calibrates_fixed_visual_heading_bias(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                cruise_speed_mps=5.1,
                max_forward_accel_mps2=20.0,
                heading_kp=1.5,
            )
        )
        controller.set_visual_heading_reference(-0.12)

        command = controller.update(
            DetectionResult(
                valid=True,
                heading_error_rad=-0.12,
                confidence=0.9,
            ),
            now=0.0,
            heading_hold_error_rad=0.0,
        )

        self.assertLess(abs(command.wz), 0.01)

    def test_large_body_heading_slows_even_when_lane_is_centered(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                cruise_speed_mps=5.1,
                max_forward_accel_mps2=20.0,
            )
        )
        centered = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            fast = controller.update(
                centered,
                now=0.05 * index,
                heading_hold_error_rad=0.0,
            )

        turned = controller.update(
            centered,
            now=1.0,
            heading_hold_error_rad=0.22,
        )

        self.assertGreater(fast.vx, 4.5)
        self.assertLess(turned.vx, fast.vx)
        self.assertLess(turned.wz, 0.0)

    def test_lateral_motion_slows_before_crossing_lane_center(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                cruise_speed_mps=5.1,
                max_forward_accel_mps2=20.0,
                max_forward_decel_mps2=20.0,
            )
        )
        centered = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            fast = controller.update(centered, now=0.05 * index)

        now = 1.0
        for lateral in (
            0.06,
            0.11,
            0.16,
            0.21,
            0.26,
            0.30,
            0.32,
            0.34,
            0.35,
            0.35,
        ):
            moving = controller.update(
                DetectionResult(
                    valid=True,
                    lateral_error=lateral,
                    confidence=1.0,
                ),
                now=now,
                heading_hold_error_rad=0.0,
            )
            now += 0.05

        self.assertGreater(fast.vx, 4.5)
        self.assertLess(moving.vx, 3.0)

    def test_large_lateral_error_forces_smooth_strong_slowdown(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                cruise_speed_mps=5.1,
                max_forward_accel_mps2=20.0,
            )
        )
        centered = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            fast = controller.update(centered, now=0.05 * index)

        speeds = []
        severe = DetectionResult(
            valid=True,
            lateral_error=0.52,
            confidence=1.0,
        )
        for index in range(1, 31):
            command = controller.update(severe, now=0.95 + 0.05 * index)
            speeds.append(command.vx)

        self.assertGreater(fast.vx, 4.5)
        self.assertLess(speeds[-1], 1.5)
        per_step_limit = controller.config.max_forward_decel_mps2 * 0.05
        self.assertTrue(
            all(
                before - after <= per_step_limit + 1e-9
                for before, after in zip([fast.vx] + speeds, speeds)
            )
        )

    def test_sustained_yaw_saturation_reduces_forward_speed(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                cruise_speed_mps=5.1,
                max_forward_accel_mps2=20.0,
                max_yaw_accel_rps2=20.0,
                max_lane_correction_heading_rad=10.0,
            )
        )
        centered = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            fast = controller.update(centered, now=0.05 * index)

        saturated = DetectionResult(
            valid=True,
            lateral_error=-1.0,
            confidence=1.0,
        )
        commands = []
        for index in range(1, 31):
            commands.append(
                controller.update(
                    saturated,
                    now=0.95 + 0.05 * index,
                )
            )

        self.assertGreater(fast.vx, 4.5)
        self.assertTrue(
            any(
                abs(command.wz) >= 0.9 * controller.config.max_yaw_rate_rps
                for command in commands
            )
        )
        self.assertLess(commands[-1].vx, 1.5)

    def test_boundary_warning_decelerates_but_keeps_correcting(self):
        controller = LaneFollowerController(
            LaneFollowerConfig(
                cruise_speed_mps=5.1,
                max_forward_accel_mps2=20.0,
            )
        )
        centered = DetectionResult(valid=True, confidence=1.0)
        for index in range(20):
            fast = controller.update(centered, now=0.05 * index)

        warning = DetectionResult(
            valid=True,
            lateral_error=-0.60,
            confidence=0.8,
            source="boundary-warning",
            boundary_risk=True,
        )
        commands = []
        for index in range(1, 31):
            commands.append(
                controller.update(
                    warning,
                    now=0.95 + 0.05 * index,
                    heading_hold_error_rad=0.0,
                )
            )

        self.assertGreater(fast.vx, 4.5)
        self.assertEqual(
            commands[-1].state,
            "LINE_FOLLOW_BOUNDARY_RECOVERY",
        )
        self.assertLess(commands[-1].vx, 1.5)
        self.assertGreater(commands[-1].wz, 0.0)


if __name__ == "__main__":
    unittest.main()
