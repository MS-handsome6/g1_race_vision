from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from g1_race_vision.camera_geometry import CameraIntrinsics
from g1_race_vision.line_detector import LaneDetectorConfig, WhiteLaneDetector
from g1_race_vision.rendering import (
    attitude_align_rgbd,
    average_rotation_matrices,
    circular_mean_angle,
    gravity_align_rgbd,
)


def synthetic_lane(shift_px: int = 0, width: int = 640, height: int = 480) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (85, 28, 28)
    left_bottom = (95 + shift_px, height - 1)
    left_top = (255 + shift_px, 175)
    right_bottom = (545 + shift_px, height - 1)
    right_top = (385 + shift_px, 175)
    cv2.line(image, left_bottom, left_top, (245, 245, 245), 14)
    cv2.line(image, right_bottom, right_top, (245, 245, 245), 14)
    return image


def horizontal_motion_blur(image: np.ndarray, width: int = 25) -> np.ndarray:
    kernel = np.full((1, width), 1.0 / width, dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


class WhiteLaneDetectorTest(unittest.TestCase):
    def test_centered_lane(self):
        result = WhiteLaneDetector().detect(synthetic_lane())
        self.assertTrue(result.valid)
        self.assertLess(abs(result.lateral_error), 0.04)
        self.assertLess(abs(result.heading_error_rad), 0.04)

    def test_low_vanishing_point_still_has_two_distinct_boundaries(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (85, 28, 28)
        # Leave the sub-pixel horizon gap seen after RGB resampling so the two
        # paint contours remain physically distinct.
        cv2.line(image, (95, 479), (312, 240), (245, 245, 245), 8)
        cv2.line(image, (545, 479), (328, 240), (245, 245, 245), 8)

        result = WhiteLaneDetector().detect(image)

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.left_line)
        self.assertIsNotNone(result.right_line)

    def test_boundaries_clipped_before_fixed_bottom_row_remain_detectable(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (70, 70, 70)
        # Both boundaries are genuinely visible over a long shared span, but
        # leave the image before the detector's fixed near-field geometry row.
        # Extrapolating to that row makes their apparent width exceed 640 px.
        cv2.line(image, (0, 400), (120, 182), (245, 245, 245), 14)
        cv2.line(image, (639, 305), (560, 182), (245, 245, 245), 14)

        result = WhiteLaneDetector().detect(image)

        self.assertTrue(result.valid)
        self.assertEqual(result.source, "two-lines")
        self.assertIsNotNone(result.left_line)
        self.assertIsNotNone(result.right_line)

    def test_latest_field_frames_lock_as_a_stable_two_line_sequence(self):
        frame_dir = (
            Path(__file__).resolve().parents[1]
            / "tests"
            / "fixtures"
            / "field_20260815_live"
        )
        detector = WhiteLaneDetector()
        frame_paths = sorted(frame_dir.glob("f*_raw_color.png"))
        self.assertEqual(len(frame_paths), 3)

        for frame_path in frame_paths:
            # ``cv2.imread`` on Windows cannot reliably open non-ASCII paths.
            # Decode bytes explicitly so this regression also runs from the
            # Chinese-named internship workspace used for deployment.
            bgr = cv2.imdecode(
                np.fromfile(frame_path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(bgr, str(frame_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(rgb)
            self.assertTrue(
                result.valid,
                f"{frame_path.name}: {result.source}",
            )
            self.assertIn(
                result.source,
                ("two-lines", "two-lines-guided"),
            )
            self.assertGreater(result.confidence, 0.25)

    def test_lane_center_to_right_is_positive(self):
        result = WhiteLaneDetector().detect(synthetic_lane(shift_px=70))
        self.assertTrue(result.valid)
        self.assertGreater(result.lateral_error, 0.15)

    def test_lane_center_to_left_is_negative(self):
        result = WhiteLaneDetector().detect(synthetic_lane(shift_px=-70))
        self.assertTrue(result.valid)
        self.assertLess(result.lateral_error, -0.15)

    def test_camera_intrinsics_control_heading_estimate(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (85, 28, 28)
        # Both boundaries converge at x=380. A calibrated principal point at
        # that same x means the robot is parallel to the lane even though the
        # camera is not optically centred in the image.
        cv2.line(image, (160, 479), (334, 180), (245, 245, 245), 14)
        cv2.line(image, (600, 479), (427, 180), (245, 245, 245), 14)

        calibrated = WhiteLaneDetector().detect(
            image,
            intrinsics=CameraIntrinsics(
                fx=800.0,
                fy=800.0,
                cx=380.0,
                cy=240.0,
                width=640,
                height=480,
            ),
        )
        image_centered = WhiteLaneDetector().detect(
            image,
            intrinsics=CameraIntrinsics(
                fx=800.0,
                fy=800.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            ),
        )

        self.assertTrue(calibrated.valid)
        self.assertTrue(image_centered.valid)
        self.assertLess(abs(calibrated.lateral_error), 0.01)
        self.assertGreater(image_centered.lateral_error, 0.15)
        self.assertAlmostEqual(calibrated.camera_center_x_px, 380.0)
        self.assertAlmostEqual(
            calibrated.lane_center_x_px,
            calibrated.camera_center_x_px,
            delta=2.0,
        )
        self.assertLess(abs(calibrated.heading_error_rad), 0.01)
        self.assertGreater(image_centered.heading_error_rad, 0.05)

    def test_one_remaining_line_is_invalid_in_strict_two_line_mode(self):
        image = synthetic_lane()
        depth = np.full(image.shape[:2], 3.0, dtype=np.float32)
        depth[:, : image.shape[1] // 2] = 0.10
        detector = WhiteLaneDetector()
        detector.detect(synthetic_lane())
        result = detector.detect(image, depth)
        self.assertFalse(result.valid)
        self.assertEqual(result.source, "none")
        self.assertIsNone(result.left_line)
        self.assertIsNone(result.right_line)

    def test_empty_image_is_invalid(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = WhiteLaneDetector().detect(image)
        self.assertFalse(result.valid)

    def test_low_light_and_color_cast_remain_detectable(self):
        image = synthetic_lane().astype(np.float32)
        dim_warm = np.clip(
            image * np.asarray([0.42, 0.34, 0.30], dtype=np.float32),
            0,
            255,
        ).astype(np.uint8)

        result = WhiteLaneDetector().detect(dim_warm)

        self.assertTrue(result.valid)
        self.assertEqual(result.source, "two-lines")
        self.assertLess(result.value_threshold, 100.0)

    def test_abrupt_shadow_after_bright_frame_does_not_drop_pair(self):
        detector = WhiteLaneDetector()
        self.assertTrue(detector.detect(synthetic_lane()).valid)
        shadow = np.clip(
            synthetic_lane().astype(np.float32) * 0.28, 0, 255
        ).astype(np.uint8)

        result = detector.detect(shadow)

        self.assertTrue(result.valid)
        self.assertLess(abs(result.lateral_error), 0.04)

    def test_horizontal_motion_blur_remains_detectable(self):
        result = WhiteLaneDetector().detect(
            horizontal_motion_blur(synthetic_lane())
        )

        self.assertTrue(result.valid)
        self.assertLess(abs(result.lateral_error), 0.05)

    def test_locked_pair_survives_large_common_camera_shake(self):
        detector = WhiteLaneDetector()
        self.assertTrue(detector.detect(synthetic_lane()).valid)

        right = detector.detect(synthetic_lane(shift_px=110))
        left = detector.detect(synthetic_lane(shift_px=-110))
        recovered = detector.detect(synthetic_lane())

        self.assertTrue(right.valid)
        self.assertTrue(left.valid)
        self.assertTrue(recovered.valid)
        self.assertGreater(right.lateral_error, 0.25)
        self.assertLess(left.lateral_error, -0.25)

    def test_fragmented_locked_boundaries_are_refit_from_real_pixels(self):
        detector = WhiteLaneDetector()
        image = synthetic_lane()
        self.assertTrue(detector.detect(image).valid)
        fragmented = image.copy()
        fragmented[225:260, :] = (85, 28, 28)
        fragmented[300:335, :] = (85, 28, 28)
        fragmented[375:410, :] = (85, 28, 28)
        fragmented[450:, :] = (85, 28, 28)

        result = detector.detect(fragmented)

        self.assertTrue(result.valid)
        self.assertEqual(result.source, "two-lines-guided")
        self.assertIsNotNone(result.left_line)
        self.assertIsNotNone(result.right_line)

    def test_impossible_heading_pair_is_rejected(self):
        detector = WhiteLaneDetector(
            LaneDetectorConfig(max_abs_heading_error_rad=0.05)
        )
        result = detector.detect(synthetic_lane(shift_px=70))
        self.assertFalse(result.valid)

    def test_locked_lane_does_not_jump_to_adjacent_pair(self):
        detector = WhiteLaneDetector(
            LaneDetectorConfig(max_boundary_step_ratio=0.18)
        )
        self.assertTrue(detector.detect(synthetic_lane()).valid)

        adjacent_lane = synthetic_lane(shift_px=190)
        result = detector.detect(adjacent_lane)

        self.assertFalse(result.valid)
        self.assertEqual(result.source, "none")

    def test_reset_restores_only_a_centered_initial_lock(self):
        detector = WhiteLaneDetector(
            LaneDetectorConfig(max_boundary_step_ratio=0.18)
        )
        self.assertTrue(detector.detect(synthetic_lane()).valid)
        self.assertFalse(detector.detect(synthetic_lane(shift_px=160)).valid)

        detector.reset()

        # Reset clears stale history, but must not make an adjacent/off-centre
        # lane eligible for the initial identity lock.
        self.assertFalse(detector.detect(synthetic_lane(shift_px=160)).valid)
        self.assertTrue(detector.detect(synthetic_lane()).valid)

    def test_reset_restarts_adaptive_exposure_for_the_new_mission(self):
        detector = WhiteLaneDetector()
        detector.detect(synthetic_lane())
        self.assertIsNotNone(detector._adaptive_value_threshold)

        detector.reset()

        self.assertIsNone(detector._adaptive_value_threshold)

    def test_large_pair_offset_enters_boundary_recovery(self):
        detector = WhiteLaneDetector()

        result = detector.detect(synthetic_lane(shift_px=190))

        self.assertFalse(result.valid)
        self.assertEqual(result.source, "none")

    def test_gradual_tracking_drift_cannot_walk_lock_to_adjacent_lane(self):
        detector = WhiteLaneDetector()
        self.assertTrue(detector.detect(synthetic_lane()).valid)

        for shift in (30, 60, 90, 120):
            self.assertTrue(detector.detect(synthetic_lane(shift_px=shift)).valid)

        adjacent = detector.detect(synthetic_lane(shift_px=190))

        self.assertFalse(adjacent.valid)
        self.assertEqual(adjacent.source, "none")

    def test_lane_lock_does_not_zero_a_real_camera_center_offset(self):
        detector = WhiteLaneDetector()
        shifted = synthetic_lane(shift_px=80)
        first = detector.detect(shifted)
        second = detector.detect(shifted)

        self.assertTrue(first.valid)
        self.assertTrue(second.valid)
        self.assertGreater(abs(first.lateral_error), 0.10)
        self.assertAlmostEqual(
            second.lateral_error,
            first.lateral_error,
            delta=0.02,
        )

    def test_confirmed_boundary_breach_cannot_relock_without_reset(self):
        detector = WhiteLaneDetector(
            LaneDetectorConfig(boundary_breach_confirm_frames=3)
        )
        self.assertTrue(detector.detect(synthetic_lane()).valid)

        # Approach the boundary continuously, as the real robot would. Once
        # the last trustworthy observation is already in the warning band,
        # repeated adjacent-lane candidates must permanently invalidate the
        # lane identity instead of silently relocking.
        for shift in (30, 60, 90, 120, 140):
            result = detector.detect(synthetic_lane(shift_px=shift))
            self.assertTrue(result.valid)
        outside = synthetic_lane(shift_px=190)

        for _ in range(3):
            result = detector.detect(outside)

        self.assertFalse(result.valid)
        self.assertEqual(result.source, "lane-identity-lost")
        still_locked_out = detector.detect(synthetic_lane())
        self.assertFalse(still_locked_out.valid)
        self.assertEqual(still_locked_out.source, "lane-identity-lost")

        detector.reset()
        self.assertTrue(detector.detect(synthetic_lane()).valid)

    def test_imu_roll_alignment_restores_shaken_lane_geometry(self):
        image = synthetic_lane()
        height, width = image.shape[:2]
        center = (0.5 * (width - 1), 0.5 * (height - 1))
        shake_degrees = 12.0
        shaken = cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D(center, shake_degrees, 1.0),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        aligned, _ = gravity_align_rgbd(
            shaken,
            np.radians(-shake_degrees),
        )
        result = WhiteLaneDetector().detect(aligned)

        self.assertTrue(result.valid)
        self.assertLess(abs(result.lateral_error), 0.05)
        self.assertLess(abs(result.heading_error_rad), 0.05)

    def test_attitude_alignment_identity_preserves_lane(self):
        image = synthetic_lane()
        camera_xmat = np.eye(3, dtype=np.float64)

        aligned, _ = attitude_align_rgbd(
            image,
            camera_xmat,
            camera_xmat,
            vertical_fov_degrees=58.0,
        )
        result = WhiteLaneDetector().detect(aligned)

        self.assertTrue(result.valid)
        self.assertLess(abs(result.lateral_error), 0.04)

    def test_rotation_average_rejects_opposite_gait_roll(self):
        angle = np.radians(12.0)
        positive = np.array(
            ((np.cos(angle), -np.sin(angle), 0.0),
             (np.sin(angle), np.cos(angle), 0.0),
             (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        negative = positive.T

        averaged = average_rotation_matrices([positive, negative])

        np.testing.assert_allclose(averaged, np.eye(3), atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(averaged), 1.0, places=7)

    def test_circular_heading_average_handles_wraparound(self):
        averaged = circular_mean_angle(
            [np.radians(179.0), np.radians(-179.0)]
        )

        self.assertAlmostEqual(abs(averaged), np.pi, places=6)

    def _num7_detector(self, **config_overrides):
        overrides = dict(initial_lock_confirm_frames=3)
        overrides.update(config_overrides)
        config = LaneDetectorConfig(**overrides)
        detector = WhiteLaneDetector(config)
        detector.configure_num7_lifecycle(True)
        detector.set_num7_mission_active(True)
        return detector

    def test_num7_preview_never_pollutes_formal_lane_identity(self):
        detector = WhiteLaneDetector(
            LaneDetectorConfig(
                initial_lock_confirm_frames=3,
                boundary_breach_confirm_frames=1,
            )
        )
        detector.configure_num7_lifecycle(True)

        for _ in range(100):
            detector.detect(synthetic_lane(shift_px=190))

        self.assertIsNone(detector._lane_anchor_geometry)
        self.assertIsNone(detector._locked_pair_geometry)
        self.assertEqual(detector._boundary_breach_frames, 0)
        self.assertFalse(detector._lane_identity_lost)
        self.assertEqual(detector._mission_state, "PREVIEW")

    def test_num7_initial_lock_requires_confirmation_frames(self):
        detector = self._num7_detector()

        first = detector.detect(synthetic_lane())
        self.assertFalse(first.valid)
        self.assertEqual(first.source, "INITIAL_LOCK")
        self.assertEqual(first.mode, "INITIAL_LOCK")
        self.assertIsNone(detector._lane_anchor_geometry)

        second = detector.detect(synthetic_lane())
        self.assertFalse(second.valid)
        self.assertEqual(second.lock_confirm, 2)

        locked = detector.detect(synthetic_lane())
        self.assertTrue(locked.valid)
        self.assertEqual(locked.mode, "LOCKED")
        self.assertIsNotNone(detector._lane_anchor_geometry)
        self.assertIsNotNone(detector._locked_pair_geometry)

    def test_num7_initial_lock_pair_jump_resets_confirmation(self):
        detector = self._num7_detector()
        detector.detect(synthetic_lane())
        self.assertEqual(detector._lock_confirm_count, 1)

        detector.detect(synthetic_lane(shift_px=60))
        self.assertEqual(detector._lock_confirm_count, 1)

    def test_num7_initial_lock_rejects_thin_centre_seam(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (70, 70, 70)
        cv2.line(image, (95, 479), (255, 175), (245, 245, 245), 14)
        cv2.line(image, (545, 479), (385, 175), (245, 245, 245), 14)
        cv2.line(image, (350, 175), (390, 479), (245, 245, 245), 2)

        detector = self._num7_detector()
        first = detector.detect(image)
        self.assertIn(
            "initial thickness",
            " ".join(first.candidate_reject_reasons),
        )

        result = None
        for _ in range(4):
            result = detector.detect(image)

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.left_line)
        self.assertIsNotNone(result.right_line)
        self.assertLess(result.left_line.slope, -0.20)
        self.assertGreater(result.right_line.slope, 0.20)

    def test_num7_initial_lock_rejects_boundary_warning_pair(self):
        detector = self._num7_detector(
            initial_min_center_margin_ratio=0.0,
            initial_lock_confirm_frames=2,
            max_pair_center_offset_ratio=0.50,
        )

        for _ in range(5):
            result = detector.detect(synthetic_lane(shift_px=132))

        self.assertNotEqual(detector._mission_state, "LOCKED")
        self.assertIsNone(detector._lane_anchor_geometry)
        self.assertIn(
            "boundary-warning", " ".join(result.candidate_reject_reasons)
        )

    def test_num7_reset_returns_to_preview_and_clears_initial_lock(self):
        detector = self._num7_detector()
        for _ in range(3):
            detector.detect(synthetic_lane())
        self.assertEqual(detector._mission_state, "LOCKED")

        detector.set_num7_mission_active(False)
        self.assertEqual(detector._mission_state, "PREVIEW")
        self.assertIsNone(detector._lane_anchor_geometry)
        self.assertFalse(detector._lane_identity_lost)

    def test_attitude_alignment_recovers_combined_camera_rotation(self):
        image = synthetic_lane()
        height, width = image.shape[:2]
        fovy = 58.0
        focal = (0.5 * height) / np.tan(np.radians(0.5 * fovy))
        intrinsic = np.array(
            (
                (focal, 0.0, 0.5 * (width - 1)),
                (0.0, focal, 0.5 * (height - 1)),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        basis = np.diag((1.0, -1.0, -1.0))

        yaw = np.radians(7.0)
        pitch = np.radians(-5.0)
        roll = np.radians(9.0)
        rz = np.array(
            ((np.cos(yaw), -np.sin(yaw), 0.0),
             (np.sin(yaw), np.cos(yaw), 0.0),
             (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        ry = np.array(
            ((np.cos(pitch), 0.0, np.sin(pitch)),
             (0.0, 1.0, 0.0),
             (-np.sin(pitch), 0.0, np.cos(pitch))),
            dtype=np.float64,
        )
        rx = np.array(
            ((1.0, 0.0, 0.0),
             (0.0, np.cos(roll), -np.sin(roll)),
             (0.0, np.sin(roll), np.cos(roll))),
            dtype=np.float64,
        )
        reference = np.eye(3, dtype=np.float64)
        current = rz @ ry @ rx
        reference_to_current = (
            intrinsic
            @ basis
            @ current.T
            @ reference
            @ basis
            @ np.linalg.inv(intrinsic)
        )
        shaken = cv2.warpPerspective(
            image,
            reference_to_current,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        aligned, _ = attitude_align_rgbd(
            shaken,
            current,
            reference,
            vertical_fov_degrees=fovy,
        )

        # Interpolation and uncovered borders prevent pixel identity, but the
        # central lane geometry should return to its original location.
        reference_result = WhiteLaneDetector().detect(image)
        aligned_result = WhiteLaneDetector().detect(aligned)
        self.assertTrue(aligned_result.valid)
        self.assertAlmostEqual(
            aligned_result.lateral_error,
            reference_result.lateral_error,
            delta=0.035,
        )
        self.assertAlmostEqual(
            aligned_result.heading_error_rad,
            reference_result.heading_error_rad,
            delta=0.04,
        )


if __name__ == "__main__":
    unittest.main()
