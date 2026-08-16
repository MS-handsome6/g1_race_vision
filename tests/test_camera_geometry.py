from __future__ import annotations

import unittest

import cv2
import numpy as np

from g1_race_vision.camera_geometry import (
    CameraIntrinsics,
    median_depth_in_neighbourhood,
)
from g1_race_vision.line_detector import (
    LaneDetectorConfig,
    LineModel,
    WhiteLaneDetector,
)


class CameraIntrinsicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            fx=320.0,
            fy=320.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

    def test_deprojection_uses_real_focal_length(self) -> None:
        points = np.asarray(((420.0, 240.0), (320.0, 280.0)))
        depths = np.asarray((3.2, 2.0))
        camera_points = self.intrinsics.deproject(points, depths)
        np.testing.assert_allclose(
            camera_points[0],
            (1.0, 0.0, 3.2),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            camera_points[1],
            (0.0, 0.25, 2.0),
            atol=1e-6,
        )

    def test_fov_derived_from_camera_info_focal_length(self) -> None:
        self.assertAlmostEqual(
            self.intrinsics.vertical_fov_degrees,
            2.0 * np.degrees(np.arctan2(240.0, 320.0)),
            places=6,
        )

    def test_median_depth_ignores_holes_and_invalid_ranges(self) -> None:
        depth = np.full((7, 7), 3.0, dtype=np.float32)
        depth[3, 3] = 0.0
        depth[2, 2] = np.nan
        depth[4, 4] = 20.0

        sampled = median_depth_in_neighbourhood(
            depth,
            3,
            3,
            radius=2,
            min_valid_depth_m=0.35,
            max_valid_depth_m=12.0,
            min_valid_pixels=3,
        )
        self.assertAlmostEqual(sampled, 3.0, places=6)

    def test_median_depth_rejects_sparse_neighbourhood(self) -> None:
        depth = np.zeros((7, 7), dtype=np.float32)
        depth[3, 3] = 3.0
        sampled = median_depth_in_neighbourhood(
            depth,
            3,
            3,
            radius=2,
            min_valid_depth_m=0.35,
            max_valid_depth_m=12.0,
            min_valid_pixels=3,
        )
        self.assertIsNone(sampled)


class PhysicalLaneWidthTest(unittest.TestCase):
    def make_detector(self) -> WhiteLaneDetector:
        return WhiteLaneDetector(
            LaneDetectorConfig(
                physical_width_sample_rows=5,
                physical_width_min_valid_samples=3,
            )
        )

    def make_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            fx=320.0,
            fy=320.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

    def test_2p1m_pair_passes_physical_width_gate(self) -> None:
        detector = self.make_detector()
        intrinsics = self.make_intrinsics()
        depth = np.full((480, 640), 3.36, dtype=np.float32)
        left = LineModel(0.0, 220.0, 0.8, 100.0, 400.0, 1000.0, 20.0)
        right = LineModel(0.0, 420.0, 0.8, 100.0, 400.0, 1000.0, 20.0)

        width_m, reliable = detector._physical_pair_width(
            left, right, depth, intrinsics, far_y=271.0, near_y=367.0
        )
        self.assertTrue(reliable)
        self.assertAlmostEqual(width_m, 2.10, places=2)

    def test_floor_seam_pair_is_rejected_by_physical_width_gate(self) -> None:
        detector = self.make_detector()
        intrinsics = self.make_intrinsics()
        depth = np.full((480, 640), 3.36, dtype=np.float32)
        left = LineModel(0.0, 280.0, 0.8, 100.0, 400.0, 1000.0, 20.0)
        seam = LineModel(0.0, 360.0, 0.8, 100.0, 400.0, 1000.0, 2.0)

        width_m, reliable = detector._physical_pair_width(
            left, seam, depth, intrinsics, far_y=271.0, near_y=367.0
        )
        self.assertTrue(reliable)
        self.assertLess(width_m, detector.config.min_lane_width_m)

    def test_depth_holes_fall_back_to_unreliable_2d_path(self) -> None:
        detector = self.make_detector()
        intrinsics = self.make_intrinsics()
        depth = np.zeros((480, 640), dtype=np.float32)
        left = LineModel(0.0, 220.0, 0.8, 100.0, 400.0, 1000.0, 20.0)
        right = LineModel(0.0, 420.0, 0.8, 100.0, 400.0, 1000.0, 20.0)

        width_m, reliable = detector._physical_pair_width(
            left, right, depth, intrinsics, far_y=271.0, near_y=367.0
        )
        self.assertFalse(reliable)
        self.assertIsNone(width_m)


class InitialLockPhysicalGateTest(unittest.TestCase):
    def test_reliable_physical_width_gate_accepts_and_rejects(self) -> None:
        detector = WhiteLaneDetector(
            LaneDetectorConfig(
                initial_min_center_margin_ratio=0.03,
                initial_min_abs_perspective_slope=0.05,
                min_lane_width_m=1.75,
                max_lane_width_m=2.45,
            )
        )
        left = LineModel(-0.50, 383.5, 0.80, 100.0, 400.0, 3000.0, 20.0)
        right = LineModel(0.50, 256.5, 0.80, 100.0, 400.0, 3000.0, 20.0)

        ok, reason = detector._initial_lock_candidate_ok(
            left,
            right,
            640,
            far_y=271.0,
            near_y=367.0,
            physical_width_m=2.10,
            physical_width_reliable=True,
        )
        self.assertTrue(ok, reason)

        ok, reason = detector._initial_lock_candidate_ok(
            left,
            right,
            640,
            far_y=271.0,
            near_y=367.0,
            physical_width_m=0.95,
            physical_width_reliable=True,
        )
        self.assertFalse(ok)
        self.assertIn("physical width", reason)


class DetectorDepthFallbackTest(unittest.TestCase):
    def test_legacy_detector_ignores_physical_depth_absence(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (85, 28, 28)
        cv2.line(image, (95, 479), (255, 175), (245, 245, 245), 14)
        cv2.line(image, (545, 479), (385, 175), (245, 245, 245), 14)
        depth = np.full((480, 640), np.nan, dtype=np.float32)
        depth[:, :] = 0.0

        detector = WhiteLaneDetector()
        result = detector.detect(image, depth)

        self.assertTrue(result.valid)
        self.assertIn(result.source, ("two-lines", "two-lines-guided"))


if __name__ == "__main__":
    unittest.main()
