from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .camera_geometry import CameraIntrinsics, median_depth_in_neighbourhood


@dataclass(frozen=True)
class LaneDetectorConfig:
    """Parameters for detecting two white lane boundaries in an RGB image."""

    roi_top_ratio: float = 0.38
    # Keep the far geometry row measurably below the vanishing point.  At a
    # sprinting torso pitch, sampling too near the horizon makes two genuine
    # boundaries cross by a few fitted pixels and falsely fail strict pairing.
    lookahead_ratio: float = 0.30
    white_value_min: int = 175
    white_saturation_max: int = 100
    adaptive_value_floor: int = 55
    adaptive_value_margin: int = 18
    adaptive_value_smoothing: float = 0.35
    max_neutral_chroma: int = 72
    local_contrast_min: int = 14
    local_contrast_kernel: int = 41
    morphology_kernel: int = 3
    vertical_close_kernel: int = 9
    min_component_area: float = 90.0
    min_vertical_span_ratio: float = 0.22
    min_lane_width_ratio: float = 0.20
    max_lane_width_ratio: float = 0.95
    max_top_to_bottom_width_ratio: float = 1.25
    min_depth_m: float = 0.35
    max_depth_m: float = 12.0
    keep_invalid_depth: bool = True
    expected_lane_width_ratio: float = 0.52
    vertical_fov_degrees: float = 72.0
    max_abs_heading_error_rad: float = 0.75
    # --- Num7 initial-lock / physical-width gates ---
    initial_lock_confirm_frames: int = 10
    initial_min_center_margin_ratio: float = 0.03
    initial_min_abs_perspective_slope: float = 0.05
    initial_lock_confirm_step_ratio: float = 0.05
    initial_lock_confirm_width_ratio: float = 0.12
    local_value_ratio: float = 0.65
    min_mean_thickness_ratio: float = 0.007
    # Physical lane-width verification with aligned depth + CameraInfo.
    expected_lane_width_m: float = 2.10
    min_lane_width_m: float = 1.75
    max_lane_width_m: float = 2.45
    physical_width_sample_rows: int = 5
    physical_depth_kernel_radius: int = 2
    physical_width_min_valid_samples: int = 3
    initial_center_weight: float = 0.80
    temporal_pair_weight: float = 5.00
    max_boundary_step_ratio: float = 0.18
    max_common_shift_ratio: float = 0.20
    max_lane_width_change_ratio: float = 0.25
    # Initial lock is allowed only on the lane containing the camera's
    # near-field ground ray. This excludes a fully visible neighbouring lane.
    max_pair_center_offset_ratio: float = 0.18
    # Unlike the short-term tracking geometry, this start-line anchor is never
    # updated. It makes a gradual walk from the selected lane to its neighbour
    # impossible even if every individual frame-to-frame step looks small.
    max_anchor_common_shift_ratio: float = 0.22
    max_anchor_boundary_deformation_ratio: float = 0.12
    max_anchor_lane_width_change_ratio: float = 0.20
    lock_update_alpha: float = 0.12
    guided_search_margin_ratio: float = 0.075
    guided_min_pixels: int = 70
    guided_min_vertical_span_ratio: float = 0.18
    max_abs_lateral_error: float = 0.58
    boundary_warning_lateral_error: float = 0.40
    boundary_recovery_lateral_error: float = 0.30
    boundary_breach_confirm_frames: int = 30


@dataclass(frozen=True)
class LineModel:
    """A lane line represented as x = slope * y + intercept."""

    slope: float
    intercept: float
    score: float
    y_min: float
    y_max: float
    area: float = 0.0
    mean_thickness_px: float = 0.0

    def x_at(self, y: float) -> float:
        return self.slope * y + self.intercept


@dataclass
class DetectionResult:
    valid: bool
    lateral_error: float = 0.0
    heading_error_rad: float = 0.0
    confidence: float = 0.0
    source: str = "none"
    boundary_risk: bool = False
    left_line: Optional[LineModel] = None
    right_line: Optional[LineModel] = None
    roi_y: int = 0
    lookahead_y: int = 0
    image_width: int = 0
    image_height: int = 0
    value_threshold: float = 0.0
    mask: Optional[np.ndarray] = None
    mode: str = "LEGACY"
    lock_confirm: int = 0
    lock_confirm_target: int = 0
    physical_lane_width_m: Optional[float] = None
    # Pixel-space alignment diagnostics. ``camera_center_x_px`` is the
    # calibrated optical principal point (CameraInfo.cx), not necessarily the
    # geometric middle of the image. ``lane_center_x_px`` is evaluated on the
    # same near-field row used by the lateral controller.
    camera_center_x_px: float = 0.0
    lane_center_x_px: Optional[float] = None
    lateral_measurement_y_px: Optional[float] = None
    candidates: list = field(default_factory=list)
    candidate_reject_reasons: list = field(default_factory=list)


class WhiteLaneDetector:
    """Detect both white lane boundaries using RGB and optional depth."""

    def __init__(self, config: LaneDetectorConfig | None = None):
        self.config = config or LaneDetectorConfig()
        self._locked_pair_geometry: Optional[np.ndarray] = None
        self._lane_anchor_geometry: Optional[np.ndarray] = None
        self._adaptive_value_threshold: Optional[float] = None
        self._last_valid_lateral_error = 0.0
        self._boundary_breach_frames = 0
        self._lane_identity_lost = False
        self._num7_lifecycle_enabled = False
        self._num7_mission_active = False
        self._mission_state = "LEGACY"
        self._lock_confirm_count = 0
        self._lock_confirm_history: list = []
        self._candidate_reject_reasons: list = []

    def reset(self) -> None:
        """Forget the selected lane so the next valid pair becomes the target."""

        self._locked_pair_geometry = None
        self._lane_anchor_geometry = None
        # A new Num7 mission is also a new exposure session.  Keeping the
        # previous mission's smoothed threshold can hide otherwise valid
        # paint for the first few frames after lighting or auto-exposure has
        # changed.
        self._adaptive_value_threshold = None
        self._last_valid_lateral_error = 0.0
        self._boundary_breach_frames = 0
        self._lane_identity_lost = False
        self._lock_confirm_count = 0
        self._lock_confirm_history.clear()

    def configure_num7_lifecycle(self, enabled: bool = True) -> None:
        """Enable the Num7 preview / initial-lock / locked state machine."""

        self._num7_lifecycle_enabled = bool(enabled)
        if self._num7_lifecycle_enabled:
            self._mission_state = "PREVIEW"
            self._lock_confirm_count = 0
            self._lock_confirm_history.clear()
        else:
            self._mission_state = "LEGACY"

    def set_num7_mission_active(self, active: bool) -> None:
        """Move between PREVIEW and an active Num7 mission.

        A rising edge clears every piece of formal lane identity and
        enters INITIAL_LOCK; a falling edge returns to PREVIEW and
        discards all mission identity so a later mission starts fresh.
        """

        if not self._num7_lifecycle_enabled:
            return
        if active and not self._num7_mission_active:
            self.reset()
            self._num7_mission_active = True
            self._mission_state = "INITIAL_LOCK"
            self._lock_confirm_count = 0
            self._lock_confirm_history.clear()
        elif not active and self._num7_mission_active:
            self._num7_mission_active = False
            self.reset()
            self._mission_state = "PREVIEW"

    def detect(
        self,
        rgb: np.ndarray,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> DetectionResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")

        height, width = rgb.shape[:2]
        if depth_m is not None and depth_m.shape != (height, width):
            raise ValueError("depth_m must match the RGB height and width")
        if intrinsics is not None and (
            intrinsics.width != width or intrinsics.height != height
        ):
            intrinsics = None
        camera_center_x = (
            float(intrinsics.cx)
            if intrinsics is not None
            else 0.5 * float(width)
        )

        self._candidate_reject_reasons = []

        roi_y = int(round(height * self.config.roi_top_ratio))
        roi_y = int(np.clip(roi_y, 0, height - 2))
        roi_height = height - roi_y
        lookahead_y = roi_y + int(round(roi_height * self.config.lookahead_ratio))
        # Evaluate lane width and lateral offset on a near-field row where a
        # D435i-class field of view still contains both 2.1 m boundaries. At
        # the literal bottom row one boundary is often outside the image; line
        # extrapolation there would falsely reject a clearly observed pair.
        bottom_y = roi_y + int(round(roi_height * 0.62))
        bottom_y = int(np.clip(bottom_y, lookahead_y + 1, height - 1))

        mask, value_threshold = self._build_white_mask(rgb, roi_y)

        if depth_m is not None:
            finite = np.isfinite(depth_m)
            measured = finite & (depth_m > 0.0)
            outside = measured & (
                (depth_m < self.config.min_depth_m)
                | (depth_m > self.config.max_depth_m)
            )
            mask[outside] = 0
            if not self.config.keep_invalid_depth:
                mask[~measured] = 0

        kernel_size = max(3, int(self.config.morphology_kernel) | 1)
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        close_height = max(kernel_size, int(self.config.vertical_close_kernel) | 1)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, close_height)
        )
        # Closing first reconnects white-line fragments produced by motion blur
        # and exposure flicker.  The smaller opening then removes isolated
        # highlights without erasing a distant, narrow boundary.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        candidates = self._extract_candidates(mask, roi_y, roi_height, width)
        pair = self._choose_pair(
            candidates,
            width,
            lookahead_y,
            bottom_y,
            depth_m,
            intrinsics,
        )
        pair_source = "two-lines"
        if pair is None:
            # A fast gait can split each physical boundary into several short
            # contours.  Search around the already locked geometry and refit
            # both boundaries from actual mask pixels.  This is still strict
            # two-line observation; no missing boundary is synthesized.
            pair = self._guided_pair_from_mask(
                mask,
                width,
                roi_y,
                roi_height,
                lookahead_y,
                bottom_y,
                depth_m,
                intrinsics,
            )
            if pair is not None:
                pair_source = "two-lines-guided"

        preview_mode = (
            self._num7_lifecycle_enabled and not self._num7_mission_active
        )
        reject_reasons = list(self._candidate_reject_reasons)

        if pair is not None:
            left, right = pair
            result = self._result_from_pair(
                left,
                right,
                width,
                height,
                roi_y,
                lookahead_y,
                bottom_y,
                mask,
                value_threshold,
                pair_source,
                focal_px=(
                    intrinsics.fx
                    if intrinsics is not None
                    else None
                ),
                principal_x_px=(
                    intrinsics.cx
                    if intrinsics is not None
                    else None
                ),
            )
            result.candidates = candidates
            result.candidate_reject_reasons = reject_reasons
            result.mode = "PREVIEW" if preview_mode else self._mission_state
            result.lock_confirm = self._lock_confirm_count
            result.lock_confirm_target = (
                self.config.initial_lock_confirm_frames
            )

            observation_rows = self._joint_observation_rows(
                left, right, lookahead_y, bottom_y
            )
            physical_width_m: Optional[float] = None
            physical_width_reliable = False
            if observation_rows is not None:
                far_y, near_y = observation_rows
                physical_width_m, physical_width_reliable = (
                    self._physical_pair_width(
                        left,
                        right,
                        depth_m,
                        intrinsics,
                        far_y,
                        near_y,
                    )
                )
            result.physical_lane_width_m = physical_width_m

            # Preview never mutates formal mission identity. It only reports
            # the mask, candidate geometry, and the currently recommended pair.
            if preview_mode:
                if (
                    abs(result.heading_error_rad)
                    > self.config.max_abs_heading_error_rad
                ):
                    result.valid = False
                    result.source = "none"
                return result

            if self._lane_identity_lost:
                result.valid = False
                result.source = "lane-identity-lost"
                result.boundary_risk = True
                result.mode = "IDENTITY_LOST"
                return result

            if self._mission_state == "INITIAL_LOCK":
                if observation_rows is None:
                    result.valid = False
                    result.source = "INITIAL_LOCK"
                    result.mode = "INITIAL_LOCK"
                    result.candidate_reject_reasons = reject_reasons
                    return result
                far_y, near_y = observation_rows
                initial_ok, reject_reason = self._initial_lock_candidate_ok(
                    left,
                    right,
                    width,
                    far_y,
                    near_y,
                    physical_width_m,
                    physical_width_reliable,
                    cx=(
                        intrinsics.cx
                        if intrinsics is not None
                        else None
                    ),
                )
                if not initial_ok:
                    self._lock_confirm_count = 0
                    self._lock_confirm_history.clear()
                    self._reject(reject_reason)
                    result.valid = False
                    result.source = "INITIAL_LOCK"
                    result.mode = "INITIAL_LOCK"
                    result.lock_confirm = 0
                    result.candidate_reject_reasons = list(
                        self._candidate_reject_reasons
                    )
                    return result

                geometry = self._pair_geometry_at(
                    left, right, width, far_y, near_y
                )
                if self._lock_confirm_count == 0:
                    self._lock_confirm_count = 1
                    self._lock_confirm_history = [geometry]
                else:
                    if self._initial_lock_consistent(geometry):
                        self._lock_confirm_count += 1
                        self._lock_confirm_history.append(geometry)
                    else:
                        self._lock_confirm_count = 1
                        self._lock_confirm_history = [geometry]
                        self._reject("initial-lock pair changed")

                result.lock_confirm = self._lock_confirm_count
                if (
                    self._lock_confirm_count
                    >= self.config.initial_lock_confirm_frames
                ):
                    robust_geometry = np.median(
                        np.asarray(self._lock_confirm_history),
                        axis=0,
                    )
                    self._lane_anchor_geometry = np.asarray(
                        robust_geometry, dtype=np.float64
                    )
                    self._locked_pair_geometry = np.asarray(
                        robust_geometry, dtype=np.float64
                    )
                    self._mission_state = "LOCKED"
                    self._lock_confirm_count = 0
                    self._lock_confirm_history.clear()
                    result.mode = "LOCKED"
                    result.valid = True
                    result.source = pair_source
                    result.lock_confirm = self.config.initial_lock_confirm_frames
                    return result

                result.valid = False
                result.source = "INITIAL_LOCK"
                result.mode = "INITIAL_LOCK"
                result.candidate_reject_reasons = list(
                    self._candidate_reject_reasons
                )
                return result

            # LOCKED Num7 tracking and legacy detector behaviour share the
            # same strict two-line safety checks below.
            if (
                abs(result.heading_error_rad)
                > self.config.max_abs_heading_error_rad
            ):
                result.valid = False
                result.source = "none"
                result.mode = self._mission_state
                return result
            if (
                abs(result.lateral_error)
                > self.config.max_abs_lateral_error
            ):
                self._record_boundary_breach()
                if self._lane_identity_lost:
                    result.valid = False
                    result.source = "lane-identity-lost"
                    result.boundary_risk = True
                    result.mode = "IDENTITY_LOST"
                    return result
                result.source = "boundary-warning"
                result.boundary_risk = True
            elif (
                abs(result.lateral_error)
                >= self.config.boundary_warning_lateral_error
            ):
                result.source = "boundary-warning"
                result.boundary_risk = True
            if (
                abs(result.lateral_error)
                <= self.config.boundary_recovery_lateral_error
            ):
                self._boundary_breach_frames = 0
            elif not result.boundary_risk:
                self._boundary_breach_frames = max(
                    0,
                    self._boundary_breach_frames - 1,
                )
            self._last_valid_lateral_error = result.lateral_error
            self._remember_pair(left, right, width, lookahead_y, bottom_y)
            return result

        # No pair was observed in this frame.
        if preview_mode:
            result = DetectionResult(
                valid=False,
                source="none",
                roi_y=roi_y,
                lookahead_y=lookahead_y,
                image_width=width,
                image_height=height,
                camera_center_x_px=camera_center_x,
                value_threshold=value_threshold,
                mask=mask,
            )
            result.mode = "PREVIEW"
            result.candidates = candidates
            result.candidate_reject_reasons = reject_reasons
            return result

        if (
            self._num7_lifecycle_enabled
            and self._mission_state == "INITIAL_LOCK"
        ):
            result = DetectionResult(
                valid=False,
                source="INITIAL_LOCK",
                roi_y=roi_y,
                lookahead_y=lookahead_y,
                image_width=width,
                image_height=height,
                camera_center_x_px=camera_center_x,
                value_threshold=value_threshold,
                mask=mask,
            )
            result.mode = "INITIAL_LOCK"
            result.lock_confirm = self._lock_confirm_count
            result.lock_confirm_target = (
                self.config.initial_lock_confirm_frames
            )
            result.candidates = candidates
            result.candidate_reject_reasons = reject_reasons
            return result

        if (
            abs(self._last_valid_lateral_error)
            >= self.config.boundary_warning_lateral_error
        ):
            self._record_boundary_breach()
        result = DetectionResult(
            valid=False,
            source=(
                "lane-identity-lost"
                if self._lane_identity_lost
                else "none"
            ),
            roi_y=roi_y,
            lookahead_y=lookahead_y,
            image_width=width,
            image_height=height,
            camera_center_x_px=camera_center_x,
            value_threshold=value_threshold,
            mask=mask,
        )
        result.mode = (
            "IDENTITY_LOST"
            if self._lane_identity_lost
            else (
                "TRACKING_LOST"
                if self._num7_lifecycle_enabled
                and self._mission_state == "LOCKED"
                else "LEGACY"
            )
        )
        result.candidates = candidates
        result.candidate_reject_reasons = reject_reasons
        return result

    def _record_boundary_breach(self) -> None:
        self._boundary_breach_frames += 1
        if (
            self._boundary_breach_frames
            >= self.config.boundary_breach_confirm_frames
        ):
            self._lane_identity_lost = True

    @staticmethod
    def _guarded_invalid_result(
        result: DetectionResult,
        source: str,
    ) -> DetectionResult:
        """Preserve observed boundaries for diagnostics but forbid steering."""

        return DetectionResult(
            valid=False,
            lateral_error=result.lateral_error,
            heading_error_rad=result.heading_error_rad,
            confidence=result.confidence,
            source=source,
            boundary_risk=True,
            left_line=result.left_line,
            right_line=result.right_line,
            roi_y=result.roi_y,
            lookahead_y=result.lookahead_y,
            image_width=result.image_width,
            image_height=result.image_height,
            value_threshold=result.value_threshold,
            mask=result.mask,
            mode=result.mode,
            lock_confirm=result.lock_confirm,
            lock_confirm_target=result.lock_confirm_target,
            physical_lane_width_m=result.physical_lane_width_m,
            camera_center_x_px=result.camera_center_x_px,
            lane_center_x_px=result.lane_center_x_px,
            lateral_measurement_y_px=result.lateral_measurement_y_px,
            candidates=result.candidates,
            candidate_reject_reasons=result.candidate_reject_reasons,
        )

    def _build_white_mask(
        self, rgb: np.ndarray, roi_y: int
    ) -> tuple[np.ndarray, float]:
        """Segment neutral bright paint under changing exposure and shadows."""

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        roi_value = value[roi_y:, :]

        if roi_value.size:
            otsu_threshold, _ = cv2.threshold(
                roi_value,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            raw_threshold = float(
                np.clip(
                    otsu_threshold + self.config.adaptive_value_margin,
                    self.config.adaptive_value_floor,
                    self.config.white_value_min,
                )
            )
        else:
            raw_threshold = float(self.config.white_value_min)

        if self._adaptive_value_threshold is None:
            self._adaptive_value_threshold = raw_threshold
        else:
            configured_alpha = float(
                np.clip(self.config.adaptive_value_smoothing, 0.0, 1.0)
            )
            # Adapt quickly when entering a shadow, but raise the threshold
            # more slowly so one bright frame cannot make the next frame fail.
            alpha = (
                max(configured_alpha, 0.70)
                if raw_threshold < self._adaptive_value_threshold
                else configured_alpha
            )
            self._adaptive_value_threshold += alpha * (
                raw_threshold - self._adaptive_value_threshold
            )
        value_threshold = float(self._adaptive_value_threshold)

        rgb_i16 = rgb.astype(np.int16)
        chroma = rgb_i16.max(axis=2) - rgb_i16.min(axis=2)
        neutral = (
            (saturation <= self.config.white_saturation_max)
            & (chroma <= self.config.max_neutral_chroma)
        )

        kernel_size = max(9, int(self.config.local_contrast_kernel) | 1)
        contrast_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        local_contrast = cv2.morphologyEx(
            value, cv2.MORPH_TOPHAT, contrast_kernel
        )
        absolute_white = value >= value_threshold
        local_value_floor = max(
            float(self.config.adaptive_value_floor),
            float(
                np.clip(self.config.local_value_ratio, 0.0, 1.0)
                * value_threshold
            ),
        )
        locally_bright = (
            (value >= local_value_floor)
            & (local_contrast >= self.config.local_contrast_min)
        )
        white = neutral & (absolute_white | locally_bright)
        white[:roi_y, :] = False

        mask = np.zeros(value.shape, dtype=np.uint8)
        mask[white] = 255
        return mask, value_threshold

    def _extract_candidates(
        self,
        mask: np.ndarray,
        roi_y: int,
        roi_height: int,
        width: int,
    ) -> list[LineModel]:
        contours, _ = cv2.findContours(
            mask[roi_y:, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        candidates: list[LineModel] = []
        min_span = roi_height * self.config.min_vertical_span_ratio

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_component_area:
                continue
            points = contour.reshape(-1, 2).astype(np.float64)
            points[:, 1] += roi_y
            y_min = float(points[:, 1].min())
            y_max = float(points[:, 1].max())
            span = y_max - y_min
            if span < min_span:
                continue

            fitted = self._fit_line(points, roi_height, y_min, y_max)
            if fitted is None:
                continue
            area_score = float(np.clip(area / (roi_height * 18.0), 0.0, 1.0))
            mean_thickness_px = area / max(span, 1.0)
            min_thickness_px = max(
                3.0,
                self.config.min_mean_thickness_ratio * float(width),
            )
            thickness_score = float(
                np.clip(
                    mean_thickness_px / (2.0 * min_thickness_px),
                    0.0,
                    1.0,
                )
            )
            score = (
                0.70 * fitted.score
                + 0.20 * area_score
                + 0.10 * thickness_score
            )
            candidates.append(
                LineModel(
                    slope=fitted.slope,
                    intercept=fitted.intercept,
                    score=score,
                    y_min=y_min,
                    y_max=y_max,
                    area=area,
                    mean_thickness_px=mean_thickness_px,
                )
            )

        return candidates

    def _choose_pair(
        self,
        candidates: list[LineModel],
        width: int,
        lookahead_y: int,
        bottom_y: int,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> Optional[tuple[LineModel, LineModel]]:
        best: Optional[tuple[LineModel, LineModel]] = None
        best_score = -np.inf
        min_width = width * self.config.min_lane_width_ratio
        max_width = width * self.config.max_lane_width_ratio
        expected = width * self.config.expected_lane_width_ratio
        image_center = (
            float(intrinsics.cx)
            if intrinsics is not None
            else 0.5 * float(width)
        )

        for index, first in enumerate(candidates):
            for second in candidates[index + 1 :]:
                left, right = sorted(
                    (first, second), key=lambda line: line.x_at(bottom_y)
                )
                observation_rows = self._joint_observation_rows(
                    left,
                    right,
                    lookahead_y,
                    bottom_y,
                )
                if observation_rows is None:
                    continue
                far_y, near_y = observation_rows

                # On the real D435i view, a genuine boundary often exits the
                # left or right edge before ``bottom_y``.  Measuring the pair
                # at that fixed row extrapolated the two fits outside the
                # image and rejected the correct lane as too wide.  Validate
                # width and centre at the nearest row where both paint
                # segments were actually observed.
                near_width = right.x_at(near_y) - left.x_at(near_y)
                far_width = right.x_at(far_y) - left.x_at(far_y)
                if not (min_width <= near_width <= max_width):
                    self._reject(
                        "pair width outside 2D range"
                    )
                    continue
                if far_width <= 0.04 * width:
                    continue
                if far_width > (
                    near_width * self.config.max_top_to_bottom_width_ratio
                ):
                    continue

                if (
                    self._num7_lifecycle_enabled
                    and self._mission_state == "INITIAL_LOCK"
                ):
                    physical_width_m, physical_width_reliable = (
                        self._physical_pair_width(
                            left,
                            right,
                            depth_m,
                            intrinsics,
                            far_y,
                            near_y,
                        )
                    )
                    ok, reason = self._initial_lock_candidate_ok(
                        left,
                        right,
                        width,
                        far_y,
                        near_y,
                        physical_width_m,
                        physical_width_reliable,
                        cx=(
                            intrinsics.cx
                            if intrinsics is not None
                            else None
                        ),
                    )
                    if not ok:
                        self._reject(reason)
                        continue

                width_match = np.exp(
                    -abs(near_width - expected) / max(expected, 1.0)
                )
                pair_center = 0.5 * (
                    left.x_at(near_y) + right.x_at(near_y)
                )
                if (
                    self._lane_anchor_geometry is None
                    and abs(pair_center - image_center)
                    > self.config.max_pair_center_offset_ratio * width
                ):
                    # A neighbouring lane can be fully visible, but it does
                    # not contain the camera's near-field ground ray. Never
                    # use such a pair for steering.
                    continue
                center_match = np.exp(
                    -abs(pair_center - image_center) / max(0.35 * width, 1.0)
                )
                temporal_match = 0.0
                if self._locked_pair_geometry is not None:
                    geometry = self._pair_geometry(
                        left, right, width, lookahead_y, bottom_y
                    )
                    if not self._matches_lane_anchor(geometry):
                        continue
                    previous = self._locked_pair_geometry
                    # Near-field boundary positions remain stable under the
                    # large torso pitch oscillation of the running gait. Far
                    # positions do not, so they must not be a hard identity
                    # gate. Switching to an adjacent lane still replaces one
                    # near boundary and produces roughly a full-lane jump.
                    near_delta = geometry[:2] - previous[:2]
                    common_shift = float(abs(np.mean(near_delta)))
                    deformation = float(
                        np.max(np.abs(near_delta - np.mean(near_delta)))
                    )
                    previous_width = previous[1] - previous[0]
                    width_change = abs(
                        (geometry[1] - geometry[0]) - previous_width
                    )
                    # Roll/yaw shake moves both boundaries together.  Treat
                    # that common image translation separately from a shape
                    # change, while retaining a tighter absolute limit that
                    # rejects a complete jump into the neighbouring lane.
                    if deformation > self.config.max_boundary_step_ratio:
                        continue
                    if common_shift > self.config.max_common_shift_ratio:
                        continue
                    if width_change > (
                        self.config.max_lane_width_change_ratio
                        * max(previous_width, 1e-6)
                    ):
                        continue
                    temporal_distance = max(deformation, 0.65 * common_shift)
                    temporal_match = float(np.exp(-temporal_distance / 0.10))

                thickness_penalty = self._thickness_score_penalty(
                    left, right, width
                )
                score = (
                    left.score
                    + right.score
                    + 0.35 * float(width_match)
                    + self.config.initial_center_weight * float(center_match)
                    + self.config.temporal_pair_weight * temporal_match
                    - thickness_penalty
                )
                if score > best_score:
                    best_score = score
                    best = (left, right)
        return best

    @staticmethod
    def _joint_observation_rows(
        left: LineModel,
        right: LineModel,
        lookahead_y: int,
        bottom_y: int,
    ) -> Optional[tuple[float, float]]:
        """Return far/near rows supported by both observed paint segments."""

        far_y = max(float(lookahead_y), left.y_min, right.y_min)
        near_y = min(float(bottom_y), left.y_max, right.y_max)
        expected_span = max(float(bottom_y - lookahead_y), 1.0)
        # Twenty percent still supplies multiple fitted cross-sections at
        # 640x480, while tolerating the common case where one thick boundary
        # reaches the image edge shortly below the lookahead row.
        minimum_overlap = max(8.0, 0.20 * expected_span)
        if near_y - far_y < minimum_overlap:
            return None
        return far_y, near_y

    def _reject(self, reason: str) -> None:
        """Record a pair/candidate rejection reason for the debug HUD."""

        if len(self._candidate_reject_reasons) < 20:
            self._candidate_reject_reasons.append(reason)

    def _thickness_score_penalty(
        self,
        left: LineModel,
        right: LineModel,
        width: int,
    ) -> float:
        """Penalise thin floor-seam candidates without killing distant paint."""

        min_thickness = max(
            3.0,
            self.config.min_mean_thickness_ratio * float(width),
        )
        penalty = 0.0
        for line in (left, right):
            ratio = line.mean_thickness_px / max(min_thickness, 1.0)
            if ratio < 1.0:
                # A seam that is half the expected tape thickness costs
                # substantially more than one that is only slightly thin.
                penalty += 0.30 * float(np.clip(1.0 - ratio, 0.0, 1.0))
        return float(penalty)

    @staticmethod
    def _clip_x(x: float, width: int) -> float:
        return float(np.clip(x, 0.0, float(width)))

    def _pair_geometry_at(
        self,
        left: LineModel,
        right: LineModel,
        width: int,
        far_y: float,
        near_y: float,
    ) -> np.ndarray:
        """Normalized four-value geometry using robust near/far rows."""

        return np.asarray(
            (
                self._clip_x(left.x_at(near_y), width) / width,
                self._clip_x(right.x_at(near_y), width) / width,
                self._clip_x(left.x_at(far_y), width) / width,
                self._clip_x(right.x_at(far_y), width) / width,
            ),
            dtype=np.float64,
        )

    def _initial_lock_candidate_ok(
        self,
        left: LineModel,
        right: LineModel,
        width: int,
        far_y: float,
        near_y: float,
        physical_width_m: Optional[float],
        physical_width_reliable: bool,
        cx: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Return whether a candidate pair may enter initial-lock confirmation."""

        left_near = self._clip_x(left.x_at(near_y), width)
        right_near = self._clip_x(right.x_at(near_y), width)
        image_center = float(cx) if cx is not None else 0.5 * width
        margin = self.config.initial_min_center_margin_ratio * width

        # Initial lock must bound the camera near-field ray on both sides.
        if left_near >= image_center - margin:
            return False, "initial center: left not left of cx"
        if right_near <= image_center + margin:
            return False, "initial center: right not right of cx"

        # Weak perspective gate for straight tracks: the left line must run
        # toward the left as y grows, and the right line toward the right.
        slope_gate = self.config.initial_min_abs_perspective_slope
        if left.slope >= -slope_gate:
            return False, "initial slope: left not negative"
        if right.slope <= slope_gate:
            return False, "initial slope: right not positive"

        min_thickness = max(
            3.0,
            self.config.min_mean_thickness_ratio * float(width),
        )
        if left.mean_thickness_px < min_thickness:
            return False, "initial thickness: left too thin"
        if right.mean_thickness_px < min_thickness:
            return False, "initial thickness: right too thin"

        # A pair that is already in the boundary-warning band must not create
        # a permanent identity; the operator has to reposition and re-enter.
        pair_center = 0.5 * (left_near + right_near)
        lateral_error = (pair_center - image_center) / (0.5 * width)
        if (
            abs(lateral_error)
            >= self.config.boundary_warning_lateral_error
        ):
            return False, "initial lock: boundary-warning rejected"

        if physical_width_reliable:
            if physical_width_m is None:
                return False, "physical width invalid"
            if not (
                self.config.min_lane_width_m
                <= physical_width_m
                <= self.config.max_lane_width_m
            ):
                return False, (
                    f"physical width {physical_width_m:.2f}m outside "
                    f"[{self.config.min_lane_width_m:.2f},"
                    f"{self.config.max_lane_width_m:.2f}]"
                )
        return True, ""

    def _initial_lock_consistent(self, geometry: np.ndarray) -> bool:
        """True when the candidate pair is consistent with recent confirmations."""

        if not self._lock_confirm_history:
            return True
        previous = np.median(
            np.asarray(self._lock_confirm_history), axis=0
        )
        near_delta = geometry[:2] - previous[:2]
        common_shift = float(abs(np.mean(near_delta)))
        deformation = float(np.max(np.abs(near_delta - np.mean(near_delta))))
        previous_width = float(previous[1] - previous[0])
        width_change = float(
            abs((geometry[1] - geometry[0]) - previous_width)
        )
        return (
            common_shift <= self.config.initial_lock_confirm_step_ratio
            and deformation <= self.config.initial_lock_confirm_step_ratio
            and width_change
            <= self.config.initial_lock_confirm_width_ratio
            * max(previous_width, 1e-6)
        )

    def _physical_pair_width(
        self,
        left: LineModel,
        right: LineModel,
        depth_m: Optional[np.ndarray],
        intrinsics: Optional[CameraIntrinsics],
        far_y: float,
        near_y: float,
    ) -> tuple[Optional[float], bool]:
        """Estimate physical boundary separation from aligned depth.

        Returns ``(width_m, reliable)``. When depth or intrinsics are not
        available, or too few samples are valid, ``reliable`` is False and the
        caller falls back to the existing 2D geometry path.
        """

        if depth_m is None or intrinsics is None:
            return None, False
        sample_rows = int(self.config.physical_width_sample_rows)
        if sample_rows < 2:
            return None, False
        ys = np.linspace(far_y, near_y, sample_rows, dtype=np.float64)
        widths: list[float] = []
        for y in ys:
            yi = int(round(float(y)))
            if yi < 0 or yi >= depth_m.shape[0]:
                continue
            xl = int(round(self._clip_x(left.x_at(float(y)), depth_m.shape[1])))
            xr = int(round(self._clip_x(right.x_at(float(y)), depth_m.shape[1])))
            zl = median_depth_in_neighbourhood(
                depth_m,
                xl,
                yi,
                radius=self.config.physical_depth_kernel_radius,
                min_valid_depth_m=self.config.min_depth_m,
                max_valid_depth_m=self.config.max_depth_m,
                min_valid_pixels=3,
            )
            zr = median_depth_in_neighbourhood(
                depth_m,
                xr,
                yi,
                radius=self.config.physical_depth_kernel_radius,
                min_valid_depth_m=self.config.min_depth_m,
                max_valid_depth_m=self.config.max_depth_m,
                min_valid_pixels=3,
            )
            if zl is None or zr is None:
                continue
            try:
                left_3d = intrinsics.deproject(
                    np.asarray(((xl, yi),)), np.asarray((zl,))
                )[0]
                right_3d = intrinsics.deproject(
                    np.asarray(((xr, yi),)), np.asarray((zr,))
                )[0]
            except (ValueError, IndexError):
                continue
            widths.append(float(np.linalg.norm(left_3d - right_3d)))
        if len(widths) < self.config.physical_width_min_valid_samples:
            return None, False
        return float(np.median(widths)), True

    def _guided_pair_from_mask(
        self,
        mask: np.ndarray,
        width: int,
        roi_y: int,
        roi_height: int,
        lookahead_y: int,
        bottom_y: int,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> Optional[tuple[LineModel, LineModel]]:
        """Refit two observed fragmented boundaries near the locked lane."""

        if self._locked_pair_geometry is None:
            return None

        y_pixels, x_pixels = np.nonzero(mask[roi_y:, :])
        if y_pixels.size == 0:
            return None
        y_pixels = y_pixels.astype(np.float64) + roi_y
        x_pixels = x_pixels.astype(np.float64)
        geometry = self._locked_pair_geometry
        margin = max(4.0, width * self.config.guided_search_margin_ratio)
        min_span = roi_height * self.config.guided_min_vertical_span_ratio

        fitted: list[LineModel] = []
        for top_x_ratio, bottom_x_ratio in (
            (geometry[2], geometry[0]),
            (geometry[3], geometry[1]),
        ):
            slope = (
                width * (bottom_x_ratio - top_x_ratio)
                / max(float(bottom_y - lookahead_y), 1.0)
            )
            predicted_x = width * top_x_ratio + slope * (
                y_pixels - lookahead_y
            )
            selected = np.abs(x_pixels - predicted_x) <= margin
            if int(np.count_nonzero(selected)) < self.config.guided_min_pixels:
                return None

            points = np.column_stack((x_pixels[selected], y_pixels[selected]))
            y_min = float(points[:, 1].min())
            y_max = float(points[:, 1].max())
            if y_max - y_min < min_span:
                return None

            line = self._fit_line(points, roi_height, y_min, y_max)
            if line is None:
                return None
            fitted.append(line)

        return self._choose_pair(
            fitted, width, lookahead_y, bottom_y, depth_m, intrinsics
        )

    @staticmethod
    def _fit_line(
        points: np.ndarray,
        roi_height: int,
        y_min: float,
        y_max: float,
    ) -> Optional[LineModel]:
        """Fit x(y) with a Huber loss so glare and blur tails have low weight."""

        if len(points) < 2:
            return None
        vx, vy, x0, y0 = cv2.fitLine(
            points.astype(np.float32).reshape(-1, 1, 2),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        if abs(float(vy)) < 1e-5:
            return None
        slope = float(vx / vy)
        intercept = float(x0 - slope * y0)
        residual = points[:, 0] - (slope * points[:, 1] + intercept)
        median_error = float(np.median(np.abs(residual)))
        span = y_max - y_min
        span_score = float(np.clip(span / max(roi_height, 1), 0.0, 1.0))
        support_score = float(
            np.clip(len(points) / max(roi_height * 10.0, 1.0), 0.0, 1.0)
        )
        straightness = float(np.exp(-median_error / 18.0))
        score = 0.50 * span_score + 0.30 * support_score + 0.20 * straightness
        return LineModel(
            slope=slope,
            intercept=intercept,
            score=score,
            y_min=y_min,
            y_max=y_max,
        )

    @staticmethod
    def _pair_geometry(
        left: LineModel,
        right: LineModel,
        width: int,
        lookahead_y: int,
        bottom_y: int,
    ) -> np.ndarray:
        """Return normalized boundary positions at near and far image rows."""

        return np.asarray(
            (
                WhiteLaneDetector._clip_x(
                    left.x_at(bottom_y), width
                ) / width,
                WhiteLaneDetector._clip_x(
                    right.x_at(bottom_y), width
                ) / width,
                WhiteLaneDetector._clip_x(
                    left.x_at(lookahead_y), width
                ) / width,
                WhiteLaneDetector._clip_x(
                    right.x_at(lookahead_y), width
                ) / width,
            ),
            dtype=np.float64,
        )

    def _remember_pair(
        self,
        left: LineModel,
        right: LineModel,
        width: int,
        lookahead_y: int,
        bottom_y: int,
    ) -> None:
        geometry = self._pair_geometry(
            left, right, width, lookahead_y, bottom_y
        )
        if self._lane_anchor_geometry is None:
            self._lane_anchor_geometry = geometry.copy()
        if self._locked_pair_geometry is None:
            self._locked_pair_geometry = geometry
            return
        alpha = self.config.lock_update_alpha
        self._locked_pair_geometry = (
            alpha * geometry + (1.0 - alpha) * self._locked_pair_geometry
        )

    def _matches_lane_anchor(self, geometry: np.ndarray) -> bool:
        """Reject every pair that cannot be the start-line target lane."""

        if self._lane_anchor_geometry is None:
            return True
        anchor = self._lane_anchor_geometry
        near_delta = geometry[:2] - anchor[:2]
        common_shift = float(abs(np.mean(near_delta)))
        deformation = float(
            np.max(np.abs(near_delta - np.mean(near_delta)))
        )
        anchor_width = float(anchor[1] - anchor[0])
        width_change = float(
            abs((geometry[1] - geometry[0]) - anchor_width)
        )
        return (
            common_shift <= self.config.max_anchor_common_shift_ratio
            and deformation
            <= self.config.max_anchor_boundary_deformation_ratio
            and width_change
            <= self.config.max_anchor_lane_width_change_ratio
            * max(anchor_width, 1e-6)
        )

    def _result_from_pair(
        self,
        left: LineModel,
        right: LineModel,
        width: int,
        height: int,
        roi_y: int,
        lookahead_y: int,
        bottom_y: int,
        mask: np.ndarray,
        value_threshold: float,
        source: str,
        focal_px: Optional[float] = None,
        principal_x_px: Optional[float] = None,
    ) -> DetectionResult:
        left_bottom = left.x_at(bottom_y)
        right_bottom = right.x_at(bottom_y)
        left_top = left.x_at(lookahead_y)
        right_top = right.x_at(lookahead_y)
        center_bottom = 0.5 * (left_bottom + right_bottom)
        center_top = 0.5 * (left_top + right_top)
        optical_center_x = (
            float(principal_x_px)
            if principal_x_px is not None
            else 0.5 * float(width)
        )
        lateral_error = (center_bottom - optical_center_x) / (0.5 * width)
        # The slope of the image-space lane center mixes lateral displacement
        # with yaw. The vanishing point of the two boundaries is independent of
        # lateral displacement on a straight track, so it is a much cleaner
        # estimate of camera/track heading.
        slope_delta = left.slope - right.slope
        if abs(slope_delta) > 1e-6:
            vanishing_y = (right.intercept - left.intercept) / slope_delta
            vanishing_x = left.x_at(vanishing_y)
            if focal_px is None:
                focal_px = (0.5 * height) / np.tan(
                    np.radians(0.5 * self.config.vertical_fov_degrees)
                )
            heading_error = np.arctan2(
                vanishing_x - optical_center_x,
                focal_px,
            )
        else:
            heading_error = np.arctan2(
                center_top - center_bottom, bottom_y - lookahead_y
            )
        confidence = float(np.clip(0.5 * (left.score + right.score), 0.0, 1.0))
        return DetectionResult(
            valid=True,
            lateral_error=float(np.clip(lateral_error, -2.0, 2.0)),
            heading_error_rad=float(heading_error),
            confidence=confidence,
            source=source,
            left_line=left,
            right_line=right,
            roi_y=roi_y,
            lookahead_y=lookahead_y,
            image_width=width,
            image_height=height,
            camera_center_x_px=optical_center_x,
            lane_center_x_px=float(center_bottom),
            lateral_measurement_y_px=float(bottom_y),
            value_threshold=value_threshold,
            mask=mask,
        )

    @staticmethod
    def annotate(
        rgb: np.ndarray,
        result: DetectionResult,
        debug_candidates: bool = False,
    ) -> np.ndarray:
        canvas = rgb.copy()
        height, width = canvas.shape[:2]
        if result.roi_y:
            cv2.line(canvas, (0, result.roi_y), (width - 1, result.roi_y), (80, 80, 255), 1)

        if debug_candidates and result.candidates:
            palette = (
                (180, 180, 180),
                (200, 160, 40),
                (40, 200, 160),
                (200, 120, 200),
                (200, 200, 80),
                (120, 200, 220),
            )
            for index, candidate in enumerate(result.candidates[:6]):
                y1 = max(result.roi_y, int(round(candidate.y_min)))
                y2 = min(height - 1, int(round(candidate.y_max)))
                if y2 <= y1:
                    y2 = y1 + 1
                color = palette[index % len(palette)]
                cv2.line(
                    canvas,
                    (int(round(candidate.x_at(y1))), y1),
                    (int(round(candidate.x_at(y2))), y2),
                    color,
                    2,
                    cv2.LINE_AA,
                )
                label = (
                    f"C{index} s={candidate.score:.2f} "
                    f"slope={candidate.slope:+.2f} "
                    f"area={candidate.area:.0f} "
                    f"span={candidate.y_max - candidate.y_min:.0f} "
                    f"th={candidate.mean_thickness_px:.1f} "
                    f"near={candidate.x_at(result.lookahead_y):.0f} "
                    f"far={candidate.x_at(max(0, result.roi_y)):.0f}"
                )
                cv2.putText(
                    canvas,
                    label,
                    (8, height - 14 - index * 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        def draw_line(
            line: Optional[LineModel],
            color: tuple[int, int, int],
            label: str,
        ) -> None:
            if line is None:
                return
            y1 = max(result.roi_y, int(round(line.y_min)))
            y2 = min(height - 1, int(round(line.y_max)))
            p1 = (int(round(line.x_at(y1))), y1)
            p2 = (int(round(line.x_at(y2))), y2)
            cv2.line(canvas, p1, p2, color, 5, cv2.LINE_AA)
            text_x = int(np.clip(p1[0] + 8, 4, max(4, width - 150)))
            text_y = int(np.clip(p1[1] + 22, 20, height - 5))
            cv2.putText(
                canvas,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

        draw_line(
            result.left_line,
            (255, 140, 40),
            "LEFT",
        )
        draw_line(
            result.right_line,
            (40, 230, 80),
            "RIGHT",
        )

        if result.valid and result.left_line and result.right_line:
            y_bottom = height - 1
            y_top = result.lookahead_y
            center_bottom = int(
                round(
                    0.5
                    * (
                        result.left_line.x_at(y_bottom)
                        + result.right_line.x_at(y_bottom)
                    )
                )
            )
            center_top = int(
                round(
                    0.5
                    * (
                        result.left_line.x_at(y_top)
                        + result.right_line.x_at(y_top)
                    )
                )
            )
            cv2.line(
                canvas,
                (center_bottom, y_bottom),
                (center_top, y_top),
                (255, 220, 20),
                3,
                cv2.LINE_AA,
            )
            camera_center_x = int(
                round(
                    result.camera_center_x_px
                    if result.camera_center_x_px > 0.0
                    else 0.5 * width
                )
            )
            cv2.line(
                canvas,
                (camera_center_x, height - 1),
                (camera_center_x, result.roi_y),
                (220, 80, 220),
                2,
                cv2.LINE_AA,
            )
            measurement_y = int(
                round(
                    result.lateral_measurement_y_px
                    if result.lateral_measurement_y_px is not None
                    else height - 1
                )
            )
            lane_center_x = int(
                round(
                    result.lane_center_x_px
                    if result.lane_center_x_px is not None
                    else center_bottom
                )
            )
            cv2.line(
                canvas,
                (camera_center_x, measurement_y),
                (lane_center_x, measurement_y),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                (camera_center_x, measurement_y),
                6,
                (220, 80, 220),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                (lane_center_x, measurement_y),
                6,
                (255, 220, 20),
                -1,
                cv2.LINE_AA,
            )

        physical = (
            f"{result.physical_lane_width_m:.2f}m"
            if result.physical_lane_width_m is not None
            else "N/A"
        )
        status = (
            f"{'VALID' if result.valid else 'LOST'} {result.source} "
            f"mode={result.mode} "
            f"lock={result.lock_confirm}/"
            f"{result.lock_confirm_target or '--'}"
        )
        errors = (
            f"center_err={result.lateral_error:+.3f} "
            f"heading={np.degrees(result.heading_error_rad):+.2f}deg "
            f"phys_width={physical}"
        )
        cv2.rectangle(canvas, (8, 8), (620, 64), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            status,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 255, 80) if result.valid else (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            errors,
            (16, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if result.candidate_reject_reasons:
            for offset, reason in enumerate(
                result.candidate_reject_reasons[-4:]
            ):
                cv2.putText(
                    canvas,
                    reason[:72],
                    (8, height - 72 - offset * 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (80, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
        return canvas
