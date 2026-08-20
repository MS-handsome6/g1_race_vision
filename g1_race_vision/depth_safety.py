from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .controller import ControllerCommand


@dataclass(frozen=True)
class DepthSafetyConfig:
    """Forward depth checks with an opt-in 100 m passing policy.

    ``allow_lane_bypass`` must only be enabled for the 100 m event.  The
    organiser permits leaving the starting lane there, but a pass is still
    authorised only when the target-side depth corridor is clear.
    """

    roi_left_ratio: float = 0.35
    roi_right_ratio: float = 0.65
    roi_top_ratio: float = 0.30
    roi_bottom_ratio: float = 0.62
    min_valid_depth_m: float = 0.25
    max_valid_depth_m: float = 12.0
    min_valid_fraction: float = 0.10
    obstacle_percentile: float = 10.0
    stop_distance_m: float = 0.80
    slow_distance_m: float = 1.50
    max_depth_age_s: float = 0.20
    allow_lane_bypass: bool = False
    side_roi_edge_margin_ratio: float = 0.05
    side_clearance_margin_m: float = 0.35
    avoidance_yaw_rps: float = 0.24
    minimum_avoidance_speed_scale: float = 0.25
    side_clear_confirm_frames: int = 3
    avoidance_clear_confirm_frames: int = 6

    def __post_init__(self) -> None:
        if not 0.0 <= self.roi_left_ratio < self.roi_right_ratio <= 1.0:
            raise ValueError("invalid horizontal depth ROI")
        if not 0.0 <= self.roi_top_ratio < self.roi_bottom_ratio <= 1.0:
            raise ValueError("invalid vertical depth ROI")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        if not 0.0 <= self.obstacle_percentile <= 100.0:
            raise ValueError("obstacle_percentile must be in [0, 100]")
        if not 0.0 < self.stop_distance_m < self.slow_distance_m:
            raise ValueError("stop distance must be below slow distance")
        if self.max_depth_age_s <= 0.0:
            raise ValueError("max_depth_age_s must be positive")
        if not 0.0 <= self.side_roi_edge_margin_ratio < 0.25:
            raise ValueError("side_roi_edge_margin_ratio must be in [0, 0.25)")
        if self.side_clearance_margin_m < 0.0:
            raise ValueError("side_clearance_margin_m must be non-negative")
        if self.avoidance_yaw_rps <= 0.0:
            raise ValueError("avoidance_yaw_rps must be positive")
        if not 0.0 < self.minimum_avoidance_speed_scale <= 1.0:
            raise ValueError(
                "minimum_avoidance_speed_scale must be in (0, 1]"
            )
        if self.side_clear_confirm_frames < 1:
            raise ValueError("side_clear_confirm_frames must be positive")
        if self.avoidance_clear_confirm_frames < 1:
            raise ValueError("avoidance_clear_confirm_frames must be positive")


@dataclass(frozen=True)
class DepthSafetyResult:
    motion_allowed: bool
    speed_scale: float
    obstacle_distance_m: float
    valid_fraction: float
    depth_age_s: float
    reason: str
    # Positive yaw turns left, negative yaw turns right.  A non-zero value is
    # emitted only after that side corridor has passed the clearance check.
    avoidance_yaw_rps: float = 0.0
    left_clearance_m: float = math.nan
    right_clearance_m: float = math.nan


class DepthSafetyGate:
    """Fail-safe depth watchdog and forward obstacle speed limiter."""

    def __init__(self, config: DepthSafetyConfig | None = None) -> None:
        self.config = config or DepthSafetyConfig()
        self._avoidance_direction = 0
        self._avoidance_clear_frames = 0
        self._candidate_direction = 0
        self._candidate_clear_frames = 0

    @property
    def avoidance_active(self) -> bool:
        return self._avoidance_direction != 0

    def reset(self) -> None:
        self._avoidance_direction = 0
        self._avoidance_clear_frames = 0
        self._candidate_direction = 0
        self._candidate_clear_frames = 0

    def evaluate(
        self,
        depth_m: np.ndarray | None,
        depth_age_s: float,
        *,
        allow_bypass: bool | None = None,
    ) -> DepthSafetyResult:
        bypass_enabled = (
            self.config.allow_lane_bypass
            if allow_bypass is None
            else bool(allow_bypass)
        )
        age = max(0.0, float(depth_age_s))
        if depth_m is None:
            self.reset()
            return self._stop("DEPTH_MISSING", age)
        if age > self.config.max_depth_age_s:
            self.reset()
            return self._stop("DEPTH_STALE", age)
        if depth_m.ndim != 2 or depth_m.size == 0:
            self.reset()
            return self._stop("DEPTH_INVALID_SHAPE", age)

        height, width = depth_m.shape
        x0 = int(round(width * self.config.roi_left_ratio))
        x1 = int(round(width * self.config.roi_right_ratio))
        y0 = int(round(height * self.config.roi_top_ratio))
        y1 = int(round(height * self.config.roi_bottom_ratio))
        x0 = int(np.clip(x0, 0, max(0, width - 1)))
        x1 = int(np.clip(x1, x0 + 1, width))
        y0 = int(np.clip(y0, 0, max(0, height - 1)))
        y1 = int(np.clip(y1, y0 + 1, height))
        roi = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float32)

        valid = (
            np.isfinite(roi)
            & (roi >= self.config.min_valid_depth_m)
            & (roi <= self.config.max_valid_depth_m)
        )
        valid_fraction = float(np.count_nonzero(valid) / max(1, roi.size))
        if valid_fraction < self.config.min_valid_fraction:
            self.reset()
            return self._stop("DEPTH_INSUFFICIENT", age, valid_fraction)

        obstacle_distance = float(
            np.percentile(roi[valid], self.config.obstacle_percentile)
        )
        if not math.isfinite(obstacle_distance):
            self.reset()
            return self._stop("DEPTH_NONFINITE", age, valid_fraction)
        if bypass_enabled:
            bypass = self._evaluate_lane_bypass(
                depth_m,
                obstacle_distance,
                valid_fraction,
                age,
                y0,
                y1,
                x0,
                x1,
            )
            if bypass is not None:
                return bypass
        else:
            self.reset()
        if obstacle_distance <= self.config.stop_distance_m:
            return DepthSafetyResult(
                motion_allowed=False,
                speed_scale=0.0,
                obstacle_distance_m=obstacle_distance,
                valid_fraction=valid_fraction,
                depth_age_s=age,
                reason="OBSTACLE_STOP",
            )
        if obstacle_distance < self.config.slow_distance_m:
            scale = (
                obstacle_distance - self.config.stop_distance_m
            ) / (
                self.config.slow_distance_m - self.config.stop_distance_m
            )
            return DepthSafetyResult(
                motion_allowed=True,
                speed_scale=float(np.clip(scale, 0.0, 1.0)),
                obstacle_distance_m=obstacle_distance,
                valid_fraction=valid_fraction,
                depth_age_s=age,
                reason="OBSTACLE_SLOW",
            )
        return DepthSafetyResult(
            motion_allowed=True,
            speed_scale=1.0,
            obstacle_distance_m=obstacle_distance,
            valid_fraction=valid_fraction,
            depth_age_s=age,
            reason="CLEAR",
        )

    def _evaluate_lane_bypass(
        self,
        depth_m: np.ndarray,
        obstacle_distance: float,
        valid_fraction: float,
        age: float,
        y0: int,
        y1: int,
        centre_x0: int,
        centre_x1: int,
    ) -> DepthSafetyResult | None:
        """Select a clear adjacent corridor without jumping into traffic."""

        width = depth_m.shape[1]
        edge = int(round(width * self.config.side_roi_edge_margin_ratio))
        left_clearance = self._corridor_clearance(
            depth_m[y0:y1, edge:centre_x0]
        )
        right_clearance = self._corridor_clearance(
            depth_m[y0:y1, centre_x1:max(centre_x1 + 1, width - edge)]
        )
        required_clearance = (
            self.config.slow_distance_m
            + self.config.side_clearance_margin_m
        )

        # Keep the selected side stable while passing.  Switching left/right
        # from frame noise is more dangerous than yielding and trying again.
        if self._avoidance_direction != 0:
            selected_clearance = (
                left_clearance
                if self._avoidance_direction > 0
                else right_clearance
            )
            if not (
                math.isfinite(selected_clearance)
                and selected_clearance >= required_clearance
            ):
                self.reset()
                return self._soft_yield(
                    obstacle_distance,
                    valid_fraction,
                    age,
                    left_clearance,
                    right_clearance,
                )

            if obstacle_distance >= self.config.slow_distance_m:
                self._avoidance_clear_frames += 1
                if (
                    self._avoidance_clear_frames
                    >= self.config.avoidance_clear_confirm_frames
                ):
                    self.reset()
                    return None
            else:
                self._avoidance_clear_frames = 0
            return self._bypass_result(
                obstacle_distance,
                valid_fraction,
                age,
                left_clearance,
                right_clearance,
            )

        if obstacle_distance >= self.config.slow_distance_m:
            self._reset_candidate()
            return None

        candidates: list[tuple[float, int]] = []
        if (
            math.isfinite(left_clearance)
            and left_clearance >= required_clearance
        ):
            candidates.append((left_clearance, 1))
        if (
            math.isfinite(right_clearance)
            and right_clearance >= required_clearance
        ):
            candidates.append((right_clearance, -1))

        if candidates:
            # Do not change lanes based on a single depth frame: aligned depth
            # can contain transient holes around a moving robot. Confirm that
            # the same target side stays clear for several consecutive frames.
            _, candidate_direction = max(candidates)
            if candidate_direction != self._candidate_direction:
                self._candidate_direction = candidate_direction
                self._candidate_clear_frames = 1
            else:
                self._candidate_clear_frames += 1
            if (
                self._candidate_clear_frames
                < self.config.side_clear_confirm_frames
            ):
                return self._soft_yield(
                    obstacle_distance,
                    valid_fraction,
                    age,
                    left_clearance,
                    right_clearance,
                    reason="OBSTACLE_PASS_CHECK",
                )
            self._avoidance_direction = candidate_direction
            self._avoidance_clear_frames = 0
            self._reset_candidate()
            return self._bypass_result(
                obstacle_distance,
                valid_fraction,
                age,
                left_clearance,
                right_clearance,
            )

        self._reset_candidate()
        if obstacle_distance <= self.config.stop_distance_m:
            # No legal pass exists.  Yield with an ordinary zero-velocity
            # command, not hard_stop, so the gait controller can settle rather
            # than performing the old abrupt emergency stop.  Sensor/watchdog
            # failures and the operator E-stop remain hard-stop conditions.
            return self._soft_yield(
                obstacle_distance,
                valid_fraction,
                age,
                left_clearance,
                right_clearance,
            )
        return None

    def _corridor_clearance(self, roi: np.ndarray) -> float:
        if roi.size == 0:
            return math.nan
        values = np.asarray(roi, dtype=np.float32)
        valid = (
            np.isfinite(values)
            & (values >= self.config.min_valid_depth_m)
            & (values <= self.config.max_valid_depth_m)
        )
        valid_fraction = float(np.count_nonzero(valid) / values.size)
        if valid_fraction < self.config.min_valid_fraction:
            return math.nan
        return float(
            np.percentile(values[valid], self.config.obstacle_percentile)
        )

    def _reset_candidate(self) -> None:
        self._candidate_direction = 0
        self._candidate_clear_frames = 0

    def _bypass_result(
        self,
        obstacle_distance: float,
        valid_fraction: float,
        age: float,
        left_clearance: float,
        right_clearance: float,
    ) -> DepthSafetyResult:
        progress = float(
            np.clip(
                (
                    obstacle_distance - self.config.stop_distance_m
                )
                / (
                    self.config.slow_distance_m
                    - self.config.stop_distance_m
                ),
                0.0,
                1.0,
            )
        )
        speed_scale = max(
            self.config.minimum_avoidance_speed_scale,
            progress,
        )
        direction_name = "LEFT" if self._avoidance_direction > 0 else "RIGHT"
        return DepthSafetyResult(
            motion_allowed=True,
            speed_scale=speed_scale,
            obstacle_distance_m=obstacle_distance,
            valid_fraction=valid_fraction,
            depth_age_s=age,
            reason=f"OBSTACLE_BYPASS_{direction_name}",
            avoidance_yaw_rps=(
                self._avoidance_direction * self.config.avoidance_yaw_rps
            ),
            left_clearance_m=left_clearance,
            right_clearance_m=right_clearance,
        )

    @staticmethod
    def _soft_yield(
        obstacle_distance: float,
        valid_fraction: float,
        age: float,
        left_clearance: float,
        right_clearance: float,
        reason: str = "OBSTACLE_YIELD",
    ) -> DepthSafetyResult:
        return DepthSafetyResult(
            motion_allowed=True,
            speed_scale=0.0,
            obstacle_distance_m=obstacle_distance,
            valid_fraction=valid_fraction,
            depth_age_s=age,
            reason=reason,
            left_clearance_m=left_clearance,
            right_clearance_m=right_clearance,
        )

    def apply(
        self,
        command: ControllerCommand,
        result: DepthSafetyResult,
    ) -> ControllerCommand:
        if not result.motion_allowed:
            return ControllerCommand(
                0.0,
                0.0,
                0.0,
                result.reason,
                False,
                hard_stop=True,
            )
        scale = float(np.clip(result.speed_scale, 0.0, 1.0))
        vx = command.vx * scale if command.vx > 0.0 else command.vx
        if result.avoidance_yaw_rps != 0.0:
            wz = result.avoidance_yaw_rps
            state = f"{command.state}|{result.reason}"
        else:
            wz = command.wz
            state = (
                command.state
                if scale >= 1.0
                else f"{command.state}|{result.reason}"
            )
        return ControllerCommand(
            vx,
            command.vy,
            wz,
            state,
            command.perception_valid,
            command.hard_stop,
        )

    @staticmethod
    def _stop(
        reason: str,
        age: float,
        valid_fraction: float = 0.0,
    ) -> DepthSafetyResult:
        return DepthSafetyResult(
            motion_allowed=False,
            speed_scale=0.0,
            obstacle_distance_m=math.nan,
            valid_fraction=valid_fraction,
            depth_age_s=age,
            reason=reason,
        )
