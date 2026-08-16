from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .controller import ControllerCommand


@dataclass(frozen=True)
class DepthSafetyConfig:
    """Conservative forward-corridor depth checks for initial G1 tests."""

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


@dataclass(frozen=True)
class DepthSafetyResult:
    motion_allowed: bool
    speed_scale: float
    obstacle_distance_m: float
    valid_fraction: float
    depth_age_s: float
    reason: str


class DepthSafetyGate:
    """Fail-safe depth watchdog and forward obstacle speed limiter."""

    def __init__(self, config: DepthSafetyConfig | None = None) -> None:
        self.config = config or DepthSafetyConfig()

    def evaluate(
        self,
        depth_m: np.ndarray | None,
        depth_age_s: float,
    ) -> DepthSafetyResult:
        age = max(0.0, float(depth_age_s))
        if depth_m is None:
            return self._stop("DEPTH_MISSING", age)
        if age > self.config.max_depth_age_s:
            return self._stop("DEPTH_STALE", age)
        if depth_m.ndim != 2 or depth_m.size == 0:
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
            return self._stop("DEPTH_INSUFFICIENT", age, valid_fraction)

        obstacle_distance = float(
            np.percentile(roi[valid], self.config.obstacle_percentile)
        )
        if not math.isfinite(obstacle_distance):
            return self._stop("DEPTH_NONFINITE", age, valid_fraction)
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
        state = command.state if scale >= 1.0 else f"{command.state}|DEPTH_SLOW"
        return ControllerCommand(
            vx,
            command.vy,
            command.wz,
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
