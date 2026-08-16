from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera model shared by the ROS CameraInfo and MuJoCo."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    model: str = "plumb_bob"
    distortion: tuple[float, ...] = ()

    @property
    def focal_px(self) -> float:
        return float(0.5 * (self.fx + self.fy))

    @property
    def vertical_fov_rad(self) -> float:
        return float(2.0 * np.arctan2(0.5 * self.height, self.fy))

    @property
    def vertical_fov_degrees(self) -> float:
        return float(np.degrees(self.vertical_fov_rad))

    def deproject(
        self,
        points: np.ndarray,
        depths_m: np.ndarray,
    ) -> np.ndarray:
        """Project image points and per-point depths into camera 3D.

        ``points`` is an ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
        ``depths_m`` is an ``(N,)`` array of positive depth values.
        Returns an ``(N, 3)`` array of camera-space ``(X, Y, Z)`` points.
        """

        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        depths = np.asarray(depths_m, dtype=np.float64).reshape(-1)
        if len(points) != len(depths):
            raise ValueError("points and depths_m length mismatch")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("intrinsics must have positive focal lengths")

        x = points[:, 0]
        y = points[:, 1]
        z = depths
        return np.column_stack(
            (
                (x - self.cx) / self.fx * z,
                (y - self.cy) / self.fy * z,
                z,
            )
        )


def median_depth_in_neighbourhood(
    depth_m: np.ndarray,
    x: int,
    y: int,
    *,
    radius: int = 2,
    min_valid_depth_m: float = 0.35,
    max_valid_depth_m: float = 12.0,
    min_valid_pixels: int = 3,
) -> float | None:
    """Return the median of valid depth pixels around ``(x, y)``.

    ``None`` is returned when the neighbourhood does not contain enough
    finite depth pixels, which callers must treat as an unreliable sample.
    """

    height, width = depth_m.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    region = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float32)
    valid = (
        np.isfinite(region)
        & (region >= min_valid_depth_m)
        & (region <= max_valid_depth_m)
    )
    if int(np.count_nonzero(valid)) < min_valid_pixels:
        return None
    return float(np.median(region[valid]))
