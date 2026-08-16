from __future__ import annotations

import cv2
import numpy as np


def average_rotation_matrices(rotations: list[np.ndarray]) -> np.ndarray:
    """Project the arithmetic mean of rotation matrices back onto SO(3)."""

    if not rotations:
        raise ValueError("at least one rotation matrix is required")
    mean = np.mean(
        [np.asarray(rotation, dtype=np.float64).reshape(3, 3)
         for rotation in rotations],
        axis=0,
    )
    u, _, vt = np.linalg.svd(mean)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def circular_mean_angle(angles_rad: list[float]) -> float:
    """Average wrapped angles without a discontinuity at +/-pi."""

    if not angles_rad:
        raise ValueError("at least one angle is required")
    angles = np.asarray(angles_rad, dtype=np.float64)
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def gravity_roll_correction_from_xmat(camera_xmat: np.ndarray) -> float:
    """Return the in-image rotation that makes projected gravity vertical."""

    rotation = np.asarray(camera_xmat, dtype=np.float64).reshape(3, 3)
    world_up_x = float(rotation[2, 0])
    world_up_y = float(rotation[2, 1])
    return float(np.arctan2(world_up_x, world_up_y))


def gravity_align_rgbd(
    rgb: np.ndarray,
    correction_rad: float,
    depth_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Roll-stabilize RGB and aligned depth using a camera/IMU attitude."""

    height, width = rgb.shape[:2]
    center = (0.5 * (width - 1), 0.5 * (height - 1))
    angle_degrees = float(
        np.degrees(np.clip(correction_rad, -np.radians(30), np.radians(30)))
    )
    affine = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    aligned_rgb = cv2.warpAffine(
        rgb,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        # Replicating an edge where a white boundary exits the image creates
        # a fake vertical white stripe after attitude correction.  Black is
        # explicitly outside the white-line mask and therefore represents an
        # unknown/unobserved pixel safely.
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if depth_m is None:
        return aligned_rgb, None
    aligned_depth = cv2.warpAffine(
        depth_m,
        affine,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return aligned_rgb, aligned_depth


def attitude_align_rgbd(
    rgb: np.ndarray,
    current_camera_xmat: np.ndarray,
    reference_camera_xmat: np.ndarray,
    vertical_fov_degrees: float,
    depth_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Warp a moving camera frame into a fixed reference orientation."""

    height, width = rgb.shape[:2]
    focal = (0.5 * height) / np.tan(
        np.radians(0.5 * vertical_fov_degrees)
    )
    intrinsic = np.asarray(
        (
            (focal, 0.0, 0.5 * (width - 1)),
            (0.0, focal, 0.5 * (height - 1)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    current = np.asarray(current_camera_xmat, dtype=np.float64).reshape(3, 3)
    reference = np.asarray(reference_camera_xmat, dtype=np.float64).reshape(3, 3)
    # MuJoCo camera coordinates use +X right, +Y up and look along -Z.
    # OpenCV's pinhole model uses +X right, +Y down and looks along +Z.
    # Conjugating the relative rotation by this basis conversion keeps yaw,
    # pitch and roll compensation consistent with the rendered image axes.
    mujoco_to_opencv = np.diag((1.0, -1.0, -1.0))
    homography = (
        intrinsic
        @ mujoco_to_opencv
        @ reference.T
        @ current
        @ mujoco_to_opencv
        @ np.linalg.inv(intrinsic)
    )
    if abs(float(homography[2, 2])) > 1e-9:
        homography /= homography[2, 2]
    aligned_rgb = cv2.warpPerspective(
        rgb,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if depth_m is None:
        return aligned_rgb, None
    aligned_depth = cv2.warpPerspective(
        depth_m,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return aligned_rgb, aligned_depth


class MujocoRgbdRenderer:
    """Renders aligned RGB and metric depth from one named MuJoCo camera."""

    def __init__(
        self,
        model,
        camera_name: str = "d435i_rgb",
        width: int = 640,
        height: int = 480,
    ):
        import mujoco

        self._mujoco = mujoco
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if self.camera_id < 0:
            raise ValueError(f"MuJoCo camera not found: {camera_name}")
        self.vertical_fov_degrees = float(model.cam_fovy[self.camera_id])
        self.renderer = mujoco.Renderer(
            model, height=self.height, width=self.width
        )

    def render_rgbd(self, data) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.render_rgb(data)
        depth_m = self.render_depth(data, update_scene=False)
        return rgb, depth_m

    def render_rgb(self, data) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        return self.renderer.render().copy()

    def render_depth(self, data, *, update_scene: bool = True) -> np.ndarray:
        self.renderer.enable_depth_rendering()
        if update_scene:
            self.renderer.update_scene(data, camera=self.camera_name)
        depth_m = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return depth_m

    def roll_correction_rad(self, data) -> float:
        return gravity_roll_correction_from_xmat(
            data.cam_xmat[self.camera_id]
        )

    def camera_xmat(self, data) -> np.ndarray:
        return np.asarray(data.cam_xmat[self.camera_id]).reshape(3, 3).copy()

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "MujocoRgbdRenderer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
