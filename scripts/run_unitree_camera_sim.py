#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_race_vision.controller import (
    ControllerCommand,
    LaneFollowerConfig,
    LaneFollowerController,
    Walk0p5mConfig,
    Walk0p5mGate,
)
from g1_race_vision.depth_safety import (
    DepthSafetyConfig,
    DepthSafetyGate,
)
from g1_race_vision.line_detector import (
    DetectionResult,
    WhiteLaneDetector,
)
from g1_race_vision.mission_lifecycle import mujoco_state_was_reset
from g1_race_vision.rendering import (
    MujocoRgbdRenderer,
    average_rotation_matrices,
    attitude_align_rgbd,
    gravity_align_rgbd,
)
from g1_race_vision.udp_command import (
    UdpCommandSender,
    UdpSkill6StatusReceiver,
    UdpVisionStatusReceiver,
    VisionMode,
)

DEBUG_WINDOW_NAME = "G1 robot camera - RGB-D lane detection"


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    """Return world yaw from a MuJoCo/Unitree w,x,y,z quaternion."""

    w, x, y, z = (float(value) for value in quaternion)
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def wrapped_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _add_panel_title(
    image: np.ndarray,
    title: str,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 38), (20, 20, 20), -1)
    cv2.putText(
        panel,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    return panel


def build_camera_dashboard(
    rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    mask: np.ndarray | None,
    depth_m: np.ndarray | None,
    source: str,
) -> np.ndarray:
    """Build a 2x2 view of every stage used by the lane controller."""
    raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detected = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    source_titles = {
        "two-lines": "DETECTION: BOTH LINES",
        "none": "DETECTION: TWO REAL LINES REQUIRED",
    }
    source_colors = {
        "two-lines": (80, 255, 80),
        "none": (80, 80, 255),
    }
    raw = _add_panel_title(raw, "RAW ROBOT RGB")
    detected = _add_panel_title(
        detected,
        source_titles.get(source, f"DETECTION: {source.upper()}"),
        source_colors.get(source, (255, 255, 255)),
    )

    if mask is None:
        mask_view = np.zeros_like(raw)
    else:
        mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_view = _add_panel_title(mask_view, "WHITE PIXEL MASK")

    depth_view = np.zeros_like(raw)
    if depth_m is not None:
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        clipped = np.clip(depth_m, 0.35, 12.0)
        normalized = ((clipped - 0.35) / (12.0 - 0.35) * 255.0).astype(
            np.uint8
        )
        depth_view = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        depth_view[~valid] = 0
    depth_view = _add_panel_title(
        depth_view,
        "METRIC DEPTH: 0.35m (RED) TO 12m (BLUE)",
    )

    return np.vstack(
        (np.hstack((raw, detected)), np.hstack((mask_view, depth_view)))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official Unitree Python DDS bridge with an RGB-D lane thread."
    )
    parser.add_argument("--unitree-mujoco", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--camera", default="d435i_rgb")
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument(
        "--depth-fps",
        type=float,
        default=10.0,
        help="Depth refresh rate; RGB control still runs at --camera-fps.",
    )
    parser.add_argument("--simulate-dt", type=float, default=0.003)
    parser.add_argument("--viewer-fps", type=float, default=50.0)
    parser.add_argument(
        "--debug-fps",
        type=float,
        default=10.0,
        help="OpenCV dashboard refresh rate; does not change control FPS.",
    )
    parser.add_argument("--speed", type=float, default=5.10)
    parser.add_argument(
        "--forward-accel",
        type=float,
        default=3.00,
        help="Maximum commanded forward acceleration in m/s^2.",
    )
    parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=0.35,
        help="Maximum absolute visual yaw-rate command in rad/s.",
    )
    parser.add_argument(
        "--max-yaw-accel",
        type=float,
        default=1.5,
        help="Maximum visual yaw-rate slew in rad/s^2.",
    )
    parser.add_argument("--lateral-kp", type=float, default=0.65)
    parser.add_argument("--heading-kp", type=float, default=0.20)
    parser.add_argument("--imu-heading-kp", type=float, default=1.20)
    parser.add_argument(
        "--error-filter-alpha",
        type=float,
        default=0.32,
        help="EMA weight for the newest lateral-error sample.",
    )
    parser.add_argument(
        "--vision-enable-delay",
        type=float,
        default=0.0,
        help="Send zero velocity until this many seconds after simulator start.",
    )
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15001)
    parser.add_argument("--status-host", default="127.0.0.1")
    parser.add_argument("--status-port", type=int, default=15002)
    parser.add_argument(
        "--skill6-stabilize-seconds",
        type=float,
        default=0.60,
        help=(
            "Short camera/IMU sampling window after Skill 6 entry before "
            "locking the starting lane."
        ),
    )
    parser.add_argument("--show-debug", action="store_true")
    parser.add_argument(
        "--debug-snapshot",
        type=Path,
        help="Continuously overwrite this PNG with the latest camera dashboard.",
    )
    parser.add_argument(
        "--print-scene-information",
        action="store_true",
        help="Print every MuJoCo link, joint, actuator, and sensor at startup.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between compact vision/pose status lines; 0 disables them.",
    )
    parser.add_argument(
        "--auto-start-policy",
        action="store_true",
        help=(
            "Simulation only: send the remote-button sequence needed to enter "
            "the selected locomotion policy automatically."
        ),
    )
    parser.add_argument(
        "--policy-backend",
        choices=("g1-running", "unitree-rl-mjlab"),
        default="g1-running",
        help=(
            "Controller whose remote-button sequence is emulated by "
            "--auto-start-policy (default: g1-running)."
        ),
    )
    parser.add_argument(
        "--auto-start-mission",
        choices=("sprint100m", "walk0p5m"),
        default="sprint100m",
        help=(
            "With --auto-start-policy, which FSM mission to enter after "
            "GetUp. sprint100m sends Num1 then Num6 (Skill 6); walk0p5m "
            "sends Num1 then Num7 (Skill 7). Simulation only."
        ),
    )
    parser.add_argument(
        "--policy-start-delay",
        type=float,
        default=0.7,
        help="Delay before the simulated LT+Up pulse when auto-start is enabled.",
    )
    parser.add_argument(
        "--startup-support-seconds",
        type=float,
        default=0.0,
        help=(
            "Simulation-only torso support duration. This holds G1 upright "
            "while FixStand moves away from the MJCF's all-zero joint pose."
        ),
    )
    parser.add_argument(
        "--startup-support-fade-seconds",
        type=float,
        default=2.0,
        help=(
            "Simulation-only duration used to fade out the torso support. "
            "A gradual release avoids an artificial camera-pose jump."
        ),
    )
    parser.add_argument(
        "--startup-support-until-mission",
        "--startup-support-until-skill6",
        dest="startup_support_until_mission",
        action="store_true",
        help=(
            "Keep the simulation safety tether active until the C++ FSM "
            "confirms the active mission (Skill 6 or Num7) AND the target "
            "lane is locked. --startup-support-until-skill6 is kept as a "
            "compatibility alias for the same behaviour."
        ),
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=0.0,
        help=(
            "Treat a sustained pelvis height below this value as a fall; "
            "0 disables detection."
        ),
    )
    parser.add_argument(
        "--fall-confirm-seconds",
        type=float,
        default=0.6,
        help="How long the pelvis must remain low before declaring a fall.",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Render metric depth too. RGB-only is faster and is the default.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the main MuJoCo viewer (the camera still renders offscreen).",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Stop automatically after this many seconds; 0 runs until closed.",
    )
    parser.add_argument(
        "--finish-line-x",
        type=float,
        default=0.0,
        help=(
            "World-frame X position of the timed finish line. Crossing it "
            "starts post-finish deceleration; 0 disables timed crossing."
        ),
    )
    parser.add_argument(
        "--stop-at-x",
        type=float,
        default=0.0,
        help=(
            "Hard safety stop at the end of the post-finish runoff; "
            "0 disables it."
        ),
    )
    parser.add_argument(
        "--finish-approach-distance",
        type=float,
        default=5.0,
        help=(
            "Distance before --finish-line-x where longitudinal lane pixels are "
            "ignored and zero yaw-rate is held through the finish stripe."
        ),
    )
    parser.add_argument(
        "--keep-open-after-finish",
        action="store_true",
        help="Freeze at the finish and keep both GUI windows open.",
    )
    parser.add_argument(
        "--mission",
        choices=("sprint100m", "walk0p5m"),
        default="sprint100m",
        help=(
            "Vision mission. sprint100m is the Skill 6 100 m race; "
            "walk0p5m is the Skill 7 slow 0.5 m walk (default: sprint100m)."
        ),
    )
    parser.add_argument(
        "--num7-target-m",
        type=float,
        default=1.00,
        help="Num7 target distance in metres (0.05..1.00).",
    )
    parser.add_argument(
        "--num7-max-duration-s",
        type=float,
        default=7.0,
        help="Num7 maximum mission duration in seconds.",
    )
    parser.add_argument(
        "--num7-distance-scale",
        type=float,
        default=0.60,
        help="Open-loop Num7 distance calibration (not odometry).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.num7_target_m = float(min(max(args.num7_target_m, 0.05), 1.00))
    args.num7_max_duration_s = float(max(args.num7_max_duration_s, 0.1))
    args.num7_distance_scale = float(
        min(max(args.num7_distance_scale, 0.10), 2.0)
    )
    if (
        args.finish_line_x > 0.0
        and args.stop_at_x > 0.0
        and args.stop_at_x <= args.finish_line_x
    ):
        raise ValueError("--stop-at-x must be beyond --finish-line-x")
    simulate_python = args.unitree_mujoco.expanduser().resolve() / "simulate_python"
    sys.path.insert(0, str(simulate_python))

    try:
        # The upstream bridge chooses its DDS message layout from config.ROBOT
        # at import time.  Force the humanoid layout for this G1-only runner.
        import config as unitree_bridge_config

        unitree_bridge_config.ROBOT = "g1"
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py_bridge import UnitreeSdk2Bridge
    except ImportError as error:
        raise RuntimeError(
            "Could not import the official Unitree Python bridge. Install "
            "unitree_sdk2_python and build unitree_mujoco/simulate_python first."
        ) from error

    model = mujoco.MjModel.from_xml_path(str(args.scene.expanduser().resolve()))
    data = mujoco.MjData(model)
    model.opt.timestep = args.simulate_dt
    lock = threading.Lock()
    stop = threading.Event()
    # A normal terminal close or stop script sends SIGTERM.  Turn it into the
    # same cooperative shutdown used by Ctrl+C so DDS/viewer resources are
    # released instead of leaving the next launch in a bad state.
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    thread_errors: queue.Queue[BaseException] = queue.Queue()
    viewer = None if args.headless else mujoco.viewer.launch_passive(model, data)
    simulation_started_at = time.monotonic()
    deadline = (
        simulation_started_at + args.run_seconds
        if args.run_seconds > 0.0
        else None
    )
    support_deadline = time.monotonic() + max(0.0, args.startup_support_seconds)
    vision_enable_deadline = time.monotonic() + max(0.0, args.vision_enable_delay)
    support_body_id = model.body("torso_link").id
    support_z = float(data.qpos[2])
    support_yaw = yaw_from_wxyz(data.qpos[3:7])
    robot_mass = float(np.sum(model.body_mass))
    support_active = args.startup_support_seconds > 0.0
    support_release_started_at: float | None = None
    fall_below_since: float | None = None
    skill6_enabled = threading.Event()
    state1_entered = threading.Event()
    num7_enabled = threading.Event()
    num7_mission_done = threading.Event()
    lane_lock_acquired = threading.Event()
    finish_line_crossed = threading.Event()
    race_finished = threading.Event()
    race_timing: dict[str, float | None] = {
        "started_at": None,
        "start_x": None,
        "heading_yaw": None,
    }
    race_stats = {
        "frames": 0,
        "valid_frames": 0,
        "max_abs_y": 0.0,
        "max_command_vx": 0.0,
    }
    if viewer is not None:
        # Keep the robot in view while still allowing mouse rotation/zoom.
        # The stock passive viewer otherwise stays at the start line.
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.cam.lookat[:] = data.xpos[support_body_id]

    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = UnitreeSdk2Bridge(model, data)
    if args.print_scene_information:
        bridge.PrintSceneInformation()

    detector = WhiteLaneDetector()
    controller = LaneFollowerController(
        LaneFollowerConfig(
            cruise_speed_mps=args.speed,
            max_forward_accel_mps2=args.forward_accel,
            max_yaw_rate_rps=args.max_yaw_rate,
            max_yaw_accel_rps2=args.max_yaw_accel,
            lateral_kp=args.lateral_kp,
            heading_kp=args.heading_kp,
            imu_heading_kp=args.imu_heading_kp,
            error_filter_alpha=args.error_filter_alpha,
        )
    )
    depth_safety = DepthSafetyGate(DepthSafetyConfig())
    num7_controller = LaneFollowerController(
        LaneFollowerConfig(
            cruise_speed_mps=0.50,
            minimum_tracking_speed_mps=0.50,
            max_yaw_rate_rps=0.25,
            max_forward_accel_mps2=0.20,
            max_forward_decel_mps2=1.20,
            max_yaw_accel_rps2=0.50,
            lateral_kp=1.20,
            heading_kp=0.20,
            imu_heading_kp=1.20,
            error_filter_alpha=0.32,
        )
    )
    sender = UdpCommandSender(args.udp_host, args.udp_port)
    status_receiver = UdpVisionStatusReceiver(
        args.status_host, args.status_port
    )
    print(
        "[vision] waiting for C++ FSM status on "
        f"{status_receiver.address[0]}:{status_receiver.address[1]}",
        flush=True,
    )

    def is_running() -> bool:
        if stop.is_set():
            return False
        if deadline is not None and time.monotonic() >= deadline:
            stop.set()
            return False
        return viewer is None or viewer.is_running()

    def physics_loop() -> None:
        nonlocal fall_below_since, support_active
        nonlocal support_release_started_at

        def apply_start_heading_restraint(force: np.ndarray) -> None:
            yaw = yaw_from_wxyz(data.qpos[3:7])
            yaw_error = wrapped_angle(yaw - support_yaw)
            force[5] = float(
                np.clip(-120.0 * yaw_error - 24.0 * data.qvel[5], -90.0, 90.0)
            )

        def apply_full_start_support(support_scale: float) -> None:
            force = data.xfrc_applied[support_body_id]
            force[:] = 0.0
            force[0] = support_scale * float(
                np.clip(-90.0 * data.qvel[0], -180.0, 180.0)
            )
            force[1] = support_scale * float(
                np.clip(-90.0 * data.qvel[1], -180.0, 180.0)
            )
            force[2] = support_scale * float(
                np.clip(
                    robot_mass * 9.81
                    + 900.0 * (support_z - data.qpos[2])
                    - 120.0 * data.qvel[2],
                    0.0,
                    robot_mass * 9.81 * 2.0,
                )
            )
            apply_start_heading_restraint(force)

        while is_running():
            started = time.perf_counter()
            if args.keep_open_after_finish and race_finished.is_set():
                time.sleep(0.05)
                continue
            with lock:
                support_now = time.monotonic()
                if support_active and args.startup_support_until_mission:
                    # The startup tether stays at full power until the C++ FSM
                    # confirms a mission entry (state 1 / Skill 6 / Num7).
                    # Only then fade it out, so the operator can take as long
                    # as needed between 0/1/6/7 without the robot collapsing.
                    if (
                        (
                            state1_entered.is_set()
                            or skill6_enabled.is_set()
                            or num7_enabled.is_set()
                        )
                        and support_release_started_at is None
                    ):
                        support_release_started_at = support_now
                        print(
                            "[policy] state 1 confirmed; fading vertical "
                            "startup support",
                            flush=True,
                        )
                    if support_release_started_at is None:
                        apply_full_start_support(1.0)
                    else:
                        fade_seconds = max(
                            args.startup_support_fade_seconds, 1e-6
                        )
                        elapsed = support_now - support_release_started_at
                        support_scale = float(
                            np.clip(1.0 - elapsed / fade_seconds, 0.0, 1.0)
                        )
                        if support_scale > 0.0:
                            apply_full_start_support(support_scale)
                        else:
                            data.xfrc_applied[support_body_id] = 0.0
                            support_active = False
                            print(
                                "[policy] startup restraint released after "
                                "policy stabilization; line detection is "
                                "correction-only",
                                flush=True,
                            )
                elif support_active and support_now < support_deadline:
                    # A virtual overhead safety tether: compensate weight,
                    # stabilize height, and damp horizontal drift. It applies
                    # no joint torque and is removed after FixStand settles.
                    remaining = support_deadline - support_now
                    fade_seconds = max(
                        args.startup_support_fade_seconds, 1e-6
                    )
                    support_scale = float(
                        np.clip(remaining / fade_seconds, 0.0, 1.0)
                    )
                    apply_full_start_support(support_scale)
                elif support_active:
                    data.xfrc_applied[support_body_id] = 0.0
                    support_active = False
                    print(
                        "[policy] startup restraint released after policy "
                        "stabilization",
                        flush=True,
                    )
                mujoco.mj_step(model, data)
                if args.fall_height > 0.0 and not race_finished.is_set():
                    fall_now = time.monotonic()
                    pelvis_height = float(data.qpos[2])
                    if pelvis_height < args.fall_height:
                        if fall_below_since is None:
                            fall_below_since = fall_now
                        elif (
                            fall_now - fall_below_since
                            >= args.fall_confirm_seconds
                        ):
                            print(
                                "[safety] FALL_DETECTED: "
                                f"pelvis_z={pelvis_height:.3f} m below "
                                f"{args.fall_height:.3f} m; requesting "
                                "clean restart",
                                flush=True,
                            )
                            stop.set()
                            return
                    else:
                        fall_below_since = None
                if (
                    args.finish_line_x > 0.0
                    and data.qpos[0] >= args.finish_line_x
                    and not finish_line_crossed.is_set()
                ):
                    started_at = race_timing["started_at"]
                    start_x = race_timing["start_x"]
                    timing = ""
                    if started_at is not None and start_x is not None:
                        elapsed = time.monotonic() - started_at
                        distance = float(data.qpos[0]) - start_x
                        timing = (
                            f"; race_time={elapsed:.2f} s, "
                            f"average={distance / max(elapsed, 1e-6):.2f} m/s"
                        )
                    frames = int(race_stats["frames"])
                    valid_ratio = (
                        float(race_stats["valid_frames"]) / frames
                        if frames > 0
                        else 0.0
                    )
                    quality = (
                        f"; two_line_ratio={valid_ratio:.3f}, "
                        f"max_abs_y={float(race_stats['max_abs_y']):.3f} m, "
                        f"max_cmd_vx={float(race_stats['max_command_vx']):.2f} m/s"
                    )
                    print(
                        f"[finish] crossed x={float(data.qpos[0]):.2f} m"
                        f"{timing}{quality}; post-finish deceleration started",
                        flush=True,
                    )
                    finish_line_crossed.set()
                if (
                    args.stop_at_x > 0.0
                    and data.qpos[0] >= args.stop_at_x
                    and not race_finished.is_set()
                ):
                    print(
                        f"[safety] runoff limit reached at "
                        f"x={float(data.qpos[0]):.2f} m; forcing stop",
                        flush=True,
                    )
                    race_finished.set()
                    if not args.keep_open_after_finish:
                        stop.set()
            delay = model.opt.timestep - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)

    def viewer_loop() -> None:
        assert viewer is not None
        period = 1.0 / args.viewer_fps
        while is_running():
            with lock:
                viewer.cam.lookat[:] = data.xpos[support_body_id]
                viewer.sync()
            time.sleep(period)

    def camera_loop() -> None:
        period = 1.0 / args.camera_fps
        renderer = MujocoRgbdRenderer(model, args.camera)
        can_copy_data = hasattr(mujoco, "mj_copyData")
        render_data = mujoco.MjData(model) if can_copy_data else data
        next_status = time.monotonic()
        next_snapshot = time.monotonic()
        next_debug = time.monotonic()
        vision_enabled = False
        skill6_ready_at: float | None = None
        saved_lock_debug = False
        saved_first_loss_debug = False
        last_depth_m = None
        next_depth = 0.0
        camera_reference_xmat = None
        camera_reference_samples: list[np.ndarray] = []
        max_camera_reference_samples = max(
            3,
            int(round(args.camera_fps * 1.0)),
        )
        walk0p5m_gate = Walk0p5mGate(
            Walk0p5mConfig(
                target_distance_m=args.num7_target_m,
                max_duration_s=args.num7_max_duration_s,
                distance_scale=args.num7_distance_scale,
            )
        )
        num7_finish_x = 0.0
        num7_start_x: float | None = None
        num7_cmd_distance_m = 0.0
        num7_last_lock_time: float | None = None
        num7_stop_sent_at: float | None = None
        num7_slide_reported = False
        _num7_prev_frame_time: float | None = None
        num7_post_stop_pending: list[dict[str, float]] = []
        last_sim_time: float | None = None
        last_pelvis_x: float | None = None

        def _reset_sprint_mission(
            *,
            now: float,
            enabled: bool,
            reason: str,
            preserve_finished_hold: bool = False,
        ) -> None:
            """Reset every latch owned by one Skill 6 race."""

            nonlocal vision_enabled, skill6_ready_at
            nonlocal saved_lock_debug, saved_first_loss_debug
            nonlocal camera_reference_xmat

            vision_enabled = False
            skill6_ready_at = (
                now + max(0.0, args.skill6_stabilize_seconds)
                if enabled
                else None
            )
            camera_reference_xmat = None
            camera_reference_samples.clear()
            saved_lock_debug = False
            saved_first_loss_debug = False
            detector.reset()
            controller.reset()
            lane_lock_acquired.clear()
            finish_line_crossed.clear()
            race_finished.clear()
            if preserve_finished_hold:
                race_finished.set()
            with lock:
                race_timing.update(
                    started_at=None,
                    start_x=None,
                    heading_yaw=None,
                )
                race_stats.update(
                    frames=0,
                    valid_frames=0,
                    max_abs_y=0.0,
                    max_command_vx=0.0,
                )
            if enabled:
                skill6_enabled.set()
            else:
                skill6_enabled.clear()
            print(f"[vision] Skill 6 mission reset: {reason}", flush=True)

        def _num7_step(
            result: DetectionResult,
            depth_m: np.ndarray | None,
            now: float,
            gate: Walk0p5mGate,
        ) -> ControllerCommand:
            nonlocal num7_last_lock_time, num7_start_x, num7_cmd_distance_m
            # White-line lock only enables bounded yaw correction. It does not
            # authorize forward motion; the locomotion policy walks straight
            # under its IMU heading reference while the detector is searching.
            if num7_last_lock_time is None:
                if result.valid and not result.boundary_risk:
                    num7_last_lock_time = now
                    lane_lock_acquired.set()
                    print(
                        "[vision] Num7 lane locked; enabling white-line yaw "
                        "correction",
                        flush=True,
                    )
            # Depth safety runs only when the RGB-D sensor is enabled; with
            # RGB-only render the depth gate stays clear.
            # Only produce non-zero motion after the physics loop has fully
            # released the starting restraint (vertical tether faded AND
            # X/Y/yaw block released). Before that, hold zero so the
            # measured displacement is not distorted by the spring forces.
            with lock:
                restraint_released = not support_active
            if not restraint_released:
                return ControllerCommand(
                    0.0, 0.0, 0.0, "NUM7_RESTRAINT_ENGAGED", False, False
                )
            if num7_start_x is None:
                with lock:
                    num7_start_x = float(data.qpos[0])
                print(
                    "[num7] straight walk started independently of line "
                    f"lock at x={num7_start_x:.3f} m",
                    flush=True,
                )
            desired = num7_controller.update(result, now=now)
            if args.use_depth:
                safe = depth_safety.apply(
                    desired, depth_safety.evaluate(depth_m, 0.0)
                )
            else:
                safe = desired
            return gate.update(safe, now=now)
        try:
            if args.show_debug:
                # The launcher arranges this dashboard beside the main viewer.
                cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(DEBUG_WINDOW_NAME, 1050, 790)
                cv2.moveWindow(DEBUG_WINDOW_NAME, 1130, 60)
            while is_running():
                started = time.perf_counter()
                render_depth = (
                    args.use_depth and time.monotonic() >= next_depth
                )
                if can_copy_data:
                    with lock:
                        mujoco.mj_copyData(render_data, model, data)
                    if render_depth:
                        rgb, last_depth_m = renderer.render_rgbd(render_data)
                    else:
                        rgb = renderer.render_rgb(render_data)
                else:
                    # MuJoCo 3.3.x has no mj_copyData Python binding. Render
                    # against the live data while holding the simulation lock.
                    with lock:
                        if render_depth:
                            rgb, last_depth_m = renderer.render_rgbd(render_data)
                        else:
                            rgb = renderer.render_rgb(render_data)
                raw_rgb = rgb
                raw_depth_m = last_depth_m if render_depth else None
                current_camera_xmat = renderer.camera_xmat(render_data)
                if camera_reference_xmat is None:
                    rgb, aligned_depth = gravity_align_rgbd(
                        raw_rgb,
                        renderer.roll_correction_rad(render_data),
                        raw_depth_m,
                    )
                else:
                    rgb, aligned_depth = attitude_align_rgbd(
                        raw_rgb,
                        current_camera_xmat,
                        camera_reference_xmat,
                        renderer.vertical_fov_degrees,
                        raw_depth_m,
                    )
                if render_depth:
                    last_depth_m = aligned_depth
                if render_depth:
                    next_depth = time.monotonic() + 1.0 / max(
                        args.depth_fps, 1e-6
                    )
                result = detector.detect(
                    rgb, last_depth_m if render_depth else None
                )
                depth_m = last_depth_m
                now = time.monotonic()
                mode_transitions = status_receiver.poll_events()
                current_mode = status_receiver.mode
                # Preserve a fast disable/re-enable pair even when both UDP
                # transitions arrived between two camera frames. Present NONE
                # for this frame so the complete existing disable/reset/report
                # path runs; the receiver retains final WALK0P5M, which arms a
                # fresh mission on the next frame.
                if (
                    VisionMode.NONE in mode_transitions
                    and num7_enabled.is_set()
                ):
                    current_mode = VisionMode.NONE
                if (
                    status_receiver.state1_entered
                    and not state1_entered.is_set()
                ):
                    state1_entered.set()
                    print(
                        "[vision] C++ FSM confirmed state 1; releasing "
                        "startup support",
                        flush=True,
                    )
                mode_sprint = current_mode == VisionMode.SPRINT100M
                mode_walk = current_mode == VisionMode.WALK0P5M
                if mode_sprint and not skill6_enabled.is_set():
                    _reset_sprint_mission(
                        now=now,
                        enabled=True,
                        reason="SPRINT100M entered",
                    )
                    print(
                        "[vision] C++ FSM confirmed visual sprint mode; "
                        "starting straight after policy stabilization; "
                        "white-line lock enables correction only",
                        flush=True,
                    )
                elif not mode_sprint and skill6_enabled.is_set():
                    _reset_sprint_mission(
                        now=now,
                        enabled=False,
                        reason="SPRINT100M exited",
                        preserve_finished_hold=race_finished.is_set(),
                    )
                with lock:
                    current_sim_time = float(data.time)
                    current_pelvis_x = float(data.qpos[0])
                if (
                    mujoco_state_was_reset(
                        last_sim_time,
                        current_sim_time,
                        last_pelvis_x,
                        current_pelvis_x,
                    )
                    and (
                        skill6_enabled.is_set()
                        or vision_enabled
                        or race_finished.is_set()
                    )
                ):
                    _reset_sprint_mission(
                        now=now,
                        enabled=mode_sprint,
                        reason=(
                            "MuJoCo state reset detected "
                            f"(time {last_sim_time:.2f}->{current_sim_time:.2f}, "
                            f"x {last_pelvis_x:.2f}->{current_pelvis_x:.2f})"
                        ),
                    )
                last_sim_time = current_sim_time
                last_pelvis_x = current_pelvis_x
                if mode_walk and not num7_enabled.is_set():
                    # Num7 owns a fresh immutable lane identity per mission.
                    # The detector may have observed state-1 frames before the
                    # button edge, so reset and re-detect this exact frame
                    # before allowing the first lock. The next 500 ms status
                    # heartbeat does not enter this edge-only branch.
                    detector.reset()
                    num7_controller.reset()
                    result = detector.detect(
                        rgb, last_depth_m if render_depth else None
                    )
                    num7_enabled.set()
                    walk0p5m_gate.activate(now=now)
                    num7_last_lock_time = None
                    print(
                        "[vision] C++ FSM confirmed Num7 walk mode; "
                        "straight 0.50 m/s motion does not wait for lane lock",
                        flush=True,
                    )
                elif not mode_walk and num7_enabled.is_set():
                    num7_enabled.clear()
                    walk0p5m_gate.deactivate()
                    # The C++ FSM owns the authoritative stop point and may
                    # stop at target-margin before the Python redundancy gate
                    # reaches its own target. Preserve this mission's measured
                    # state across the mode-disable transition so we can prove
                    # one full second of post-stop zero output and no residual
                    # starting-restraint force.
                    with lock:
                        fsm_stop_x = float(data.qpos[0])
                    fsm_start_x = (
                        num7_start_x if num7_start_x is not None else fsm_stop_x
                    )
                    num7_post_stop_pending.append(
                        {
                            "stopped_at": now,
                            "stop_x": fsm_stop_x,
                            "start_x": fsm_start_x,
                            "cmd_distance": num7_cmd_distance_m,
                        }
                    )
                    print(
                        "[num7] FSM mission disabled; "
                        f"cmd_distance={num7_cmd_distance_m:.3f} m, "
                        f"qpos[0]={fsm_stop_x:.3f} m, "
                        f"actual_displacement={fsm_stop_x - fsm_start_x:.3f} m, "
                        f"start_x={fsm_start_x:.3f} m",
                        flush=True,
                    )
                    # A second Num7 entry must rebuild its visual correction
                    # reference and distance lifecycle from scratch.
                    lane_lock_acquired.clear()
                    num7_mission_done.clear()
                    num7_last_lock_time = None
                    num7_finish_x = 0.0
                    num7_start_x = None
                    num7_cmd_distance_m = 0.0
                    num7_stop_sent_at = None
                    num7_slide_reported = False
                    _num7_prev_frame_time = None
                if (
                    skill6_enabled.is_set()
                    and not vision_enabled
                    and camera_reference_xmat is None
                ):
                    camera_reference_samples.append(current_camera_xmat.copy())
                    if (
                        len(camera_reference_samples)
                        > max_camera_reference_samples
                    ):
                        del camera_reference_samples[
                            :-max_camera_reference_samples
                        ]
                if race_finished.is_set():
                    command = ControllerCommand(
                        0.0, 0.0, 0.0, "FINISHED_WINDOW_HELD", False
                    )
                elif num7_enabled.is_set():
                    command = _num7_step(
                        result=result,
                        depth_m=depth_m,
                        now=now,
                        gate=walk0p5m_gate,
                    )
                    if num7_start_x is not None and not command.hard_stop:
                        # Integrate the sent safe vx for the report. This is a
                        # commanded-distance estimate, not a measured value.
                        num7_cmd_distance_m += (
                            max(0.0, command.vx)
                            * (now - _num7_prev_frame_time)
                            if _num7_prev_frame_time is not None
                            else 0.0
                        )
                    _num7_prev_frame_time = now
                    if command.state == "NUM7_DISTANCE" or command.state == "NUM7_TIMEOUT":
                        if not num7_mission_done.is_set():
                            num7_mission_done.set()
                            with lock:
                                num7_finish_x = float(data.qpos[0])
                            start_x = (
                                num7_start_x if num7_start_x is not None else 0.0
                            )
                            print(
                                "[num7] mission finished "
                                f"({command.state}); "
                                f"cmd_distance={num7_cmd_distance_m:.3f} m, "
                                f"qpos[0]={num7_finish_x:.3f} m, "
                                f"actual_displacement="
                                f"{num7_finish_x - start_x:.3f} m, "
                                f"start_x={start_x:.3f} m",
                                flush=True,
                            )
                    if (
                        command.state == "NUM7_DISTANCE"
                        and num7_stop_sent_at is None
                    ):
                        num7_stop_sent_at = now
                elif not skill6_enabled.is_set():
                    command = ControllerCommand(
                        0.0,
                        0.0,
                        0.0,
                        "WAIT_FOR_SKILL6",
                        result.valid,
                    )
                elif support_active:
                    # The mechanical startup aid is a policy-stabilization
                    # guard only. Once it fades, straight motion begins even
                    # if no white-line pair has been acquired yet.
                    command = ControllerCommand(
                        0.0,
                        0.0,
                        0.0,
                        "WAIT_FOR_POLICY_SUPPORT_RELEASE",
                        result.valid,
                    )
                elif now < max(
                    vision_enable_deadline,
                    skill6_ready_at if skill6_ready_at is not None else now,
                ):
                    command = ControllerCommand(
                        0.0,
                        0.0,
                        0.0,
                        "WAIT_FOR_STABLE_POLICY",
                        result.valid,
                    )
                else:
                    sprint_started_now = False
                    with lock:
                        if race_timing["started_at"] is None:
                            race_timing["started_at"] = now
                            race_timing["start_x"] = float(data.qpos[0])
                            race_timing["heading_yaw"] = support_yaw
                            sprint_started_now = True
                    if sprint_started_now:
                        print(
                            "[race] straight sprint started under IMU heading "
                            "hold; vision remains correction-only",
                            flush=True,
                        )
                    if not vision_enabled:
                        # Standing up changes the camera pose substantially.
                        # Select the target lane only after the running policy
                        # is stable, then keep that lane identity locked for
                        # the entire race.
                        if camera_reference_xmat is None:
                            camera_reference_xmat = average_rotation_matrices(
                                camera_reference_samples
                                if camera_reference_samples
                                else [current_camera_xmat]
                            )
                        rgb, aligned_depth = attitude_align_rgbd(
                            raw_rgb,
                            current_camera_xmat,
                            camera_reference_xmat,
                            renderer.vertical_fov_degrees,
                            raw_depth_m,
                        )
                        if render_depth:
                            last_depth_m = aligned_depth
                        detector.reset()
                        result = detector.detect(
                            rgb, last_depth_m if render_depth else None
                        )
                        lock_quality_ok = (
                            result.valid
                            and not result.boundary_risk
                            and result.confidence >= 0.50
                            and abs(result.lateral_error) <= 0.25
                        )
                        if lock_quality_ok:
                            controller.set_visual_heading_reference(
                                result.heading_error_rad
                            )
                            vision_enabled = True
                            lane_lock_acquired.set()
                            if args.debug_snapshot is not None:
                                lock_path = (
                                    args.debug_snapshot.expanduser().resolve()
                                    .with_name("g1_lane_lock_frame.png")
                                )
                                cv2.imwrite(
                                    str(lock_path),
                                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                )
                                saved_lock_debug = True
                            print(
                                "[vision] final-height target lane locked; "
                                "camera optical centre remains the lane "
                                "target; initial centre error="
                                f"{result.lateral_error:+.3f}; "
                                "starting-block IMU heading reference="
                                f"{race_timing['heading_yaw']:+.3f} rad; "
                                "enabling bounded white-line correction",
                                flush=True,
                            )
                        elif result.valid:
                            # Do not calibrate correction from a gait-phase
                            # frame near a boundary. Straight IMU motion keeps
                            # running while the detector retries.
                            result.valid = False
                            result.source = "lane-lock-wait"
                    finish_approach = False
                    start_x = race_timing["start_x"]
                    heading_reference = race_timing["heading_yaw"]
                    heading_hold_error = None
                    if heading_reference is not None:
                        with lock:
                            current_yaw = yaw_from_wxyz(data.qpos[3:7])
                        heading_hold_error = wrapped_angle(
                            current_yaw - heading_reference
                        )
                    finish_target_x = (
                        args.finish_line_x
                        if args.finish_line_x > 0.0
                        else args.stop_at_x
                    )
                    if (
                        finish_target_x > 0.0
                        and args.finish_approach_distance > 0.0
                        and start_x is not None
                    ):
                        with lock:
                            current_x = float(data.qpos[0])
                        remaining = finish_target_x - current_x
                        finish_approach = (
                            0.0 <= remaining <= args.finish_approach_distance
                        )
                    if finish_line_crossed.is_set():
                        command = controller.brake_after_finish(
                            result=result,
                            now=now,
                            heading_hold_error_rad=heading_hold_error,
                        )
                    elif finish_approach:
                        command = controller.follow_lane_through_finish(
                            result=result,
                            now=now,
                            heading_hold_error_rad=heading_hold_error,
                        )
                    else:
                        command = controller.update(
                            result,
                            now=now,
                            heading_hold_error_rad=heading_hold_error,
                        )
                    if start_x is not None:
                        with lock:
                            current_y = float(data.qpos[1])
                        race_stats["frames"] += 1
                        race_stats["valid_frames"] += int(result.valid)
                        race_stats["max_abs_y"] = max(
                            float(race_stats["max_abs_y"]), abs(current_y)
                        )
                        race_stats["max_command_vx"] = max(
                            float(race_stats["max_command_vx"]), command.vx
                        )
                    if (
                        vision_enabled
                        and saved_lock_debug
                        and not result.valid
                        and not saved_first_loss_debug
                        and args.debug_snapshot is not None
                    ):
                        loss_path = (
                            args.debug_snapshot.expanduser().resolve()
                            .with_name("g1_first_loss_frame.png")
                        )
                        cv2.imwrite(
                            str(loss_path),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        )
                        saved_first_loss_debug = True
                # Report each completed FSM mission after observing at least
                # one second of zero output. Pending reports intentionally
                # survive Num7 deactivate/reset and therefore cover a second
                # entry independently.
                for pending in list(num7_post_stop_pending):
                    if now - pending["stopped_at"] < 1.0:
                        continue
                    with lock:
                        slide_x = float(data.qpos[0])
                        residual_force = float(
                            np.linalg.norm(data.xfrc_applied[support_body_id])
                        )
                    print(
                        "[num7] post-stop report: "
                        f"final_qpos0={slide_x:.3f} m, "
                        f"displacement={slide_x - pending['start_x']:.3f} m, "
                        f"slide_after_stop={slide_x - pending['stop_x']:.3f} m, "
                        f"cmd_distance={pending['cmd_distance']:.3f} m, "
                        f"zero_command={command.vx == 0.0 and command.vy == 0.0 and command.wz == 0.0}, "
                        f"residual_xfrc_norm={residual_force:.4f} N "
                        f"(0 = restraint fully released)",
                        flush=True,
                    )
                    num7_post_stop_pending.remove(pending)

                sender.send(command)
                if (
                    finish_line_crossed.is_set()
                    and command.state == "POST_FINISH_STOP"
                    and not race_finished.is_set()
                ):
                    with lock:
                        stopped_x = float(data.qpos[0])
                    print(
                        f"[stop] post-finish deceleration complete at "
                        f"x={stopped_x:.2f} m",
                        flush=True,
                    )
                    race_finished.set()
                    if not args.keep_open_after_finish:
                        stop.set()

                if (
                    args.status_interval > 0.0
                    and time.monotonic() >= next_status
                ):
                    with lock:
                        x, y, z = (float(value) for value in data.qpos[:3])
                        yaw = yaw_from_wxyz(data.qpos[3:7])
                    print(
                        "[vision] "
                        f"t={time.monotonic() - simulation_started_at:.2f}s "
                        f"source={result.source} valid={result.valid} "
                        f"confidence={result.confidence:.2f} "
                        f"offset={result.lateral_error:+.3f} "
                        f"heading={result.heading_error_rad:+.3f} "
                        f"state={command.state} "
                        f"cmd=({command.vx:+.2f},{command.vy:+.2f},"
                        f"{command.wz:+.2f}) "
                        f"pelvis=({x:+.2f},{y:+.2f},{z:+.2f}) "
                        f"yaw={yaw:+.3f}",
                        flush=True,
                    )
                    next_status = time.monotonic() + args.status_interval

                if args.show_debug and time.monotonic() >= next_debug:
                    debug = detector.annotate(rgb, result)
                    cv2.putText(
                        debug,
                        f"{command.state} vx={command.vx:.2f} wz={command.wz:+.3f}",
                        (12, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    dashboard = build_camera_dashboard(
                        raw_rgb,
                        debug,
                        result.mask,
                        depth_m,
                        result.source,
                    )
                    cv2.imshow(DEBUG_WINDOW_NAME, dashboard)
                    if (
                        args.debug_snapshot is not None
                        and time.monotonic() >= next_snapshot
                    ):
                        snapshot = args.debug_snapshot.expanduser().resolve()
                        snapshot.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(snapshot), dashboard)
                        next_snapshot = time.monotonic() + 1.0
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        stop.set()
                    next_debug = time.monotonic() + 1.0 / max(
                        args.debug_fps, 1e-6
                    )

                delay = period - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
        finally:
            renderer.close()
            sender.stop()

    def auto_start_policy_loop() -> None:
        def set_buttons(byte2: int, byte3: int) -> None:
            bridge.low_state.wireless_remote[2] = byte2
            bridge.low_state.wireless_remote[3] = byte3

        def wait_for_state1(timeout_s: float = 8.0) -> bool:
            """Wait for the post-InitRL state-1 acknowledgement."""

            deadline = time.monotonic() + timeout_s
            while not state1_entered.is_set():
                if stop.wait(0.02):
                    return False
                if time.monotonic() >= deadline:
                    print(
                        "[policy] state 1 acknowledgement timed out; "
                        "mission button not sent",
                        flush=True,
                    )
                    return False
            return True

        # Unitree remote byte layout:
        # byte 2: [reserved, reserved, LT, RT, select, start, LB, RB]
        # byte 3: [left, down, right, up, Y, X, B, A]
        if stop.wait(max(0.0, args.policy_start_delay)):
            return

        if args.policy_backend == "g1-running":
            print("[policy] sending A -> GetUp", flush=True)
            set_buttons(0, 0b00000001)
            if stop.wait(0.15):
                return
            set_buttons(0, 0)

            # The GetUp interpolation in rl_sar lasts 2 seconds. The extra
            # margin makes sure its FSM accepts the next transition.
            if stop.wait(2.35):
                return

            # Num1 -> state 1 (RLFSMStateRLRoboMimicLocomotion), which both
            # Skill 6 and Skill 7 missions require as the stable entry point.
            print("[policy] sending RB+Up -> Locomotion (state 1)", flush=True)
            set_buttons(0b00000001, 0b00010000)
            if stop.wait(0.15):
                return
            set_buttons(0, 0)

            if not wait_for_state1():
                return

            if args.auto_start_mission == "walk0p5m":
                # Let the locomotion policy stand stably before Num7.
                if stop.wait(0.50):
                    return
                # Num7 = LB + DPadLeft -> RLFSMStateRLVisionWalk0p5m.
                print("[policy] sending LB+Left -> Num7 (walk0p5m)", flush=True)
                set_buttons(0b00000010, 0b10000000)
                if stop.wait(0.15):
                    return
                set_buttons(0, 0)
                return

            # Default: Skill 6 (sprint100m). Num6 = LB + DPadDown.
            # State 1 has acknowledged completion of InitRL above. Only a short
            # input-edge guard is needed before Num6.
            if stop.wait(0.20):
                return
            print("[policy] sending LB+Down -> Num6 (sprint100m)", flush=True)
            set_buttons(0b00000010, 0b00100000)
            if stop.wait(0.15):
                return
            set_buttons(0, 0)
            return

        # The SDK filters analog triggers, so hold LT/RT first and press the
        # digital direction/face button only after the trigger is recognized.
        set_buttons(0b00100000, 0)
        if stop.wait(0.25):
            return
        print("[policy] sending LT+Up -> FixStand", flush=True)
        set_buttons(0b00100000, 0b00010000)
        if stop.wait(0.12):
            return
        set_buttons(0b00100000, 0)
        if stop.wait(0.12):
            return
        set_buttons(0, 0)

        if stop.wait(3.00):
            return
        set_buttons(0b00010000, 0)
        if stop.wait(0.25):
            return
        print("[policy] sending RT+A -> Velocity", flush=True)
        set_buttons(0b00010000, 0b00000001)
        if stop.wait(0.12):
            return
        set_buttons(0b00010000, 0)
        if stop.wait(0.12):
            return
        set_buttons(0, 0)

    def finish_policy_idle_loop() -> None:
        """Return rl_sar to Passive after the post-finish stop."""

        if args.policy_backend != "g1-running":
            return
        while not stop.wait(0.10):
            if race_finished.is_set():
                break
        if stop.is_set():
            return
        print("[policy] sending LB+X -> Passive after finish", flush=True)
        # X is bit 2 of byte 3; LB is bit 1 of byte 2.
        bridge.low_state.wireless_remote[2] = 0b00000010
        bridge.low_state.wireless_remote[3] = 0b00000100
        if stop.wait(0.15):
            return
        bridge.low_state.wireless_remote[2] = 0
        bridge.low_state.wireless_remote[3] = 0

    def run_thread(target) -> None:
        try:
            target()
        except BaseException as error:
            thread_errors.put(error)
            stop.set()

    threads = [
        threading.Thread(target=run_thread, args=(physics_loop,), name="physics"),
        threading.Thread(target=run_thread, args=(camera_loop,), name="camera"),
        threading.Thread(
            target=run_thread,
            args=(finish_policy_idle_loop,),
            name="finish-policy-idle",
        ),
    ]
    if viewer is not None:
        threads.append(
            threading.Thread(target=run_thread, args=(viewer_loop,), name="viewer")
        )
    if args.auto_start_policy:
        threads.append(
            threading.Thread(
                target=run_thread,
                args=(auto_start_policy_loop,),
                name="auto-start-policy",
            )
        )
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        sender.close()
        status_receiver.close()
        if viewer is not None:
            viewer.close()
        for thread in threads:
            thread.join(timeout=2.0)
    if not thread_errors.empty():
        error = thread_errors.get()
        raise RuntimeError("A simulation thread failed") from error


if __name__ == "__main__":
    main()
