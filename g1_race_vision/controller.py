from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .line_detector import DetectionResult


@dataclass(frozen=True)
class LaneFollowerConfig:
    cruise_speed_mps: float = 0.55
    minimum_tracking_speed_mps: float = 0.20
    # Strong correction authority: once a real departure is detected, pull back
    # toward the lane centre hard. The previous 0.65 gain + 0.10 rad heading
    # limit let a 0.4 m drift grow for ~16 m before the boundary stop kicked in.
    lateral_kp: float = 1.20
    lateral_kd: float = 0.08
    heading_kp: float = 0.20
    imu_heading_kp: float = 1.20
    error_filter_alpha: float = 0.32
    heading_filter_alpha: float = 0.16
    max_lateral_error_rate_per_s: float = 1.20
    max_visual_heading_rate_rps: float = 1.20
    max_visual_heading_error_rad: float = 0.35
    max_heading_imu_disagreement_rad: float = 0.25
    heading_confidence_floor: float = 0.45
    # The trained policy owns the straight run. Vision enters a bounded
    # correction mode only after leaving this corridor and stays quiet again
    # after returning to the smaller exit threshold (Schmitt hysteresis).
    #
    # A large departure enters correction immediately. Smaller, steady centre
    # offsets use the persistence gate below so gait sway is ignored but the
    # robot still converges to the camera optical centre instead of preserving
    # its starting offset for the whole lane.
    correction_enter_lateral_error: float = 0.26
    correction_exit_lateral_error: float = 0.035
    center_alignment_enter_lateral_error: float = 0.075
    center_alignment_confirm_s: float = 0.60
    # A real slow drift should not have to reach the wide static-offset gate.
    # Enter early only when an offset is already meaningful and keeps moving
    # outward in one direction for several frames. Gait sway reverses too often
    # to accumulate this confirmation time.
    predictive_enter_lateral_error: float = 0.14
    predictive_outward_rate_per_s: float = 0.055
    predictive_confirm_s: float = 0.30
    # Strong correction authority: once a real departure is detected, pull back
    # toward the lane centre hard. The previous 0.10 rad heading limit and 0.65
    # lateral gain let a 0.4 m drift grow for ~16 m before the boundary stop
    # kicked in; with a stronger correction the robot recentres much sooner and
    # the speed ramp stays mostly inactive.
    max_lane_correction_heading_rad: float = 0.18
    # Gentle speed ramp so a real (but bounded) correction slows the robot only
    # mildly; full slow-down is reserved for large departures.
    lateral_slow_start: float = 0.16
    lateral_full_slow: float = 0.50
    lateral_rate_slow_start_per_s: float = 0.12
    lateral_rate_full_slow_per_s: float = 0.45
    imu_heading_slow_start_rad: float = 0.05
    imu_heading_full_slow_rad: float = 0.20
    heading_slow_start_rad: float = 0.10
    heading_full_slow_rad: float = 0.35
    steering_slow_start_ratio: float = 0.65
    # Keep the aggressive floor (0.22 of cruise) so genuine departures still
    # collapse the sprint into a cautious slow-down; the wide enter threshold
    # above already keeps small static offsets out of the correction ramp.
    minimum_speed_scale: float = 0.22
    yaw_saturation_ratio: float = 0.92
    yaw_saturation_slow_after_s: float = 0.18
    yaw_saturation_full_slow_s: float = 0.65
    max_forward_accel_mps2: float = 0.35
    max_forward_decel_mps2: float = 3.00
    max_yaw_rate_rps: float = 0.35
    max_yaw_accel_rps2: float = 1.50
    # Retained for configuration compatibility. Visual availability no longer
    # gates forward motion: the running policy and IMU own the straight path,
    # while white lines only add bounded yaw correction when they are valid.
    hold_last_command_s: float = 0.35
    slow_after_lost_s: float = 0.90
    stop_after_lost_s: float = 1.50


@dataclass(frozen=True)
class ControllerCommand:
    vx: float
    vy: float
    wz: float
    state: str
    perception_valid: bool
    hard_stop: bool = False


class LaneFollowerController:
    """Turns visual lane errors into conservative velocity commands."""

    def __init__(self, config: LaneFollowerConfig | None = None):
        self.config = config or LaneFollowerConfig()
        self._filtered_lateral = 0.0
        self._filtered_lateral_rate = 0.0
        self._filtered_heading = 0.0
        self._previous_lateral = 0.0
        self._previous_wz = 0.0
        self._previous_vx = 0.0
        self._yaw_saturation_duration_s = 0.0
        self._correction_direction = 0.0
        self._outward_drift_direction = 0.0
        self._outward_drift_duration_s = 0.0
        self._center_offset_direction = 0.0
        self._center_offset_duration_s = 0.0
        self._visual_heading_reference_rad = 0.0
        self._last_time: float | None = None
        self._last_valid_time: float | None = None

    def reset(self) -> None:
        self._filtered_lateral = 0.0
        self._filtered_lateral_rate = 0.0
        self._filtered_heading = 0.0
        self._previous_lateral = 0.0
        self._previous_wz = 0.0
        self._previous_vx = 0.0
        self._yaw_saturation_duration_s = 0.0
        self._correction_direction = 0.0
        self._outward_drift_direction = 0.0
        self._outward_drift_duration_s = 0.0
        self._center_offset_direction = 0.0
        self._center_offset_duration_s = 0.0
        self._visual_heading_reference_rad = 0.0
        self._last_time = None
        self._last_valid_time = None

    def reset_lane_guidance(self) -> None:
        """Forget the old lane after a legal 100 m obstacle pass.

        Forward-speed history is preserved so accepting the newly occupied
        lane cannot create a second, artificial stop after the bypass.
        """

        previous_vx = self._previous_vx
        last_time = self._last_time
        self.reset()
        self._previous_vx = previous_vx
        self._last_time = last_time

    def set_visual_heading_reference(self, heading_error_rad: float) -> None:
        """Zero a fixed camera/mounting vanishing-point bias at lane lock."""

        self._visual_heading_reference_rad = float(heading_error_rad)
        self._filtered_heading = 0.0

    def update(
        self,
        result: DetectionResult,
        now: float | None = None,
        heading_hold_error_rad: float | None = None,
    ) -> ControllerCommand:
        now = time.monotonic() if now is None else float(now)
        dt = self._time_step(now)

        if result.valid:
            wz = self._visual_steering(
                result, dt, heading_hold_error_rad
            )
            self._last_valid_time = now

            confidence_scale = 0.72 + 0.28 * float(
                np.clip(result.confidence, 0.0, 1.0)
            )
            correcting = self._correction_direction != 0.0
            lateral_scale = (
                self._safety_scale(
                    abs(self._filtered_lateral),
                    self.config.lateral_slow_start,
                    self.config.lateral_full_slow,
                    self.config.minimum_speed_scale,
                )
                if correcting
                else 1.0
            )
            lateral_rate_scale = (
                self._safety_scale(
                    abs(self._filtered_lateral_rate),
                    self.config.lateral_rate_slow_start_per_s,
                    self.config.lateral_rate_full_slow_per_s,
                    self.config.minimum_speed_scale,
                )
                if correcting
                else 1.0
            )
            # Visual heading is diagnostic only during the straight corridor;
            # body yaw is held by the IMU reference captured at mission start.
            heading_scale = 1.0
            imu_heading_scale = self._safety_scale(
                abs(heading_hold_error_rad or 0.0),
                self.config.imu_heading_slow_start_rad,
                self.config.imu_heading_full_slow_rad,
                self.config.minimum_speed_scale,
            )
            steering_ratio = abs(wz) / max(self.config.max_yaw_rate_rps, 1e-6)
            steering_scale = self._safety_scale(
                steering_ratio,
                self.config.steering_slow_start_ratio,
                1.0,
                max(self.config.minimum_speed_scale, 0.42),
            )
            saturation_scale = self._safety_scale(
                self._yaw_saturation_duration_s,
                self.config.yaw_saturation_slow_after_s,
                self.config.yaw_saturation_full_slow_s,
                self.config.minimum_speed_scale,
            )
            boundary_scale = (
                self.config.minimum_speed_scale
                if result.boundary_risk
                else 1.0
            )
            safety_scale = min(
                confidence_scale,
                lateral_scale,
                lateral_rate_scale,
                heading_scale,
                imu_heading_scale,
                steering_scale,
                saturation_scale,
                boundary_scale,
            )
            vx = max(
                self.config.minimum_tracking_speed_mps,
                self.config.cruise_speed_mps * safety_scale,
            )
            vx = self._rate_limit_vx(vx, dt)
            state = (
                "LINE_FOLLOW_BOUNDARY_RECOVERY"
                if result.boundary_risk
                else "LINE_FOLLOW"
            )
            return ControllerCommand(vx, 0.0, wz, state, True)

        # White-line perception is correction-only. Before the first lock and
        # during any later dropout, keep following the immutable IMU heading
        # and ramp toward the policy's normal straight-line speed. Independent
        # depth, command-freshness, distance, timeout and E-stop gates remain
        # responsible for stopping the robot.
        wz = self._rate_limit(
            self._heading_hold_target(heading_hold_error_rad), dt
        )
        vx = self._rate_limit_vx(self.config.cruise_speed_mps, dt)
        state = (
            "STRAIGHT_IMU_NO_LINE"
            if self._last_valid_time is None
            else "STRAIGHT_IMU_LINE_LOST"
        )
        return ControllerCommand(vx, 0.0, wz, state, False)

    def follow_lane_through_finish(
        self,
        result: DetectionResult,
        now: float | None = None,
        heading_hold_error_rad: float | None = None,
    ) -> ControllerCommand:
        """Keep normal two-line correction active until 100 m is crossed."""

        command = self.update(
            result,
            now=now,
            heading_hold_error_rad=heading_hold_error_rad,
        )
        return ControllerCommand(
            command.vx,
            command.vy,
            command.wz,
            "FINISH_APPROACH_LANE_FOLLOW",
            command.perception_valid,
        )

    def brake_after_finish(
        self,
        result: DetectionResult | None = None,
        now: float | None = None,
        heading_hold_error_rad: float | None = None,
    ) -> ControllerCommand:
        """Decelerate after 100 m while still following both lane lines."""

        now = time.monotonic() if now is None else float(now)
        dt = self._time_step(now)
        vx = self._rate_limit_vx(0.0, dt)
        perception_valid = result is not None and result.valid
        if perception_valid:
            wz = self._visual_steering(
                result, dt, heading_hold_error_rad
            )
            self._last_valid_time = now
        else:
            wz = self._rate_limit(
                self._heading_hold_target(heading_hold_error_rad), dt
            )
        state = "POST_FINISH_STOP" if vx <= 1e-3 else "POST_FINISH_DECEL"
        return ControllerCommand(vx, 0.0, wz, state, perception_valid)

    def _visual_steering(
        self,
        result: DetectionResult,
        dt: float,
        heading_hold_error_rad: float | None,
    ) -> float:
        alpha = float(np.clip(self.config.error_filter_alpha, 0.0, 1.0))
        lateral_target = (
            alpha * result.lateral_error
            + (1.0 - alpha) * self._filtered_lateral
        )
        max_lateral_delta = self.config.max_lateral_error_rate_per_s * dt
        self._filtered_lateral += float(
            np.clip(
                lateral_target - self._filtered_lateral,
                -max_lateral_delta,
                max_lateral_delta,
            )
        )
        derivative_raw = (
            self._filtered_lateral - self._previous_lateral
        ) / dt
        self._filtered_lateral_rate = (
            0.18 * derivative_raw
            + 0.82 * self._filtered_lateral_rate
        )
        self._previous_lateral = self._filtered_lateral
        self._update_correction_mode(dt)
        self._update_filtered_heading(
            result,
            dt,
            heading_hold_error_rad,
        )
        # Treat the lane error as a bounded desired body heading.  Directly
        # mapping a large lateral error to yaw rate creates an under-damped
        # weave at sprint speed: the robot crosses the centre while still
        # carrying a large yaw angle.  The inner IMU loop damps that angle.
        desired_heading = self._desired_lane_heading()
        heading_error = (
            (heading_hold_error_rad or 0.0) - desired_heading
        )
        # Do not continuously steer from the camera's visual heading. The
        # trained running policy plus IMU heading hold produce the straight
        # trajectory; vision only supplies the bounded lateral recovery angle.
        raw_wz = -(self.config.imu_heading_kp * heading_error)
        saturated = (
            abs(raw_wz)
            >= self.config.yaw_saturation_ratio
            * self.config.max_yaw_rate_rps
        )
        if saturated:
            self._yaw_saturation_duration_s += dt
        else:
            self._yaw_saturation_duration_s = max(
                0.0,
                self._yaw_saturation_duration_s - 2.0 * dt,
            )
        target_wz = float(
            np.clip(
                raw_wz,
                -self.config.max_yaw_rate_rps,
                self.config.max_yaw_rate_rps,
            )
        )
        return self._rate_limit(target_wz, dt)

    def _desired_lane_heading(self) -> float:
        if self._correction_direction == 0.0:
            return 0.0

        # Keep one correction direction until the robot is back inside the
        # exit corridor. This prevents left/right command chatter when gait
        # sway makes the measured centre cross zero on consecutive frames.
        signed_error = self._correction_direction * self._filtered_lateral
        lateral = self._correction_direction * max(
            0.0,
            signed_error - self.config.correction_exit_lateral_error,
        )
        lateral_rate = self._correction_direction * max(
            0.0,
            self._correction_direction * self._filtered_lateral_rate,
        )
        lane_guidance = (
            self.config.lateral_kp * lateral
            + self.config.lateral_kd * lateral_rate
        )
        return float(
            np.clip(
                -lane_guidance / max(self.config.imu_heading_kp, 1e-6),
                -self.config.max_lane_correction_heading_rad,
                self.config.max_lane_correction_heading_rad,
            )
        )

    def _update_correction_mode(self, dt: float) -> None:
        magnitude = abs(self._filtered_lateral)
        if self._correction_direction == 0.0:
            if magnitude >= self.config.correction_enter_lateral_error:
                self._correction_direction = float(
                    np.sign(self._filtered_lateral)
                )
                self._clear_outward_drift_confirmation()
                self._clear_center_offset_confirmation()
                return

            if self._confirm_persistent_center_offset(magnitude, dt):
                self._clear_outward_drift_confirmation()
                return

            direction = float(np.sign(self._filtered_lateral))
            outward_rate = direction * self._filtered_lateral_rate
            predictive = (
                direction != 0.0
                and magnitude >= self.config.predictive_enter_lateral_error
                and outward_rate
                >= self.config.predictive_outward_rate_per_s
            )
            if predictive:
                if direction != self._outward_drift_direction:
                    self._outward_drift_direction = direction
                    self._outward_drift_duration_s = 0.0
                self._outward_drift_duration_s += dt
                if (
                    self._outward_drift_duration_s
                    >= self.config.predictive_confirm_s
                ):
                    self._correction_direction = direction
                    self._clear_outward_drift_confirmation()
            else:
                # Decay faster than evidence accumulates so alternating gait
                # sway cannot stitch unrelated half-cycles into a correction.
                self._outward_drift_duration_s = max(
                    0.0,
                    self._outward_drift_duration_s - 2.0 * dt,
                )
                if self._outward_drift_duration_s == 0.0:
                    self._outward_drift_direction = 0.0
            return
        if magnitude <= self.config.correction_exit_lateral_error:
            self._correction_direction = 0.0
            self._clear_outward_drift_confirmation()
            self._clear_center_offset_confirmation()

    def _confirm_persistent_center_offset(
        self,
        magnitude: float,
        dt: float,
    ) -> bool:
        """Enter correction for a stable off-centre camera observation.

        Alternating torso sway changes sign and cannot accumulate confirmation,
        while a real lane-centre offset keeps one sign long enough to trigger.
        """

        direction = float(np.sign(self._filtered_lateral))
        persistent = (
            direction != 0.0
            and magnitude
            >= self.config.center_alignment_enter_lateral_error
        )
        if persistent:
            if direction != self._center_offset_direction:
                self._center_offset_direction = direction
                self._center_offset_duration_s = 0.0
            self._center_offset_duration_s += dt
            if (
                self._center_offset_duration_s
                >= self.config.center_alignment_confirm_s
            ):
                self._correction_direction = direction
                self._clear_center_offset_confirmation()
                return True
        else:
            self._center_offset_duration_s = max(
                0.0,
                self._center_offset_duration_s - 2.0 * dt,
            )
            if self._center_offset_duration_s == 0.0:
                self._center_offset_direction = 0.0
        return False

    def _clear_outward_drift_confirmation(self) -> None:
        self._outward_drift_direction = 0.0
        self._outward_drift_duration_s = 0.0

    def _clear_center_offset_confirmation(self) -> None:
        self._center_offset_direction = 0.0
        self._center_offset_duration_s = 0.0

    def _update_filtered_heading(
        self,
        result: DetectionResult,
        dt: float,
        heading_hold_error_rad: float | None,
    ) -> None:
        """Reject camera-shake spikes and fuse visual heading with the IMU."""

        visual_heading = float(
            np.arctan2(
                np.sin(
                    result.heading_error_rad
                    - self._visual_heading_reference_rad
                ),
                np.cos(
                    result.heading_error_rad
                    - self._visual_heading_reference_rad
                ),
            )
        )
        imu_heading = (
            None
            if heading_hold_error_rad is None
            else float(heading_hold_error_rad)
        )
        visual_is_plausible = (
            result.confidence >= self.config.heading_confidence_floor
            and abs(visual_heading)
            <= self.config.max_visual_heading_error_rad
        )
        if (
            visual_is_plausible
            and imu_heading is not None
            and abs(visual_heading - imu_heading)
            > self.config.max_heading_imu_disagreement_rad
        ):
            visual_is_plausible = False

        if visual_is_plausible:
            measurement = visual_heading
        elif imu_heading is not None:
            measurement = imu_heading
        else:
            measurement = self._filtered_heading

        measurement = float(
            np.clip(
                measurement,
                -self.config.max_visual_heading_error_rad,
                self.config.max_visual_heading_error_rad,
            )
        )
        alpha = float(np.clip(self.config.heading_filter_alpha, 0.0, 1.0))
        target = (
            alpha * measurement
            + (1.0 - alpha) * self._filtered_heading
        )
        max_delta = self.config.max_visual_heading_rate_rps * dt
        self._filtered_heading += float(
            np.clip(
                target - self._filtered_heading,
                -max_delta,
                max_delta,
            )
        )

    @staticmethod
    def _safety_scale(
        value: float,
        slow_start: float,
        full_slow: float,
        minimum_scale: float,
    ) -> float:
        """Return a continuous 1..minimum_scale safety speed multiplier."""

        if value <= slow_start:
            return 1.0
        if full_slow <= slow_start:
            return float(minimum_scale)
        progress = float(
            np.clip(
                (value - slow_start) / (full_slow - slow_start),
                0.0,
                1.0,
            )
        )
        return float(1.0 - progress * (1.0 - minimum_scale))

    def _time_step(self, now: float) -> float:
        dt = (
            1.0 / 30.0
            if self._last_time is None
            else max(now - self._last_time, 1e-3)
        )
        self._last_time = now
        return dt

    def _heading_hold_target(
        self, heading_error_rad: float | None
    ) -> float:
        if heading_error_rad is None:
            return 0.0
        return float(
            np.clip(
                -self.config.imu_heading_kp * heading_error_rad,
                -self.config.max_yaw_rate_rps,
                self.config.max_yaw_rate_rps,
            )
        )

    def _rate_limit(self, target_wz: float, dt: float) -> float:
        max_delta = self.config.max_yaw_accel_rps2 * dt
        delta = float(np.clip(target_wz - self._previous_wz, -max_delta, max_delta))
        self._previous_wz += delta
        return self._previous_wz

    def _rate_limit_vx(self, target_vx: float, dt: float) -> float:
        rate = (
            self.config.max_forward_accel_mps2
            if target_vx >= self._previous_vx
            else self.config.max_forward_decel_mps2
        )
        max_delta = rate * dt
        self._previous_vx += float(
            np.clip(target_vx - self._previous_vx, -max_delta, max_delta)
        )
        return self._previous_vx


@dataclass(frozen=True)
class Walk0p5mConfig:
    """Python-side redundant safety limits for the Num7 1.0 m walk.

    ``Walk0p5m`` remains the protocol/class name for backward compatibility.
    """

    max_vx_mps: float = 0.50
    max_wz_rps: float = 0.25
    target_distance_m: float = 1.00
    # Open-loop calibration only; this is not measured odometry.
    distance_scale: float = 0.60
    max_duration_s: float = 7.0
    # Kept for compatibility with existing launch files. Line loss no longer
    # stops Num7; independent safety gates still latch hard stops.
    stop_after_lost_s: float = 0.30


class Walk0p5mGate:
    """Python redundancy for the Num7 mission.

    This is a secondary guard on top of the C++ FSM: it re-clamps vx/vy/wz,
    integrates the *sent* safe vx, latches zero + hard_stop on distance,
    timeout or an independent safety failure, and keeps sending zero after a
    stop. White-line validity is deliberately not a motion gate: the policy
    walks straight under IMU heading hold when vision is unavailable.

    The gate has an explicit lifecycle. It does NOT start counting when
    constructed: time and distance accumulate only after ``activate()`` is
    called, which happens once the C++ FSM reports ``G1_VISION_WALK_0P5M 1``.
    Before activation the gate outputs zero and never inherits a previous
    mission's hard_stop / distance / timeout.
    """

    def __init__(self, config: Walk0p5mConfig | None = None) -> None:
        self.config = config or Walk0p5mConfig()
        self.reset()

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._started = False
        self._stopped = False
        self._distance_m = 0.0
        self._start_time: float | None = None
        self._last_time: float | None = None

    def activate(self, now: float | None = None) -> None:
        """Start a fresh Num7 mission. Time/distance reset to zero."""
        self._active = True
        self._started = False
        self._stopped = False
        self._distance_m = 0.0
        now = time.monotonic() if now is None else float(now)
        self._start_time = now
        self._last_time = None

    def deactivate(self) -> None:
        """Leave the Num7 mission: output zero, clear all task state so the
        next activation is a brand-new mission."""
        self._active = False
        self.reset()

    def update(
        self,
        desired: ControllerCommand,
        now: float | None = None,
    ) -> ControllerCommand:
        now = time.monotonic() if now is None else float(now)

        if not self._active:
            # Not in a Num7 mission yet: hold zero, do not accumulate time or
            # distance, and do not emit a hard_stop that could be inherited by
            # a future session.
            return ControllerCommand(0.0, 0.0, 0.0, "NUM7_INACTIVE", False, False)

        dt = (
            1.0 / 30.0
            if self._last_time is None
            else max(now - self._last_time, 1e-3)
        )
        self._last_time = now

        if self._start_time is None:
            self._start_time = now

        if self._stopped:
            # Keep sending zero + hard_stop until the FSM exits the state.
            return ControllerCommand(0.0, 0.0, 0.0, "NUM7_STOPPED", False, True)

        # Re-clamp to the Num7 hard limits.
        vx = float(np.clip(desired.vx, 0.0, self.config.max_vx_mps))
        vz = float(
            np.clip(
                desired.wz,
                -self.config.max_wz_rps,
                self.config.max_wz_rps,
            )
        )
        if desired.hard_stop:
            self._stopped = True
            return ControllerCommand(
                0.0,
                0.0,
                0.0,
                desired.state,
                desired.perception_valid,
                True,
            )

        clamped = ControllerCommand(
            vx, 0.0, vz, desired.state, desired.perception_valid, False
        )

        # Integrate the sent safe vx (Python redundancy).
        if self._started:
            self._distance_m += vx * dt * self.config.distance_scale
        elif vx > 0.0:
            self._started = True
            self._distance_m += vx * dt * self.config.distance_scale

        if self._distance_m >= self.config.target_distance_m:
            self._stopped = True
            return ControllerCommand(0.0, 0.0, 0.0, "NUM7_DISTANCE", True, True)

        if now - self._start_time >= self.config.max_duration_s:
            self._stopped = True
            return ControllerCommand(0.0, 0.0, 0.0, "NUM7_TIMEOUT", True, True)

        return clamped
