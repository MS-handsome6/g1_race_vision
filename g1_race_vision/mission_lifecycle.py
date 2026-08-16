from __future__ import annotations

from typing import Protocol


class Resettable(Protocol):
    def reset(self) -> None: ...


class ActivatableGate(Protocol):
    @property
    def active(self) -> bool: ...

    def activate(self, now: float | None = None) -> None: ...

    def deactivate(self) -> None: ...


def apply_num7_mode_transition(
    current_mode: str,
    detector: Resettable,
    controller: Resettable,
    gate: ActivatableGate,
    *,
    now: float,
) -> str | None:
    """Apply one effective Num7 mode edge on the camera-owner thread.

    Returns ``"enabled"`` or ``"disabled"`` when a lifecycle transition was
    applied, otherwise ``None``. Repeated enable heartbeats are idempotent so
    they cannot erase the lane anchor or restart the distance timer.
    """

    if current_mode == "WALK0P5M":
        if gate.active:
            return None
        detector.reset()
        controller.reset()
        activate_detector = getattr(
            detector, "set_num7_mission_active", None
        )
        if callable(activate_detector):
            activate_detector(True)
        gate.activate(now=now)
        return "enabled"

    if gate.active:
        deactivate_detector = getattr(
            detector, "set_num7_mission_active", None
        )
        if callable(deactivate_detector):
            deactivate_detector(False)
        gate.deactivate()
        return "disabled"
    return None


def mujoco_state_was_reset(
    previous_time: float | None,
    current_time: float,
    previous_x: float | None,
    current_x: float,
    *,
    minimum_time_rewind_s: float = 0.5,
    minimum_backward_jump_m: float = 1.0,
) -> bool:
    """Detect a MuJoCo viewer reset from discontinuous simulation state.

    The passive viewer resets ``MjData`` directly and does not notify the
    Python vision mission or the C++ FSM. Simulation time rewinding is the
    authoritative signal; the X jump is a fallback for reset implementations
    that preserve time.
    """

    time_rewound = (
        previous_time is not None
        and current_time < previous_time - minimum_time_rewind_s
    )
    position_rewound = (
        previous_x is not None
        and current_x < previous_x - minimum_backward_jump_m
    )
    return time_rewound or position_rewound
