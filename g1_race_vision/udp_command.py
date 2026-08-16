from __future__ import annotations

import socket

from .controller import ControllerCommand


class UdpCommandSender:
    """Sends an ASCII `vx vy wz` or `vx vy wz hard_stop` datagram."""

    def __init__(self, host: str = "127.0.0.1", port: int = 15001):
        self.address = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: ControllerCommand) -> None:
        payload = (
            f"{command.vx:.6f} {command.vy:.6f} {command.wz:.6f} "
            f"{1 if command.hard_stop else 0}\n"
        )
        self.socket.sendto(payload.encode("ascii"), self.address)

    def stop(self) -> None:
        self.socket.sendto(b"0.000000 0.000000 0.000000 1\n", self.address)

    def close(self) -> None:
        self.stop()
        self.socket.close()

    def __enter__(self) -> "UdpCommandSender":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class VisionMode:
    """Names of the vision command modes shared with the C++ FSM."""

    NONE = "NONE"
    SPRINT100M = "SPRINT100M"
    WALK0P5M = "WALK0P5M"


class UdpVisionStatusReceiver:
    """Receives C++ FSM vision mode transitions on the status port.

    Supports NONE / SPRINT100M / WALK0P5M plus the legacy state-1 notification.
    """

    SPRINT_ENABLED_PAYLOAD = b"G1_VISION_SPRINT 1"
    SPRINT_DISABLED_PAYLOAD = b"G1_VISION_SPRINT 0"
    WALK_ENABLED_PAYLOAD = b"G1_VISION_WALK_0P5M 1"
    WALK_DISABLED_PAYLOAD = b"G1_VISION_WALK_0P5M 0"
    STATE1_PAYLOAD = b"G1_VISION_STATE1 1"

    def __init__(self, host: str = "127.0.0.1", port: int = 15002):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, int(port)))
        self.socket.setblocking(False)
        self.address = self.socket.getsockname()
        self.mode = VisionMode.NONE
        self.state1_entered = False

    def poll(self) -> str:
        """Drain pending datagrams and return the final current mode."""
        self.poll_events()
        return self.mode

    def poll_events(self) -> list[str]:
        """Drain datagrams and return every effective mode transition.

        UDP preserves datagram order on loopback, but callers that only inspect
        the final mode can miss a fast WALK0P5M -> NONE -> WALK0P5M sequence.
        Returning the effective transitions lets mission lifecycle code process
        the mandatory disable before the next enable.
        """
        transitions: list[str] = []
        while True:
            try:
                payload, _address = self.socket.recvfrom(256)
            except BlockingIOError:
                break
            message = payload.strip()
            previous = self.mode
            if message == self.SPRINT_ENABLED_PAYLOAD:
                self.mode = VisionMode.SPRINT100M
            elif message == self.SPRINT_DISABLED_PAYLOAD:
                if self.mode == VisionMode.SPRINT100M:
                    self.mode = VisionMode.NONE
            elif message == self.WALK_ENABLED_PAYLOAD:
                self.mode = VisionMode.WALK0P5M
            elif message == self.WALK_DISABLED_PAYLOAD:
                if self.mode == VisionMode.WALK0P5M:
                    self.mode = VisionMode.NONE
            elif message == self.STATE1_PAYLOAD:
                self.state1_entered = True
            if self.mode != previous:
                transitions.append(self.mode)
        return transitions

    def close(self) -> None:
        self.socket.close()

    def __enter__(self) -> "UdpVisionStatusReceiver":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class UdpSkill6StatusReceiver(UdpVisionStatusReceiver):
    """Backwards-compatible Skill 6 status receiver.

    Kept so existing callers that use ``.enabled`` continue to work unchanged.
    """

    ENABLED_PAYLOAD = UdpVisionStatusReceiver.SPRINT_ENABLED_PAYLOAD
    DISABLED_PAYLOAD = UdpVisionStatusReceiver.SPRINT_DISABLED_PAYLOAD
    STATE1_PAYLOAD = UdpVisionStatusReceiver.STATE1_PAYLOAD

    def poll(self) -> bool:
        """Drain pending datagrams and return whether Skill 6 is enabled."""
        super().poll()
        return self.enabled

    @property
    def enabled(self) -> bool:
        return self.mode == VisionMode.SPRINT100M
