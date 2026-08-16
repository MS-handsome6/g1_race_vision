from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .controller import ControllerCommand
from .udp_command import UdpCommandSender


ZERO_COMMAND = ControllerCommand(0.0, 0.0, 0.0, "COMMAND_DISABLED", False, True)


class CommandSender(Protocol):
    def send(self, command: ControllerCommand) -> None: ...

    def close(self) -> None: ...


class CommandOutputGate:
    """Keeps robot command output physically disconnected until enabled."""

    def __init__(
        self,
        enabled: bool = False,
        host: str = "127.0.0.1",
        port: int = 15001,
        sender_factory: Callable[[str, int], CommandSender] = UdpCommandSender,
    ) -> None:
        self.enabled = bool(enabled)
        self._sender = (
            sender_factory(host, int(port)) if self.enabled else None
        )

    def emit(self, desired: ControllerCommand) -> ControllerCommand:
        """Send and return the command only when explicitly armed."""

        if self._sender is None:
            return ZERO_COMMAND
        self._sender.send(desired)
        return desired

    def close(self) -> None:
        if self._sender is not None:
            self._sender.close()
            self._sender = None
