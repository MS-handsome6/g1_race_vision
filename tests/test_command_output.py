from __future__ import annotations

from pathlib import Path
import unittest

from g1_race_vision.command_output import CommandOutputGate
from g1_race_vision.controller import ControllerCommand


VISION_ROOT = Path(__file__).resolve().parents[1]
REALSENSE_SCRIPT = VISION_ROOT / "scripts/run_realsense_ros2.py"


class FakeSender:
    def __init__(self, host: str, port: int) -> None:
        self.address = (host, port)
        self.commands: list[ControllerCommand] = []
        self.closed = False

    def send(self, command: ControllerCommand) -> None:
        self.commands.append(command)

    def close(self) -> None:
        self.closed = True


class CommandOutputGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.created_senders: list[FakeSender] = []

    def sender_factory(self, host: str, port: int) -> FakeSender:
        sender = FakeSender(host, port)
        self.created_senders.append(sender)
        return sender

    def test_output_is_physically_disconnected_by_default(self) -> None:
        gate = CommandOutputGate(sender_factory=self.sender_factory)
        desired = ControllerCommand(0.2, 0.0, 0.1, "LINE_FOLLOW", True)

        actual = gate.emit(desired)

        self.assertFalse(gate.enabled)
        self.assertEqual(self.created_senders, [])
        self.assertEqual((actual.vx, actual.vy, actual.wz), (0.0, 0.0, 0.0))
        gate.close()
        self.assertEqual(self.created_senders, [])

    def test_enabled_output_sends_and_returns_desired_command(self) -> None:
        gate = CommandOutputGate(
            enabled=True,
            host="127.0.0.1",
            port=15001,
            sender_factory=self.sender_factory,
        )
        desired = ControllerCommand(0.2, 0.0, -0.1, "LINE_FOLLOW", True)

        actual = gate.emit(desired)

        self.assertTrue(gate.enabled)
        self.assertIs(actual, desired)
        self.assertEqual(len(self.created_senders), 1)
        self.assertEqual(self.created_senders[0].address, ("127.0.0.1", 15001))
        self.assertEqual(self.created_senders[0].commands, [desired])

        gate.close()
        self.assertTrue(self.created_senders[0].closed)


class RealSenseDryRunConfigurationTest(unittest.TestCase):
    def test_realsense_node_defaults_to_dry_run(self) -> None:
        source = REALSENSE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'self.declare_parameter("command_output_enabled", False)',
            source,
        )
        self.assertIn('"/g1/race/desired_cmd_vel"', source)
        self.assertIn('"/g1/race/safe_cmd_vel"', source)
        self.assertIn('"/g1/race/safety_state"', source)
        self.assertIn('"/g1/race/cmd_vel"', source)
        self.assertIn('self.declare_parameter("audit_udp_enabled", False)', source)
        self.assertIn("DRY RUN - ROBOT COMMAND DISABLED", source)


if __name__ == "__main__":
    unittest.main()
