from __future__ import annotations

import socket
import time
import unittest

from g1_race_vision.udp_command import (
    UdpCommandSender,
    UdpSkill6StatusReceiver,
    UdpVisionStatusReceiver,
    VisionMode,
)
from g1_race_vision.controller import ControllerCommand


class UdpVisionStatusReceiverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.receiver = UdpVisionStatusReceiver(port=0)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tearDown(self) -> None:
        self.sender.close()
        self.receiver.close()

    def wait_for_mode(self, expected: str) -> None:
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            if self.receiver.poll() == expected:
                return
            time.sleep(0.005)
        self.fail(f"mode did not become {expected}")

    def test_starts_none(self) -> None:
        self.assertEqual(self.receiver.poll(), VisionMode.NONE)

    def test_sprint_enable_disable(self) -> None:
        self.sender.sendto(b"G1_VISION_SPRINT 1\n", self.receiver.address)
        self.wait_for_mode(VisionMode.SPRINT100M)
        self.sender.sendto(b"G1_VISION_SPRINT 0\n", self.receiver.address)
        self.wait_for_mode(VisionMode.NONE)

    def test_walk_enable_disable(self) -> None:
        self.sender.sendto(b"G1_VISION_WALK_0P5M 1\n", self.receiver.address)
        self.wait_for_mode(VisionMode.WALK0P5M)
        self.sender.sendto(b"G1_VISION_WALK_0P5M 0\n", self.receiver.address)
        self.wait_for_mode(VisionMode.NONE)

    def test_num6_num7_do_not_cross_contaminate(self) -> None:
        self.sender.sendto(b"G1_VISION_SPRINT 1\n", self.receiver.address)
        self.wait_for_mode(VisionMode.SPRINT100M)
        self.sender.sendto(b"G1_VISION_WALK_0P5M 1\n", self.receiver.address)
        self.wait_for_mode(VisionMode.WALK0P5M)
        # A disabled sprint while walking must NOT disable the walk.
        self.sender.sendto(b"G1_VISION_SPRINT 0\n", self.receiver.address)
        self.wait_for_mode(VisionMode.WALK0P5M)
        self.sender.sendto(b"G1_VISION_WALK_0P5M 0\n", self.receiver.address)
        self.wait_for_mode(VisionMode.NONE)

    def test_unknown_payload_ignored(self) -> None:
        self.sender.sendto(b"G1_VISION_UNKNOWN 1\n", self.receiver.address)
        self.assertEqual(self.receiver.poll(), VisionMode.NONE)

    def test_state1_notification(self) -> None:
        self.sender.sendto(b"G1_VISION_STATE1 1\n", self.receiver.address)
        self.receiver.poll()
        self.assertTrue(self.receiver.state1_entered)

    def test_fast_disable_reenable_preserves_both_transitions(self) -> None:
        self.sender.sendto(
            b"G1_VISION_WALK_0P5M 1\n", self.receiver.address
        )
        self.sender.sendto(
            b"G1_VISION_WALK_0P5M 0\n", self.receiver.address
        )
        self.sender.sendto(
            b"G1_VISION_WALK_0P5M 1\n", self.receiver.address
        )
        time.sleep(0.02)
        self.assertEqual(
            self.receiver.poll_events(),
            [VisionMode.WALK0P5M, VisionMode.NONE, VisionMode.WALK0P5M],
        )


class UdpCommandSenderTest(unittest.TestCase):
    def test_send_includes_hard_stop_zero_by_default(self) -> None:
        # Bind a raw socket first to learn a usable local port, then send to it.
        recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv.bind(("127.0.0.1", 0))
        recv.setblocking(False)
        port = recv.getsockname()[1]
        sender = UdpCommandSender(host="127.0.0.1", port=port)
        try:
            sender.send(ControllerCommand(0.05, 0.0, 0.01, "T", True, False))
            payload, _ = recv.recvfrom(256)
            fields = payload.decode("ascii").split()
            self.assertEqual(len(fields), 4)
            self.assertEqual(fields[3], "0")
            sender.send(ControllerCommand(0.0, 0.0, 0.0, "S", False, True))
            payload, _ = recv.recvfrom(256)
            fields = payload.decode("ascii").split()
            self.assertEqual(fields[3], "1")
        finally:
            sender.close()
            recv.close()


class Skill6CompatTest(unittest.TestCase):
    def test_skill6_receiver_is_bool_compatible(self) -> None:
        receiver = UdpSkill6StatusReceiver(port=0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.assertIs(receiver.poll(), False)
            sender.sendto(b"G1_VISION_SPRINT 1\n", receiver.address)
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline and not receiver.poll():
                time.sleep(0.005)
            self.assertTrue(receiver.enabled)
            self.assertEqual(receiver.mode, VisionMode.SPRINT100M)
        finally:
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
