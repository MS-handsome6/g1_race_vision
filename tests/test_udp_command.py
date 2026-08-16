from __future__ import annotations

import socket
import time
import unittest

from g1_race_vision.udp_command import UdpSkill6StatusReceiver


class UdpSkill6StatusReceiverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.receiver = UdpSkill6StatusReceiver(port=0)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tearDown(self) -> None:
        self.sender.close()
        self.receiver.close()

    def wait_for_state(self, expected: bool) -> None:
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            if self.receiver.poll() is expected:
                return
            time.sleep(0.005)
        self.fail(f"Skill 6 status did not become {expected}")

    def test_enabled_and_disabled_status(self) -> None:
        self.assertFalse(self.receiver.poll())
        self.sender.sendto(b"G1_VISION_SPRINT 1\n", self.receiver.address)
        self.wait_for_state(True)
        self.sender.sendto(b"G1_VISION_SPRINT 0\n", self.receiver.address)
        self.wait_for_state(False)

    def test_unknown_payload_is_ignored(self) -> None:
        self.sender.sendto(b"not-a-skill6-status", self.receiver.address)
        self.assertFalse(self.receiver.poll())
