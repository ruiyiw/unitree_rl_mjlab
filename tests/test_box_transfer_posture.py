"""Regression tests for loaded versus unloaded completion posture."""
import math
import unittest

import torch

from unitree_box_transfer.five_requirements import posture_ok, upright_putdown_ok


def pitch(angle):
    return torch.tensor([[math.cos(angle/2),0.,math.sin(angle/2),0.]])


class PostureTests(unittest.TestCase):
    def test_loaded_reference_satisfies_its_own_requirement(self):
        self.assertTrue(posture_ok(pitch(.1186),pitch(.1186),True).item())

    def test_small_deviation_from_carrying_pose_is_allowed(self):
        self.assertTrue(posture_ok(pitch(.16),pitch(.1186),True).item())

    def test_large_reference_error_is_rejected(self):
        self.assertFalse(posture_ok(pitch(-.08),pitch(.1186),True).item())

    def test_reference_cannot_override_absolute_tilt_limit(self):
        self.assertFalse(posture_ok(pitch(.4),pitch(.4),True).item())

    def test_unloaded_standing_keeps_strict_upright_requirement(self):
        self.assertFalse(posture_ok(pitch(.1186),pitch(.1186),False).item())
        self.assertTrue(posture_ok(pitch(.05),pitch(0.),False).item())

    def test_old_putdown_squat_and_lean_rejected_even_if_terminal_pose_is_upright(self):
        for height, angle in ((.603,0.),(.76,math.radians(-25)),(.604,math.radians(-15))):
            self.assertFalse(upright_putdown_ok(torch.tensor([height]),pitch(angle)).item())

    def test_upright_putdown_allows_small_balance_motion(self):
        self.assertTrue(upright_putdown_ok(torch.tensor([.755]),pitch(math.radians(2))).item())


if __name__ == "__main__":
    unittest.main()
