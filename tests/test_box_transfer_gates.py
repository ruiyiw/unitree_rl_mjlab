"""GPU regression: the reference cannot lower before release or turn before arms down."""
import unittest
from unittest.mock import patch

import torch
from mjlab.envs import ManagerBasedRlEnv

from unitree_box_transfer.five_task import env_cfg
from unitree_box_transfer import task as tk


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU")
class PutdownGateTests(unittest.TestCase):
    def test_release_and_arm_lowering_are_required_in_order(self):
        torch.set_num_threads(1)
        env = ManagerBasedRlEnv(env_cfg("put-down",1,play=True),device="cuda:0")
        try:
            env.reset()
            t = tk.task(env)
            t.clock[:,0] = t.release_frame
            t.fresh[:] = False
            t.release_allowed[:] = False
            t.turn_allowed[:] = False
            zero = torch.zeros(1,device="cuda:0")
            no = zero.bool()
            yes = ~no
            with patch("unitree_box_transfer.five_task.robot_box_contact",return_value=zero), patch("unitree_box_transfer.task.r_prop_contact",return_value=zero):
                with patch.object(t,"released",return_value=no):
                    for _ in range(30):
                        t._update_command(None)
                    self.assertEqual(int(t.clock[0,0]),t.release_frame)
                with patch.object(t,"released",return_value=yes), patch.object(t,"arms_lowered",return_value=no):
                    for _ in range(30):
                        t._update_command(None)
                    self.assertTrue(bool(t.release_allowed[0]))
                    t.clock[:,0] = t.turn_frame
                    for _ in range(30):
                        t._update_command(None)
                    self.assertFalse(bool(t.turn_allowed[0]))
                    self.assertEqual(int(t.clock[0,0]),t.turn_frame)
                with patch.object(t,"released",return_value=yes), patch.object(t,"arms_lowered",return_value=yes):
                    for _ in range(30):
                        t._update_command(None)
                    self.assertTrue(bool(t.turn_allowed[0]))
                    self.assertGreater(int(t.clock[0,0]),t.turn_frame)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
