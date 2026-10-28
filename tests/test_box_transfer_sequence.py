"""GPU regressions for continuous phase switching and actor observations."""
import unittest
import torch
from mjlab.envs import ManagerBasedRlEnv
from unitree_box_transfer.five_sequence import sequence_cfg, CompositePolicy, phase_timeout
from unitree_box_transfer import task as tk


@unittest.skipUnless(torch.cuda.is_available(),'requires GPU')
class SequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env=ManagerBasedRlEnv(sequence_cfg(2),device='cuda:0')

    @classmethod
    def tearDownClass(cls):cls.env.close()

    def setUp(self):self.obs,_=self.env.reset()

    def test_handoffs_preserve_physics_and_action_history(self):
        # Nonzero history catches accidental clearing that a reset-only test misses.
        self.env.step(torch.full((2,23),.1,device='cuda:0'))
        t=tk.task(self.env)
        q=self.env.sim.data.qpos.clone();v=self.env.sim.data.qvel.clone()
        a=self.env.action_manager.action.clone()
        self.assertTrue(a.abs().sum()>0)
        ids=torch.tensor([0],device='cuda:0')
        for phase in range(1,5):
            t.advance(ids)
            self.assertEqual(t.plan_ids.tolist(),[phase,0])
            self.assertTrue(torch.equal(q,self.env.sim.data.qpos))
            self.assertTrue(torch.equal(v,self.env.sim.data.qvel))
            self.assertTrue(torch.equal(a,self.env.action_manager.action))
            self.assertEqual(t.clock[0,0].item(),0)

    def test_navigation_actor_is_shared_and_uses_native_observation_shape(self):
        p=CompositePolicy(self.env,self.obs)
        self.assertEqual(set(p.actors),{'navigation','take','carry','put-down'})
        actual=p(self.obs)
        expected=p.actors['navigation'](self.obs['navigation'])
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(tuple(actual.shape),(2,23))
        # Manipulation observations extend the native navigation prefix.
        n=self.obs['navigation'].shape[-1]
        self.assertTrue(torch.equal(self.obs['navigation'],self.obs['manipulation'][:,:n]))

    def test_single_bundle_matches_all_four_frozen_actors(self):
        from unitree_box_transfer.policy import BundledPolicy
        original=CompositePolicy(self.env,self.obs)
        bundled=BundledPolicy(self.env)
        t=tk.task(self.env)
        for stage in range(5):
            t.plan_ids[:]=stage
            tk._invalidate(self.env)
            obs=self.env.observation_manager.compute()
            self.assertTrue(torch.equal(original(obs),bundled(obs)),stage)

    def test_completed_return_can_hold_beyond_phase_deadline(self):
        t=tk.task(self.env)
        t.plan_ids[:]=4;t.phase_elapsed[:]=100.
        t.completed[0,4]=True
        self.assertEqual(phase_timeout(self.env).tolist(),[False,True])

    def test_completion_does_not_advance_before_reference_end(self):
        t=tk.task(self.env)
        t.clock[:]=0;t.fresh[:]=False
        for _ in range(5):t._update_command(None)
        self.assertEqual(t.plan_ids.tolist(),[0,0])
        self.assertFalse(t.completed.any())


if __name__=='__main__':unittest.main()
