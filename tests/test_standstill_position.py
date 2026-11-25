"""Braking grace, fixed world-XY anchors and strict command/reset semantics."""
import unittest
import torch
from offline_support import FLAT, env_cfg, load_module
from test_baseline import rewards, sample_inputs
State = load_module('stop_state', FLAT / 'reward_state.py').CommandRewardState


class StandstillPositionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = env_cfg()
        self.cfg.stand_still_position_delay_s = .5
        self.cfg.stand_still_position_ramp_s = .5
        self.s = State(self.cfg, 4, 'cpu')
        self.cmd = torch.zeros(4, 3)
        self.pos = torch.zeros(4, 3)

    def step(self):
        return self.s.update_stop_position(self.cmd, self.pos, .125)

    def settle(self):
        for _ in range(9):
            result = self.step()
        return result

    def test_grace_captures_end_of_braking_then_anchor_stays_fixed(self):
        for _ in range(4):
            self.pos[:, 0] += .1  # large braking displacement must remain free
            error, gate = self.step()
            self.assertFalse(self.s.stop_anchor_valid.any())
            self.assertEqual(error.abs().sum().item(), 0.)
        self.pos[:, 0] = .5
        error, gate = self.step()  # .5 s since first zero sample
        self.assertTrue(self.s.stop_anchor_valid.all())
        torch.testing.assert_close(error, torch.zeros(4, 2))
        self.assertEqual(gate.sum().item(), 0.)
        self.pos[:, 0] = .6
        self.step()
        error, gate = self.step()  # .25 s fade-in => .5
        torch.testing.assert_close(gate, torch.full((4,), .5))
        torch.testing.assert_close(error[:, 0], torch.full((4,), .1))
        self.settle()
        torch.testing.assert_close(self.s.stop_anchor_xy[:, 0], torch.full((4,), .5))
        _, gate = self.step()
        torch.testing.assert_close(gate, torch.ones(4))

    def test_nonzero_any_axis_cancels_and_reentry_reanchors(self):
        self.settle()
        self.cmd[:3] = torch.eye(3) * 1e-8
        self.pos[:, 0] = 1.
        error, gate = self.step()
        self.assertFalse(self.s.stop_anchor_valid[:3].any())
        torch.testing.assert_close(error[:3], torch.zeros(3, 2))
        torch.testing.assert_close(gate, torch.tensor([0., 0., 0., 1.]))
        self.cmd.zero_()
        self.step()
        self.assertEqual(self.s.stop_age[:3].sum().item(), 0.)
        self.settle()
        torch.testing.assert_close(self.s.stop_anchor_xy[:, 0], torch.tensor([1., 1., 1., 0.]))

    def test_partial_reset_and_height_change(self):
        self.settle()
        self.pos[:, 2] = .45  # world Z does not affect horizontal position
        self.s.reset(torch.tensor([1]))
        error, gate = self.step()
        torch.testing.assert_close(gate, torch.tensor([1., 0., 1., 1.]))
        self.assertFalse(self.s.stop_anchor_valid[1])
        self.assertEqual(error.abs().sum().item(), 0.)
        self.pos[:, 0] = .1
        error, _ = self.step()
        torch.testing.assert_close(error[:, 0], torch.tensor([.1, 0., .1, .1]))

    def test_world_translation_invariance_and_no_speed_based_reanchor(self):
        other = State(self.cfg, 4, 'cpu')
        offset = torch.tensor([100., -40., 5.])
        for i in range(12):
            self.pos[:, 0] = i * .02
            error, gate = self.step()
            other_error, other_gate = other.update_stop_position(self.cmd, self.pos + offset, .125)
            torch.testing.assert_close(error, other_error, atol=5e-6, rtol=1e-4)
            torch.testing.assert_close(gate, other_gate)
        self.assertTrue((error[:, 0] > .1).all())  # moving forever does not disable it

    def test_position_cost_tolerance_radial_distance_gate_and_dt(self):
        # Retain explicit opt-in coverage; this term is disabled by default.
        self.cfg.rewards['stand_still_position'] = -.5
        x = sample_inputs(6)
        x['stand_still_position_error_xy'] = torch.tensor([[0.,0.],[.006,.008],[.06,0.],[.036,.048],[.11,0.],[.11,0.]])
        x['stand_still_position_gate'] = torch.tensor([1.,1.,1.,1.,.5,1.])
        x['command'][5, 2] = 1e-8  # even a tiny nonzero yaw disables position holding
        terms = rewards.compute_flat_reward_terms(self.cfg, **x)
        torch.testing.assert_close(terms['stand_still_position'], torch.tensor([0.,0.,1.,1.,2.,0.]), atol=1e-6, rtol=1e-5)
        for dt in (.02,.01):
            w = rewards.weight_reward_terms(terms, self.cfg.rewards, dt)
            torch.testing.assert_close(w['stand_still_position'], torch.tensor([0.,0.,1.,1.,2.,0.]) * -.5 * dt)

    def test_default_position_reward_is_zero_even_for_large_drift(self):
        self.assertEqual(self.cfg.rewards['stand_still_position'], 0.)
        x = sample_inputs(4)
        x['stand_still_position_error_xy'].fill_(20.)
        x['stand_still_position_gate'].fill_(1.)
        terms = rewards.compute_flat_reward_terms(self.cfg, **x)
        weighted = rewards.weight_reward_terms(terms, self.cfg.rewards, .02)
        torch.testing.assert_close(weighted['stand_still_position'], torch.zeros(4))


if __name__ == '__main__':
    unittest.main()
