"""Command transitions, geometric leg rates and actual-body standstill costs."""
import ast
import unittest

import torch

from offline_support import ROOT, FLAT, env_cfg, load_module
from test_baseline import rewards, sample_inputs

State = load_module('reward_state', FLAT / 'reward_state.py').CommandRewardState


class RewardStateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = env_cfg()
        self.state = State(self.cfg, 4, 'cpu')
        self.inputs = sample_inputs()
        self.arm = torch.zeros(4)

    def update(self):
        x = self.inputs
        return self.state.update(x['command'], x['wheel_pos_b'], .02)

    def settle(self):
        for _ in range(60):
            result = self.update()
        return result

    def test_step_grace_smooth_recovery_and_partial_reset(self):
        steady, motion, rate = self.settle()
        torch.testing.assert_close(steady, torch.ones(4))
        self.inputs['command'][0, 0] = -.8
        self.inputs['command'][1, 2] = 1.
        steady, motion, _ = self.update()
        torch.testing.assert_close(steady, torch.tensor([0., 0., 1., 1.]))
        for _ in range(20):
            steady, _, _ = self.update()
        self.assertLess(steady[0].item(), 1e-10)
        for _ in range(15):
            steady, _, _ = self.update()
        self.assertAlmostEqual(steady[0].item(), .5, places=5)
        self.settle()
        self.state.reset(torch.tensor([1]))
        steady, _, rate = self.update()
        torch.testing.assert_close(steady, torch.tensor([1., 0., 1., 1.]))
        self.assertEqual(rate[1].abs().sum().item(), 0.)

    def test_only_velocity_ramps_hold_transition_height_and_arm_do_not(self):
        self.settle()
        for _ in range(70):
            self.inputs['command'][0, 0] += .002  # .1 m/s2
            self.inputs['height_cmd'][1] += .0006  # .03 m/s
            self.arm[2] += .01  # .5 rad/s
            self.inputs['command'][3, 2] += .004  # .2 rad/s2
            steady, motion, _ = self.update()
        torch.testing.assert_close(steady, torch.tensor([0., 1., 1., 0.]))
        torch.testing.assert_close(motion, steady)
        steady, _, _ = self.settle()
        torch.testing.assert_close(steady, torch.ones(4))
        self.inputs['command'][1, 1] = .1
        steady, motion, _ = self.update()
        torch.testing.assert_close(steady, torch.tensor([1., 0., 1., 1.]))
        torch.testing.assert_close(motion, steady)

    def test_angle_wrap_first_sample_and_opposite_legs_do_not_cancel(self):
        angles = torch.tensor([[3.13, -3.13]]).repeat(4, 1)
        def set_angles():
            self.inputs['wheel_pos_b'][..., 0] = -.3 * angles.sin()
            self.inputs['wheel_pos_b'][..., 2] = -.3 * angles.cos()
        set_angles()
        _, _, rate = self.update()
        self.assertEqual(rate.abs().sum().item(), 0.)
        angles[:, 0] = -3.13
        angles[:, 1] = 3.13
        set_angles()
        _, _, rate = self.update()
        expected = torch.tensor([(2 * torch.pi - 6.26) / .02, -(2 * torch.pi - 6.26) / .02])
        torch.testing.assert_close(rate, expected.repeat(4, 1), atol=2e-5, rtol=2e-5)
        self.inputs['virtual_leg_pitch_rate'] = rate
        terms = rewards.compute_flat_reward_terms(self.cfg, **self.inputs)
        self.assertTrue((terms['virtual_leg_pitch_rate'] > 2.).all())
        self.state.reset(torch.tensor([0]))
        angles += .1
        set_angles()
        _, _, rate = self.update()
        self.assertEqual(rate[0].abs().sum().item(), 0.)
        torch.testing.assert_close(rate[1:], torch.full((3, 2), 5.), atol=2e-5, rtol=2e-5)

    def test_orientation_full_strength_independent_of_phase(self):
        x = self.inputs
        x['projected_gravity_b'][:, :2] = torch.tensor([.15, -.2])
        steady = rewards.compute_flat_reward_terms(self.cfg, **x)
        x['command_steady_gate'] = torch.tensor([0., .25, .5, 1.])
        transition = rewards.compute_flat_reward_terms(self.cfg, **x)
        for axis in ('y', 'x'):
            for suffix in ('', '_v', '_exp'):
                key = f'flat_orientation_{axis}{suffix}'
                torch.testing.assert_close(transition[key], steady[key])
        for key in ('ang_vel_xy', 'termination', 'track_lin_vel_xy'):
            torch.testing.assert_close(transition[key], steady[key])

    def test_leg_cost_only_steady_tolerance_and_dt(self):
        x = self.inputs
        x['command_steady_gate'] = torch.tensor([0., .5, 1., 1.])
        x['virtual_leg_pitch_rate'] = torch.tensor([[1.1, -1.1], [1.1, -1.1], [1.1, -1.1], [.1, -.09]])
        terms = rewards.compute_flat_reward_terms(self.cfg, **x)
        torch.testing.assert_close(terms['virtual_leg_pitch_rate'], torch.tensor([0., 1., 2., 0.]))
        weighted = rewards.weight_reward_terms(terms, self.cfg.rewards, .02)
        torch.testing.assert_close(weighted['virtual_leg_pitch_rate'], torch.tensor([0., -.002, -.004, 0.]))

    def test_body_drift_strict_gate_no_speed_tolerance_or_wheel_dependency(self):
        x = sample_inputs(8)
        x['command'] = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                                     [1e-8, 0., 0.], [0., 0., 1e-8], [0., 1e-8, 0.],
                                     [0., 0., 1.], [.09, 0., 0.]])
        x['root_lin_vel_b'][:, :2] = torch.tensor([.1, -.1])
        x['root_ang_vel_b'][:, 2] = .2
        x['root_lin_vel_b'][2, :2] = .001  # tiny drift still costs
        x['motion_steady_gate'][1] = 0.
        # Height/arm transients must not switch off strict-zero body costs.
        x['command_steady_gate'][:] = 0.
        for enabled in (True, False):
            self.cfg.stand_still_deadzone_enabled = enabled
            terms = rewards.compute_flat_reward_terms(self.cfg, **x)
            torch.testing.assert_close(terms['stand_still_body_lin_vel'],
                                       torch.tensor([2., .5, .0002, 0., 2., 0., 2., 0.]))
            torch.testing.assert_close(terms['stand_still_body_ang_vel'],
                                       torch.tensor([1., .25, 1., 0., 0., 0., 0., 0.]))
            self.assertEqual(terms['stand_still_wheel_vel'].sum().item(), 0.)
            weighted = rewards.weight_reward_terms(terms, self.cfg.rewards, .02)
            self.assertAlmostEqual(weighted['stand_still_body_lin_vel'][0].item(), -.02)

    def test_invalid_phase_parameters_rejected(self):
        for name, value in [('stand_still_position_ramp_s', 0.), ('stand_still_position_scale_m', 0.),
                            ('stand_still_position_delay_s', -1.), ('stand_still_position_tolerance_m', -1.),
                            ('command_steady_ramp_s', 0.), ('command_steady_delay_s', -.1),
                            ('front_wheel_impact_force_threshold_n', -1.),
                            ('front_wheel_impact_force_scale_n', 0.),
                            ('front_wheel_impact_force_threshold_n', float('inf')),
                            ('stand_still_body_lin_vel_scale_m_s', float('nan'))]:
            cfg = env_cfg()
            setattr(cfg, name, value)
            # The helper returns a namespace: execute the real post-init AST.
            tree = ast.parse((FLAT / 'env_cfg.py').read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MongoFlatEnvCfg')
            post = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
            with self.assertRaises(ValueError):
                scope = {}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[post], type_ignores=[])), '<post_init>', 'exec'), scope)
                scope['__post_init__'](cfg)

    def test_default_solver_tgs_12_1_external_forces(self):
        asset = ast.parse((ROOT / 'source/mongo_world/mongo_world/assets/mongo.py').read_text())
        scene = ast.parse((FLAT / 'env_cfg.py').read_text())
        def keywords(tree, call_name):
            call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, (ast.Name, ast.Attribute))
                        and (n.func.id if isinstance(n.func, ast.Name) else n.func.attr) == call_name)
            return {k.arg: k.value for k in call.keywords}
        articulation = keywords(asset, 'ArticulationRootPropertiesCfg')
        physx = keywords(scene, 'PhysxCfg')
        self.assertEqual(ast.literal_eval(articulation['solver_position_iteration_count']), 12)
        self.assertEqual(ast.literal_eval(articulation['solver_velocity_iteration_count']), 1)
        self.assertEqual(ast.literal_eval(physx['solver_type']), 1)
        self.assertEqual(ast.unparse(physx[None]), 'external_force_iteration_kwargs(PhysxCfg)')


if __name__ == '__main__':
    unittest.main()
