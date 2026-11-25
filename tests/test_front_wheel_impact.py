"""Light caster contact is free; hard contact costs grow without command gating."""
import ast
import unittest
import torch
from offline_support import FLAT, env_cfg
from test_baseline import rewards, sample_inputs


class FrontWheelImpactTests(unittest.TestCase):
    def test_light_contact_threshold_quadratic_and_time_scale(self):
        cfg, x = env_cfg(), sample_inputs(7)
        x['front_wheel_contact_peak'] = torch.tensor([0., 20., 50., 100., 150., 500., 1100.])
        terms = rewards.compute_flat_reward_terms(cfg, **x)
        expected = torch.tensor([0., 0., 0., .25, 1., 20.25, 110.25])
        torch.testing.assert_close(terms['front_wheel_impact'], expected)
        for dt in (.02, .01):
            weighted = rewards.weight_reward_terms(terms, cfg.rewards, dt)
            torch.testing.assert_close(weighted['front_wheel_impact'], -dt * expected)
        masked = rewards.weight_reward_terms(terms, cfg.rewards, .02, torch.ones(7, dtype=torch.bool))
        self.assertEqual(masked['front_wheel_impact'].abs().sum().item(), 0.)

    def test_impact_not_disabled_by_motion_or_transition(self):
        cfg, x = env_cfg(), sample_inputs()
        x['front_wheel_contact_peak'][:] = 150.
        x['command'][:, 0] = torch.tensor([0., 1., -1., 0.])
        x['command'][3, 2] = 2.
        x['command_steady_gate'] = torch.tensor([0., 1., 0., 1.])
        x['motion_steady_gate'] = x['command_steady_gate']
        torch.testing.assert_close(rewards.compute_flat_reward_terms(cfg, **x)['front_wheel_impact'], torch.ones(4))

    def test_contact_history_covers_control_period(self):
        tree = ast.parse((FLAT / 'env_cfg.py').read_text())
        sensor = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == 'ContactSensorCfg')
        kwargs = {k.arg: ast.literal_eval(k.value) for k in sensor.keywords}
        self.assertGreaterEqual(kwargs['history_length'], env_cfg().decimation)
        self.assertEqual(kwargs['update_period'], .002)


if __name__ == '__main__':
    unittest.main()
