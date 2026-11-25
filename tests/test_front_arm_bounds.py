"""Static front-arm bounds and smooth training/Play target transitions."""
import ast
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from offline_support import ROOT, env_cfg, isolated_env_class


def make_env(count=1):
    env = isolated_env_class({'_compute_front_arm_command_upper', '_update_front_arm_commands',
                             '_rate_limit_front_arm_target', '_step_front_arm_command'})()
    env.cfg, env.step_dt = env_cfg(), .02
    env.height_cmd = torch.full((count,), .42)
    env.height_goal_cmd = env.height_cmd.clone()
    env.front_arm_target_cmd = torch.zeros(count)
    env.front_arm_goal_cmd = torch.zeros(count)
    env.front_arm_command_upper = torch.zeros(count)
    env.robot = SimpleNamespace(_ALL_INDICES=torch.arange(count))
    env.command = torch.zeros(count, 3)
    env.command_generator = SimpleNamespace(vel_command_b=torch.zeros(count, 3))
    return env


class FrontArmBoundsTests(unittest.TestCase):
    def test_margin_and_absolute_cap_at_requested_heights(self):
        env = make_env()
        self.assertEqual(env.cfg.front_arm_command_upper_margin_deg, 8.)
        self.assertEqual(env.cfg.front_arm_command_upper_limit_deg, 50.)
        height = torch.tensor([.1, .29, .355, .42, .6])
        actual = torch.rad2deg(env._compute_front_arm_command_upper(height))
        torch.testing.assert_close(actual, torch.tensor([-15.11, 2.0791, 25.8974, 50., 50.]),
                                   atol=1e-4, rtol=1e-5)

    def test_reset_sampling_uses_new_height_bounds(self):
        env = make_env(4096)
        env.height_cmd = torch.linspace(.29, .42, 4096)
        env._update_front_arm_commands(env.robot._ALL_INDICES, resample=True, snap_to_target=True)
        self.assertTrue((env.front_arm_target_cmd <= env.front_arm_command_upper).all())
        self.assertTrue((env.front_arm_target_cmd >= math.radians(-15.11)).all())
        self.assertTrue((env.front_arm_target_cmd <= math.radians(50.)).all())
        torch.testing.assert_close(env.front_arm_target_cmd, env.front_arm_goal_cmd)

    def test_training_goal_tightens_without_target_jump(self):
        env = make_env()
        env.front_arm_target_cmd[:] = env.front_arm_goal_cmd[:] = math.radians(50.)
        env.height_cmd[:] = .29
        env._update_front_arm_commands(env.robot._ALL_INDICES, resample=False)
        self.assertAlmostEqual(torch.rad2deg(env.front_arm_target_cmd).item(), 50., places=4)
        self.assertLess(torch.rad2deg(env.front_arm_goal_cmd).item(), 3.)
        for _ in range(100):
            old = env.front_arm_target_cmd.clone()
            env._rate_limit_front_arm_target()
            self.assertLessEqual((env.front_arm_target_cmd-old).abs().max().item(), math.radians(.6)+1e-7)
        torch.testing.assert_close(env.front_arm_target_cmd, env.front_arm_goal_cmd)

    def test_real_play_injection_and_mujoco_cap_compatibility(self):
        tree = ast.parse((ROOT/'scripts/rsl_rl/play.py').read_text())
        inject = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_inject_command')
        scope = {'torch':torch, 'math':math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[inject],type_ignores=[])), '<play>', 'exec'),scope)
        env = make_env()
        env.front_arm_target_cmd[:] = math.radians(50.)
        for i in range(200):
            old = env.front_arm_target_cmd.clone()
            scope['_inject_command'](env, 0., 0., 0., .29 if i%2 else .42, float(i%3!=0))
            self.assertLessEqual((env.front_arm_target_cmd-old).abs().max().item(), math.radians(.6)+1e-7)
            self.assertLessEqual(env.front_arm_target_cmd.max().item(), math.radians(50.)+1e-7)
        tree = ast.parse((ROOT/'mujoco/controller.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='front_upper')
        scope = {'np':np, 'math':math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])), '<mujoco>', 'exec'),scope)
        for h in [.29,.355,.42,.6]:
            actual = scope['front_upper'](h, vars(env.cfg))
            expected = env._compute_front_arm_command_upper(torch.tensor([h])).item()
            self.assertAlmostEqual(actual, expected, places=5)
        # Old archives have no absolute-cap field: their recorded formula stays usable.
        old = dict(vars(env.cfg)); old.pop('front_arm_command_upper_limit_deg'); old['front_arm_command_upper_margin_deg']=4.
        self.assertAlmostEqual(math.degrees(scope['front_upper'](.42, old)), 59.)


if __name__ == '__main__':
    unittest.main()
