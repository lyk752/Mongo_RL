"""Check propulsive cut-off, full braking, symmetry and shared backend parity."""
import json
import sys
from types import SimpleNamespace
import unittest

import torch
import numpy as np

from offline_support import ROOT, runner_cfg, load_module

sys.path.insert(0, str(ROOT / 'source/mongo_world'))
from mongo_world.motor_limits import speed_limited_effort


class MotorSpeedTests(unittest.TestCase):
    def test_all_motors_taper_drive_and_retain_braking_at_overspeed(self):
        profile = json.loads((ROOT / 'model/motor_config_24v.json').read_text())
        fraction = profile['speed_taper_start_fraction']
        for motor in profile['motors'].values():
            torque, limit = motor['effort_limit_nm'], motor['velocity_limit_rad_s']
            ratios = torch.tensor([0., .9, .95, 1., 1.2], dtype=torch.float64)
            vel = ratios * limit
            drive = torch.full_like(vel, 1e6)
            positive = speed_limited_effort(drive, vel, torch.full_like(vel, torque),
                                             torch.full_like(vel, limit), fraction, torch)
            torch.testing.assert_close(positive, torque * torch.tensor([1.,1.,.5,0.,0.], dtype=torch.float64))
            brake = speed_limited_effort(-drive, vel, torch.full_like(vel, torque),
                                         torch.full_like(vel, limit), fraction, torch)
            torch.testing.assert_close(brake, torch.full_like(vel, -torque))
            reverse = speed_limited_effort(-drive, -vel, torch.full_like(vel, torque),
                                           torch.full_like(vel, limit), fraction, torch)
            torch.testing.assert_close(reverse, -positive)
            # Zero torque remains zero even under externally imposed overspeed.
            torch.testing.assert_close(speed_limited_effort(vel*0., vel, vel*0.+torque,
                                       vel*0.+limit, fraction, torch), vel*0.)

    def test_numpy_and_torch_match_for_randomized_batch(self):
        rng = np.random.default_rng(42)
        effort = rng.uniform(-100, 100, (128,7))
        velocity = rng.uniform(-120, 120, (128,7))
        limit = np.array([4.712389]*4 + [33.510322,74.351026,74.351026])
        maximum = np.array([25.]*4 + [4.,2.42,2.42])
        expected = speed_limited_effort(effort, velocity, maximum, limit, .9, np)
        actual = speed_limited_effort(*(torch.from_numpy(x) for x in (effort, velocity, maximum, limit)), .9, torch)
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)

    def test_fixed_lr_overrides_restored_optimizer_rate(self):
        cli = load_module('motor_test_cli', ROOT/'scripts/rsl_rl/cli_args.py')
        cfg = runner_cfg()['algorithm']
        self.assertEqual((cfg['schedule'],cfg['learning_rate']), ('fixed',1e-4))
        runner = SimpleNamespace(alg=SimpleNamespace(learning_rate=.01,
            optimizer=SimpleNamespace(param_groups=[{'lr':.01}, {'lr':.005}])))
        cli.apply_fixed_learning_rate(runner, SimpleNamespace(**cfg))
        self.assertEqual(runner.alg.learning_rate, 1e-4)
        self.assertEqual([x['lr'] for x in runner.alg.optimizer.param_groups], [1e-4,1e-4])


if __name__ == '__main__':
    unittest.main()
