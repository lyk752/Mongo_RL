"""Identical encoder semantics across Torch training and NumPy deployment."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from offline_support import FLAT, ROOT, env_cfg, isolated_env_class, load_module

torch_velocity = load_module('torch_feedback', FLAT / 'velocity_feedback.py').encoder_velocity
numpy_velocity = load_module('numpy_feedback', ROOT / 'mujoco/feedback.py').encoder_velocity


class FeedbackTests(unittest.TestCase):
    def test_backend_parity_and_wheel_wrap(self):
        previous = np.array([[0., 0., .1, .1, .2, np.pi - .01, -np.pi + .02]])
        position = previous + [[.005, -.005, .01, -.01, .02, -2*np.pi+.03, 2*np.pi-.04]]
        result = numpy_velocity(position, previous, .005)
        np.testing.assert_allclose(result, [[1, -1, 2, -2, 4, 6, -8]], atol=1e-12)
        np.testing.assert_allclose(result, torch_velocity(torch.tensor(position), torch.tensor(previous), .005))
        # Bounded leg hinges must not wrap away a real discontinuity.
        position[0, 0] += 2*np.pi
        self.assertGreater(numpy_velocity(position, previous, .005)[0, 0], 1000)

    def test_new_flat_training_defaults_to_encoder_feedback(self):
        cfg = env_cfg()
        self.assertEqual(cfg.joint_velocity_source, 'position_delta')

    def test_fresh_buffers_and_partial_reset_use_native_initial_velocity(self):
        cls = isolated_env_class(['_sample_joint_feedback'])
        env = cls()
        env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005))
        env._actuate_idx = list(range(7))
        env.joint_pos = torch.full((2, 7), -99.)  # stale constructor alias
        env.joint_vel = torch.full((2, 7), -99.)
        data = SimpleNamespace(joint_pos=torch.full((2, 7), .1), joint_vel=torch.full((2, 7), 3.))
        env.robot = SimpleNamespace(data=data)
        env._encoder_previous = torch.full((2, 7), .095)
        env._encoder_valid = torch.tensor([True, False])
        env._encoder_velocity = torch.zeros(2, 7)
        env._sample_joint_feedback()
        torch.testing.assert_close(env._encoder_velocity[0], torch.ones(7), atol=2e-5, rtol=0)
        torch.testing.assert_close(env._encoder_velocity[1], torch.full((7,), 3.))
        # Replacing the source tensor must be observed on the next physics tick.
        data.joint_pos = torch.full((2, 7), .11)
        env._sample_joint_feedback()
        torch.testing.assert_close(env._encoder_velocity, torch.full((2, 7), 2.), atol=2e-5, rtol=0)

    def test_partial_reset_retains_other_robots_sensor_phase(self):
        cls = isolated_env_class(['_reset_sensor_feedback'])
        env = cls()
        env._actuate_idx = list(range(7))
        for name, size in [('obs_joint_pos', 7), ('obs_joint_vel', 7),
                           ('obs_root_ang_vel_b', 3), ('obs_projected_gravity_b', 3)]:
            setattr(env, name, torch.full((2, size), 17.))
        env._obs_buffers_valid = False
        env._reset_sensor_feedback(torch.tensor([1]), torch.ones(1, 7), torch.zeros(1, 7),
                                   torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]))
        self.assertTrue(env._obs_buffers_valid)
        for name in ['obs_joint_pos', 'obs_joint_vel', 'obs_root_ang_vel_b', 'obs_projected_gravity_b']:
            self.assertTrue(torch.all(getattr(env, name)[0] == 17.))
        torch.testing.assert_close(env.obs_joint_vel[1], torch.zeros(7))


if __name__ == '__main__':
    unittest.main()
