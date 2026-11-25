"""Exercise the installed stock RSL-RL runner without a simulator."""
import copy
import importlib.util
import tempfile
from pathlib import Path
import unittest

import torch

from offline_support import env_cfg, runner_cfg


@unittest.skipUnless(importlib.util.find_spec('rsl_rl'), 'RSL-RL is not installed')
class PPOTrainingTests(unittest.TestCase):
    def test_stock_runner_update_checkpoint_and_inference(self):
        from tensordict import TensorDict
        from rsl_rl.algorithms import PPO
        from rsl_rl.runners import OnPolicyRunner

        task_cfg = env_cfg()

        class TensorEnv:
            num_envs, num_actions, max_episode_length, device = 16, 6, 100, 'cpu'
            cfg = {}

            def __init__(self):
                self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
                self.state = torch.randn(self.num_envs, 31) * .1
                self.history = self.state[:, None, :].repeat(1, task_cfg.num_obs_hist, 1)

            def get_observations(self):
                critic = torch.cat([self.state, torch.zeros(self.num_envs, 46)], dim=1)
                return TensorDict({'policy': self.history.flatten(1).clone(), 'critic': critic},
                                  batch_size=[self.num_envs])

            def step(self, actions):
                self.episode_length_buf += 1
                self.state[:, :6] = .9 * self.state[:, :6] + .1 * actions
                self.history = torch.roll(self.history, shifts=-1, dims=1)
                self.history[:, -1] = self.state
                reward = 1. - actions.square().mean(dim=1)
                return self.get_observations(), reward, torch.zeros(self.num_envs, dtype=torch.long), {}

        torch.set_num_threads(1)
        torch.manual_seed(7)
        cfg = runner_cfg()
        with tempfile.TemporaryDirectory() as directory:
            runner = OnPolicyRunner(TensorEnv(), copy.deepcopy(cfg), log_dir=directory, device='cpu')
            self.assertIs(type(runner.alg), PPO)
            self.assertEqual(tuple(runner.alg.policy.actor[0].weight.shape), (256, 155))
            self.assertEqual(tuple(runner.alg.policy.critic[0].weight.shape), (256, 77))
            before = [p.detach().clone() for p in runner.alg.policy.parameters()]
            runner.learn(num_learning_iterations=1)
            after = list(runner.alg.policy.parameters())
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
            self.assertTrue(all(torch.isfinite(p).all() for p in after))
            path = str(Path(directory) / 'model_0.pt')
            runner.save(path)
            restored = OnPolicyRunner(TensorEnv(), copy.deepcopy(cfg), device='cpu')
            restored.load(path, load_optimizer=True)
            obs = runner.env.get_observations()
            with torch.inference_mode():
                expected = runner.get_inference_policy()(obs)
                actual = restored.get_inference_policy()(obs)
            self.assertEqual(actual.shape, (16, 6))
            torch.testing.assert_close(actual, expected)
            runner.writer.close()


if __name__ == '__main__':
    unittest.main()
