"""Check real Isaac Sim reset/step and Mongo observation/action contracts."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mongo_bootstrap

mongo_bootstrap.ensure_source_packages(verbose=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Mongo-Flat-v0')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import mongo_tasks  # noqa: F401


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    try:
        obs, _ = env.reset()
        assert cfg.action_space == 6
        assert obs['policy'].shape == (args.num_envs, cfg.observation_space)
        assert obs['critic'].shape == (args.num_envs, cfg.state_space)
        with torch.inference_mode():
            for _ in range(args.steps):
                actions = .1 * torch.randn(args.num_envs, 6, device=env.unwrapped.device)
                obs, reward, _, _, _ = env.step(actions)
                assert torch.isfinite(reward).all(), 'Nonfinite rewards'
                assert all(torch.isfinite(value).all() for value in obs.values()), 'Nonfinite observations'
                assert torch.isfinite(env.unwrapped.robot.data.joint_pos).all(), 'Nonfinite joints'
                assert not env.unwrapped._numerical_safety_reset_buf.any(), 'Numerical safety reset'
        print(f'PASS: {args.num_envs} robots, {args.steps} control steps, '
              f'{cfg.observation_space}/{cfg.state_space} observations, 6 actions')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        launcher.app.close()
