"""Visualise a Mongo0909 environment with a zero-action (or random action) agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# make ``source/mongo_world`` / ``source/mongo_tasks`` importable without an
# editable install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mongo_bootstrap  # noqa: E402

mongo_bootstrap.ensure_source_packages(verbose=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab environment viewer.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--random", action="store_true", default=False, help="Sample random actions instead of zeros."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import mongo_world  # noqa: F401
import mongo_tasks  # noqa: F401


def main():
    """Run the environment with a fixed or random action agent."""
    if args_cli.task and args_cli.task.startswith("="):
        args_cli.task = args_cli.task[1:]
        print(f"[INFO] 已清理任务名称，去除前导 '=' 符号: {args_cli.task}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.random:
                actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            else:
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
