"""Train a Mongo0909 RL agent with RSL-RL (plain PPO)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

# make ``source/mongo_world`` / ``source/mongo_tasks`` importable without an
# editable install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mongo_bootstrap  # noqa: E402

mongo_bootstrap.ensure_source_packages(verbose=True)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from log_recovery import learn_with_log_recovery

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length", type=int, default=200, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--resume_same_log_dir",
    action="store_true",
    default=False,
    help=(
        "When used with --resume_training, append logs and checkpoints to the directory "
        "containing --checkpoint instead of creating a timestamped run directory."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import warnings
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.runners import OnPolicyRunner

import mongo_world  # noqa: F401
import mongo_tasks  # noqa: F401
# silence the very noisy quat_rotate deprecation warning from isaaclab.utils.math
warnings.filterwarnings("ignore", message=".*quat_rotate.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="isaaclab.utils.math")
logging.getLogger("isaaclab.utils.math").setLevel(logging.ERROR)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _load_policy_for_finetune(runner, checkpoint_path: str) -> None:
    """Require the baseline architecture, then load only policy weights."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
    try:
        runner.alg.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint does not match the configured observation dimensions and network "
            "(default: 155/77 inputs, [256, 128, 64] hidden layers). "
            "Train from scratch or use a checkpoint produced by this library."
        ) from error
    runner.current_learning_iteration = 0


def _attach_checkpoint_metadata(env_cfg, agent_cfg, checkpoint_path: str | None) -> None:
    """Record CLI checkpoint provenance in the dumped training YAML files."""
    if checkpoint_path is None:
        return
    checkpoint_path = os.path.abspath(checkpoint_path)
    metadata = {
        "path": checkpoint_path,
        "dir": os.path.dirname(checkpoint_path),
        "file": os.path.basename(checkpoint_path),
        "resume_training": bool(args_cli.resume_training),
    }
    setattr(env_cfg, "launch_checkpoint", metadata)
    setattr(agent_cfg, "launch_checkpoint", metadata)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with a plain PPO RSL-RL agent."""
    if args_cli.resume_training and args_cli.checkpoint is None:
        raise ValueError("--resume_training requires --checkpoint")
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    if args_cli.resume_same_log_dir:
        if not args_cli.resume_training or args_cli.checkpoint is None:
            raise ValueError(
                "--resume_same_log_dir requires both --resume_training and --checkpoint"
            )
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        log_dir = os.path.dirname(checkpoint_path)
        print(f"[INFO] Continuing logs in checkpoint directory: {log_dir}")
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.resume_training:
        from play_config import resolve_joint_velocity_source
        feedback_resolution = resolve_joint_velocity_source(env_cfg, args_cli.checkpoint)
        print(f"[INFO] Resuming recorded joint velocity feedback: {feedback_resolution}")

    # create the Isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap the environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create the runner (plain PPO -> RSL-RL's stock OnPolicyRunner)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if hasattr(runner, "add_git_repo_to_log"):
        runner.add_git_repo_to_log(__file__)

    # optionally load a checkpoint
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.resume_training:
            load_signature = inspect.signature(runner.load)
            if "load_iteration" in load_signature.parameters:
                checkpoint_infos = runner.load(
                    resume_path, load_optimizer=True, load_iteration=True
                )
            else:
                checkpoint_infos = runner.load(resume_path, load_optimizer=True)
            runner.current_learning_iteration += 1
            print(f"[INFO]: Resuming training from next iteration: {runner.current_learning_iteration}")
        else:
            _load_policy_for_finetune(runner, resume_path)

    cli_args.apply_fixed_learning_rate(runner, agent_cfg.algorithm)
    _attach_checkpoint_metadata(env_cfg, agent_cfg, args_cli.checkpoint)

    # dump the configuration into the log directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    from mongo_tasks.direct.mongo.flat.physics_snapshot import runtime_snapshot
    with open(os.path.join(log_dir, "params", "physics_env0.json"), "w", encoding="utf-8") as stream:
        json.dump(runtime_snapshot(env.unwrapped), stream, indent=2)

    # run training
    try:
        learn_with_log_recovery(
            runner, iterations=agent_cfg.max_iterations, log_dir=log_dir,
            recovery_root=Path(__file__).resolve().parents[2] / "outputs/training_recovery",
            env_cfg=env_cfg, agent_cfg=agent_cfg, dump_yaml=dump_yaml,
        )
    except BaseException:
        try:
            env.close()
        except Exception as close_error:
            warnings.warn(f"Environment cleanup failed after training error: {close_error}")
        raise
    else:
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
