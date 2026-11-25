"""RSL-RL command line arguments shared by ``train.py`` and ``play.py``."""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING


def apply_fixed_learning_rate(runner, algorithm_cfg):
    """A restored Adam state must not override the configured fixed schedule."""
    if algorithm_cfg.schedule == 'fixed':
        runner.alg.learning_rate = float(algorithm_cfg.learning_rate)
        for group in runner.alg.optimizer.param_groups:
            group['lr'] = runner.alg.learning_rate

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser."""
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name", type=str, default=None, help="Run name suffix to the log directory."
    )
    # -- load arguments
    arg_group.add_argument(
        "--resume", action="store_true", default=False, help="(Placeholder) Resume from a previous run."
    )
    arg_group.add_argument(
        "--load_run", type=str, default=None, help="(Placeholder) Run id/name to load for resuming."
    )
    arg_group.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file to load weights from."
    )
    arg_group.add_argument(
        "--resume_training",
        action="store_true",
        default=False,
        help=(
            "Resume the optimizer state and iteration counter from --checkpoint instead of only "
            "initialising the policy weights."
        ),
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )
    arg_group.add_argument(
        "--clip_actions",
        type=float,
        default=None,
        help="[RSL-RL] Override agent clip_actions before environment action scaling.",
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> "RslRlOnPolicyRunnerCfg":
    """Load and update the RSL-RL configuration registered for ``task_name``."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    rslrl_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(rslrl_cfg, args_cli)


def _adapt_rsl_rl_cfg(agent_cfg, rsl_rl_version: str):
    """Use Isaac Lab's migration helper when supplied by the installed release."""
    from importlib import import_module

    package = import_module("isaaclab_rl.rsl_rl")
    helper = getattr(package, "handle_deprecated_rsl_rl_cfg", None)
    if helper is None:
        try:
            utils = import_module("isaaclab_rl.rsl_rl.utils")
        except ModuleNotFoundError as error:
            # Older Isaac Lab releases have no utils module. Do not hide an
            # unrelated missing dependency inside an existing utils module.
            if error.name != "isaaclab_rl.rsl_rl.utils":
                raise
            return agent_cfg
        helper = getattr(utils, "handle_deprecated_rsl_rl_cfg", None)
    if helper is None:
        return agent_cfg
    return helper(agent_cfg, rsl_rl_version)


def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Update the RSL-RL configuration from the command line arguments."""
    from importlib.metadata import version

    rsl_rl_version = version("rsl-rl-lib")
    agent_cfg = _adapt_rsl_rl_cfg(agent_cfg, rsl_rl_version)
    # Isaac Lab main currently exposes this rsl-rl >= 4 option even when used
    # with rsl-rl 3.x; its compatibility helper does not remove it yet.
    if int(rsl_rl_version.split(".", maxsplit=1)[0]) < 4 and hasattr(
        agent_cfg.algorithm, "share_cnn_encoders"
    ):
        del agent_cfg.algorithm.share_cnn_encoders

    exp_name = getattr(args_cli, "experiment_name", None)
    if exp_name is not None and exp_name != "":
        agent_cfg.experiment_name = exp_name

    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed

    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    clip_actions = getattr(args_cli, "clip_actions", None)
    if clip_actions is not None:
        agent_cfg.clip_actions = clip_actions

    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"}:
        task_suffix = getattr(args_cli, "task", "") or ""
        if args_cli.log_project_name is not None:
            agent_cfg.wandb_project = args_cli.log_project_name
            agent_cfg.neptune_project = args_cli.log_project_name
        else:
            agent_cfg.wandb_project += "." + task_suffix
            agent_cfg.neptune_project += "." + task_suffix

    return agent_cfg
