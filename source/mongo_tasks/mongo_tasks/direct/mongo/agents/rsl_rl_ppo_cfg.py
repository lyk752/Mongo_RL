# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
"""Stock PPO with a fixed learning rate for the 24V motor training plant."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MongoFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "mongo_flat_direct"
    empirical_normalization = False
    # The environment returns a privileged "critic" observation stream as well.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        """Fill in fields that only exist on newer Isaac Lab versions.

        Isaac Lab >= 2.2 requires the observation normalization flags on the
        policy configuration, while older releases (e.g. 2.1) do not define
        them at all.  Setting them defensively keeps this configuration
        portable across both.
        """
        for field_name in ("actor_obs_normalization", "critic_obs_normalization"):
            if hasattr(self.policy, field_name) and getattr(self.policy, field_name) is MISSING:
                setattr(self.policy, field_name, False)


__all__ = ["MongoFlatPPORunnerCfg"]
