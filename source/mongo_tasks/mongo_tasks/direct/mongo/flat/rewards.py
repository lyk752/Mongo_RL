# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
"""Wheelbipe V14 flat-v0 reward equations, adapted to Mongo motor indices.

Source: wheeled-legged_RL, wheelbipe25_v3/env.py::_get_rewards and
wheelbipe_V14/env_cfg.py::WheelbipeV14FlatEnvCfg. This module runs on CPU.
The directly commanded front arm is excluded from motor regularizers. Mongo
adds front-wheel impact, steady-leg costs and strict standstill costs.
"""

from __future__ import annotations

import torch


def _tracking_error_square(error: torch.Tensor, constraint: float | None) -> torch.Tensor:
    if constraint is not None:
        error = torch.clamp(error, -constraint, constraint)
    return torch.square(error)


def compute_flat_reward_terms(
    cfg, *, command, root_lin_vel_b, root_ang_vel_b, projected_gravity_b,
    pitch, height, height_cmd, wheel_pos_b, joint_acc, joint_vel, joint_pos,
    applied_torque, leg_idx, wheel_idx, actions, previous_actions,
    before_previous_actions, terminated, undesired_contact, wheel_contact_peaks,
    stand_still_position_error_xy, stand_still_position_gate,
    front_wheel_contact_peak, command_steady_gate, motion_steady_gate, virtual_leg_pitch_rate,
) -> dict[str, torch.Tensor]:
    """Return flat-v0 terms and Mongo's task-specific costs."""
    leg_dim = len(leg_idx)
    policy_motor_idx = list(leg_idx) + list(wheel_idx)
    linear_mask = torch.zeros_like(height, dtype=torch.bool)
    yaw_mask = torch.zeros_like(linear_mask)
    tracking_command = command
    if cfg.stand_still_deadzone_enabled:
        threshold = max(float(cfg.stand_still_deadzone_threshold), 0.0)
        linear_mask = torch.abs(command[:, 0]) < threshold
        yaw_mask = torch.abs(command[:, 2]) < threshold
        tracking_command = command.clone()
        tracking_command[linear_mask, 0] = 0.0
        tracking_command[yaw_mask, 2] = 0.0

    lin_err = tracking_command[:, 0] - root_lin_vel_b[:, 0] * torch.cos(pitch)
    yaw_err = tracking_command[:, 2] - root_ang_vel_b[:, 2]
    height_err = height - height_cmd
    lin_error_sq = _tracking_error_square(lin_err, cfg.lin_vel_err_constraint)
    yaw_error_sq = _tracking_error_square(yaw_err, cfg.ang_vel_err_constraint)
    height_error_sq = _tracking_error_square(height_err, cfg.height_err_constraint)
    pgb = projected_gravity_b[:, :2]
    speed_cmd_sq = torch.square(command[:, 0])
    action_second_diff = actions - 2 * previous_actions + before_previous_actions
    wheel_vel = joint_vel[:, wheel_idx]
    # Check the raw command, independently of the tracking deadzone above.
    # Any nonzero vx, vy or yaw command disables this additional cost.
    zero_velocity_command = (command == 0.0).all(dim=1)
    zero_linear_command = (command[:, :2] == 0.0).all(dim=1)
    stand_still_wheel_cost = torch.square(torch.clamp(
        torch.abs(wheel_vel) - cfg.stand_still_wheel_vel_deadzone_rad_s, min=0.0
    )).sum(dim=1)
    wheel_count = min(wheel_contact_peaks.shape[1], wheel_vel.shape[1])
    both_contact = (wheel_contact_peaks > cfg.desired_contact_force_threshold).all(dim=1)
    if wheel_count < 2:
        both_contact = torch.zeros_like(linear_mask)
    wheel_air_spin = torch.sum(
        torch.square(torch.nan_to_num(wheel_vel[:, :wheel_count], nan=0.0, posinf=0.0, neginf=0.0))
        * (wheel_contact_peaks[:, :wheel_count] <= cfg.desired_contact_force_threshold).float(), dim=1
    )
    wheel_diff_x = wheel_pos_b[:, 0, 0] - wheel_pos_b[:, 1, 0]
    wheel_diff_z = wheel_pos_b[:, 0, 2] - wheel_pos_b[:, 1, 2]
    # Explicit leg order: [left hip, right hip, left parallel_1, right parallel_1].
    pair_delta = joint_pos[:, leg_idx[::2]] - joint_pos[:, leg_idx[1::2]]
    pair_delta = torch.remainder(pair_delta + torch.pi, 2 * torch.pi) - torch.pi
    stop_scale = (cfg.stand_still_body_transition_scale
                 + (1.0 - cfg.stand_still_body_transition_scale) * motion_steady_gate)
    # Preserve both common-mode and differential leg oscillation; averaging
    # signed rates first would let opposite leg motions cancel each other.
    leg_rate_excess = torch.clamp(
        virtual_leg_pitch_rate.abs() - cfg.virtual_leg_pitch_rate_deadzone_rad_s, min=0.0)
    return {
        "termination": terminated.float(),
        "leg_joint_acc": torch.sum(torch.square(joint_acc[:, leg_idx]), dim=1),
        "leg_joint_vel": torch.sum(torch.square(joint_vel[:, leg_idx]), dim=1),
        "leg_joint_pair_pos_diff": torch.sum(torch.square(pair_delta), dim=1),
        "joint_torque": torch.sum(torch.square(applied_torque[:, policy_motor_idx]), dim=1),
        "wheel_acc": torch.sum(torch.square(joint_acc[:, wheel_idx]), dim=1),
        "wheel_vel": torch.sum(torch.square(wheel_vel), dim=1),
        "wheel_power": torch.sum(torch.clamp(applied_torque[:, wheel_idx] * wheel_vel, min=0), dim=1),
        "wheel_air_spin": wheel_air_spin,
        "lin_vel_z": torch.square(root_lin_vel_b[:, 2]),
        "ang_vel_xy": torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1),
        "action_rate": torch.sum(torch.square(actions - previous_actions), dim=1),
        "action_smoothness_leg": torch.sum(torch.square(action_second_diff[:, :leg_dim]), dim=1),
        "action_smoothness_wheel": torch.sum(torch.square(action_second_diff[:, leg_dim:]), dim=1),
        "flat_orientation_y": torch.square(cfg.orientation_y_square_sigma * pgb[:, 0]),
        "flat_orientation_y_v": torch.square((cfg.orientation_y_A * torch.exp(
            -speed_cmd_sq / cfg.orientation_y_sigma) + cfg.orientation_y_bias) * pgb[:, 0]),
        "flat_orientation_y_exp": torch.exp(-torch.square(pgb[:, 0]) / cfg.orientation_y_exp_sigma),
        "flat_orientation_x": torch.square(cfg.orientation_x_square_sigma * pgb[:, 1]),
        "flat_orientation_x_v": torch.square((cfg.orientation_x_A * torch.exp(
            -speed_cmd_sq / cfg.orientation_x_sigma) + cfg.orientation_x_bias) * pgb[:, 1]),
        "flat_orientation_x_exp": torch.exp(-torch.square(pgb[:, 1]) / cfg.orientation_x_exp_sigma),
        "track_lin_vel_xy": torch.exp(-lin_error_sq / cfg.lin_vel_xy_sigma),
        "track_lin_vel_xy_tight": torch.exp(-lin_error_sq / cfg.lin_vel_xy_tight_sigma),
        "track_lin_vel_xy_square": torch.square(lin_err * cfg.lin_vel_xy_square_sigma),
        "track_ang_vel_z": torch.exp(-yaw_error_sq / cfg.ang_vel_z_sigma),
        "track_ang_vel_z_square": torch.square(yaw_err * cfg.ang_vel_z_square_sigma),
        "stand_still_lin_vel": torch.sum(torch.abs(root_lin_vel_b[:, :2]), dim=1) * linear_mask.float(),
        "stand_still": (torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1) * linear_mask.float()
                        + torch.square(root_ang_vel_b[:, 2]) * yaw_mask.float()),
        "stand_still_wheel_vel": torch.where(
            zero_velocity_command, stand_still_wheel_cost, torch.zeros_like(height)
        ),
        "stand_still_body_lin_vel": torch.where(
            zero_linear_command,
            stop_scale * (root_lin_vel_b[:, :2] / cfg.stand_still_body_lin_vel_scale_m_s).square().sum(dim=1),
            torch.zeros_like(height),
        ),
        "stand_still_body_ang_vel": torch.where(
            zero_velocity_command,
            stop_scale * (root_ang_vel_b[:, 2] / cfg.stand_still_body_ang_vel_scale_rad_s).square(),
            torch.zeros_like(height),
        ),
        "stand_still_position": torch.where(
            zero_velocity_command,
            stand_still_position_gate * (
                (torch.linalg.vector_norm(stand_still_position_error_xy, dim=1)
                 - cfg.stand_still_position_tolerance_m).clamp(min=0.0)
                / cfg.stand_still_position_scale_m).square(),
            torch.zeros_like(height),
        ),
        "virtual_leg_pitch_rate": command_steady_gate * (
            leg_rate_excess / cfg.virtual_leg_pitch_rate_scale_rad_s).square().sum(dim=1),
        "track_height_exp": torch.exp(-height_error_sq / cfg.height_sigma),
        "track_height_exp_soft": torch.exp(-height_error_sq / cfg.height_soft_sigma),
        "track_height_exp_tight": torch.exp(-height_error_sq / cfg.height_tight_sigma),
        "track_height_square": torch.square(cfg.height_square_sigma * height_err),
        "track_height_exp_both_wheels_contact": torch.exp(-height_error_sq / cfg.height_soft_sigma) * both_contact.float(),
        "no_fork": (torch.abs(wheel_diff_x) > cfg.no_fork_distance).float(),
        "no_fork_square": torch.square(wheel_diff_x * cfg.no_fork_square_sigma),
        "no_fork_exp": 1.0 - torch.exp(-torch.clamp(torch.abs(wheel_diff_x) - cfg.no_fork_distance, min=0.0) / cfg.no_fork_exp_sigma),
        "no_fork_z_exp": 1.0 - torch.exp(-torch.clamp(torch.abs(wheel_diff_z) - getattr(cfg, "no_fork_z_distance", cfg.no_fork_distance), min=0.0) / getattr(cfg, "no_fork_z_exp_sigma", cfg.no_fork_exp_sigma)),
        "undesired_contact": undesired_contact.float(),
        "front_wheel_impact": torch.clamp(
            (front_wheel_contact_peak - cfg.front_wheel_impact_force_threshold_n)
            / cfg.front_wheel_impact_force_scale_n, min=0.0,
        ).square(),
    }


def weight_reward_terms(terms, weights, step_dt, numerical_reset=None):
    """Reference scales every term by dt and drops numerically invalid states."""
    missing = set(weights) - set(terms)
    if missing:
        raise KeyError(f"Reward weights have no implementation: {sorted(missing)}")
    rewards = {}
    for key, weight in weights.items():
        value = torch.nan_to_num(terms[key] * weight * step_dt, nan=0.0, posinf=0.0, neginf=0.0)
        if numerical_reset is not None:
            value = value.masked_fill(numerical_reset, 0.0)
        rewards[key] = value
    return rewards
