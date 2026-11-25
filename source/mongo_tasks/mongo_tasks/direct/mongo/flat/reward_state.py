"""Per-environment reward history, independent of Isaac Sim and observations."""

import torch


class CommandRewardState:
    """Detect quiet velocity commands and differentiate virtual-leg geometry.

    Update once per policy step, before scheduling the next commands. Never use
    tracking errors to decide steadiness: a poor policy must not disable costs.
    """

    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.previous_command = torch.zeros(num_envs, 3, device=device)
        self.previous_angle = torch.zeros(num_envs, 2, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motion_age = torch.zeros(num_envs, device=device)
        self.stop_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stop_age = torch.zeros(num_envs, device=device)
        self.stop_anchor_valid = torch.zeros_like(self.stop_active)
        self.stop_anchor_xy = torch.zeros(num_envs, 2, device=device)

    def reset(self, env_ids):
        self.valid[env_ids] = False
        self.motion_age[env_ids] = 0.0
        self.stop_active[env_ids] = False
        self.stop_age[env_ids] = 0.0
        self.stop_anchor_valid[env_ids] = False
        self.stop_anchor_xy[env_ids] = 0.0

    def _gate(self, age):
        t = ((age - self.cfg.command_steady_delay_s)
             / self.cfg.command_steady_ramp_s).clamp(0.0, 1.0)
        return t.square() * (3.0 - 2.0 * t)

    def update(self, command, wheel_pos_b, dt):
        # Only issued vx/vy/yaw commands determine the phase. Height, arm and
        # measured robot motion do not enter this decision.
        change_rate = (command - self.previous_command).abs() / dt
        motion_changed = (
            (change_rate[:, :2] > self.cfg.command_steady_lin_rate_m_s2).any(dim=1)
            | (change_rate[:, 2] > self.cfg.command_steady_yaw_rate_rad_s2)
        )
        self.motion_age = torch.where(
            self.valid & ~motion_changed, self.motion_age + dt, 0.0)
        steady_gate = self._gate(self.motion_age)

        # Same body-relative definition used in fixed_velocity_test.py.
        angle = torch.atan2(-wheel_pos_b[..., 0], -wheel_pos_b[..., 2])
        delta = angle - self.previous_angle
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        angle_rate = torch.where(self.valid[:, None], delta / dt, 0.0)
        self.previous_command.copy_(command)
        self.previous_angle.copy_(angle)
        self.valid.fill_(True)
        # Preserve both existing reward/log interfaces as aliases of one gate.
        return steady_gate, steady_gate, angle_rate

    def update_stop_position(self, command, root_pos_w, dt):
        """Anchor after a braking grace period; never chase a drifting robot.

        Exact raw zero commands only. Height/arm changes do not restart the
        timer. Capture XY at the end of grace, independently of actual speed.
        """
        zero = (command == 0.0).all(dim=1)
        self.stop_age = torch.where(zero & self.stop_active, self.stop_age + dt, 0.0)
        self.stop_active.copy_(zero)
        self.stop_anchor_valid.logical_and_(zero)
        ready = zero & (self.stop_age >= self.cfg.stand_still_position_delay_s)
        capture = ready & ~self.stop_anchor_valid
        self.stop_anchor_xy[capture] = root_pos_w[capture, :2]
        self.stop_anchor_valid.logical_or_(capture)
        t = ((self.stop_age - self.cfg.stand_still_position_delay_s)
             / self.cfg.stand_still_position_ramp_s).clamp(0.0, 1.0)
        gate = t.square() * (3.0 - 2.0 * t) * self.stop_anchor_valid.float()
        error = torch.where(self.stop_anchor_valid[:, None],
                            root_pos_w[:, :2] - self.stop_anchor_xy, 0.0)
        return error, gate
