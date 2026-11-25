"""Explicit PD motors with directional speed protection at every physics tick."""
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass

from .motor_limits import speed_limited_effort


class SpeedLimitedPDActuator(IdealPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if not 0.0 <= cfg.speed_taper_start_fraction < 1.0:
            raise ValueError('speed_taper_start_fraction must be finite and in [0, 1).')
        if not torch.all(torch.isfinite(self.velocity_limit) & (self.velocity_limit > 0)):
            raise ValueError('Speed-limited motors require finite positive velocity_limit.')
        self._speed_guard_velocity = torch.zeros_like(self.computed_effort)

    def compute(self, control_action, joint_pos, joint_vel):
        # Use physical speed for motor protection. Encoder feedback for PD is
        # still supplied by the environment's velocity-target compensation.
        self._speed_guard_velocity.copy_(joint_vel)
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort):
        if not self.cfg.speed_limit_enabled:
            return super()._clip_effort(effort)
        return speed_limited_effort(effort, self._speed_guard_velocity,
                                    self.effort_limit, self.velocity_limit,
                                    self.cfg.speed_taper_start_fraction, torch)


@configclass
class SpeedLimitedPDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = SpeedLimitedPDActuator
    speed_limit_enabled: bool = True
    speed_taper_start_fraction: float = 0.9
