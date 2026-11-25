"""Encoder velocity contract: backward difference at each actuator tick (500 Hz by default)."""
import math

import torch


def encoder_velocity(position, previous, dt):
    """Seven active hinges in policy order; only the two continuous wheels wrap.

    Sampling must satisfy abs(wheel_speed * dt) < pi. No offset, filter or
    clipping is applied here. Episode resets seed previous with the reset pose.
    """
    delta = position - previous
    delta = delta.clone()
    wheel = delta[..., 5:]
    delta[..., 5:] = torch.where(torch.abs(wheel) > math.pi,
                                torch.remainder(wheel + math.pi, 2 * math.pi) - math.pi, wheel)
    return delta / dt
