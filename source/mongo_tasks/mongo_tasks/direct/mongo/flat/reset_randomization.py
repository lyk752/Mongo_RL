"""Task-space reset helpers for the Mongo0909 rear-leg linkage.

The rear leg is the same six-revolute closed-chain layout used by the
``wheelbipe25_v3`` reference task.  Sampling a virtual leg length and angle and
solving all six joint positions together keeps the loop geometry consistent;
independently perturbing the passive joint positions does not.
"""

from __future__ import annotations

import math

import torch


# Geometry read from ``model/urdf/wl0901.urdf``.  ``SHORT_DIAGONAL`` is the
# length of the 98 mm x 65 mm offset link.  The zero offsets convert geometric
# linkage angles into the URDF joint coordinates.
MONGO_LEG_SHORT_LINK = 0.098
MONGO_LEG_SHORT_DIAGONAL = math.hypot(0.098, 0.065)
MONGO_LEG_LONG_LINK = 0.215
MONGO_LEG_DIAGONAL_ANGLE = math.atan2(0.065, 0.098)
MONGO_LEG_ALPHA_OFFSETS = (
    0.0,
    math.pi,
    MONGO_LEG_DIAGONAL_ANGLE,
    math.pi - 2.0 * MONGO_LEG_DIAGONAL_ANGLE,
    MONGO_LEG_DIAGONAL_ANGLE,
    MONGO_LEG_DIAGONAL_ANGLE,
)

# Per-side output order.  The environment interleaves left/right instances of
# these names before writing the state to the articulation.
MONGO_LEG_IK_JOINT_SUFFIXES = (
    "parallel_1_joint",
    "hip_joint",
    "parallel_2_joint",
    "parallel_3_joint",
    "parallel_4_joint",
    "knee_joint",
)


def mongo_leg_inverse_kinematics(
    leg_length: torch.Tensor,
    leg_angle: torch.Tensor,
    *,
    acos_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Solve the Mongo rear-leg closed chain from virtual length and angle.

    Args:
        leg_length: Virtual hip-to-wheel-centre distance in metres.  The last
            dimension normally contains ``[left, right]``.
        leg_angle: Virtual-leg angle in radians, positive from the downward
            body-z axis towards body-backward.  Shape must match
            ``leg_length``.
        acos_epsilon: Margin used when clamping cosine-law arguments.

    Returns:
        Joint positions with shape ``leg_length.shape + (6,)`` and per-side
        order given by :data:`MONGO_LEG_IK_JOINT_SUFFIXES`.
    """
    if leg_length.shape != leg_angle.shape:
        raise ValueError(
            "leg_length and leg_angle must have identical shapes; got "
            f"{tuple(leg_length.shape)} and {tuple(leg_angle.shape)}."
        )

    eps = min(max(float(acos_epsilon), 1.0e-9), 0.25)
    length = torch.clamp(leg_length, min=eps)
    a = MONGO_LEG_SHORT_LINK
    b = MONGO_LEG_SHORT_DIAGONAL
    c = MONGO_LEG_LONG_LINK
    a2, b2, c2 = a * a, b * b, c * c
    length2 = torch.square(length)

    beta_cos = (a2 * length2 + c2 * (a2 - b2)) / (2.0 * c * a2 * length)
    gamma_cos = (a2 + b2 - a2 * length2 / c2) / (2.0 * a * b)
    reachability_tolerance = 1.0e-5
    unreachable = (
        (beta_cos < -1.0 - reachability_tolerance)
        | (beta_cos > 1.0 + reachability_tolerance)
        | (gamma_cos < -1.0 - reachability_tolerance)
        | (gamma_cos > 1.0 + reachability_tolerance)
    )
    if torch.any(unreachable):
        bad_length = length[unreachable][0]
        raise ValueError(
            "Mongo reset sampled an unreachable virtual leg length: "
            f"{float(bad_length):.6f} m. Tighten reset_leg_length_range."
        )
    clamp_min, clamp_max = -1.0 + eps, 1.0 - eps
    beta = torch.acos(torch.clamp(beta_cos, clamp_min, clamp_max))
    gamma = torch.acos(torch.clamp(gamma_cos, clamp_min, clamp_max))

    raw = torch.empty(*length.shape, 6, dtype=length.dtype, device=length.device)
    raw[..., 0] = leg_angle + 0.5 * math.pi - beta
    raw[..., 1] = leg_angle + 0.5 * math.pi + beta
    raw[..., 2] = gamma
    raw[..., 3] = 2.0 * math.pi - (raw[..., 1] - raw[..., 0]) - 2.0 * gamma
    raw[..., 4] = gamma
    raw[..., 5] = gamma

    offsets = torch.as_tensor(
        MONGO_LEG_ALPHA_OFFSETS, dtype=length.dtype, device=length.device
    )
    joint_pos = torch.empty_like(raw)
    joint_pos[..., 0] = raw[..., 0] - offsets[0]
    joint_pos[..., 1] = raw[..., 1] - offsets[1]
    joint_pos[..., 2] = -(raw[..., 2] - offsets[2])
    joint_pos[..., 3] = -(raw[..., 3] - offsets[3])
    joint_pos[..., 4] = -(raw[..., 4] - offsets[4])
    joint_pos[..., 5] = raw[..., 5] - offsets[5]
    return joint_pos


__all__ = [
    "MONGO_LEG_IK_JOINT_SUFFIXES",
    "MONGO_LEG_ALPHA_OFFSETS",
    "MONGO_LEG_DIAGONAL_ANGLE",
    "MONGO_LEG_LONG_LINK",
    "MONGO_LEG_SHORT_DIAGONAL",
    "MONGO_LEG_SHORT_LINK",
    "mongo_leg_inverse_kinematics",
]
