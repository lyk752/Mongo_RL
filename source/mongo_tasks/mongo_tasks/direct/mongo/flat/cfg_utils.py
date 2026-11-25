"""Shared constants and observation scaling tables for the Mongo0909 flat task.

The policy observation layout follows the wheelbipe V14 flat task of
``wheeled-legged_RL`` and is adapted to the Mongo0909 joint count.  Body linear
velocity is a critic-only privileged input. Actor frames have 31 dimensions
and are stacked over five policy steps (155 inputs); Critic uses one 77-D
frame, with no displacement input. The front arm is commanded
directly from a sampled task command, so it is sensed like an actuated joint
but is not part of the policy action vector. Joint positions and the issued
front-arm target are absolute articulation angles in radians, without
default-pose subtraction or division by the leg action scale:

===========  ==========================  ==========================
block        wheelbipe V14 (6 actions)   Mongo0909 (6 actions)
===========  ==========================  ==========================
command      3                           3
height_cmd   1                           1
ang_vel_b    3                           3
proj gravity 3                           3
joint_pos    6                           7   (5 leg + 2 wheel)
joint_vel    6                           7   (5 leg + 2 wheel)
actions      6                           6
front target 0                           1
-------------  --------------------------  --------------------------
policy       28                          31
critic adds  lin_vel_b(3)+height(1)
critic       32                          35
===========  ==========================  ==========================
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

MONGO_MOTOR_PROFILE = json.loads(
    (Path(__file__).resolve().parents[6] / "model/motor_config_24v.json").read_text()
)
MONGO_PHYSICS_DT = MONGO_MOTOR_PROFILE["physics_dt_s"]
MONGO_POLICY_DECIMATION = MONGO_MOTOR_PROFILE["policy_decimation"]
MONGO_MOTOR_PROFILE_NAME = MONGO_MOTOR_PROFILE["profile"]
MONGO_WHEEL_TARGET_VEL_LIMIT = MONGO_MOTOR_PROFILE["motors"]["wheel"]["rated_velocity_rad_s"]
MONGO_WHEEL_EFFORT_LIMIT = MONGO_MOTOR_PROFILE["motors"]["wheel"]["effort_limit_nm"]

# ---------------------------------------------------------------------------
# Observation dimensions
# ---------------------------------------------------------------------------

#: Number of rear leg position joints controlled by the policy.  The front arm
#: is a separately sampled direct motor command.
MONGO_LEG_ACTION_DIM = 4
#: Number of drive wheels controlled by the policy.
MONGO_WHEEL_ACTION_DIM = 2
#: Full policy action dimension.
MONGO_ACTION_DIM = MONGO_LEG_ACTION_DIM + MONGO_WHEEL_ACTION_DIM

#: Number of physically actuated joints observed by the policy and critic:
#: four rear-leg joints, one directly commanded front arm, and two wheels.
MONGO_ACTUATED_JOINT_COUNT = 7
#: Policy observation dimension (no privileged information): command(3),
#: height(1), angular velocity(3), gravity(3), joint position/velocity(7+7),
#: previous policy action(6), and sampled front-arm target(1).
MONGO_BASE_POLICY_OBS_DIM = 31

# ---------------------------------------------------------------------------
# Privileged (critic-only) extra observation layout
# ---------------------------------------------------------------------------
# Mirrors the ``privileged_extra_obs`` block of the wheelbipe V14 flat task,
# adapted to the Mongo0909 joint/body counts.  One entry per observation block:
#
#   joint_stiffness          (7)   actuated joint stiffness
#   joint_damping            (7)   actuated joint damping
#   joint_torque             (7)   applied torque of the actuated joints
#   obs_delay_steps          (4)   per-group observation delay (physics steps)
#   act_delay_steps          (2)   per-group action delay (physics steps)
#   wheel_body_lin_vel       (2x3) wheel link velocity in the base frame
#   wheel_contact_state      (2)   1 if the wheel contact force exceeds the threshold
#   body_mass_scale          (1)   randomized base mass / default mass
#   body_material            (2x3) wheel static/dynamic friction + restitution

#: Number of actuated joints included in the privileged block (5 leg + 2 wheel).
MONGO_PRIV_JOINT_COUNT = MONGO_ACTUATED_JOINT_COUNT
#: Number of drive wheels included in the privileged block.
MONGO_PRIV_WHEEL_COUNT = 2
#: Number of rigid bodies whose mass scale is observed (the base).
MONGO_PRIV_BODY_COUNT = 1
#: Number of bodies whose physics material is observed (the two drive wheels).
MONGO_PRIV_MATERIAL_BODY_COUNT = 2
#: Number of observation-delay groups (see ``obs_delay_cfg``).
MONGO_OBS_DELAY_GROUP_COUNT = 4
#: Number of action-delay groups (see ``act_delay_cfg``).
MONGO_ACT_DELAY_GROUP_COUNT = 2

MONGO_PRIV_EXTRA_OBS_DIM = (
    MONGO_PRIV_JOINT_COUNT * 3  # stiffness + damping + torque
    + MONGO_OBS_DELAY_GROUP_COUNT  # observation delay (physics steps)
    + MONGO_ACT_DELAY_GROUP_COUNT  # action delay (physics steps)
    + MONGO_PRIV_WHEEL_COUNT * 3  # wheel link linear velocity (base frame)
    + MONGO_PRIV_WHEEL_COUNT  # wheel contact state
    + MONGO_PRIV_BODY_COUNT  # base mass scale
    + MONGO_PRIV_MATERIAL_BODY_COUNT * 3  # wheel physics material
)  # = 42

#: Critic-only blocks that are not part of the privileged extras.
MONGO_CRITIC_EXTRA_BASE_DIM = 4  # root_lin_vel_b(3) + obs_height(1)

#: Privileged / critic observation dimension with the privileged block enabled.
MONGO_BASE_PRIVILEGED_OBS_DIM = (
    MONGO_BASE_POLICY_OBS_DIM + MONGO_CRITIC_EXTRA_BASE_DIM + MONGO_PRIV_EXTRA_OBS_DIM
)  # = 77

#: Privileged / critic observation dimension with the privileged block disabled.
MONGO_MIN_PRIVILEGED_OBS_DIM = MONGO_BASE_POLICY_OBS_DIM + MONGO_CRITIC_EXTRA_BASE_DIM  # = 35

# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------

#: Order of all position-controlled leg joints inside the joint observation;
#: the policy action uses the first four and the front arm is commanded directly.
MONGO_ORDERED_LEG_ACT_JOINT_NAMES = (
    "left_hip_joint",
    "right_hip_joint",
    "left_parallel_1_joint",
    "right_parallel_1_joint",
    "front_hip_joint",
)

#: Left/right joint pairs that should keep a symmetric posture.
MONGO_LEG_JOINT_PAIRS = (
    ("left_hip_joint", "right_hip_joint"),
    ("left_parallel_1_joint", "right_parallel_1_joint"),
)

# ---------------------------------------------------------------------------
# Observation clipping / scaling (mirrors V14_BASIC_OBS_CLIP / V14_BASIC_OBS_SCALE)
# ---------------------------------------------------------------------------

MONGO_BASIC_OBS_CLIP: dict = {
    "command": 100.0,
    "height_cmd": [0.0, 1.0],
    "root_ang_vel_b": 100.0,
    "projected_gravity_b": 100.0,
    "joint_pos": 100.0,
    "joint_vel_leg": 200.0,
    "joint_vel_wheel": 200.0,
    "actions": 100.0,
    "front_arm_target": 10.0,
    "root_lin_vel_b": 100.0,
    "obs_height": [-10.0, 10.0],
}

MONGO_BASIC_OBS_SCALE: dict = {
    "command": {
        "lin_vel_x": 1.0,
        "lin_vel_y": 1.0,
        "ang_vel_z": 1.0,
    },
    "height_cmd": 5.0,
    "root_ang_vel_b": 0.5,
    "joint_vel_leg": 0.1,
    "joint_vel_wheel": 0.1,
    "obs_height": 5.0,
    "front_arm_target": 1.0,
    "root_lin_vel_b": 1.0,
}

MONGO_EXTRA_OBS_CLIP = copy.deepcopy(MONGO_BASIC_OBS_CLIP)
MONGO_EXTRA_OBS_CLIP.update(
    {
        "joint_torque": 100.0,
        "joint_acc": 1000.0,
        "joint_stiffness": 100.0,
        "joint_damping": 100.0,
        "joint_friction": 100.0,
        "obs_delay_steps": 100.0,
        "act_delay_steps": 100.0,
        "wheel_body_lin_vel": 100.0,
        "wheel_contact_state": 1.0,
        "wheel_contact_force": 5000.0,
        "body_mass": 100.0,
        "body_mass_scale": 10.0,
        "body_inertia_diag": 100.0,
        "body_material": 100.0,
        "body_com": 100.0,
    }
)

MONGO_EXTRA_OBS_SCALE = copy.deepcopy(MONGO_BASIC_OBS_SCALE)
MONGO_EXTRA_OBS_SCALE.update(
    {
        "joint_torque": 0.05,
        "joint_acc": 0.01,
        "joint_stiffness": 1.0,
        "joint_damping": 1.0,
        "joint_friction": 1.0,
        "obs_delay_steps": 1.0,
        "act_delay_steps": 1.0,
        "wheel_body_lin_vel": 1.0,
        "wheel_contact_state": 1.0,
        "wheel_contact_force": 0.01,
        "body_mass": 0.1,
        "body_mass_scale": 1.0,
        "body_inertia_diag": 1.0,
        "body_material": 1.0,
        "body_com": 1.0,
    }
)
__all__ = [
    "MONGO_LEG_ACTION_DIM",
    "MONGO_WHEEL_ACTION_DIM",
    "MONGO_ACTION_DIM",
    "MONGO_ACTUATED_JOINT_COUNT",
    "MONGO_BASE_POLICY_OBS_DIM",
    "MONGO_BASE_PRIVILEGED_OBS_DIM",
    "MONGO_MIN_PRIVILEGED_OBS_DIM",
    "MONGO_CRITIC_EXTRA_BASE_DIM",
    "MONGO_PRIV_EXTRA_OBS_DIM",
    "MONGO_PRIV_JOINT_COUNT",
    "MONGO_PRIV_WHEEL_COUNT",
    "MONGO_PRIV_BODY_COUNT",
    "MONGO_PRIV_MATERIAL_BODY_COUNT",
    "MONGO_OBS_DELAY_GROUP_COUNT",
    "MONGO_ACT_DELAY_GROUP_COUNT",
    "MONGO_ORDERED_LEG_ACT_JOINT_NAMES",
    "MONGO_LEG_JOINT_PAIRS",
    "MONGO_BASIC_OBS_CLIP",
    "MONGO_BASIC_OBS_SCALE",
    "MONGO_EXTRA_OBS_CLIP",
    "MONGO_EXTRA_OBS_SCALE",
]
