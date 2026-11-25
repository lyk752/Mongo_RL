"""Isaac Lab articulation configuration for the Mongo0909 wheeled-legged robot.

The kinematic model (USD), the default standing pose and the actuator gains are
taken from the ``Mongo_RL_0909`` project.  The actuator *structure* follows the
``wheeled-legged_RL`` convention: explicit PD actuator groups named
``legs_act`` / ``legs_inact`` / ``wheel`` plus a dedicated passive
``front_caster`` group, so that linkage and caster friction can be calibrated
independently. Active motors use SpeedLimitedPDActuatorCfg for speed protection.

Compared with the wheelbipe robot this asset has:

* one extra position-controlled joint (``front_hip_joint``, the front swing arm),
* no gimbal,
* no spring joint.
"""

from __future__ import annotations

import math
import json
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from mongo_world.actuators import SpeedLimitedPDActuatorCfg

# ---------------------------------------------------------------------------
# USD resolution
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the ``Mongo_RL`` repository root."""

    return Path(__file__).resolve().parents[4]


def _resolve_usd_path() -> str:
    """Resolve the Mongo0909 USD path.

    The robot model is **vendored inside this repository**, so by default the
    library is completely self-contained and never reads anything outside
    ``<Mongo_RL>/``:

    * ``<Mongo_RL>/model/usd/Mongo0909_caster.usda``

    The ASCII wrapper sublayers the original converted USD and applies the
    front-caster-only physics overrides (supported spherical-joint axis and a
    finite torsional contact patch) without modifying the vendored binary USD.

    ``MONGO0909_USD_PATH`` can be used to point at a different USD explicitly;
    it is optional and unused by default.

    The path is resolved through symlinks on purpose: the USD references its
    ``urdf`` folder relatively (``model/usd/urdf -> ../urdf``), so Isaac Sim
    must see the real file location.
    """

    candidates: list[Path] = []

    override = os.environ.get("MONGO0909_USD_PATH")
    if override:
        candidates.append(Path(override).expanduser())

    repo_root = _repo_root()
    candidates.append(repo_root / "model" / "usd" / "Mongo0909_caster.usda")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return str(resolved)

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Mongo0909 USD was not found. The model is expected inside the "
        "repository (model/usd/Mongo0909_caster.usda); set MONGO0909_USD_PATH to use a "
        "different file. Searched:\n  " + searched
    )


MONGO0909_USD_PATH = _resolve_usd_path()

# ---------------------------------------------------------------------------
# Joint / body grouping
# ---------------------------------------------------------------------------

# The five joints the policy controls with position targets, in policy order.
MONGO0909_LEG_ACT_JOINT_NAMES = [
    "left_hip_joint",
    "right_hip_joint",
    "left_parallel_1_joint",
    "right_parallel_1_joint",
    "front_hip_joint",
]

# The two drive wheels, in policy order.
MONGO0909_WHEEL_JOINT_NAMES = [
    "left_podarthrum_joint",
    "right_podarthrum_joint",
]

# Passive joints: present in the articulation, zero stiffness, zero effort.
MONGO0909_LINKAGE_INACT_JOINT_NAMES = [
    "left_knee_joint",
    "right_knee_joint",
    "left_parallel_2_joint",
    "right_parallel_2_joint",
    "left_parallel_3_joint",
    "right_parallel_3_joint",
    "left_parallel_4_joint",
    "right_parallel_4_joint",
]

# The front auxiliary wheel is a spherical (ball-caster) joint in the USD, so
# Isaac Sim exposes it as three DOFs: front_podarthrum_joint:0 / :1 / :2.
MONGO0909_FRONT_CASTER_JOINT_EXPR = "front_podarthrum_joint.*"
MONGO0909_INACT_JOINT_NAMES = MONGO0909_LINKAGE_INACT_JOINT_NAMES + [
    MONGO0909_FRONT_CASTER_JOINT_EXPR
]

# Conservative ball-transfer-unit baseline.  In Isaac Sim >= 5 these friction
# values are joint efforts (N m for rotational DOFs), while viscous friction is
# N m s/rad.  At the 30.3 mm ball radius, 0.010 N m dynamic friction corresponds
# to roughly 0.33 N of rolling resistance before contact-patch torsion.
MONGO0909_FRONT_CASTER_STATIC_FRICTION = 0.015
MONGO0909_FRONT_CASTER_DYNAMIC_FRICTION = 0.010
MONGO0909_FRONT_CASTER_VISCOUS_FRICTION = 0.001

MONGO0909_ACTIVE_JOINT_NAMES = MONGO0909_LEG_ACT_JOINT_NAMES + MONGO0909_WHEEL_JOINT_NAMES

MONGO0909_BASE_BODY_NAME = "wlbody"
MONGO0909_WHEEL_BODY_NAMES = ["left_wheel", "right_wheel"]
MONGO0909_FRONT_WHEEL_BODY_NAME = "front_wheel"
# Bodies that are allowed to touch the ground without penalty.
MONGO0909_ALLOWED_CONTACT_BODY_NAMES = MONGO0909_WHEEL_BODY_NAMES + [
    MONGO0909_FRONT_WHEEL_BODY_NAME
]

# Lateral offsets (m) of the left/right leg root, used by the leg-axis reward.
MONGO0909_LEFT_LEG_ROOT_Y = 0.09205
MONGO0909_RIGHT_LEG_ROOT_Y = -0.09205

# Effective drive wheel radius (m), used for height observation / reward.
MONGO0909_WHEEL_RADIUS = 0.13

# Rigid-body angular speed bound in SI units, including the drive wheels.
MONGO0909_MAX_BODY_ANGULAR_VELOCITY_RAD_S = 100.0

# ---------------------------------------------------------------------------
# Actuator conventions (follow wheeled-legged_RL)
# ---------------------------------------------------------------------------

# Shared with the independent MuJoCo runner; no Isaac imports in the JSON.
MONGO0909_MOTOR_PROFILE = json.loads((_repo_root() / "model/motor_config_24v.json").read_text())
_LEG_MOTOR = MONGO0909_MOTOR_PROFILE["motors"]["legs_act"]
_FRONT_MOTOR = MONGO0909_MOTOR_PROFILE["motors"]["front_arm"]
_WHEEL_MOTOR = MONGO0909_MOTOR_PROFILE["motors"]["wheel"]
# 1 g cm^2 = 1e-7 kg m^2. Reflected rotor inertia is J_rotor * ratio^2.
# MG6012: 930e-7 * 36^2; MG5010 reference: 850e-7 * 10^2.
MONGO0909_LEG_ARMATURE = _LEG_MOTOR["armature_kg_m2"]
MONGO0909_FRONT_ARM_ARMATURE = _FRONT_MOTOR["armature_kg_m2"]
# User-selected direct-drive wheel model: retain CAD rigid-body inertia only.
MONGO0909_WHEEL_ARMATURE = _WHEEL_MOTOR["armature_kg_m2"]

# ---------------------------------------------------------------------------
# Default state (from Mongo_RL_0909)
# ---------------------------------------------------------------------------

# Stable standing assembly identified from the linkage solver.  Values are in
# radians because Isaac Lab articulation states use SI units.
MONGO0909_STANDING_JOINT_POS = {
    "left_hip_joint": math.radians(-3.0),
    "right_hip_joint": math.radians(-3.0),
    "left_parallel_1_joint": math.radians(30.0),
    "right_parallel_1_joint": math.radians(30.0),
    "front_hip_joint": math.radians(28.0),
    "left_podarthrum_joint": math.radians(0.0),
    "right_podarthrum_joint": math.radians(0.0),
    "left_knee_joint": math.radians(19.899820),
    "right_knee_joint": math.radians(19.907970),
    "left_parallel_2_joint": math.radians(-19.905178),
    "right_parallel_2_joint": math.radians(-19.901633),
    "left_parallel_3_joint": math.radians(6.810596),
    "right_parallel_3_joint": math.radians(6.815199),
    "left_parallel_4_joint": math.radians(-19.912850),
    "right_parallel_4_joint": math.radians(-19.920997),
    "front_podarthrum_joint.*": math.radians(0.0),
}

# Nominal base height of the default standing pose.
#
# Measured against the real wheel geometry (drive wheel radius 0.1300 m, front
# caster radius 0.0303 m, both taken from the collision meshes): with the
# standing joint pose below, the drive-wheel centres sit 0.2102 m *below* the
# base, so the wheels just touch the ground at ``0.2102 + 0.1300 = 0.3402 m``
# (the front caster then clears the ground by 1.4 cm, which is by design).
#
# The historic value of 0.30 m buried the drive wheels 4.0 cm and the caster
# 2.6 cm into the ground.  PhysX then depenetrated them at up to
# ``max_depenetration_velocity = 5 m/s`` and launched the robot: measured
# +11.6 cm of base height within 0.2 s, first ground contact only after 0.8 s,
# and a first-impact contact force of ~1100 N versus a 127 N static load.  Every
# episode therefore started with a violent transient instead of a quiet stance.
MONGO0909_DEFAULT_ROOT_HEIGHT = 0.34

# ---------------------------------------------------------------------------
# Articulation config
# ---------------------------------------------------------------------------

MONGO0909_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=MONGO0909_USD_PATH,
        activate_contact_sensors=True,
        # ``copy_from_source=False`` keeps the USD references (meshes / urdf
        # sublayers) relative to the original model folder, which is what the
        # Mongo_RL_0909 project relies on as well.
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            # Isaac's rigid-body schema uses deg/s; motor targets use rad/s.
            max_angular_velocity=math.degrees(MONGO0909_MAX_BODY_ANGULAR_VELOCITY_RAD_S),
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, MONGO0909_DEFAULT_ROOT_HEIGHT),
        joint_pos=MONGO0909_STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Active motors taper propulsive effort near their no-load speed.
        # Passive joints retain IdealPD with zero effort. No solver state clip.
        #
        # Policy-controlled position joints (5): left/right hip, left/right
        # parallel linkage, and the front swing arm.
        "legs_act": SpeedLimitedPDActuatorCfg(
            speed_limit_enabled=True,
            speed_taper_start_fraction=MONGO0909_MOTOR_PROFILE["speed_taper_start_fraction"],
            joint_names_expr=[
                "left_hip_joint",
                "right_hip_joint",
                "left_parallel_1_joint",
                "right_parallel_1_joint",
            ],
            stiffness={
                name: _LEG_MOTOR["stiffness"] for name in MONGO0909_LEG_ACT_JOINT_NAMES[:4]
            },
            damping={
                name: _LEG_MOTOR["damping"] for name in MONGO0909_LEG_ACT_JOINT_NAMES[:4]
            },
            effort_limit={
                name: _LEG_MOTOR["effort_limit_nm"] for name in MONGO0909_LEG_ACT_JOINT_NAMES[:4]
            },
            velocity_limit={
                name: _LEG_MOTOR["velocity_limit_rad_s"] for name in MONGO0909_LEG_ACT_JOINT_NAMES[:4]
            },
            armature=MONGO0909_LEG_ARMATURE,
        ),
        "front_arm": SpeedLimitedPDActuatorCfg(
            speed_limit_enabled=True,
            speed_taper_start_fraction=MONGO0909_MOTOR_PROFILE["speed_taper_start_fraction"],
            joint_names_expr=["front_hip_joint"],
            stiffness={"front_hip_joint": _FRONT_MOTOR["stiffness"]},
            damping={"front_hip_joint": _FRONT_MOTOR["damping"]},
            effort_limit={"front_hip_joint": _FRONT_MOTOR["effort_limit_nm"]},
            velocity_limit={"front_hip_joint": _FRONT_MOTOR["velocity_limit_rad_s"]},
            armature=MONGO0909_FRONT_ARM_ARMATURE,
        ),
        # Drive wheels: velocity PD with the same physical speed guard.
        "wheel": SpeedLimitedPDActuatorCfg(
            speed_limit_enabled=True,
            speed_taper_start_fraction=MONGO0909_MOTOR_PROFILE["speed_taper_start_fraction"],
            joint_names_expr=[
                "left_podarthrum_joint",
                "right_podarthrum_joint",
            ],
            stiffness={
                name: _WHEEL_MOTOR["stiffness"] for name in MONGO0909_WHEEL_JOINT_NAMES
            },
            damping={
                name: _WHEEL_MOTOR["damping"] for name in MONGO0909_WHEEL_JOINT_NAMES
            },
            effort_limit={
                name: _WHEEL_MOTOR["effort_limit_nm"] for name in MONGO0909_WHEEL_JOINT_NAMES
            },
            velocity_limit={
                name: _WHEEL_MOTOR["velocity_limit_rad_s"] for name in MONGO0909_WHEEL_JOINT_NAMES
            },
            armature=MONGO0909_WHEEL_ARMATURE,
        ),
        # Closed-loop linkage joints.  They are passive: zero stiffness and
        # zero effort limit.  Their friction is randomized by the task.
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=MONGO0909_LINKAGE_INACT_JOINT_NAMES,
            stiffness=0.0,
            damping=0.01,
            effort_limit=0.0,
            velocity_limit=100.0,
            armature=0.0001,
        ),
        # Passive spherical front ball caster.  Coulomb + viscous joint
        # friction models bearing/rolling resistance; the contact-normal spin
        # resistance is supplied by the Sphere collision override in
        # Mongo0909_caster.usda.  Keep explicit PD gains at zero: these are
        # solver friction properties, not a motor drive.
        "front_caster": IdealPDActuatorCfg(
            joint_names_expr=[MONGO0909_FRONT_CASTER_JOINT_EXPR],
            stiffness=0.0,
            damping=0.0,
            effort_limit=0.0,
            velocity_limit=100.0,
            armature=0.0001,
            friction=MONGO0909_FRONT_CASTER_STATIC_FRICTION,
            dynamic_friction=MONGO0909_FRONT_CASTER_DYNAMIC_FRICTION,
            viscous_friction=MONGO0909_FRONT_CASTER_VISCOUS_FRICTION,
        ),
    },
)

# Convenience alias so downstream code keeps the wheelbipe naming.
Mongo0909_CFG = MONGO0909_CFG
