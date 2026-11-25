"""Domain randomization event terms for the Mongo0909 tasks.

These helpers mirror the ones used by the ``wheeled-legged_RL`` project so that
the Mongo task inherits exactly the same domain randomization pipeline.  They
are reproduced here (instead of imported) so that ``Mongo_RL`` stays a
self-contained library.
"""

from __future__ import annotations

from typing import Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators import IdealPDActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


def _sample_event_distribution(
    distribution_params: tuple[float, float],
    shape: tuple[int, ...],
    device: torch.device | str,
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    if distribution == "uniform":
        return math_utils.sample_uniform(*distribution_params, shape, device=device)
    if distribution == "log_uniform":
        return math_utils.sample_log_uniform(*distribution_params, shape, device=device)
    if distribution == "gaussian":
        return math_utils.sample_gaussian(*distribution_params, shape, device=device)
    raise ValueError(f"Unsupported distribution: {distribution}")


def _validate_scale_range(distribution_params, name: str) -> None:
    """Reject non-positive scale ranges (scaling by <= 0 is not physical)."""

    lower, upper = distribution_params
    if torch.is_tensor(lower):
        lower = float(lower.min())
    if torch.is_tensor(upper):
        upper = float(upper.max())
    if lower <= 0.0:
        raise ValueError(
            f"The lower bound of the scale range for '{name}' must be greater than zero."
        )
    if upper < lower:
        raise ValueError(
            f"The upper bound of the scale range for '{name}' must be greater than the lower bound."
        )


# ---------------------------------------------------------------------------
# External force / torque (per axis ranges)
# ---------------------------------------------------------------------------


def apply_external_force_torque_xyz(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    torque_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply random external forces and torques with independent per-axis ranges.

    Args:
        force_range: ``((fx_min, fx_max), (fy_min, fy_max), (fz_min, fz_max))``.
        torque_range: ``((tx_min, tx_max), (ty_min, ty_max), (tz_min, tz_max))``.
    """

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    forces = torch.cat(
        [
            math_utils.sample_uniform(
                force_range[axis][0], force_range[axis][1], (len(env_ids), num_bodies, 1), asset.device
            )
            for axis in range(3)
        ],
        dim=2,
    )
    torques = torch.cat(
        [
            math_utils.sample_uniform(
                torque_range[axis][0], torque_range[axis][1], (len(env_ids), num_bodies, 1), asset.device
            )
            for axis in range(3)
        ],
        dim=2,
    )

    # Note: these are only applied when ``asset.write_data_to_sim()`` is called.
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


# ---------------------------------------------------------------------------
# Explicit actuator effort output randomization
# ---------------------------------------------------------------------------


def _resolve_actuator_joint_indices(
    asset_cfg: SceneEntityCfg, actuator, device: torch.device
) -> tuple[slice | torch.Tensor, slice | torch.Tensor] | None:
    if isinstance(asset_cfg.joint_ids, slice):
        actuator_indices = slice(None)
        if isinstance(actuator.joint_indices, slice):
            global_indices = slice(None)
        elif isinstance(actuator.joint_indices, torch.Tensor):
            global_indices = actuator.joint_indices.to(device=device)
        else:
            global_indices = torch.as_tensor(actuator.joint_indices, device=device, dtype=torch.long)
        return actuator_indices, global_indices

    asset_joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=device, dtype=torch.long)
    if isinstance(actuator.joint_indices, slice):
        return asset_joint_ids, asset_joint_ids

    actuator_joint_indices = (
        actuator.joint_indices.to(device=device)
        if isinstance(actuator.joint_indices, torch.Tensor)
        else torch.as_tensor(actuator.joint_indices, device=device, dtype=torch.long)
    )
    actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
    if actuator_indices.numel() == 0:
        return None
    return actuator_indices, actuator_joint_indices[actuator_indices]


def _install_effort_output_randomizer(actuator) -> None:
    if getattr(actuator, "_mongo_effort_output_randomizer_installed", False):
        return

    actuator._mongo_effort_output_scale = torch.ones_like(actuator.computed_effort)
    actuator._mongo_effort_output_bias = torch.zeros_like(actuator.computed_effort)
    actuator._mongo_effort_output_noise_std = torch.zeros_like(actuator.computed_effort)
    actuator._mongo_effort_output_original_compute = actuator.compute

    def compute_with_effort_output_randomization(control_action, joint_pos, joint_vel):
        result = actuator._mongo_effort_output_original_compute(control_action, joint_pos, joint_vel)
        base_effort = actuator.computed_effort
        if base_effort is None or base_effort.shape != actuator._mongo_effort_output_scale.shape:
            base_effort = result.joint_efforts
        disturbed_effort = base_effort * actuator._mongo_effort_output_scale + actuator._mongo_effort_output_bias
        noise_std = actuator._mongo_effort_output_noise_std
        if torch.any(noise_std > 0.0):
            disturbed_effort = disturbed_effort + torch.randn_like(disturbed_effort) * noise_std
        actuator.computed_effort = disturbed_effort
        actuator.applied_effort = actuator._clip_effort(disturbed_effort)
        result.joint_efforts = actuator.applied_effort
        result.joint_positions = None
        result.joint_velocities = None
        return result

    actuator.compute = compute_with_effort_output_randomization
    actuator._mongo_effort_output_randomizer_installed = True


class randomize_actuator_effort_output(ManagerTermBase):
    """Randomize the torque output of explicit ``IdealPD``-family actuators.

    The term installs a light runtime wrapper around every selected actuator and
    perturbs the torque *after* the nominal PD computation but *before* the
    actuator specific effort clipping::

        tau = clip(tau_nominal * scale + bias + N(0, noise_std))

    Implicit PhysX actuators are not affected.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, Articulation):
            raise TypeError("randomize_actuator_effort_output only supports Articulation assets.")
        if not hasattr(self.asset, "actuators"):
            raise TypeError("Asset does not expose explicit actuator models.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        effort_scale_distribution_params: tuple[float, float] | None = None,
        effort_bias_distribution_params: tuple[float, float] | None = None,
        effort_noise_std_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        reset_to_default: bool = True,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        for actuator in self.asset.actuators.values():
            if not isinstance(actuator, IdealPDActuator):
                continue

            resolved_indices = _resolve_actuator_joint_indices(self.asset_cfg, actuator, self.asset.device)
            if resolved_indices is None:
                continue
            actuator_indices, _ = resolved_indices
            _install_effort_output_randomizer(actuator)

            if reset_to_default:
                actuator._mongo_effort_output_scale[env_ids[:, None], actuator_indices] = 1.0
                actuator._mongo_effort_output_bias[env_ids[:, None], actuator_indices] = 0.0
                actuator._mongo_effort_output_noise_std[env_ids[:, None], actuator_indices] = 0.0

            num_actuator_joints = (
                actuator.num_joints if isinstance(actuator_indices, slice) else int(actuator_indices.numel())
            )
            shape = (int(env_ids.numel()), num_actuator_joints)
            env_index = env_ids[:, None]

            if effort_scale_distribution_params is not None:
                scale = _sample_event_distribution(
                    effort_scale_distribution_params, shape, self.asset.device, distribution
                )
                actuator._mongo_effort_output_scale[env_index, actuator_indices] = torch.clamp(scale, min=0.0)
            if effort_bias_distribution_params is not None:
                bias = _sample_event_distribution(
                    effort_bias_distribution_params, shape, self.asset.device, distribution
                )
                actuator._mongo_effort_output_bias[env_index, actuator_indices] = bias
            if effort_noise_std_distribution_params is not None:
                noise_std = _sample_event_distribution(
                    effort_noise_std_distribution_params, shape, self.asset.device, distribution
                )
                actuator._mongo_effort_output_noise_std[env_index, actuator_indices] = torch.clamp(noise_std, min=0.0)


# ---------------------------------------------------------------------------
# Rigid body inertia
# ---------------------------------------------------------------------------


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the inertia tensor of the selected rigid bodies.

    A single sampling factor is used for all inertia components of a body so the
    tensor stays positive definite (PhysX otherwise raises a non-positive
    definite error).
    """

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # PhysX stores inertia as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, 0, 0, 0].
    inertias = asset.root_physx_view.get_inertias().clone()
    rand_samples = math_utils.sample_uniform(
        inertia_distribution_params[0],
        inertia_distribution_params[1],
        (len(env_ids), len(body_ids), 1),
        device="cpu",
    )

    if operation == "scale":
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), :6] *= rand_samples
    elif operation == "add":
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), :3] += rand_samples
    elif operation == "abs":
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), :3] = rand_samples
        inertias[env_ids.unsqueeze(1), body_ids.unsqueeze(0), 3:6] = 0.0
    else:
        raise ValueError(f"Invalid operation: {operation}. Must be 'add', 'scale', or 'abs'.")

    asset.root_physx_view.set_inertias(inertias, env_ids)


# ---------------------------------------------------------------------------
# Joint parameters (friction / armature / limits)
# ---------------------------------------------------------------------------


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]

    n_dim_1 = data.shape[1] if isinstance(dim_1_ids, slice) else len(dim_1_ids)

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization. "
            "Please use 'uniform', 'log_uniform', 'gaussian'."
        )

    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. "
            "Please use 'add', 'scale', or 'abs'."
        )
    return data


class randomize_joint_parameters_ext(ManagerTermBase):
    """Randomize simulated joint friction / armature / position limits.

    Only the properties for which distribution parameters are provided get
    modified.  This term is intended to run at ``startup`` because it uses CPU
    tensors to assign the joint properties.

    The difference with the Isaac Lab built-in ``randomize_joint_parameters`` is
    that static / dynamic / viscous friction can each be sampled from their own
    range, which is what the wheelbipe domain randomization relies on.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if cfg.params["operation"] == "scale":
            for key in (
                "static_friction_distribution_params",
                "dynamic_friction_distribution_params",
                "viscous_friction_distribution_params",
                "armature_distribution_params",
            ):
                if key in cfg.params:
                    _validate_scale_range(cfg.params[key], key)
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_joint_parameters_ext' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        static_friction_distribution_params: tuple[float, float] | None = None,
        dynamic_friction_distribution_params: tuple[float, float] | None = None,
        viscous_friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        if self.asset_cfg.joint_ids == slice(None):
            joint_ids: slice | torch.Tensor = slice(None)
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # -- joint friction coefficient
        if static_friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                static_friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            friction_coeff = torch.clamp(friction_coeff, min=0.0)
            static_friction_coeff = friction_coeff[env_ids[:, None], joint_ids]
        else:
            static_friction_coeff = None

        # Isaac Sim < 5.0 only supports the static friction coefficient.
        major_version = int(env.sim.get_version()[0])
        if major_version >= 5:
            if dynamic_friction_distribution_params is not None:
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    dynamic_friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            elif static_friction_distribution_params is not None:
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    static_friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            else:
                dynamic_friction_coeff = None

            if dynamic_friction_coeff is not None and static_friction_coeff is not None:
                dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
                dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)
                dynamic_friction_coeff = dynamic_friction_coeff[env_ids[:, None], joint_ids]
            else:
                dynamic_friction_coeff = None

            if viscous_friction_distribution_params is not None:
                viscous_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_viscous_friction_coeff.clone(),
                    viscous_friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)
                viscous_friction_coeff = viscous_friction_coeff[env_ids[:, None], joint_ids]
            else:
                viscous_friction_coeff = None
        else:
            dynamic_friction_coeff = None
            viscous_friction_coeff = None

        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=static_friction_coeff,
            joint_dynamic_friction_coeff=dynamic_friction_coeff,
            joint_viscous_friction_coeff=viscous_friction_coeff,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )

        # -- joint armature
        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # -- joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "randomize_joint_parameters is producing lower joint limits greater than upper "
                    "joint limits. Please check the distribution parameters."
                )
            self.asset.write_joint_position_limit_to_sim(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )
