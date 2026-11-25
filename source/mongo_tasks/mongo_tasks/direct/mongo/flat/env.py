"""Mongo kinematics, reset and front-arm interface; V14 flat-v0 reward adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_from_euler_xyz,
    quat_mul,
    wrap_to_pi,
)

from .env_cfg import MongoFlatEnvCfg
from .rewards import compute_flat_reward_terms, weight_reward_terms
from .reward_state import CommandRewardState
from .action_pipeline import condition_policy_actions
from .velocity_feedback import encoder_velocity
from .cfg_utils import MONGO_ORDERED_LEG_ACT_JOINT_NAMES
from .reset_randomization import MONGO_LEG_IK_JOINT_SUFFIXES, mongo_leg_inverse_kinematics
from mongo_world.assets.mongo import MONGO0909_USD_PATH, MONGO0909_WHEEL_JOINT_NAMES

# isaaclab.sim is imported lazily inside ``_setup_scene`` to keep module import
# cheap when only the configuration is inspected.


class MongoFlatEnv(DirectRLEnv):
    """Two-wheel-only flat-ground wheeled-legged locomotion task."""

    cfg: MongoFlatEnvCfg

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self, cfg: MongoFlatEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra may change history lengths after config construction. Resolve
        # spaces before DirectRLEnv creates the Gym observation interfaces.
        cfg.__post_init__()
        super().__init__(cfg, render_mode, **kwargs)

        # -- joint groups -------------------------------------------------
        # The leg action order is resolved from an *explicit ordered* list of
        # joint names.  With ``preserve_order=True`` Isaac Lab returns the
        # joints in the order of the requested list, which is what keeps the
        # action clamping table and the observation layout aligned.  Resolving
        # the regex actuator group instead would return the articulation order
        # (``left_hip, right_hip, front_hip, left_parallel_1,
        # right_parallel_1``) and silently break both.
        self._legs_act_idx, self._legs_act_names = self.robot.find_joints(
            list(MONGO_ORDERED_LEG_ACT_JOINT_NAMES), preserve_order=True
        )
        self._legs_inact_idx, _ = self.robot.find_joints(self.cfg.legs_inact_name)
        self._wheel_idx, self._wheel_names = self.robot.find_joints(
            list(MONGO0909_WHEEL_JOINT_NAMES), preserve_order=True
        )
        self._policy_leg_idx = [
            idx for idx, name in zip(self._legs_act_idx, self._legs_act_names)
            if name != "front_hip_joint"
        ]
        self._policy_leg_names = [
            name for name in self._legs_act_names if name != "front_hip_joint"
        ]
        self._actuate_idx = list(self._legs_act_idx) + list(self._wheel_idx)
        self._policy_actuate_idx = list(self._policy_leg_idx) + list(self._wheel_idx)
        self._controlled_leg_count = len(self._legs_act_idx)
        self._leg_action_dim = len(self._policy_leg_idx)
        self._wheel_action_dim = len(self._wheel_idx)
        self._actuated_joint_count = self._controlled_leg_count + self._wheel_action_dim

        if (
            self._controlled_leg_count != 5
            or self._leg_action_dim != 4
            or self._wheel_action_dim != 2
        ):
            raise RuntimeError(
                "Mongo0909 expects 4 policy leg joints, 1 direct front arm, and 2 drive "
                f"wheels; got {self._leg_action_dim}, "
                f"{self._controlled_leg_count - self._leg_action_dim}, and "
                f"{self._wheel_action_dim}. Resolved leg joints: {self._legs_act_names}, "
                f"wheel joints: {self._wheel_names}."
            )
        self._front_arm_local_idx = self._legs_act_names.index("front_hip_joint")
        self._front_arm_joint_idx = self._legs_act_idx[self._front_arm_local_idx]

        # IK reset order: all left/right instances of each closed-chain joint
        # are interleaved so ``ik.transpose(-2, -1).reshape(N, -1)`` can be
        # written without relying on articulation order.
        reset_leg_joint_names = [
            f"{side}_{suffix}"
            for suffix in MONGO_LEG_IK_JOINT_SUFFIXES
            for side in ("left", "right")
        ]
        self._reset_leg_joint_idx, self._reset_leg_joint_names = self.robot.find_joints(
            reset_leg_joint_names, preserve_order=True
        )
        if list(self._reset_leg_joint_names) != reset_leg_joint_names:
            raise RuntimeError(
                "Mongo reset IK joint mapping is incomplete or reordered: expected "
                f"{reset_leg_joint_names}, got {self._reset_leg_joint_names}."
            )

        # -- bodies -------------------------------------------------------
        self._wheel_body_idx, self._wheel_body_names = self.robot.find_bodies(
            self.cfg.wheel_body_names
        )
        self._base_body_idx, _ = self.robot.find_bodies(self.cfg.base_body_name)

        self._wheel_contact_idx, _ = self.contact_sensor.find_bodies(self.cfg.wheel_body_names)
        # Front caster allows light contact; large impacts have a separate cost.
        front_name = str(getattr(self.cfg, "front_wheel_body_name", "front_wheel"))
        front_body_idx, _ = self.robot.find_bodies([front_name])
        self._front_wheel_body_idx = front_body_idx[0] if front_body_idx else None
        if front_name in self.contact_sensor.body_names:
            self._front_wheel_contact_idx = self.contact_sensor.body_names.index(front_name)
        else:
            raise ValueError(
                f"Front wheel body '{front_name}' is missing from the contact sensor; "
                "front-wheel impact reward requires this body."
            )
        allowed_contact_idx, _ = self.contact_sensor.find_bodies(
            list(self.cfg.wheel_body_names) + ["front_wheel"]
        )
        allowed_set = set(allowed_contact_idx)
        self._undesired_contact_idx = [
            idx for idx in range(len(self.contact_sensor.body_names)) if idx not in allowed_set
        ]

        # -- action scaling ----------------------------------------------
        self.leg_action_scale = float(self.cfg.leg_action_scale)
        self.use_wheel_vel_control = bool(self.cfg.use_wheel_vel_control)
        self.wheel_action_scale = float(self.cfg.wheel_vel_action_scale)
        self.max_wheel_vel = float(self.cfg.max_wheel_vel)
        self.max_wheel_torque = float(self.cfg.max_wheel_torque)
        self.use_action_low_pass_filter = bool(self.cfg.use_action_low_pass_filter)
        self.leg_action_low_pass_prev_weight = float(
            self.cfg.leg_action_low_pass_prev_weight
        )
        self.leg_action_low_pass_curr_weight = float(
            self.cfg.leg_action_low_pass_curr_weight
        )
        self.wheel_action_low_pass_prev_weight = float(
            self.cfg.wheel_action_low_pass_prev_weight
        )
        self.wheel_action_low_pass_curr_weight = float(
            self.cfg.wheel_action_low_pass_curr_weight
        )
        for group, previous_weight, current_weight in (
            (
                "leg",
                self.leg_action_low_pass_prev_weight,
                self.leg_action_low_pass_curr_weight,
            ),
            (
                "wheel",
                self.wheel_action_low_pass_prev_weight,
                self.wheel_action_low_pass_curr_weight,
            ),
        ):
            if not (
                0.0 <= previous_weight <= 1.0
                and 0.0 <= current_weight <= 1.0
                and math.isclose(previous_weight + current_weight, 1.0, abs_tol=1.0e-6)
            ):
                raise ValueError(
                    f"{group} action low-pass weights must be in [0, 1] and sum to 1; "
                    f"got previous={previous_weight}, current={current_weight}."
                )
        self.use_action_slew_rate_limit = bool(self.cfg.use_action_slew_rate_limit)
        self.leg_action_max_delta = float(self.cfg.leg_action_max_delta)
        self.wheel_action_max_delta = float(self.cfg.wheel_action_max_delta)
        if self.leg_action_max_delta <= 0.0 or self.wheel_action_max_delta <= 0.0:
            raise ValueError(
                "Action slew-rate limits must be positive; got "
                f"leg={self.leg_action_max_delta}, wheel={self.wheel_action_max_delta}."
            )
        self._action_max_delta = torch.full(
            (self.cfg.action_space,), self.wheel_action_max_delta, device=self.device
        )
        self._action_max_delta[: self._leg_action_dim] = self.leg_action_max_delta
        self._action_current_weight = torch.full(
            (self.cfg.action_space,),
            self.wheel_action_low_pass_curr_weight,
            device=self.device,
        )
        self._action_current_weight[: self._leg_action_dim] = (
            self.leg_action_low_pass_curr_weight
        )

        self._leg_clip_lower = torch.tensor(
            self.cfg.leg_action_clip_lower, dtype=torch.float, device=self.device
        )
        self._leg_clip_upper = torch.tensor(
            self.cfg.leg_action_clip_upper, dtype=torch.float, device=self.device
        )
        if (
            self._leg_clip_lower.numel() != self._leg_action_dim
            or self._leg_clip_upper.numel() != self._leg_action_dim
        ):
            raise ValueError(
                "leg action clip tables must match the four policy-controlled leg joints; "
                f"got lower={self._leg_clip_lower.numel()}, upper={self._leg_clip_upper.numel()}."
            )

        # -- state aliases (buffers updated in place by the articulation) --
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._encoder_previous = self.joint_pos[:, self._actuate_idx].clone()
        self._encoder_velocity = self.joint_vel[:, self._actuate_idx].clone()
        self._encoder_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.default_joint_pos = self.robot.data.default_joint_pos
        # -- direct front-arm command -------------------------------------
        self.front_arm_target_cmd = self.robot.data.default_joint_pos[
            :, self._front_arm_joint_idx
        ].clone()
        self.front_arm_goal_cmd = self.front_arm_target_cmd.clone()
        self.front_arm_command_upper = self.front_arm_target_cmd.clone()
        self._front_arm_command_time_left = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # -- action buffers ----------------------------------------------
        self._reward_state = CommandRewardState(self.cfg, self.num_envs, self.device)
        # Policy actions drive the reference action-rate/smoothness terms.
        self._policy_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self._actions = torch.zeros_like(self._policy_actions)
        self._previous_actions = torch.zeros_like(self._actions)
        self._before_previous_actions = torch.zeros_like(self._actions)
        self.last_actions = torch.zeros_like(self._actions)
        self.leg_actions = torch.zeros(self.num_envs, self._leg_action_dim, device=self.device)
        self.wheel_actions = torch.zeros(self.num_envs, self._wheel_action_dim, device=self.device)
        # Normalized targets reaching the actuator interface. Their physics-tick
        # differences are accumulated across every physics sub-step so changes
        # introduced by the measured command delay are not hidden at the 50 Hz
        # boundary. Retained for command/actuator diagnostics.
        self._issued_targets = torch.zeros_like(self._actions)
        self._previous_issued_targets = torch.zeros_like(self._actions)
        self._issued_target_rate_accum = torch.zeros(self.num_envs, device=self.device)

        # -- observation buffers -----------------------------------------
        self.obs = torch.zeros(self.num_envs, self.cfg.num_single_obs, device=self.device)
        self.prev_obs = torch.zeros_like(self.obs)
        self.priv = torch.zeros(
            self.num_envs, self.cfg.num_single_privileged_obs, device=self.device
        )
        self.prev_priv = torch.zeros_like(self.priv)
        self._policy_obs_history = torch.zeros(
            self.num_envs,
            max(int(self.cfg.num_obs_hist), 1),
            self.cfg.num_single_obs,
            device=self.device,
        )
        self._critic_obs_history = torch.zeros(
            self.num_envs,
            max(int(self.cfg.num_privileged_obs_hist), 1),
            self.cfg.num_single_privileged_obs,
            device=self.device,
        )
        # A reset environment is filled with its first valid frame instead of
        # four zero frames, which avoids an artificial transient at episode start.
        self._obs_history_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._critic_history_valid = torch.zeros_like(self._obs_history_valid)

        # -- commands -----------------------------------------------------
        self.command_generator = self.cfg.commands.class_type(self.cfg.commands, self)
        self.command = self.command_generator.command.clone()
        self.height_cmd = torch.full(
            (self.num_envs,), float(self.cfg.default_height_cmd), device=self.device
        )
        self.height_goal_cmd = self.height_cmd.clone()
        self._height_command_time_left = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # Last sampled reset state, retained for TensorBoard diagnostics.
        self._reset_leg_length = torch.zeros(self.num_envs, 2, device=self.device)
        self._reset_leg_angle = torch.zeros_like(self._reset_leg_length)
        self._reset_root_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self._reset_root_lin_vel_b = torch.zeros_like(self._reset_root_rpy)
        self._reset_root_ang_vel_b = torch.zeros_like(self._reset_root_rpy)

        # -- sim2real: observation / action delay buffers (physics-step level) --
        self._setup_delay_buffers()

        # -- sim2real: sensor / actuator noise models ---------------------
        self._setup_noise_models()

        # -- privileged (critic-only) extra observation entities -----------
        self._setup_privileged_extra_obs()

        # -- delayed sensor buffers (filled on the first physics step) -----
        self.obs_root_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.obs_projected_gravity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.obs_joint_pos = torch.zeros(
            self.num_envs, self._actuated_joint_count, device=self.device
        )
        self.obs_joint_vel = torch.zeros(
            self.num_envs, self._actuated_joint_count, device=self.device
        )
        # ``False`` right after a reset: the policy then sees the un-delayed state
        # (an empty delay buffer cannot be stale).
        self._obs_buffers_valid = False
        self._obs_dim_checked = False
        self._reset_clearance_checked = False

        # -- termination bookkeeping -------------------------------------
        self._termination_duration_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._numerical_safety_reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # -- logging ------------------------------------------------------
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.cfg.rewards
        }
        self._last_reward_terms: dict[str, torch.Tensor] = {}

        print(f"[Mongo0909] Policy leg action joints  : {self._policy_leg_names}")
        print("[Mongo0909] Direct front-arm joint    : front_hip_joint")
        print(f"[Mongo0909] USD asset                  : {self.cfg.robot_cfg.spawn.usd_path}")
        print(f"[Mongo0909] Resolved wheel joints      : {self._wheel_names}")
        print(
            "[Mongo0909] Action smoothing           : "
            f"LPF={'on' if self.use_action_low_pass_filter else 'off'}, "
            f"slew={'on' if self.use_action_slew_rate_limit else 'off'}, "
            f"leg max da={self.leg_action_max_delta:.2f}, "
            f"wheel max da={self.wheel_action_max_delta:.2f}"
        )
        print(f"[Mongo0909] Passive joints             : {self._legs_inact_idx}")
        print(f"[Mongo0909] Wheel contact bodies       : {self._wheel_contact_idx}")
        print(
            "[Mongo0909] Undesired contact bodies   : "
            f"{[self.contact_sensor.body_names[i] for i in self._undesired_contact_idx]}"
        )
        print(
            f"[Mongo0909] Observation frames       : policy "
            f"{self.cfg.num_single_obs} x {self.cfg.num_obs_hist} = "
            f"{self.cfg.observation_space}, critic "
            f"{self.cfg.num_single_privileged_obs} x "
            f"{self.cfg.num_privileged_obs_hist} = {self.cfg.state_space}"
        )
        print("[Mongo0909] Support mode               : two-wheel only")

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        import isaaclab.sim as sim_utils

        # robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # terrain (a simple plane, exactly like the wheelbipe flat task)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # sim2real helpers: delay buffers, noise models, privileged observations
    # ------------------------------------------------------------------
    def _setup_delay_buffers(self) -> None:
        """Create the observation / action delay buffers (physics-step level)."""
        # -- observation delay (sampled at every physics step) --------------
        self.obs_delay: dict[str, DelayBuffer] = {}
        self.use_obs_delay = False
        obs_delay_cfg = getattr(self.cfg, "obs_delay_cfg", None)
        if bool(getattr(self.cfg, "use_obs_delay", False)) and obs_delay_cfg:
            max_delay = max(int(v[1]) for v in obs_delay_cfg.values())
            history_len = int(getattr(self.cfg, "obs_history_len", 0))
            if history_len < max_delay:
                raise ValueError(
                    f"obs_history_len ({history_len}) must be >= the maximum observation delay "
                    f"({max_delay} physics steps)."
                )
            for key in obs_delay_cfg:
                buffer = DelayBuffer(history_len, self.num_envs, self.device)
                buffer.set_time_lag(int(getattr(self.cfg, "obs_default_time_lag", 0)))
                self.obs_delay[key] = buffer
            self.use_obs_delay = True
            dt_ms = self.cfg.sim.dt * 1000.0
            print(
                f"[Mongo0909][Sim2Real] observation delay enabled - groups: {list(self.obs_delay)}"
            )
            print(f"[Mongo0909][Sim2Real]   range {obs_delay_cfg}, buffer {history_len} steps")
            print(
                f"[Mongo0909][Sim2Real]   physics step {dt_ms:.1f} ms, default lag "
                f"{int(getattr(self.cfg, 'obs_default_time_lag', 0))} step"
            )

        # -- action delay ---------------------------------------------------
        self.act_delay: dict[str, DelayBuffer] = {}
        self.use_act_delay = False
        act_delay_cfg = getattr(self.cfg, "act_delay_cfg", None)
        if bool(getattr(self.cfg, "use_act_delay", False)) and act_delay_cfg:
            max_delay = max(int(v[1]) for v in act_delay_cfg.values())
            history_len = max_delay + 2
            for key in act_delay_cfg:
                self.act_delay[key] = DelayBuffer(history_len, self.num_envs, self.device)
            self.use_act_delay = True
            print(
                f"[Mongo0909][Sim2Real] action delay enabled - groups: {list(self.act_delay)}"
            )
            print(f"[Mongo0909][Sim2Real]   range {act_delay_cfg}, buffer {history_len} steps")

        if self.use_obs_delay:
            self._resample_time_lags(self.obs_delay, obs_delay_cfg, None)
        if self.use_act_delay:
            self._set_action_delay_time_lags(None)

    def _resample_time_lags(self, buffers: dict, delay_cfg: dict, env_ids) -> None:
        """Sample inclusive per-environment delays and support fixed ``[n, n]``."""
        for key, buffer in buffers.items():
            low, high = delay_cfg[key]
            low, high = int(low), int(high)
            if low < 0 or high < low:
                raise ValueError(
                    f"Invalid delay range for {key}: expected 0 <= low <= high, got "
                    f"[{low}, {high}]."
                )
            count = self.num_envs if env_ids is None else len(env_ids)
            if low == high:
                lag = torch.full((count,), low, dtype=torch.int, device=self.device)
            else:
                # torch.randint excludes its upper bound; +1 implements the
                # inclusive ranges documented in the task configuration.
                lag = torch.randint(
                    low, high + 1, (count,), dtype=torch.int, device=self.device
                )
            if env_ids is None:
                buffer.set_time_lag(lag)
            else:
                buffer.set_time_lag(lag, env_ids)

    def _set_action_delay_time_lags(self, env_ids) -> None:
        """Sample configured Mongo actuator delays without curriculum overrides."""
        self._resample_time_lags(self.act_delay, self.cfg.act_delay_cfg, env_ids)


    def _setup_noise_models(self) -> None:
        """Instantiate the sensor (and optional actuator) noise models."""
        self.self_obs_noise: dict = {}
        self.use_self_obs_noise = False
        obs_noise_cfg = getattr(self.cfg, "self_obs_noise_cfg", None)
        if obs_noise_cfg:
            for key, model_cfg in obs_noise_cfg.items():
                self.self_obs_noise[key] = model_cfg.class_type(
                    model_cfg, num_envs=self.num_envs, device=self.device
                )
            self.use_self_obs_noise = True
            print(f"[Mongo0909][Sim2Real] observation noise enabled - groups: {list(self.self_obs_noise)}")

        self.self_act_noise: dict = {}
        self.use_self_act_noise = False
        act_noise_cfg = getattr(self.cfg, "self_act_noise_cfg", None)
        if act_noise_cfg:
            for key, model_cfg in act_noise_cfg.items():
                self.self_act_noise[key] = model_cfg.class_type(
                    model_cfg, num_envs=self.num_envs, device=self.device
                )
            self.use_self_act_noise = True
            print(f"[Mongo0909][Sim2Real] actuator noise enabled - groups: {list(self.self_act_noise)}")

    def _setup_privileged_extra_obs(self) -> None:
        """Resolve the entities used by the privileged (critic-only) block."""
        cfg = self.cfg
        self._privileged_extra_obs_enabled = bool(
            getattr(cfg, "privileged_extra_obs_enabled", False)
        )
        self._privileged_extra_obs_dim = int(getattr(cfg, "privileged_extra_obs_dim", 0) or 0)
        self._privileged_extra_joint_count = int(
            getattr(cfg, "privileged_extra_joint_count", 0) or 0
        )
        self._privileged_extra_wheel_count = int(
            getattr(cfg, "privileged_extra_wheel_count", 0) or 0
        )
        self._privileged_extra_body_count = int(
            getattr(cfg, "privileged_extra_body_count", 0) or 0
        )
        self._privileged_extra_material_body_count = int(
            getattr(cfg, "privileged_extra_material_body_count", 0) or 0
        )

        if not self._privileged_extra_obs_enabled:
            self.static_priv_obs = torch.zeros(self.num_envs, 0, device=self.device)
            return

        joint_names = tuple(getattr(cfg, "privileged_extra_joint_names", ()) or ())
        if joint_names:
            self._privileged_extra_actuate_idx, _ = self.robot.find_joints(
                list(joint_names), preserve_order=True
            )
        else:
            self._privileged_extra_actuate_idx = list(self._actuate_idx)

        wheel_names = tuple(getattr(cfg, "privileged_extra_wheel_body_names", ()) or ())
        self._privileged_extra_wheel_link_idx, _ = self.robot.find_bodies(list(wheel_names))
        body_names = tuple(getattr(cfg, "privileged_extra_body_names", ()) or ())
        self._privileged_extra_body_indices, _ = self.robot.find_bodies(list(body_names))
        material_names = tuple(
            getattr(cfg, "privileged_extra_material_body_names", ()) or ()
        )
        self._privileged_extra_material_indices, _ = self.robot.find_bodies(
            list(material_names)
        )

        # The physics material is stored per *shape*, so the flat shape offset of
        # every rigid body is resolved once (same approach as Isaac Lab's
        # ``randomize_rigid_body_material``).
        self._num_shapes_per_body: list[int] = []
        try:
            for link_path in self.robot.root_physx_view.link_paths[0]:
                link_view = self.robot._physics_sim_view.create_rigid_body_view(link_path)
                self._num_shapes_per_body.append(int(link_view.max_shapes))
        except Exception as exc:  # pragma: no cover - depends on the Isaac Sim version
            print(f"[Mongo0909][WARN] could not resolve the per-body shape layout: {exc}")
            self._num_shapes_per_body = []

        # wheel contact state follows the contact-sensor body order
        self._privileged_extra_wheel_contact_idx = []
        for body_id in self._privileged_extra_wheel_link_idx:
            body_name = self.robot.body_names[body_id]
            if body_name in self.contact_sensor.body_names:
                self._privileged_extra_wheel_contact_idx.append(
                    self.contact_sensor.body_names.index(body_name)
                )

        self.static_priv_obs = self._get_static_priv_obs()
        print(
            f"[Mongo0909] Privileged critic block: {self._privileged_extra_obs_dim} dims "
            f"(joints={self._privileged_extra_joint_count}, wheels={self._privileged_extra_wheel_count}, "
            f"bodies={self._privileged_extra_body_count}, "
            f"material_bodies={self._privileged_extra_material_body_count})"
        )

    def _check_reset_ground_clearance(self) -> None:
        """One-off sanity check that the reset pose does not bury the wheels.

        The drive wheels have to start *on* the ground.  If the configured
        default root height is too low, PhysX depenetrates them (up to
        ``max_depenetration_velocity = 5 m/s``) and throws the robot into the
        air at the start of every episode.  That transient is very hard to spot
        in the training curves, so it is checked and reported explicitly.
        """
        if self._reset_clearance_checked:
            return
        self._reset_clearance_checked = True

        ground_z = self.scene.env_origins[:, 2].mean()
        wheel_centre_z = self.robot.data.body_pos_w[:, self._wheel_body_idx, 2]
        clearance = wheel_centre_z - float(self.cfg.wheel_radius) - ground_z
        print(
            f"[Mongo0909] reset ground clearance (drive wheels): "
            f"min {clearance.min():+.4f} m  mean {clearance.mean():+.4f} m"
        )
        if float(clearance.min()) < -0.01:
            configured = self.robot.data.default_root_state[:, 2].mean()
            needed = float(configured) - float(clearance.min())
            print(
                "[Mongo0909][WARNING] the drive wheels start below the ground -> "
                "PhysX will depenetrate them and launch the robot at every reset! "
                f"Configured default root height {float(configured):.4f} m, "
                f"needs about {needed:.4f} m."
            )

    def _update_obs(self) -> None:
        """Sample sensor signals at the physics rate and push them through delays."""
        data = self.robot.data
        ang_vel = data.root_ang_vel_b.clone()
        proj_g = data.projected_gravity_b.clone()
        # Joint observations use absolute articulation angles in radians.
        joint_pos = self.joint_pos[:, self._actuate_idx].clone()
        joint_vel = (self._encoder_velocity.clone()
                     if self.cfg.joint_velocity_source == "position_delta"
                     else self.joint_vel[:, self._actuate_idx].clone())

        if self.use_obs_delay:
            buffer = self.obs_delay.get("root_ang_vel_b")
            if buffer is not None:
                ang_vel = buffer.compute(ang_vel)
            buffer = self.obs_delay.get("projected_gravity_b")
            if buffer is not None:
                proj_g = buffer.compute(proj_g)
            buffer = self.obs_delay.get("joint_pos")
            if buffer is not None:
                joint_pos = buffer.compute(joint_pos)
            buffer = self.obs_delay.get("joint_vel")
            if buffer is not None:
                joint_vel = buffer.compute(joint_vel)
            # grouped aliases: they delay the concatenated block and take priority
            buffer = self.obs_delay.get("motor")
            if buffer is not None:
                motor = buffer.compute(torch.cat([joint_pos, joint_vel], dim=-1))
                joint_pos = motor[:, : self._actuated_joint_count]
                joint_vel = motor[:, self._actuated_joint_count :]
            buffer = self.obs_delay.get("imu")
            if buffer is not None:
                imu = buffer.compute(torch.cat([ang_vel, proj_g], dim=-1))
                ang_vel = imu[:, :3]
                proj_g = imu[:, 3:]

        self.obs_root_ang_vel_b = ang_vel
        self.obs_projected_gravity_b = proj_g
        self.obs_joint_pos = joint_pos
        self.obs_joint_vel = joint_vel
        self._obs_buffers_valid = True

    # -- privileged-observation building blocks --------------------------
    def _pad_flat_features(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Flatten ``tensor`` to (num_envs, target_dim), zero padding or truncating."""
        flat = tensor.reshape(tensor.shape[0], -1)
        target_dim = int(target_dim)
        if flat.shape[-1] == target_dim:
            return flat
        if flat.shape[-1] > target_dim:
            return flat[:, :target_dim]
        pad = torch.zeros(
            flat.shape[0], target_dim - flat.shape[-1], dtype=flat.dtype, device=flat.device
        )
        return torch.cat([flat, pad], dim=-1)

    def _select_entity_features(
        self,
        tensor: torch.Tensor | None,
        indices,
        target_count: int,
        feature_dim: int | None = None,
    ) -> torch.Tensor:
        """Select the features of ``indices`` and pad/truncate to ``target_count`` entities."""
        target_count = int(target_count)
        if target_count <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        if tensor is None:
            width = target_count * int(feature_dim or 1)
            return torch.zeros(self.num_envs, width, dtype=torch.float, device=self.device)

        tensor = torch.as_tensor(tensor, dtype=torch.float, device=self.device)
        if tensor.ndim < 2:
            return self._pad_flat_features(tensor, target_count * int(feature_dim or 1))

        valid = [int(i) for i in indices if 0 <= int(i) < tensor.shape[1]][:target_count]
        if valid:
            selected = tensor[:, valid]
        else:
            selected = torch.zeros(
                (self.num_envs, 0) + tuple(tensor.shape[2:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        inferred = (
            int(torch.tensor(selected.shape[2:]).prod().item()) if selected.ndim > 2 else 1
        )
        width = target_count * int(feature_dim or inferred)
        return self._pad_flat_features(selected, width)

    def _clip_scale_critic_component(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        return self._scale_obs_component(
            key,
            self._clip_obs_component(key, tensor, stream_name="critic"),
            stream_name="critic",
        )

    def _get_body_masses_tensor(self) -> torch.Tensor | None:
        masses = torch.as_tensor(
            self.robot.root_physx_view.get_masses(), dtype=torch.float, device=self.device
        )
        if masses.ndim == 3 and masses.shape[-1] == 1:
            masses = masses.squeeze(-1)
        return masses

    def _get_body_mass_scale_tensor(self) -> torch.Tensor | None:
        masses = self._get_body_masses_tensor()
        default_masses = getattr(self.robot.data, "default_mass", None)
        if masses is None or default_masses is None:
            return None
        default_masses = torch.as_tensor(default_masses, dtype=torch.float, device=self.device)
        if default_masses.ndim == 3 and default_masses.shape[-1] == 1:
            default_masses = default_masses.squeeze(-1)
        if default_masses.shape != masses.shape:
            return None
        scale = masses / torch.clamp(default_masses, min=1.0e-6)
        return torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)

    def _get_body_material_tensor(self) -> torch.Tensor | None:
        """Return the physics material of the bodies as (num_envs, n_bodies, 3)."""
        if not self._privileged_extra_material_indices:
            return None
        materials = self.robot.root_physx_view.get_material_properties()
        if materials is None:
            return None
        materials = torch.as_tensor(materials, dtype=torch.float, device=self.device)
        count = self._privileged_extra_material_body_count
        if not self._num_shapes_per_body:
            # Without the per-body shape layout the body -> shape mapping is not
            # available; fall back to the first `count` shapes so the block keeps
            # its dimension instead of crashing.
            return materials[:, :count, :]

        offsets = [
            sum(self._num_shapes_per_body[: int(body_id)])
            for body_id in self._privileged_extra_material_indices
        ]
        offsets = [offset for offset in offsets if 0 <= offset < materials.shape[1]][:count]
        if not offsets:
            return None
        return materials[:, offsets, :]

    def _get_wheel_lin_vel_b(self) -> torch.Tensor:
        """Wheel link velocities in the base frame (legacy planar projection)."""
        if not self._privileged_extra_wheel_link_idx:
            return torch.zeros(self.num_envs, 0, device=self.device)
        root_quat_inv = quat_inv(self.robot.data.root_quat_w)
        relative_vel_w = (
            self.robot.data.body_lin_vel_w[:, self._privileged_extra_wheel_link_idx]
            - self.robot.data.root_lin_vel_w.unsqueeze(1)
        )
        vel_b = quat_apply(
            root_quat_inv.unsqueeze(1).expand(-1, relative_vel_w.shape[1], -1), relative_vel_w
        )
        vel_b = vel_b.clone()
        vel_b[:, :, 1] = 0.0
        return vel_b.reshape(self.num_envs, -1)

    def _get_wheel_contact_state(self) -> torch.Tensor:
        count = self._privileged_extra_wheel_count
        if count <= 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        peaks = self._get_wheel_contact_force_peaks(
            self.contact_sensor.data.net_forces_w_history
        )
        if peaks.numel() == 0 or peaks.shape[1] == 0:
            return torch.zeros(self.num_envs, count, device=self.device)
        filled = self._pad_flat_features(peaks, count)
        return (filled > float(self.cfg.desired_contact_force_threshold)).float()

    def _get_delay_time_lag_obs(self, buffers: dict, delay_cfg: dict | None, enabled: bool) -> torch.Tensor:
        """Active delay of every delay group, in physics steps."""
        if not enabled or not buffers or not delay_cfg:
            return torch.zeros(self.num_envs, 0, device=self.device)
        blocks = []
        for key in delay_cfg:
            buffer = buffers.get(key)
            if buffer is None:
                blocks.append(torch.zeros(self.num_envs, 1, device=self.device))
                continue
            lags = torch.as_tensor(buffer.time_lags, dtype=torch.float, device=self.device)
            blocks.append(lags.reshape(self.num_envs, 1))
        return torch.cat(blocks, dim=-1) if blocks else torch.zeros(self.num_envs, 0, device=self.device)

    def _get_dynamic_priv_obs(self) -> torch.Tensor:
        data = self.robot.data
        joint_count = self._privileged_extra_joint_count
        idx = self._privileged_extra_actuate_idx

        joint_stiffness = self._select_entity_features(
            getattr(data, "joint_stiffness", None), idx, joint_count
        )
        joint_damping = self._select_entity_features(
            getattr(data, "joint_damping", None), idx, joint_count
        )
        applied_torque = self._select_entity_features(data.applied_torque, idx, joint_count)
        obs_delay_steps = self._get_delay_time_lag_obs(
            self.obs_delay, getattr(self.cfg, "obs_delay_cfg", None), self.use_obs_delay
        )
        act_delay_steps = self._get_delay_time_lag_obs(
            self.act_delay, getattr(self.cfg, "act_delay_cfg", None), self.use_act_delay
        )
        wheel_lin_vel_b = self._get_wheel_lin_vel_b()
        wheel_contact_state = self._get_wheel_contact_state()

        blocks = [
            self._clip_scale_critic_component("joint_stiffness", joint_stiffness),
            self._clip_scale_critic_component("joint_damping", joint_damping),
            self._clip_scale_critic_component("joint_torque", applied_torque),
            self._clip_scale_critic_component("obs_delay_steps", obs_delay_steps),
            self._clip_scale_critic_component("act_delay_steps", act_delay_steps),
            self._clip_scale_critic_component("wheel_body_lin_vel", wheel_lin_vel_b),
            self._clip_scale_critic_component("wheel_contact_state", wheel_contact_state),
        ]
        return torch.cat(blocks, dim=-1)

    def _get_static_priv_obs(self) -> torch.Tensor:
        body_mass_scale = self._select_entity_features(
            self._get_body_mass_scale_tensor(),
            self._privileged_extra_body_indices,
            self._privileged_extra_body_count,
        )
        width = self._privileged_extra_material_body_count * 3
        material = self._get_body_material_tensor()
        if material is None:
            body_material = torch.zeros(self.num_envs, width, device=self.device)
        else:
            # ``_get_body_material_tensor`` already selected the target bodies
            body_material = self._pad_flat_features(material, width)
        blocks = [
            self._clip_scale_critic_component("body_mass_scale", body_mass_scale),
            self._clip_scale_critic_component("body_material", body_material),
        ]
        return torch.cat(blocks, dim=-1)

    def _get_privileged_extra_obs(self) -> torch.Tensor:
        if not self._privileged_extra_obs_enabled:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        dynamic_obs = self._get_dynamic_priv_obs()
        extra_obs = torch.cat([dynamic_obs, self.static_priv_obs], dim=-1)
        extra_obs = self._pad_flat_features(extra_obs, self._privileged_extra_obs_dim)
        return torch.nan_to_num(extra_obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # action pipeline
    # ------------------------------------------------------------------


    def _compute_front_arm_command_upper(self, height_command: torch.Tensor) -> torch.Tensor:
        """Return the height-derived upper bound from the degree-valued formula."""
        height_mm = height_command * 1000.0
        asin_arg = (
            height_mm - float(self.cfg.front_arm_command_height_offset_mm)
        ) / float(self.cfg.front_arm_command_length_mm)
        upper_deg = torch.rad2deg(
            torch.asin(torch.clamp(asin_arg, min=-1.0, max=1.0))
        ) - float(self.cfg.front_arm_command_angle_offset_deg) - float(
            self.cfg.front_arm_command_upper_margin_deg
        )
        upper_deg = torch.clamp(upper_deg, min=float(self.cfg.front_arm_command_lower_deg),
                                max=float(self.cfg.front_arm_command_upper_limit_deg))
        return torch.deg2rad(upper_deg)

    def _step_height_command(self) -> None:
        """Resample height goals periodically and rate-limit the issued command."""
        if bool(getattr(self.cfg, "play", False)):
            return
        period = max(float(self.cfg.height_command_resampling_time_s), self.step_dt)
        self._height_command_time_left -= self.step_dt
        due = torch.nonzero(self._height_command_time_left <= 0.0).flatten()
        if due.numel() > 0:
            self.height_goal_cmd[due] = self._sample_height_command(due)
            self._height_command_time_left[due] += period
        max_delta = float(self.cfg.height_command_max_rate_m_s) * self.step_dt
        self.height_cmd.add_(
            torch.clamp(
                self.height_goal_cmd - self.height_cmd,
                min=-max_delta,
                max=max_delta,
            )
        )

    def _step_front_arm_command(self) -> None:
        """Resample arm goals periodically and rate-limit the issued command."""
        if bool(getattr(self.cfg, "play", False)):
            return
        # Height can now move during an episode.  Refresh the feasible arm range
        # every policy step and constrain the goal, while leaving the issued
        # target to approach that goal only through the rate limiter below.
        self._update_front_arm_commands(
            self.robot._ALL_INDICES, resample=False, snap_to_target=False
        )
        period = max(float(self.cfg.front_arm_command_resampling_time_s), self.step_dt)
        self._front_arm_command_time_left -= self.step_dt
        due = torch.nonzero(self._front_arm_command_time_left <= 0.0).flatten()
        if due.numel() > 0:
            self._update_front_arm_commands(due, resample=True, snap_to_target=False)
            self._front_arm_command_time_left[due] += period
        self._rate_limit_front_arm_target()

    def _rate_limit_front_arm_target(self) -> None:
        """Move the issued front-arm target toward its goal at the configured rate."""
        max_delta = math.radians(float(self.cfg.front_arm_command_max_rate_deg_s)) * self.step_dt
        delta = torch.clamp(
            self.front_arm_goal_cmd - self.front_arm_target_cmd,
            min=-max_delta,
            max=max_delta,
        )
        self.front_arm_target_cmd.add_(delta)

    def _update_front_arm_commands(
        self,
        env_ids: torch.Tensor,
        *,
        resample: bool = True,
        snap_to_target: bool = False,
    ) -> None:
        """Sample/constrain the arm goal and optionally snap the reset state.

        In-episode training resamples only the goal; the per-step update
        rate-limits the motor target.  Resets snap because the simulated joint
        state is reset to the same sampled angle.
        """
        upper = self._compute_front_arm_command_upper(self.height_cmd[env_ids])
        lower = torch.full_like(
            upper, math.radians(float(self.cfg.front_arm_command_lower_deg))
        )
        upper = torch.maximum(upper, lower)
        self.front_arm_command_upper[env_ids] = upper
        if resample:
            goals = lower + torch.rand_like(upper) * (upper - lower)
        else:
            goals = torch.maximum(self.front_arm_goal_cmd[env_ids], lower)
            goals = torch.minimum(goals, upper)
        self.front_arm_goal_cmd[env_ids] = goals
        if snap_to_target:
            self.front_arm_target_cmd[env_ids] = goals

    def _low_pass_action_filter(self, actions: torch.Tensor) -> torch.Tensor:
        return condition_policy_actions(
            actions,
            self.last_actions,
            self._action_current_weight,
            self._action_max_delta,
            True,
            False,
        )

    def _limit_action_slew_rate(self, actions: torch.Tensor) -> torch.Tensor:
        """Limit one-policy-step action changes separately for legs and wheels."""
        return condition_policy_actions(
            actions,
            self.last_actions,
            self._action_current_weight,
            self._action_max_delta,
            False,
            True,
        )

    def _decode_leg_policy_actions(self, leg_policy_actions: torch.Tensor) -> torch.Tensor:
        """Map normalised policy actions to absolute joint position targets."""
        default_leg_pos = self.robot.data.default_joint_pos[:, self._policy_leg_idx]
        return self.leg_action_scale * leg_policy_actions + default_leg_pos


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        # Reference smoothness uses the sampled actions and zero history at reset.
        self._before_previous_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._issued_target_rate_accum.zero_()
        self._policy_actions.copy_(actions)
        self._actions.copy_(condition_policy_actions(
            actions, self.last_actions, self._action_current_weight, self._action_max_delta,
            self.use_action_low_pass_filter, self.use_action_slew_rate_limit,
        ))
        self.last_actions.copy_(self._actions)
        self.leg_actions = self._decode_leg_policy_actions(self._actions[:, :self._leg_action_dim])
        self.wheel_actions = self.wheel_action_scale * self._actions[:, self._leg_action_dim:]


    def _sample_joint_feedback(self) -> None:
        # Isaac's lazy data buffers can replace their tensor at refresh.
        # Constructor aliases may still contain the previous substep until
        # the actuator model reads the fresh buffer. Refresh BEFORE sampling.
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        position = self.joint_pos[:, self._actuate_idx]
        estimate = encoder_velocity(position, self._encoder_previous, self.cfg.sim.dt)
        self._encoder_velocity.copy_(torch.where(
            self._encoder_valid[:, None], estimate,
            self.joint_vel[:, self._actuate_idx]))
        self._encoder_previous.copy_(position)
        self._encoder_valid.fill_(True)

    def _reset_sensor_feedback(self, env_ids, joint_pos, joint_vel, angular_b, gravity_b):
        """Reset only selected rows, retaining other robots' sampling phase."""
        self.obs_joint_pos[env_ids] = joint_pos[:, self._actuate_idx]
        self.obs_joint_vel[env_ids] = joint_vel[:, self._actuate_idx]
        self.obs_root_ang_vel_b[env_ids] = angular_b
        self.obs_projected_gravity_b[env_ids] = gravity_b
        self._obs_buffers_valid = True

    def _apply_action(self) -> None:
        if self.cfg.joint_velocity_source == "position_delta":
            self._sample_joint_feedback()
        # Sample the sensors once per physics step and push them through the
        # observation delay buffers (mirrors the wheelbipe ``_update_obs``).
        self._update_obs()

        # -- legs: position control, clamped to the Mongo0909 working range
        leg_targets = torch.clamp(self.leg_actions, self._leg_clip_lower, self._leg_clip_upper)
        if self.use_self_act_noise:
            leg_targets = self.self_act_noise["leg_actions"](leg_targets)
        if self.use_act_delay:
            buffer = self.act_delay.get("leg_actions")
            if buffer is not None:
                leg_targets = buffer.compute(leg_targets)
        self.robot.set_joint_position_target(leg_targets, joint_ids=self._policy_leg_idx)
        # The front-arm target is an environment command, not a policy action.
        # It is written independently and directly to the position motor.
        self.robot.set_joint_position_target(
            self.front_arm_target_cmd.unsqueeze(-1), joint_ids=[self._front_arm_joint_idx]
        )
        default_leg_pos = self.robot.data.default_joint_pos[:, self._policy_leg_idx]
        self._issued_targets[:, : self._leg_action_dim] = (
            leg_targets - default_leg_pos
        ) / self.leg_action_scale

        # -- wheels
        if self.use_wheel_vel_control:
            wheel_targets = torch.clamp(self.wheel_actions, -self.max_wheel_vel, self.max_wheel_vel)
            if self.use_self_act_noise:
                wheel_targets = self.self_act_noise["wheel_actions"](wheel_targets)
            if self.use_act_delay:
                buffer = self.act_delay.get("wheel_actions")
                if buffer is not None:
                    wheel_targets = buffer.compute(wheel_targets)
            self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_idx)
            self._issued_targets[:, self._leg_action_dim :] = (
                wheel_targets / self.wheel_action_scale
            )
            # Hold the current wheel position as the position target so the
            # actuator produces damping when the velocity target is zero.
            self.robot.set_joint_position_target(
                self.joint_pos[:, self._wheel_idx], joint_ids=self._wheel_idx
            )
        else:
            wheel_targets = torch.clamp(
                self.wheel_actions, -self.max_wheel_torque, self.max_wheel_torque
            )
            self.robot.set_joint_effort_target(wheel_targets, joint_ids=self._wheel_idx)
            self._issued_targets[:, self._leg_action_dim :] = (
                wheel_targets / self.wheel_action_scale
            )

        if self.cfg.joint_velocity_source == "position_delta":
            # IdealPDActuator subtracts native joint_vel internally. This target
            # compensation makes its damping exactly D*(desired-encoder_vel),
            # before effort saturation, without changing the engine state.
            correction = self.joint_vel[:, self._actuate_idx] - self._encoder_velocity
            self.robot.set_joint_velocity_target(
                correction[:, :self._controlled_leg_count], joint_ids=self._legs_act_idx)
            if self.use_wheel_vel_control:
                self.robot.set_joint_velocity_target(
                    wheel_targets + correction[:, self._controlled_leg_count:],
                    joint_ids=self._wheel_idx)

        issued_target_diff = self._issued_targets - self._previous_issued_targets
        self._issued_target_rate_accum += torch.sum(torch.square(issued_target_diff), dim=1)
        self._previous_issued_targets.copy_(self._issued_targets)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _apply_height_obs_clip(self, height: torch.Tensor) -> torch.Tensor:
        if not bool(self.cfg.height_obs_clip_enabled):
            return height
        clip_range = self.cfg.height_obs_clip_range
        if clip_range is None:
            return height
        lower, upper = clip_range
        lower_f = None if lower is None else float(lower)
        upper_f = None if upper is None else float(upper)
        if lower_f is None and upper_f is None:
            return height
        return torch.clamp(height, min=lower_f, max=upper_f)

    def _get_observed_height(self) -> torch.Tensor:
        """Return the height signal used by the observations and the reward.

        Mongo0909 runs on a plane and has no terrain, therefore the "lightweight
        absolute height" path of the V14 flat task is used: the world frame base
        height.
        """
        if bool(self.cfg.use_leg_length_as_height):
            root_quat_inv = self.robot.data.root_quat_w
            wheel_pos_b = quat_apply_inverse(
                root_quat_inv.unsqueeze(1).expand(-1, len(self._wheel_body_idx), -1),
                self.robot.data.body_pos_w[:, self._wheel_body_idx]
                - self.robot.data.root_pos_w.unsqueeze(1),
            )
            wheel_pos_b = wheel_pos_b.clone()
            wheel_pos_b[:, :, 1] = 0.0
            height = torch.norm(wheel_pos_b, dim=-1).mean(dim=1) + float(self.cfg.wheel_radius)
        else:
            height = self.robot.data.root_pos_w[:, 2]
        return self._apply_height_obs_clip(height)

    # -- observation normalisation helpers ------------------------------
    def _stream_is_scaled(self, stream_name: str | None) -> bool:
        if not bool(self.cfg.obs_input_scale_enabled):
            return False
        streams = tuple(getattr(self.cfg, "obs_input_scale_streams", ("policy", "critic")))
        return stream_name is None or stream_name in streams

    def _clip_obs_component(
        self, key: str, value: torch.Tensor, stream_name: str | None = None
    ) -> torch.Tensor:
        if not self._stream_is_scaled(stream_name):
            return value
        clip = self.cfg.obs_input_clip_cfg.get(key)
        if clip is None:
            return value
        if isinstance(clip, (list, tuple)):
            return torch.clamp(value, min=clip[0], max=clip[1])
        return torch.clamp(value, -float(clip), float(clip))

    def _scale_obs_component(
        self, key: str, value: torch.Tensor, stream_name: str | None = None
    ) -> torch.Tensor:
        if not self._stream_is_scaled(stream_name):
            return value
        scale = self.cfg.obs_input_scale_cfg.get(key)
        if scale is None:
            return value
        if isinstance(scale, dict):
            if len(scale) != value.shape[-1]:
                # layout mismatch: leave the block untouched rather than
                # silently broadcasting the wrong scale
                return value
            scale_tensor = torch.tensor(
                list(scale.values()), dtype=value.dtype, device=value.device
            )
            return value * scale_tensor
        return value * float(scale)

    def _append_observation_history(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Append one frame and flatten history in oldest-to-newest order."""
        history.copy_(torch.roll(history, shifts=-1, dims=1))
        history[:, -1].copy_(current)
        # On the first observation after reset, repeat the valid reset frame
        # across the window rather than exposing zero padding to the policy.
        invalid = ~valid
        history[invalid] = current[invalid].unsqueeze(1).expand(-1, history.shape[1], -1)
        valid.fill_(True)
        return history.reshape(self.num_envs, -1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # one-off reset sanity check (the first call happens right after the
        # initial reset, i.e. when the forward kinematics of the reset pose is
        # available)
        self._check_reset_ground_clearance()

        # refresh the command buffer that the policy will act on
        self.command = self.command_generator.command.clone()

        data = self.robot.data
        leg_obs_dim = self._controlled_leg_count

        # Use the (delayed) sensor buffers sampled in ``_update_obs`` and apply
        # the configured sensor noise to the raw blocks, exactly like the
        # wheelbipe task does.
        if self._obs_buffers_valid:
            joint_pos_obs = self.obs_joint_pos.clone()
            joint_vel_obs = self.obs_joint_vel.clone()
            raw_ang_vel = self.obs_root_ang_vel_b
            raw_proj_g = self.obs_projected_gravity_b
        else:
            # immediately after a reset the delay buffers are empty, so the
            # current (undelayed) state is used
            joint_pos_obs = self.joint_pos[:, self._actuate_idx].clone()
            joint_vel_obs = self.joint_vel[:, self._actuate_idx].clone()
            raw_ang_vel = data.root_ang_vel_b
            raw_proj_g = data.projected_gravity_b
        if self.use_self_obs_noise:
            raw_ang_vel = self._apply_obs_noise("root_ang_vel_b", raw_ang_vel)
            raw_proj_g = self._apply_obs_noise("projected_gravity_b", raw_proj_g)
            joint_pos_obs = self._apply_obs_noise("joint_pos", joint_pos_obs)
            joint_vel_leg = self._apply_obs_noise(
                "leg_joint_vel", joint_vel_obs[:, :leg_obs_dim]
            )
            joint_vel_wheel = self._apply_obs_noise(
                "wheel_joint_vel", joint_vel_obs[:, leg_obs_dim:]
            )
        else:
            joint_vel_leg = joint_vel_obs[:, :leg_obs_dim]
            joint_vel_wheel = joint_vel_obs[:, leg_obs_dim:]

        if bool(self.cfg.mute_wheel_pos_obs):
            # zero the wheel entries *after* the noise so they stay exactly zero
            joint_pos_obs = joint_pos_obs.clone()
            joint_pos_obs[:, leg_obs_dim:] = 0.0

        height_cmd = self.height_cmd.unsqueeze(-1)
        obs_height = self._get_observed_height().unsqueeze(-1)
        # The measured front-arm angle and issued target both use absolute
        # radians; action scaling is independent of observation encoding.
        front_arm_target_obs = self.front_arm_target_cmd.unsqueeze(-1)
        self.obs = torch.cat(
            [
                self._scale_obs_component(
                    "command", self._clip_obs_component("command", self.command)
                ),
                self._scale_obs_component(
                    "height_cmd", self._clip_obs_component("height_cmd", height_cmd)
                ),
                self._scale_obs_component(
                    "root_ang_vel_b", self._clip_obs_component("root_ang_vel_b", raw_ang_vel)
                ),
                self._scale_obs_component(
                    "projected_gravity_b",
                    self._clip_obs_component("projected_gravity_b", raw_proj_g),
                ),
                self._scale_obs_component(
                    "joint_pos", self._clip_obs_component("joint_pos", joint_pos_obs)
                ),
                self._scale_obs_component(
                    "joint_vel_leg",
                    self._clip_obs_component("joint_vel_leg", joint_vel_leg),
                ),
                self._scale_obs_component(
                    "joint_vel_wheel",
                    self._clip_obs_component("joint_vel_wheel", joint_vel_wheel),
                ),
                self._scale_obs_component(
                    "actions", self._clip_obs_component("actions", self._actions)
                ),
                # The target is not inferable from height because it is sampled
                # inside the height-dependent valid interval.
                self._scale_obs_component(
                    "front_arm_target",
                    self._clip_obs_component("front_arm_target", front_arm_target_obs),
                ),
            ],
            dim=-1,
        )
        self.obs = torch.nan_to_num(self.obs, nan=0.0, posinf=0.0, neginf=0.0)

        if self.cfg.state_space:
            true_joint_pos = self.joint_pos[:, self._actuate_idx].clone()
            if self.cfg.mute_wheel_pos_obs:
                true_joint_pos[:, leg_obs_dim:] = 0.0
            clean_blocks = [
                ("command", self.command), ("height_cmd", height_cmd),
                ("root_ang_vel_b", data.root_ang_vel_b),
                ("projected_gravity_b", data.projected_gravity_b),
                ("joint_pos", true_joint_pos),
                ("joint_vel_leg", self.joint_vel[:, self._legs_act_idx]),
                ("joint_vel_wheel", self.joint_vel[:, self._wheel_idx]),
                ("actions", self._actions), ("front_arm_target", front_arm_target_obs),
            ]
            clean_obs = torch.cat([
                self._scale_obs_component(k, self._clip_obs_component(k, v, stream_name="critic"),
                                          stream_name="critic") for k, v in clean_blocks
            ], dim=-1)
            priv_blocks = [
                clean_obs,
                self._scale_obs_component(
                    "root_lin_vel_b",
                    self._clip_obs_component(
                        "root_lin_vel_b", data.root_lin_vel_b, stream_name="critic"
                    ),
                    stream_name="critic",
                ),
                self._scale_obs_component(
                    "obs_height",
                    self._clip_obs_component("obs_height", obs_height, stream_name="critic"),
                    stream_name="critic",
                ),
            ]
            if self._privileged_extra_obs_enabled:
                priv_blocks.append(self._get_privileged_extra_obs())
            self.priv = torch.cat(priv_blocks, dim=-1)
            self.priv = torch.nan_to_num(self.priv, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(self.cfg.use_frame_stack):
            policy_obs = self._append_observation_history(
                self.obs, self._policy_obs_history, self._obs_history_valid
            )
        else:
            policy_obs = self.obs

        if self.cfg.state_space and int(self.cfg.num_privileged_obs_hist) > 1:
            critic_obs = self._append_observation_history(
                self.priv,
                self._critic_obs_history,
                self._critic_history_valid,
            )
        else:
            critic_obs = self.priv

        if not self._obs_dim_checked:
            expected_policy_frame = int(self.cfg.num_single_obs)
            expected_critic_frame = int(self.cfg.num_single_privileged_obs)
            if self.obs.shape[-1] != expected_policy_frame:
                raise RuntimeError(
                    f"Mongo0909 policy frame dimension is {self.obs.shape[-1]}, "
                    f"expected {expected_policy_frame}."
                )
            if self.priv.shape[-1] != expected_critic_frame:
                raise RuntimeError(
                    f"Mongo0909 critic frame dimension is {self.priv.shape[-1]}, "
                    f"expected {expected_critic_frame}."
                )
            if policy_obs.shape[-1] != int(self.cfg.observation_space):
                raise RuntimeError(
                    f"Mongo0909 stacked policy dimension is {policy_obs.shape[-1]}, "
                    f"expected {self.cfg.observation_space}."
                )
            if critic_obs.shape[-1] != int(self.cfg.state_space):
                raise RuntimeError(
                    f"Mongo0909 stacked critic dimension is {critic_obs.shape[-1]}, "
                    f"expected {self.cfg.state_space}."
                )
            self._obs_dim_checked = True

        return {"policy": policy_obs.clone(), "critic": critic_obs.clone()}

    def _apply_obs_noise(self, key: str, value: torch.Tensor) -> torch.Tensor:
        model = self.self_obs_noise.get(key)
        return model(value) if model is not None else value

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------
    def _get_wheel_contact_force_peaks(self, net_contact_forces: torch.Tensor) -> torch.Tensor:
        """Peak wheel contact force over the history window, shape (num_envs, num_wheels)."""
        if net_contact_forces is None:
            return torch.zeros(self.num_envs, len(self._wheel_contact_idx), device=self.device)
        wheel_forces = net_contact_forces[:, :, self._wheel_contact_idx, :]
        return torch.norm(wheel_forces, dim=-1).amax(dim=1)

    def _get_rewards(self) -> torch.Tensor:
        data = self.robot.data
        wheel_pos_b = quat_apply_inverse(
            data.root_quat_w.unsqueeze(1).expand(-1, len(self._wheel_body_idx), -1),
            data.body_link_pos_w[:, self._wheel_body_idx] - data.root_pos_w.unsqueeze(1),
        )
        steady_gate, motion_gate, leg_pitch_rate = self._reward_state.update(
            self.command, wheel_pos_b, self.step_dt)
        stop_error, stop_gate = self._reward_state.update_stop_position(
            self.command, data.root_pos_w, self.step_dt)
        pitch = wrap_to_pi(euler_xyz_from_quat(data.root_quat_w)[1])
        forces = self.contact_sensor.data.net_forces_w_history
        wheel_peaks = self._get_wheel_contact_force_peaks(forces)
        front_peak = torch.linalg.vector_norm(
            forces[:, :, self._front_wheel_contact_idx, :], dim=-1).amax(dim=1)
        undesired = torch.zeros(self.num_envs, device=self.device)
        if forces is not None and self._undesired_contact_idx:
            undesired = (torch.linalg.vector_norm(forces[:, :, self._undesired_contact_idx], dim=-1)
                         > self.cfg.undesired_contact_force_threshold).any(dim=1).any(dim=1).float()
        terms = compute_flat_reward_terms(
            self.cfg, command=self.command, root_lin_vel_b=data.root_lin_vel_b,
            root_ang_vel_b=data.root_ang_vel_b, projected_gravity_b=data.projected_gravity_b,
            pitch=pitch, height=self._get_observed_height(), height_cmd=self.height_cmd,
            wheel_pos_b=wheel_pos_b, joint_acc=data.joint_acc, joint_vel=self.joint_vel,
            joint_pos=self.joint_pos, applied_torque=data.applied_torque,
            # Front arm is directly commanded. Regularize only the six policy motors.
            leg_idx=self._policy_leg_idx, wheel_idx=self._wheel_idx,
            actions=self._actions, previous_actions=self._previous_actions,
            before_previous_actions=self._before_previous_actions,
            terminated=self.reset_terminated, undesired_contact=undesired,
            wheel_contact_peaks=wheel_peaks, front_wheel_contact_peak=front_peak,
            stand_still_position_error_xy=stop_error, stand_still_position_gate=stop_gate,
            command_steady_gate=steady_gate, motion_steady_gate=motion_gate,
            virtual_leg_pitch_rate=leg_pitch_rate,
        )
        rewards = weight_reward_terms(terms, self.cfg.rewards, self.step_dt,
                                     self._numerical_safety_reset_buf)
        self._last_reward_terms = {k: v.detach() for k, v in rewards.items()}
        for key, value in rewards.items():
            self._episode_sums[key] += value
        total = torch.stack(list(rewards.values()), dim=-1).sum(dim=-1)
        # RSL-RL retains each step's dict until rollout logging. A fresh dict
        # prevents aliasing all samples and carrying old episode reset metrics.
        log = {}
        self.extras["log"] = log
        log.update({f"Reward/{k}": v.mean() for k, v in rewards.items()})
        log["Reward/total"] = total.mean()
        log["Metrics/front_wheel_contact_peak_mean_n"] = front_peak.mean()
        log["Metrics/front_wheel_contact_peak_max_n"] = front_peak.amax()
        log["Metrics/front_wheel_impact_fraction"] = (
            front_peak > self.cfg.front_wheel_impact_force_threshold_n).float().mean()
        log["Metrics/command_steady_gate"] = steady_gate.mean()
        anchored = self._reward_state.stop_anchor_valid
        anchor_count = anchored.sum().clamp(min=1)
        log["Metrics/stand_still_position_gate"] = stop_gate.mean()
        log["Metrics/stand_still_position_active_fraction"] = (stop_gate > 0).float().mean()
        log["Metrics/stand_still_position_error_m"] = torch.where(
            anchored, stop_error.norm(dim=1), 0.0).sum() / anchor_count
        linear_zero = (self.command[:, :2] == 0.0).all(dim=1)
        log["Metrics/zero_linear_command_body_speed_m_s"] = torch.where(
            linear_zero, data.root_lin_vel_b[:, :2].norm(dim=1), 0.0).sum() / linear_zero.sum().clamp(min=1)
        log["Metrics/motion_steady_gate"] = motion_gate.mean()
        log["Metrics/virtual_leg_pitch_rate_abs_rad_s"] = leg_pitch_rate.abs().mean()
        zero_command = (self.command == 0.0).all(dim=1)
        zero_count = zero_command.sum().clamp(min=1)
        log["Metrics/zero_command_fraction"] = zero_command.float().mean()
        log["Metrics/zero_command_body_speed_m_s"] = torch.where(
            zero_command, data.root_lin_vel_b[:, :2].norm(dim=1), 0.0).sum() / zero_count
        log["Metrics/zero_command_yaw_speed_rad_s"] = torch.where(
            zero_command, data.root_ang_vel_b[:, 2].abs(), 0.0).sum() / zero_count
        log["Metrics/height_abs_error_m"] = torch.abs(self._get_observed_height() - self.height_cmd).mean()
        log["Metrics/lin_vel_abs_error_m_s"] = torch.abs(
            self.command[:, 0] - data.root_lin_vel_b[:, 0] * torch.cos(pitch)).mean()
        log["Metrics/front_arm_position_error_deg"] = torch.rad2deg(torch.abs(
            self.joint_pos[:, self._front_arm_joint_idx] - self.front_arm_target_cmd)).mean()
        # Keep Mongo's height-conditioned direct-arm command scheduling.
        self._step_height_command()
        self._step_front_arm_command()
        self.command_generator.compute(self.step_dt)
        return total


    # ------------------------------------------------------------------
    # terminations
    # ------------------------------------------------------------------

    def _apply_termination_duration(self, terminate: torch.Tensor) -> torch.Tensor:
        steps = int(self.cfg.termination_duration_steps)
        if not bool(self.cfg.termination_duration_enabled) or steps <= 1:
            return terminate
        self._termination_duration_buf = torch.where(
            terminate,
            self._termination_duration_buf + 1,
            torch.zeros_like(self._termination_duration_buf),
        )
        return self._termination_duration_buf >= steps

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # -- undesired contacts (base and legs touching the ground)
        net_contact_forces = self.contact_sensor.data.net_forces_w
        if net_contact_forces is not None and self._undesired_contact_idx:
            contact_norm = torch.norm(net_contact_forces[:, self._undesired_contact_idx, :], dim=-1)
            terminate = torch.any(contact_norm > 1.0, dim=-1)
        else:
            terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- numerical safety
        joint_vel = self.joint_vel[:, self._actuate_idx]
        root_lin_vel_b = data.root_lin_vel_b
        root_ang_vel_b = data.root_ang_vel_b
        nan_terminate = (
            ~torch.isfinite(joint_vel).all(dim=-1)
            | ~torch.isfinite(root_lin_vel_b).all(dim=-1)
            | ~torch.isfinite(root_ang_vel_b).all(dim=-1)
        )
        outlier_terminate = (
            (torch.amax(torch.abs(joint_vel), dim=-1) > float(self.cfg.terminate_joint_vel_abs))
            | (
                torch.amax(torch.abs(root_ang_vel_b), dim=-1)
                > float(self.cfg.terminate_root_ang_vel_abs)
            )
            | (
                torch.amax(torch.abs(root_lin_vel_b), dim=-1)
                > float(self.cfg.terminate_root_lin_vel_abs)
            )
        )
        numerical_terminate = nan_terminate | outlier_terminate
        immediate_terminate = numerical_terminate
        self._numerical_safety_reset_buf.copy_(numerical_terminate)
        terminate = terminate | immediate_terminate

        # -- orientation
        root_rpy = euler_xyz_from_quat(data.root_quat_w)
        roll_limit = math.radians(float(self.cfg.termination_roll_deg))
        pitch_limit = math.radians(float(self.cfg.termination_pitch_deg))
        terminate = terminate | (torch.abs(wrap_to_pi(root_rpy[0])) > roll_limit)
        terminate = terminate | (torch.abs(wrap_to_pi(root_rpy[1])) > pitch_limit)

        # -- terminate duration (numerical safety is immediate)
        delayed_terminate = self._apply_termination_duration(terminate & ~immediate_terminate)
        terminate = delayed_terminate | immediate_terminate

        if self.cfg.play and not bool(self.cfg.play_keep_done_reset):
            terminate = torch.zeros_like(terminate)

        return terminate, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _sample_height_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        h_min, h_max = self.cfg.height_range
        if h_max <= h_min:
            return torch.full((len(env_ids),), float(h_min), device=self.device)
        return torch.empty(len(env_ids), device=self.device).uniform_(float(h_min), float(h_max))

    def _sample_uniform_reset_range(
        self, count: int, value_range: Sequence[float], *, columns: int | None = None
    ) -> torch.Tensor:
        """Sample a scalar config range on the environment device."""
        shape = (count,) if columns is None else (count, columns)
        low, high = float(value_range[0]), float(value_range[1])
        if high < low:
            raise ValueError(f"Invalid reset range ({low}, {high}): upper must be >= lower.")
        if high == low:
            return torch.full(shape, low, dtype=torch.float, device=self.device)
        return torch.empty(shape, dtype=torch.float, device=self.device).uniform_(low, high)

    def _apply_reset_randomization(
        self,
        env_ids: torch.Tensor,
        root_state: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a geometrically coherent randomized root and rear-leg state.

        Returns the local absolute root height, which becomes the initial height
        goal/issued command so reset does not introduce a height-command step.
        """
        count = len(env_ids)
        if not bool(self.cfg.reset_randomization_enabled):
            return root_state[:, 2] - self.scene.env_origins[env_ids, 2]

        common_length = self._sample_uniform_reset_range(
            count, self.cfg.reset_leg_length_range
        ).unsqueeze(-1)
        side_length_offset = self._sample_uniform_reset_range(
            count, self.cfg.reset_leg_length_side_offset_range, columns=2
        )
        length_min, length_max = map(float, self.cfg.reset_leg_length_range)
        leg_length = torch.clamp(common_length + side_length_offset, length_min, length_max)

        common_angle_deg = self._sample_uniform_reset_range(
            count, self.cfg.reset_leg_angle_range_deg
        ).unsqueeze(-1)
        side_angle_offset_deg = self._sample_uniform_reset_range(
            count, self.cfg.reset_leg_angle_side_offset_range_deg, columns=2
        )
        angle_min_deg, angle_max_deg = map(float, self.cfg.reset_leg_angle_range_deg)
        leg_angle = torch.deg2rad(
            torch.clamp(
                common_angle_deg + side_angle_offset_deg,
                angle_min_deg,
                angle_max_deg,
            )
        )

        ik_joint_pos = mongo_leg_inverse_kinematics(
            leg_length,
            leg_angle,
            acos_epsilon=float(self.cfg.reset_ik_acos_epsilon),
        )
        if not torch.isfinite(ik_joint_pos).all():
            raise RuntimeError("Mongo reset IK produced NaN/Inf joint positions.")
        reset_active_pos = torch.stack(
            (
                ik_joint_pos[:, 0, 1],
                ik_joint_pos[:, 1, 1],
                ik_joint_pos[:, 0, 0],
                ik_joint_pos[:, 1, 0],
            ),
            dim=-1,
        )
        outside_work_range = (reset_active_pos < self._leg_clip_lower) | (
            reset_active_pos > self._leg_clip_upper
        )
        if torch.any(outside_work_range):
            bad = reset_active_pos[outside_work_range][0]
            raise ValueError(
                "Mongo reset IK sampled an active-joint position outside its working "
                f"range ({float(torch.rad2deg(bad)):.3f} deg). Tighten the reset leg "
                "length/angle ranges."
            )
        joint_pos[:, self._reset_leg_joint_idx] = ik_joint_pos.transpose(-2, -1).reshape(
            count, -1
        )

        # Only the four motorized rear-leg velocities are randomized.  The
        # passive closed-chain joints and directly commanded front arm start at
        # zero velocity, matching the reference reset design without injecting
        # a passive-constraint velocity discontinuity.
        joint_vel[:, self._policy_leg_idx] = self._sample_uniform_reset_range(
            count, self.cfg.reset_leg_joint_velocity_range, columns=len(self._policy_leg_idx)
        )
        joint_pos[:, self._wheel_idx] = self._sample_uniform_reset_range(
            count, self.cfg.reset_wheel_angle_range, columns=len(self._wheel_idx)
        )
        joint_vel[:, self._wheel_idx] = self._sample_uniform_reset_range(
            count, self.cfg.reset_wheel_joint_velocity_range, columns=len(self._wheel_idx)
        )

        roll = torch.deg2rad(
            self._sample_uniform_reset_range(count, self.cfg.reset_root_roll_range_deg)
        )
        pitch = torch.deg2rad(
            self._sample_uniform_reset_range(count, self.cfg.reset_root_pitch_range_deg)
        )
        yaw = torch.deg2rad(
            self._sample_uniform_reset_range(count, self.cfg.reset_root_yaw_range_deg)
        )
        orientation_delta = quat_from_euler_xyz(roll, pitch, yaw)
        root_state[:, 3:7] = quat_mul(root_state[:, 3:7], orientation_delta)

        # Build the two virtual wheel-centre vectors in the body frame, rotate
        # them with the sampled root attitude, and place the lowest wheel at the
        # requested clearance.  Accounting for roll/pitch here avoids the small
        # initial wheel penetration produced by a level-body height estimate.
        leg_root_y = torch.tensor(
            [float(self.cfg.left_leg_root_y), float(self.cfg.right_leg_root_y)],
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0).expand(count, -1)
        wheel_position_b = torch.stack(
            (
                -leg_length * torch.sin(leg_angle),
                leg_root_y,
                -leg_length * torch.cos(leg_angle),
            ),
            dim=-1,
        )
        root_quat_for_legs = root_state[:, None, 3:7].expand(-1, 2, -1)
        wheel_position_w_from_root = quat_apply(root_quat_for_legs, wheel_position_b)
        clearance = self._sample_uniform_reset_range(
            count, self.cfg.reset_wheel_clearance_range
        )
        local_root_height = (
            -torch.amin(wheel_position_w_from_root[..., 2], dim=-1)
            + float(self.cfg.wheel_radius)
            + clearance
        )
        root_height_min, root_height_max = map(float, self.cfg.reset_root_height_range)
        local_root_height = torch.clamp(local_root_height, root_height_min, root_height_max)
        root_state[:, 2] = self.scene.env_origins[env_ids, 2] + local_root_height

        lin_ranges = self.cfg.reset_root_lin_vel_b_range
        ang_ranges = self.cfg.reset_root_ang_vel_b_range
        if len(lin_ranges) != 3 or len(ang_ranges) != 3:
            raise ValueError("reset root linear/angular velocity ranges must each contain x/y/z.")
        lin_vel_b = torch.stack(
            [self._sample_uniform_reset_range(count, axis_range) for axis_range in lin_ranges],
            dim=-1,
        )
        ang_vel_b = torch.stack(
            [self._sample_uniform_reset_range(count, axis_range) for axis_range in ang_ranges],
            dim=-1,
        )
        root_state[:, 7:10] = quat_apply(root_state[:, 3:7], lin_vel_b)
        root_state[:, 10:13] = quat_apply(root_state[:, 3:7], ang_vel_b)

        self._reset_leg_length[env_ids] = leg_length
        self._reset_leg_angle[env_ids] = leg_angle
        self._reset_root_rpy[env_ids] = torch.stack((roll, pitch, yaw), dim=-1)
        self._reset_root_lin_vel_b[env_ids] = lin_vel_b
        self._reset_root_ang_vel_b[env_ids] = ang_vel_b
        return local_root_height

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # reset the articulation and the scene (this also applies the
        # reset-mode domain randomization events)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # -- clear the per-episode buffers
        self._policy_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._before_previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.leg_actions[env_ids] = 0.0
        self.wheel_actions[env_ids] = 0.0
        self._issued_targets[env_ids] = 0.0
        self._previous_issued_targets[env_ids] = 0.0
        self._issued_target_rate_accum[env_ids] = 0.0
        self._front_arm_command_time_left[env_ids] = float(
            self.cfg.front_arm_command_resampling_time_s
        )
        self._height_command_time_left[env_ids] = float(
            self.cfg.height_command_resampling_time_s
        )
        self.obs[env_ids] = 0.0
        self.prev_obs[env_ids] = 0.0
        self._policy_obs_history[env_ids] = 0.0
        self._obs_history_valid[env_ids] = False
        if self.cfg.state_space:
            self.priv[env_ids] = 0.0
            self.prev_priv[env_ids] = 0.0
            self._critic_obs_history[env_ids] = 0.0
            self._critic_history_valid[env_ids] = False
        self._termination_duration_buf[env_ids] = 0
        self._numerical_safety_reset_buf[env_ids] = False
        self._reward_state.reset(env_ids)

        # -- sim2real: clear the delay / noise state and re-sample the delays --
        if self.use_obs_delay:
            for buffer in self.obs_delay.values():
                buffer.reset(env_ids)
            self._resample_time_lags(self.obs_delay, self.cfg.obs_delay_cfg, env_ids)
        if self.use_act_delay:
            for buffer in self.act_delay.values():
                buffer.reset(env_ids)
            self._set_action_delay_time_lags(env_ids)
        if self.use_self_obs_noise:
            for model in self.self_obs_noise.values():
                model.reset(env_ids)
        if self.use_self_act_noise:
            for model in self.self_act_noise.values():
                model.reset(env_ids)
        # the delay buffers are empty again -> the first observation after the
        # reset is served from the current state
        self._obs_buffers_valid = False

        # -- new velocity command
        self.command_generator.reset(env_ids)
        self.command = self.command_generator.command.clone()

        # -- randomized, closed-chain-consistent reset pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        sampled_height = self._apply_reset_randomization(
            env_ids, default_root_state, joint_pos, joint_vel
        )

        # Synchronize height/front-arm commands with the randomized physical
        # pose.  The direct arm remains exactly at its sampled motor target.
        self.height_goal_cmd[env_ids] = sampled_height
        self.height_cmd[env_ids] = sampled_height
        self._update_front_arm_commands(env_ids, resample=True, snap_to_target=True)
        joint_pos[:, self._front_arm_joint_idx] = self.front_arm_target_cmd[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:13], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._encoder_previous[env_ids] = joint_pos[:, self._actuate_idx]
        self._encoder_velocity[env_ids] = joint_vel[:, self._actuate_idx]
        self._encoder_valid[env_ids] = False
        if self.cfg.joint_velocity_source == "position_delta":
            # A partial reset must not switch every other robot back to native
            # velocity for one policy frame. Replace only the reset sensor rows.
            down = torch.zeros((len(env_ids), 3), device=self.device)
            down[:, 2] = -1.
            self._reset_sensor_feedback(
                env_ids, joint_pos, joint_vel,
                quat_apply_inverse(default_root_state[:, 3:7], default_root_state[:, 10:13]),
                quat_apply_inverse(default_root_state[:, 3:7], down))

        # avoid a target jump on the first physics step after the reset
        self.robot.set_joint_position_target(
            joint_pos[:, self._legs_act_idx], joint_ids=self._legs_act_idx, env_ids=env_ids
        )
        self.robot.set_joint_velocity_target(
            joint_vel[:, self._wheel_idx], joint_ids=self._wheel_idx, env_ids=env_ids
        )
        self.robot.set_joint_position_target(
            joint_pos[:, self._wheel_idx], joint_ids=self._wheel_idx, env_ids=env_ids
        )

        # -- logging
        extras = {}
        episode_total = torch.stack(list(self._episode_sums.values()), dim=-1).sum(dim=-1)
        extras["Episode/reward_total"] = (
            episode_total[env_ids].mean() / self.max_episode_length_s
        )
        for key in self._episode_sums:
            extras[f"Episode/Reward/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode/Reset/terminate"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode/Reset/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        if bool(self.cfg.reset_randomization_enabled):
            extras["Reset/root_height_m"] = sampled_height.mean()
            extras["Reset/mean_leg_length_m"] = self._reset_leg_length[env_ids].mean()
            extras["Reset/mean_abs_leg_angle_deg"] = torch.rad2deg(
                torch.abs(self._reset_leg_angle[env_ids])
            ).mean()
            extras["Reset/mean_abs_roll_deg"] = torch.rad2deg(
                torch.abs(self._reset_root_rpy[env_ids, 0])
            ).mean()
            extras["Reset/mean_abs_pitch_deg"] = torch.rad2deg(
                torch.abs(self._reset_root_rpy[env_ids, 1])
            ).mean()
            extras["Reset/mean_lin_speed_m_s"] = torch.linalg.vector_norm(
                self._reset_root_lin_vel_b[env_ids], dim=-1
            ).mean()
        # NOTE: merge instead of overwrite.  ``_get_rewards`` has already filled
        # ``extras["log"]`` with this step's ``Reward/*`` and ``Metrics/*`` terms;
        # overwriting here silently dropped them on every step where any
        # environment reset (i.e. on ~81 % of the iterations in the 5000-iteration
        # run, which is why the reward breakdown was so sparse in TensorBoard).
        # The episode summaries use their own ``Episode/*`` prefix, so both can be
        # logged together.
        self.extras.setdefault("log", {}).update(extras)


if __name__ == "__main__":
    gym.logger.warn(
        "Use scripts/rsl_rl/train.py or scripts/rsl_rl/play.py to launch Isaac Sim and this task."
    )
