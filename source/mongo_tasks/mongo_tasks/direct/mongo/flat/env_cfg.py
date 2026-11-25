"""Mongo flat task with V14 inputs/rewards and Mongo stability costs."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import mongo_tasks.mdp as mdp
from mongo_tasks.direct.mongo.flat.sim_compat import external_force_iteration_kwargs
from mongo_tasks.direct.mongo.flat.cfg_utils import (
    MONGO_PHYSICS_DT,
    MONGO_POLICY_DECIMATION,
    MONGO_MOTOR_PROFILE_NAME,
    MONGO_WHEEL_TARGET_VEL_LIMIT,
    MONGO_WHEEL_EFFORT_LIMIT,
    MONGO_ACTION_DIM,
    MONGO_BASE_POLICY_OBS_DIM,
    MONGO_BASE_PRIVILEGED_OBS_DIM,
    MONGO_BASIC_OBS_CLIP,
    MONGO_BASIC_OBS_SCALE,
    MONGO_CRITIC_EXTRA_BASE_DIM,
    MONGO_EXTRA_OBS_CLIP,
    MONGO_EXTRA_OBS_SCALE,
    MONGO_MIN_PRIVILEGED_OBS_DIM,
    MONGO_PRIV_BODY_COUNT,
    MONGO_PRIV_EXTRA_OBS_DIM,
    MONGO_PRIV_JOINT_COUNT,
    MONGO_PRIV_MATERIAL_BODY_COUNT,
    MONGO_PRIV_WHEEL_COUNT,
)
from mongo_world.assets.mongo import (
    MONGO0909_ACTIVE_JOINT_NAMES,
    MONGO0909_BASE_BODY_NAME,
    MONGO0909_CFG,
    MONGO0909_DEFAULT_ROOT_HEIGHT,
    MONGO0909_INACT_JOINT_NAMES,
    MONGO0909_LEFT_LEG_ROOT_Y,
    MONGO0909_RIGHT_LEG_ROOT_Y,
    MONGO0909_WHEEL_BODY_NAMES,
    MONGO0909_WHEEL_RADIUS,
)

# Regular expressions used both by the domain randomization events and by the
# environment itself.
_LEG_ACT_JOINT_EXPR = [".*_hip_joint", ".*_parallel_1_joint"]
_WHEEL_JOINT_EXPR = ["left_podarthrum_joint", "right_podarthrum_joint"]
_LINKAGE_INACT_JOINT_EXPR = [
    ".*_knee_joint",
    ".*_parallel_2_joint",
    ".*_parallel_3_joint",
    ".*_parallel_4_joint",
]
_FRONT_CASTER_JOINT_EXPR = ["front_podarthrum_joint.*"]
_INACT_JOINT_EXPR = _LINKAGE_INACT_JOINT_EXPR + _FRONT_CASTER_JOINT_EXPR
_LEG_BODY_EXPR = [
    ".*_thigh.*",
    ".*_calf",
    ".*_parallel_.*",
    "front_leg",
]


@configclass
class EventCfg:
    """Domain randomization, mirroring ``EventCfgV14`` of wheeled-legged_RL.

    Only the wheelbipe specific entries (gimbal, spring, guide link,
    ``*_front2/3/4`` linkage) were dropped; every range is kept identical where
    the corresponding part exists on Mongo0909.
    """

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MONGO0909_BASE_BODY_NAME),
            "mass_distribution_params": (0.9, 1.3),
            "operation": "scale",
        },
    )
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_LEG_BODY_EXPR),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MONGO0909_BASE_BODY_NAME),
            "com_range": {"x": (-0.04, 0.04), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    # Base inertia randomization is available but disabled, matching V14.
    base_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MONGO0909_BASE_BODY_NAME),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_inertia = None

    base_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MONGO0909_BASE_BODY_NAME),
            "static_friction_range": (0.01, 0.1),
            "dynamic_friction_range": (0.01, 0.1),
            "restitution_range": (0.02, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.02, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    leg_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_ext,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_ACT_JOINT_EXPR),
            "static_friction_distribution_params": (0.25, 1.0),
            "dynamic_friction_distribution_params": (0.25, 1.0),
            "viscous_friction_distribution_params": (0.05, 0.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    wheel_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_ext,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINT_EXPR),
            "static_friction_distribution_params": (0.05, 0.25),
            "viscous_friction_distribution_params": (0.0, 0.01),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    legs_inact_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_ext,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LINKAGE_INACT_JOINT_EXPR),
            "static_friction_distribution_params": (0.05, 0.1),
            "viscous_friction_distribution_params": (0.01, 0.025),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    front_caster_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_ext,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_FRONT_CASTER_JOINT_EXPR),
            # Independently scale the nominal 0.015 / 0.010 / 0.001
            # N-m(-s/rad) ball-caster values within the same range.  This
            # covers bearing preload and wear without applying the much larger
            # closed-linkage friction range to the light 60 g ball.
            "static_friction_distribution_params": (0.67, 1.5),
            "dynamic_friction_distribution_params": (0.67, 1.5),
            "viscous_friction_distribution_params": (0.67, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    leg_effort_noise = EventTerm(
        func=mdp.randomize_actuator_effort_output,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_ACT_JOINT_EXPR),
            "effort_scale_distribution_params": (0.8, 1.1),
            "effort_bias_distribution_params": (0.0, 0.0),
            "effort_noise_std_distribution_params": (0.0, 0.0),
            "distribution": "uniform",
        },
    )
    wheel_effort_noise = EventTerm(
        func=mdp.randomize_actuator_effort_output,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINT_EXPR),
            "effort_scale_distribution_params": (0.9, 1.1),
            "effort_bias_distribution_params": (0.0, 0.0),
            "effort_noise_std_distribution_params": (0.0, 0.0),
            "distribution": "uniform",
        },
    )
    # Disable actuator-side scale/noise randomization together with the sensor
    # noise models.  PPO's stochastic exploration remains a training algorithm
    # property and is intentionally not treated as plant/sensor noise here.
    leg_effort_noise = None
    wheel_effort_noise = None
    # Root/joint reset randomization is applied inside ``MongoFlatEnv._reset_idx``
    # after reset-mode events.  A manager reset event would be overwritten by
    # that method when it writes the coherent IK pose to the articulation.
    reset_base = None

    # ------------------------------------------------------------------
    # interval
    # ------------------------------------------------------------------
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}},
    )
    base_external_force_torque_xyz = EventTerm(
        func=mdp.apply_external_force_torque_xyz,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[MONGO0909_BASE_BODY_NAME]
            ),
            "force_range": ((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
            "torque_range": ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        },
    )
    # Temporarily disable all interval disturbances while diagnosing the
    # controller's intrinsic smoothness.  Keep the definitions above so the
    # exact training perturbations can be restored without reconstructing them.
    push_robot = None
    base_external_force_torque_xyz = None


@configclass
class MongoFlatEnvCfg(DirectRLEnvCfg):
    """Flat-ground locomotion task for Mongo0909."""

    # ------------------------------------------------------------------
    # environment
    # ------------------------------------------------------------------
    decimation = MONGO_POLICY_DECIMATION
    motor_profile = MONGO_MOTOR_PROFILE_NAME
    episode_length_s = 20.0
    termination_roll_deg = 40.0
    termination_pitch_deg = 40.0
    termination_duration_enabled: bool = True
    termination_duration_steps: int = 20
    value_space = 1
    play: bool = False
    play_keep_done_reset: bool = False
    # New flat training uses encoder feedback. Play/resume restore the recorded
    # choice (native for archives predating this field).
    joint_velocity_source: str = "position_delta"

    # termination safety margins (wheelbipe style numerical safety)
    terminate_joint_vel_abs: float = 500.0
    terminate_root_ang_vel_abs: float = 200.0
    terminate_root_lin_vel_abs: float = 100.0

    # ------------------------------------------------------------------
    # reset-state randomization
    # ------------------------------------------------------------------
    reset_randomization_enabled: bool = True
    # The root height is derived from the rotated virtual wheel-centre vectors,
    # drive-wheel radius and clearance, then clamped to this absolute range.
    # The height command is synchronized to the resulting reset height.
    reset_root_height_range: tuple[float, float] = (0.29, 0.45)
    reset_wheel_clearance_range: tuple[float, float] = (0.002, 0.008)
    reset_root_roll_range_deg: tuple[float, float] = (-3.0, 3.0)
    reset_root_pitch_range_deg: tuple[float, float] = (-3.0, 3.0)
    reset_root_yaw_range_deg: tuple[float, float] = (-180.0, 180.0)
    # Linear and angular velocity samples are expressed in the randomized body
    # frame, then rotated to the world frame required by Isaac Lab.
    reset_root_lin_vel_b_range: tuple[tuple[float, float], ...] = (
        (-0.20, 0.20),
        (-0.05, 0.05),
        (0.0, 0.0),
    )
    reset_root_ang_vel_b_range: tuple[tuple[float, float], ...] = (
        (-0.10, 0.10),
        (-0.10, 0.10),
        (-0.25, 0.25),
    )

    # A common task-space pose is sampled first, followed by small independent
    # left/right offsets.  This retains asymmetric recovery examples without
    # producing the large one-wheel suspension gaps of two fully independent
    # samples.  Angles are measured from body-down toward body-backward.
    reset_leg_length_range: tuple[float, float] = (0.16, 0.275)
    reset_leg_angle_range_deg: tuple[float, float] = (-20.0, 15.0)
    reset_leg_length_side_offset_range: tuple[float, float] = (-0.005, 0.005)
    reset_leg_angle_side_offset_range_deg: tuple[float, float] = (-2.0, 2.0)
    reset_ik_acos_epsilon: float = 1.0e-6

    reset_wheel_angle_range: tuple[float, float] = (
        -2.0 * 3.141592653589793,
        2.0 * 3.141592653589793,
    )
    # Match the reference design by randomizing the four policy-controlled rear
    # leg velocities and both drive-wheel velocities; passive-link and direct
    # front-arm velocities remain zero to avoid a constraint impulse.
    reset_leg_joint_velocity_range: tuple[float, float] = (-0.50, 0.50)
    reset_wheel_joint_velocity_range: tuple[float, float] = (-10.0, 10.0)

    # ------------------------------------------------------------------
    # actions (four rear-leg position targets + two drive-wheel velocities)
    # ------------------------------------------------------------------
    action_space = MONGO_ACTION_DIM
    leg_action_encoding: str = "raw"
    leg_action_scale = 0.5
    use_wheel_vel_control: bool = True
    wheel_vel_action_scale: float = 10.0
    max_wheel_vel: float = MONGO_WHEEL_TARGET_VEL_LIMIT
    max_wheel_torque: float = MONGO_WHEEL_EFFORT_LIMIT

    # The wheelbipe task clamps leg targets to a global +/-3.14 rad.  Mongo0909
    # has asymmetric mechanical stops, so the clamp uses the per-joint working
    # ranges of the Mongo_RL_0909 configuration (3 deg inside the stops).
    leg_action_clip_lower = [
        -3.08923278,  # left_hip_joint
        -3.08923278,  # right_hip_joint
        -0.46408305,  # left_parallel_1_joint
        -0.46408305,  # right_parallel_1_joint
    ]
    leg_action_clip_upper = [
        0.25708700,  # left_hip_joint
        0.25708700,  # right_hip_joint
        3.08923278,  # left_parallel_1_joint
        3.08923278,  # right_parallel_1_joint
    ]

    # Action conditioning is temporarily disabled.  Training, Play and the
    # exported real-robot action processor therefore all pass policy actions
    # through unchanged before joint mapping and mechanical target clipping.
    # The coefficients/limits remain as dormant configuration so a future
    # controlled ablation can re-enable either mechanism without changing the
    # deployment interface.
    use_action_low_pass_filter: bool = False
    leg_action_low_pass_prev_weight: float = 0.7
    leg_action_low_pass_curr_weight: float = 0.3
    wheel_action_low_pass_prev_weight: float = 0.7
    wheel_action_low_pass_curr_weight: float = 0.3
    use_action_slew_rate_limit: bool = False
    leg_action_max_delta: float = 0.10
    wheel_action_max_delta: float = 0.20

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    # Actor: five 50 Hz frames (oldest to newest); Critic: current state only.
    num_obs_hist = 5
    num_privileged_obs_hist = 1
    num_single_obs = MONGO_BASE_POLICY_OBS_DIM
    num_single_privileged_obs = MONGO_BASE_PRIVILEGED_OBS_DIM
    observation_space = 5 * MONGO_BASE_POLICY_OBS_DIM
    state_space = MONGO_BASE_PRIVILEGED_OBS_DIM
    use_frame_stack: bool = True

    # observation input normalisation (mirrors V14_EXTRA_OBS_CLIP / SCALE).
    # The EXTRA tables are a superset of the BASIC ones; the shared blocks keep
    # identical values, the extra keys cover the privileged (critic) blocks.
    obs_input_clip_cfg: dict = MONGO_EXTRA_OBS_CLIP
    obs_input_scale_enabled: bool = True
    obs_input_scale_streams: tuple[str, ...] = ("policy", "critic")
    obs_input_scale_cfg: dict = MONGO_EXTRA_OBS_SCALE

    # ------------------------------------------------------------------
    # privileged (critic-only) extra observations
    # ------------------------------------------------------------------
    # Mirrors the ``privileged_extra_obs`` block of WheelbipeV14FlatEnvCfg:
    # the critic additionally sees the actuator gains, the applied torque, the
    # active delay (in physics steps), the wheel kinematics/contact state and the
    # randomized base mass / wheel physics material.
    privileged_extra_obs_enabled: bool = True
    privileged_extra_obs_dim: int = MONGO_PRIV_EXTRA_OBS_DIM
    privileged_extra_joint_count: int = MONGO_PRIV_JOINT_COUNT
    privileged_extra_wheel_count: int = MONGO_PRIV_WHEEL_COUNT
    privileged_extra_body_count: int = MONGO_PRIV_BODY_COUNT
    privileged_extra_material_body_count: int = MONGO_PRIV_MATERIAL_BODY_COUNT
    # Entity selection by name (no reliance on the internal body/joint order).
    privileged_extra_joint_names: tuple[str, ...] = (
        "left_hip_joint",
        "right_hip_joint",
        "left_parallel_1_joint",
        "right_parallel_1_joint",
        "front_hip_joint",
        "left_podarthrum_joint",
        "right_podarthrum_joint",
    )
    privileged_extra_wheel_body_names: tuple[str, ...] = (
        "left_wheel",
        "right_wheel",
    )
    privileged_extra_body_names: tuple[str, ...] = (MONGO0909_BASE_BODY_NAME,)
    privileged_extra_material_body_names: tuple[str, ...] = (
        "left_wheel",
        "right_wheel",
    )

    # ------------------------------------------------------------------
    # sim2real: observation / action delay (physics-step level)
    # ------------------------------------------------------------------
    # Delay models are temporarily disabled for both training and Play.  The
    # reserved trial ranges below remain dormant until hardware identification.
    # The old 5--20 ms interval rounds to 6--20 ms on a 2 ms physics grid.
    use_obs_delay: bool = False
    obs_delay_cfg = {
        "root_ang_vel_b": [3, 10],
        "projected_gravity_b": [3, 10],
        "joint_pos": [3, 10],
        "joint_vel": [3, 10],
    }
    obs_history_len: int = 10
    obs_default_time_lag: int = 3

    use_act_delay: bool = False
    # Dormant 10 ms end-to-end latency setting (five 500 Hz physics steps). It is
    # not applied while ``use_act_delay`` is false.  The range is inclusive and
    # supports a fixed [n, n] value when delay experiments are resumed.
    act_delay_cfg = {
        "leg_actions": [5, 5],
        "wheel_actions": [5, 5],
    }

    # ------------------------------------------------------------------
    # sim2real: sensor noise (applied to the raw, post-delay blocks)
    # ------------------------------------------------------------------
    # Sensor/action noise is temporarily disabled while the intrinsic closed-loop
    # oscillation is addressed.  Restore only from measured hardware residuals.
    self_obs_noise_cfg: dict | None = None
    self_act_noise_cfg: dict | None = None

    # ------------------------------------------------------------------
    # deployment simulation: 500 Hz physics/servo, 50 Hz policy
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=MONGO_PHYSICS_DT,
        render_interval=decimation,
        # Every environment stands on the same unbounded ground plane, so all of
        # them generate contact patches simultaneously.  At 4096 environments
        # PhysX asks for ~4.3e5 patches, which is above the default
        # ``5 * 2**15`` buffer; overflowing it silently drops contacts and lets
        # the robots sink through the floor.  This was measured and fixed in
        # Mongo_RL_0909, so the same (larger) buffers are reserved here.
        physx=PhysxCfg(
            solver_type=1,  # TGS; articulation iterations are 12/1.
            **external_force_iteration_kwargs(PhysxCfg),
            gpu_max_rigid_patch_count=2**20,
            gpu_max_rigid_contact_count=2**23,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=4.0, replicate_physics=True
    )

    # ------------------------------------------------------------------
    # viewer camera (viewport / --video / livestream)
    # ------------------------------------------------------------------
    # Default = Isaac Lab 的世界视角，对准世界原点。env_spacing=4.0 时画面里是
    # 0 号机器人及其周围的若干台，也就是"一次看到多台"，这是训练录像想要的画面。
    #
    # 想要单机特写（跟随某台机器人）有两种方式：
    #   1) 用 play.py 的 --viewer_robot（最方便）；
    #   2) 直接把这里改成 asset_root 视角：
    #        origin_type="asset_root", asset_name="robot", env_index=0,
    #        eye=(-1.5, -1.1, 0.8), lookat=(0.35, 0.0, 0.28)
    viewer = ViewerCfg(
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="world",
        resolution=(1280, 720),
    )

    # ------------------------------------------------------------------
    # robot
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = MONGO0909_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,  # Cover all ten physics substeps per control step.
        update_period=0.002,
        track_air_time=True,
    )

    # joint groups resolved from the asset
    legs_act_name = robot_cfg.actuators["legs_act"].joint_names_expr + robot_cfg.actuators[
        "front_arm"
    ].joint_names_expr
    legs_inact_name = MONGO0909_INACT_JOINT_NAMES
    wheel_name = robot_cfg.actuators["wheel"].joint_names_expr
    active_joint_names = MONGO0909_ACTIVE_JOINT_NAMES
    base_body_name = MONGO0909_BASE_BODY_NAME
    wheel_body_names = MONGO0909_WHEEL_BODY_NAMES
    wheel_radius = MONGO0909_WHEEL_RADIUS
    left_leg_root_y = MONGO0909_LEFT_LEG_ROOT_Y
    right_leg_root_y = MONGO0909_RIGHT_LEG_ROOT_Y

    # ------------------------------------------------------------------
    # commands (kept from Mongo_RL_0909)
    # ------------------------------------------------------------------
    # Full Mongo command range is available from the first rollout.
    commands = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-4.0, 4.0),
        ),
    )
    # Height uses the same goal/issued-target split as the front arm.  A new
    # goal is sampled periodically; the command observed by the policy and used
    # by the reward approaches it at a bounded physical rate.
    default_height_cmd = MONGO0909_DEFAULT_ROOT_HEIGHT
    # The natural stance is 0.34 m; commands cover 0.29--0.45 m.
    height_range = [0.29, 0.45]
    height_command_resampling_time_s: float = 4.0
    height_command_max_rate_m_s: float = 0.03

    # ------------------------------------------------------------------
    # two-wheel-only front-arm command
    # ------------------------------------------------------------------
    # Given a sampled body-height command h in millimetres, sample the front-arm
    # motor target uniformly in degrees from
    #   [-15.11 deg, asin(clamp((h - 145.3) / 240, -1, 1)) deg - 27 deg
    #    - safety_margin].
    # Isaac/PhysX still receives radians; the degree-valued formula is converted
    # only after evaluating asin and subtracting the geometric offset and margin.
    front_arm_command_lower_deg: float = -15.11
    front_arm_command_height_offset_mm: float = 145.3
    front_arm_command_length_mm: float = 240.0
    front_arm_command_angle_offset_deg: float = 27.0
    # Dynamic clearance below the static geometric furthest-down boundary.
    # The commanded height can temporarily exceed the actual height during
    # acceleration/reversing, so the zero-margin boundary is not motion-safe.
    front_arm_command_upper_margin_deg: float = 8.0
    # Absolute cap after the height-derived bound, in the motor coordinates.
    front_arm_command_upper_limit_deg: float = 50.0
    # During an episode a new arm goal is sampled every four seconds.  The
    # motor command approaches it at a bounded rate instead of stepping.
    front_arm_command_resampling_time_s: float = 4.0
    front_arm_command_max_rate_deg_s: float = 30.0
    # Physical radius is retained only for clearance/contact diagnostics.
    front_wheel_radius: float = 0.0303

    # Mongoose has no terrain, so the height signal is the absolute base height
    # (same "lightweight absolute height path" the V14 flat task uses).
    use_absolute_height: bool = True
    use_leg_length_as_height: bool = False
    height_obs_clip_enabled: bool = False
    height_obs_clip_range = [None, None]
    # The wheel absolute angle carries no useful information, so (exactly like
    # the V14 flat task) the wheel part of the joint position observation is
    # zeroed while the observation layout keeps its slot.
    mute_wheel_pos_obs: bool = True

    # ------------------------------------------------------------------
    # domain randomization
    # ------------------------------------------------------------------
    events: EventCfg = EventCfg()

    # V14 weights plus Mongo impact, standstill and steady-leg costs.
    # Every term, including termination, is multiplied by step_dt.
    rewards = {
        "termination": -200.0,
        "leg_joint_acc": -5e-07,
        "leg_joint_vel": -0.005,
        "leg_joint_pair_pos_diff": -0.0,
        "joint_torque": -0.0001,
        "wheel_acc": -1e-08,
        "wheel_vel": -1e-05,
        "wheel_power": -0.0001,
        "wheel_air_spin": 0.0,
        "lin_vel_z": -0.5,
        "ang_vel_xy": -0.05,
        "action_smoothness_leg": -0.05,
        "action_rate": -0.01,
        "action_smoothness_wheel": -0.01,
        "flat_orientation_y": -0.0,
        "flat_orientation_y_v": -2.0,
        "flat_orientation_y_exp": 1.0,
        "flat_orientation_x": -0.0,
        "flat_orientation_x_v": -2.0,
        "flat_orientation_x_exp": 1.0,
        "track_lin_vel_xy": 1.5,
        "track_lin_vel_xy_tight": 0.0,
        "track_lin_vel_xy_square": -1.0,
        "track_ang_vel_z": 1.0,
        "track_ang_vel_z_square": -1.0,
        "stand_still_lin_vel": -1.0,
        "stand_still": -0.0,
        "stand_still_wheel_vel": -0.10,
        "stand_still_body_lin_vel": -0.5,
        "stand_still_body_ang_vel": -0.1,
        "stand_still_position": 0.0,  # Disabled: unobserved accumulated drift destabilizes critic targets.
        "virtual_leg_pitch_rate": -0.1,
        "track_height_exp": 0.0,
        "track_height_exp_soft": 0.0,
        "track_height_exp_tight": 1.0,
        "track_height_square": -1.0,
        "track_height_exp_both_wheels_contact": 0.0,
        "no_fork": -1.0,
        "no_fork_square": -1.0,
        "no_fork_exp": -0.0,
        "no_fork_z_exp": -0.0,
        "undesired_contact": -2.0,
        "front_wheel_impact": -1.0,
    }

    lin_vel_err_constraint = 1.0
    ang_vel_err_constraint = 0.8
    height_err_constraint = 0.15
    lin_vel_xy_sigma = 0.5
    lin_vel_xy_tight_sigma = 0.1
    lin_vel_xy_soft_sigma = 1.5
    lin_vel_xy_square_sigma = 0.25
    ang_vel_z_sigma = 0.25
    ang_vel_z_square_sigma = 0.5
    height_sigma = 0.025
    height_soft_sigma = 0.1
    height_tight_sigma = 0.001
    height_square_sigma = 10.0
    orientation_y_exp_sigma = 0.02
    orientation_x_exp_sigma = 0.01
    orientation_x_bias = 2.0
    orientation_x_sigma = 3.0
    orientation_x_A = 2.0
    orientation_y_bias = 2.0
    orientation_y_sigma = 3.0
    orientation_y_A = 2.0
    orientation_x_square_sigma = 4.0
    orientation_y_square_sigma = 2.0
    no_fork_distance = 0.05
    no_fork_square_sigma = 5.0
    no_fork_exp_sigma = 0.0001
    undesired_contact_force_threshold = 3.0
    desired_contact_force_threshold = 5.0
    stand_still_deadzone_enabled = True
    stand_still_deadzone_threshold = 0.1
    # Actual wheel speed tolerance for balance corrections, in rad/s. This
    # separate cost requires exactly zero vx, vy AND yaw commands; the 0.1
    # tracking-command deadzone above does not activate it.
    stand_still_wheel_vel_deadzone_rad_s: float = 0.3

    # Reward phase depends only on issued vx/vy/yaw command rates.
    # After a change: .4 s grace, then .6 s smoothstep recovery to full weight.
    command_steady_delay_s: float = 0.4
    command_steady_ramp_s: float = 0.6
    command_steady_lin_rate_m_s2: float = 0.05
    command_steady_yaw_rate_rad_s2: float = 0.1
    virtual_leg_pitch_rate_deadzone_rad_s: float = 0.1
    virtual_leg_pitch_rate_scale_rad_s: float = 1.0
    # Planar body cost: exact zero vx/vy, including turns; yaw cost: all three zero.
    # No actual-speed deadzone or saturation:
    # even slow residual drift costs something, and faster drift costs more.
    stand_still_body_lin_vel_scale_m_s: float = 0.1
    stand_still_body_ang_vel_scale_rad_s: float = 0.2
    stand_still_body_transition_scale: float = 0.25
    # Strict zero vx/vy/yaw: brake first, then anchor world XY and fade in cost.
    stand_still_position_delay_s: float = 0.6
    stand_still_position_ramp_s: float = 0.4
    stand_still_position_tolerance_m: float = 0.01
    stand_still_position_scale_m: float = 0.05

    # Allow light contact; penalize the excess peak resultant force quadratically.
    front_wheel_impact_force_threshold_n: float = 50.0
    front_wheel_impact_force_scale_n: float = 100.0
    front_wheel_body_name = "front_wheel"
    # Keep Mongo sensor/action delay and noise settings; no command curriculum.

    def __post_init__(self):
        """Resolve dimensions and joint-order dependent defaults."""
        if hasattr(self, "contact_sensor") and hasattr(self, "sim"):
            self.contact_sensor.update_period = self.sim.dt
            self.contact_sensor.history_length = max(
                self.contact_sensor.history_length, int(self.decimation))
        if self.joint_velocity_source not in ("native", "position_delta"):
            raise ValueError("joint_velocity_source must be native or position_delta")
        if not (0.0 <= float(self.front_arm_command_upper_margin_deg) < float("inf")):
            raise ValueError("front_arm_command_upper_margin_deg must be finite and nonnegative.")
        if not (float(self.front_arm_command_lower_deg) <= float(self.front_arm_command_upper_limit_deg) < float("inf")):
            raise ValueError("front_arm_command_upper_limit_deg must be finite and >= the lower limit.")
        for name in (
            "front_wheel_impact_force_scale_n",
            "stand_still_position_ramp_s", "stand_still_position_scale_m",
            "command_steady_ramp_s", "command_steady_lin_rate_m_s2",
            "command_steady_yaw_rate_rad_s2", "virtual_leg_pitch_rate_scale_rad_s",
            "stand_still_body_lin_vel_scale_m_s", "stand_still_body_ang_vel_scale_rad_s",
        ):
            if not (0.0 < float(getattr(self, name)) < float("inf")):
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("command_steady_delay_s", "virtual_leg_pitch_rate_deadzone_rad_s",
                     "front_wheel_impact_force_threshold_n",
                     "stand_still_position_delay_s", "stand_still_position_tolerance_m"):
            if not (0.0 <= float(getattr(self, name)) < float("inf")):
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("stand_still_body_transition_scale",):
            if not (0.0 <= float(getattr(self, name)) <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1].")
        if not (0.0 <= float(self.stand_still_wheel_vel_deadzone_rad_s) < float("inf")):
            raise ValueError("stand_still_wheel_vel_deadzone_rad_s must be finite and nonnegative.")
        if int(self.num_obs_hist) < 1 or int(self.num_privileged_obs_hist) < 1:
            raise ValueError("Observation history lengths must be positive.")
        # Keep the observation dimensions consistent with the frame stacking
        # option (same convention as the wheelbipe task).
        if self.use_frame_stack:
            self.observation_space = self.num_obs_hist * self.num_single_obs
        else:
            self.observation_space = self.num_single_obs

        # Recompute the privileged block size from the configured entity counts
        # so that editing the counts never desynchronizes the critic dimension.
        obs_delay_groups = len(self.obs_delay_cfg or {})
        act_delay_groups = len(self.act_delay_cfg or {})
        self.privileged_extra_obs_dim = int(
            self.privileged_extra_joint_count * 3  # stiffness + damping + torque
            + obs_delay_groups
            + act_delay_groups
            + self.privileged_extra_wheel_count * 3  # wheel link linear velocity
            + self.privileged_extra_wheel_count  # wheel contact state
            + self.privileged_extra_body_count  # base mass scale
            + self.privileged_extra_material_body_count * 3  # wheel material
        )
        extra_dim = (
            self.privileged_extra_obs_dim if self.privileged_extra_obs_enabled else 0
        )
        self.num_single_privileged_obs = (
            self.num_single_obs + MONGO_CRITIC_EXTRA_BASE_DIM + extra_dim
        )
        self.state_space = self.num_privileged_obs_hist * self.num_single_privileged_obs


__all__ = ["MongoFlatEnvCfg", "EventCfg"]
