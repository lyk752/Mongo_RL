"""Play a trained Mongo0909 checkpoint with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import datetime
import json
import os
import sys

# make ``source/mongo_world`` / ``source/mongo_tasks`` importable without an
# editable install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mongo_bootstrap  # noqa: E402

mongo_bootstrap.ensure_source_packages(verbose=True)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from play_config import resolve_external_force_iterations, resolve_joint_velocity_source, resolve_motor_speed_limits
from playback import script_phases, TimeSeriesPlots, torque_differences
from reset_snapshots import PlayResetSnapshots

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent trained with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument(
    "--video_length", type=int, default=200, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Mongo-Flat-Play-v0", help="Name of the task.")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Evaluation seed. Defaults to the seed in the RSL-RL task config.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--external_force_iterations", choices=("auto", "on", "off"), default="auto",
    help=("TGS external forces per iteration: auto uses the recorded training value; "
          "legacy snapshots without this field use off for compatibility. "
          "Explicit on/off overrides this playback setting only."),
)
parser.add_argument(
    "--export_dir", type=str, default=None,
    help="Policy export directory; defaults to <checkpoint_dir>/exported.",
)
parser.add_argument('--motor_speed_limits', choices=('auto', 'on', 'off'), default='auto',
                    help='auto preserves recorded motor speed protection; on tests an old actor with the new guard.')
parser.add_argument(
    "--max_steps", type=int, default=0, help="Stop play after this many steps; 0 runs until closed."
)
parser.add_argument(
    "--viewer_robot",
    action="store_true",
    default=False,
    help=(
        "Point the viewport camera at robot 0 (close-up 3/4 rear view). "
        "Without this flag the task's default world camera is used, which frames "
        "several robots at once - that is what training-time recordings show."
    ),
)
parser.add_argument(
    "--script",
    type=str,
    default="none",
    choices=("none", "dynamic"),
    help=(
        "Run the built-in scripted command test instead of a constant policy rollout. "
        "'dynamic' runs a 15-phase front-arm / height / speed / yaw sequence "
        "(see _script_phases).  Defaults to num_envs=1 (an explicit --num_envs enables "
        "batched robustness evaluation) and disables episode resets so the commands are "
        "not overwritten."
    ),
)
parser.add_argument(
    "--script_duration",
    type=float,
    default=8.0,
    help="Duration of each basic script phase in seconds (default 8 s).",
)
parser.add_argument(
    "--script_coupled_duration",
    type=float,
    default=None,
    help=(
        "Duration of the final coupled-motion phase in seconds. "
        "Defaults to twice --script_duration (16 s; 128 s total by default)."
    ),
)
parser.add_argument(
    "--script_height_center",
    type=float,
    default=0.37,
    help="Centre of the height sweep used by the script (default range is [0.29, 0.45]).",
)
parser.add_argument(
    "--script_height_amp",
    type=float,
    default=0.08,
    help="Amplitude of the height sweep (0.08 -> 0.29 .. 0.45 m).",
)
parser.add_argument("--script_vel_max", type=float, default=0.8, help="Peak |vx| of the script (m/s).")
parser.add_argument("--script_yaw_max", type=float, default=2.0, help="Peak |wz| of the script (rad/s).")
parser.add_argument(
    "--front_arm_upper_margin_deg",
    type=float,
    default=0.0,
    help=(
        "Additional evaluation-only safety margin, on top of the configured "
        "training margin, subtracted from the height-dependent furthest-down "
        "front-arm angle (degrees, default 0)."
    ),
)
parser.add_argument(
    "--camera",
    type=str,
    default="global",
    choices=("global", "tracking", "both"),
    help=(
        "Viewport camera for the recording. 'global' = fixed wide shot of the world, "
        "'tracking' = close-up that follows robot 0, 'both' runs the whole sequence "
        "twice (once per camera) and writes two videos."
    ),
)
parser.add_argument(
    "--plot_dir",
    type=str,
    default=None,
    help=(
        "Directory for play CSV/curve outputs. Defaults to "
        "<checkpoint_dir>/play_plots/<timestamp>."
    ),
)
parser.add_argument(
    "--plot_env_index",
    type=int,
    default=0,
    help="Environment index whose time series is exported when num_envs > 1 (default: 0).",
)
parser.add_argument(
    "--no_plot_export",
    action="store_true",
    default=False,
    help="Disable the default CSV and PNG time-series export.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None or not os.path.isfile(os.path.abspath(args_cli.checkpoint)):
    parser.error("--checkpoint must point to an existing checkpoint file.")
if args_cli.script_duration <= 0 or args_cli.script_height_amp < 0:
    parser.error("--script_duration must be positive and --script_height_amp non-negative.")
if args_cli.script_coupled_duration is not None and args_cli.script_coupled_duration <= 0:
    parser.error("--script_coupled_duration must be positive.")
if args_cli.max_steps < 0 or args_cli.video_length <= 0:
    parser.error("--max_steps must be non-negative and --video_length positive.")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import time

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

from rsl_rl.runners import OnPolicyRunner

import mongo_world  # noqa: F401
import mongo_tasks  # noqa: F401
from mongo_tasks.direct.mongo.flat.action_pipeline import MongoActionProcessor
from mongo_tasks.direct.mongo.flat.physics_snapshot import actuator_effort_limits



def _play_metrics(unwrapped) -> dict[str, torch.Tensor]:
    """Compute per-environment diagnostics without depending on training log keys."""
    data = unwrapped.robot.data
    pitch = wrap_to_pi(euler_xyz_from_quat(data.root_quat_w)[1])
    force = torch.zeros_like(pitch)
    clearance = torch.zeros_like(pitch)
    contact_idx = unwrapped._front_wheel_contact_idx
    history = unwrapped.contact_sensor.data.net_forces_w_history
    if contact_idx is not None and history is not None:
        force = torch.linalg.vector_norm(history[:, :, contact_idx, :], dim=-1).amax(dim=1)
    body_idx = unwrapped._front_wheel_body_idx
    if body_idx is not None:
        clearance = (
            data.body_pos_w[:, body_idx, 2] - unwrapped.scene.env_origins[:, 2]
            - float(unwrapped.cfg.front_wheel_radius)
        )
    return {
        "Metrics/pitch_deg": torch.rad2deg(pitch),
        "Metrics/front_wheel_force": force,
        "Metrics/front_wheel_clearance": clearance,
        "Metrics/filtered_action_delta_rms": (
            unwrapped._actions - unwrapped._previous_actions
        ).square().mean(dim=-1).sqrt(),
        "Metrics/issued_target_delta_rms": (
            unwrapped._issued_target_rate_accum
            / (unwrapped.cfg.decimation * unwrapped.cfg.action_space)
        ).sqrt(),
    }


class PlayTimeSeriesRecorder(TimeSeriesPlots):
    """Collect one environment's deployable commands, state and policy outputs."""

    def __init__(self, unwrapped, env_index: int = 0):
        if env_index < 0 or env_index >= int(unwrapped.num_envs):
            raise ValueError(
                f"--plot_env_index={env_index} is outside [0, {int(unwrapped.num_envs) - 1}]."
            )
        self.unwrapped = unwrapped
        self.env_index = int(env_index)
        self.step_dt = float(unwrapped.step_dt)
        self.rows: list[dict[str, float | int | str]] = []
        # Velocities/accelerations cover the complete articulation, including
        # passive closed-chain joints.  Policy actions intentionally cover only
        # the four rear-leg outputs and two drive-wheel outputs.
        self.joint_names = list(unwrapped.robot.joint_names)
        self.joint_indices = list(range(len(self.joint_names)))
        self.action_names = list(unwrapped._policy_leg_names) + list(unwrapped._wheel_names)
        self.actuated_joint_names = list(unwrapped._legs_act_names) + list(
            unwrapped._wheel_names
        )
        self.actuated_joint_indices = list(unwrapped._actuate_idx)
        self._previous_applied_torque: torch.Tensor | None = None
        self._before_previous_applied_torque: torch.Tensor | None = None
        self._last_episode_step: int | None = None

    @staticmethod
    def _cpu_values(value: torch.Tensor) -> list[float]:
        return value.detach().float().cpu().reshape(-1).tolist()

    def capture_command(self) -> dict[str, float]:
        """Capture the command paired with the next policy action."""
        env_id = self.env_index
        command = self._cpu_values(self.unwrapped.command[env_id])
        return {
            "command_vx_m_s": command[0],
            "command_vy_m_s": command[1],
            "command_wz_rad_s": command[2],
            "command_height_m": float(self.unwrapped.height_cmd[env_id].detach().cpu()),
            "command_height_goal_m": float(
                self.unwrapped.height_goal_cmd[env_id].detach().cpu()
            ),
            "command_front_arm_deg": math.degrees(
                float(self.unwrapped.front_arm_target_cmd[env_id].detach().cpu())
            ),
        }

    def record(
        self,
        actions: torch.Tensor,
        phase: str = "policy_rollout",
        command_sample: dict[str, float] | None = None,
    ) -> None:
        """Record the post-step robot state and the action that produced it."""
        env_id = self.env_index
        data = self.unwrapped.robot.data
        roll, pitch, _ = euler_xyz_from_quat(data.root_quat_w[env_id : env_id + 1])
        if command_sample is None:
            command_sample = self.capture_command()
        root_lin_vel = self._cpu_values(data.root_lin_vel_b[env_id])
        root_ang_vel = self._cpu_values(data.root_ang_vel_b[env_id])
        joint_pos = self._cpu_values(data.joint_pos[env_id, self.joint_indices])
        joint_vel = self._cpu_values(data.joint_vel[env_id, self.joint_indices])
        joint_acc = self._cpu_values(data.joint_acc[env_id, self.joint_indices])
        policy_actions = self._cpu_values(actions[env_id])
        applied_torque_tensor = data.applied_torque[
            env_id, self.actuated_joint_indices
        ].detach()
        effort_limit_tensor = actuator_effort_limits(
            self.unwrapped.robot, self.actuated_joint_names, env_id)
        episode_step = int(self.unwrapped.episode_length_buf[env_id].detach().cpu())
        # A reset is not an actuator impulse.  Restart both histories when the
        # per-environment episode counter wraps.
        if self._last_episode_step is None or episode_step <= self._last_episode_step:
            self._previous_applied_torque = None
            self._before_previous_applied_torque = None
        normalized_torque_delta_tensor, normalized_torque_second_diff_tensor = torque_differences(
            applied_torque_tensor, effort_limit_tensor, self._previous_applied_torque,
            self._before_previous_applied_torque, torch)
        applied_torque = self._cpu_values(applied_torque_tensor)
        normalized_torque_delta = self._cpu_values(normalized_torque_delta_tensor)
        normalized_torque_second_diff = self._cpu_values(
            normalized_torque_second_diff_tensor
        )
        self._before_previous_applied_torque = self._previous_applied_torque
        self._previous_applied_torque = applied_torque_tensor.clone()
        self._last_episode_step = episode_step
        front_joint_pos = float(
            data.joint_pos[env_id, self.unwrapped._front_arm_joint_idx].detach().cpu()
        )
        log = _play_metrics(self.unwrapped)

        def _logged_scalar(key: str, default: float = 0.0) -> float:
            value = log.get(key)
            if value is None:
                return default
            if isinstance(value, torch.Tensor):
                return float((value if value.ndim == 0 else value[env_id]).detach().cpu())
            return float(value)

        actual_front_arm_deg = math.degrees(front_joint_pos)
        front_arm_position_error_deg = (
            actual_front_arm_deg - float(command_sample["command_front_arm_deg"])
        )

        row: dict[str, float | int | str] = {
            "step": len(self.rows),
            "time_s": len(self.rows) * self.step_dt,
            "phase": phase,
            **command_sample,
            "actual_vx_m_s": root_lin_vel[0],
            "actual_vy_m_s": root_lin_vel[1],
            "actual_wz_rad_s": root_ang_vel[2],
            "actual_height_m": float(data.root_pos_w[env_id, 2].detach().cpu()),
            "actual_front_arm_deg": actual_front_arm_deg,
            "front_arm_position_error_deg": front_arm_position_error_deg,
            "front_wheel_contact_force_n": _logged_scalar("Metrics/front_wheel_force"),
            "front_wheel_clearance_m": _logged_scalar("Metrics/front_wheel_clearance"),
            "roll_deg": math.degrees(float(wrap_to_pi(roll)[0].detach().cpu())),
            "pitch_deg": math.degrees(float(wrap_to_pi(pitch)[0].detach().cpu())),
        }
        for name, value in zip(self.joint_names, joint_pos):
            row[f"joint_position_rad/{name}"] = value
        for name, value in zip(self.joint_names, joint_vel):
            row[f"joint_velocity_rad_s/{name}"] = value
        for name, value in zip(self.joint_names, joint_acc):
            row[f"joint_acceleration_rad_s2/{name}"] = value
        for name, value in zip(self.action_names, policy_actions):
            row[f"policy_action/{name}"] = value
        for name, value in zip(self.actuated_joint_names, applied_torque):
            row[f"applied_torque_nm/{name}"] = value
        for name, value in zip(self.actuated_joint_names, self._cpu_values(effort_limit_tensor)):
            row[f"effort_limit_nm/{name}"] = value
        for name, value in zip(self.actuated_joint_names, normalized_torque_delta):
            row[f"normalized_torque_delta/{name}"] = value
        for name, value in zip(
            self.actuated_joint_names, normalized_torque_second_diff
        ):
            row[f"normalized_torque_second_diff/{name}"] = value
        self.rows.append(row)



def _load_checkpoint_for_play(runner, checkpoint_path: str) -> None:
    """Load a checkpoint matching the configured PPO observation dimensions."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
    try:
        runner.alg.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint does not match Mongo's configured observation dimensions and network "
            "(default: 155/77 inputs, [256, 128, 64] hidden layers). "
            "Use a checkpoint produced by this library."
        ) from error
    runner.alg.policy.eval()


# ---------------------------------------------------------------------------
# dynamic command script
# ---------------------------------------------------------------------------
#: Camera presets used by ``--camera``.  ``global`` is a fixed wide shot of the
#: world near the initial robot position, ``tracking`` is a close-up camera rigidly
#: attached to robot 0 (``asset_root`` origin), i.e. it follows the robot.
CAMERA_PRESETS = {
    # Raise and pull back the global camera for a wider, more elevated view.
    # Shift the framing right/down so robot 0 starts toward the upper left.
    "global": {"origin": "world", "eye": (8.1, 9.7, 5.2), "lookat": (0.0, 1.6, 0.0)},
    # Rigidly attached to robot 0 (``asset_root`` origin).  NOTE: ``eye`` and
    # ``lookat`` are *offsets from the robot's root*, which already sits ~0.33 m
    # above the ground - so the look-at z is slightly negative to aim at the
    # body centre and keep the robot in the middle of the frame while it spins.
    "tracking": {"origin": "asset_root", "eye": (-1.15, -0.85, 0.28), "lookat": (0.0, 0.0, -0.06)},
}


def _script_phases(
    duration: float, height_center: float, height_amp: float,
    vel_max: float, yaw_max: float, coupled_duration: float | None = None,
):
    """Shared Isaac/MuJoCo profiles; see scripts/playback.py."""
    return script_phases(duration, height_center, height_amp, vel_max, yaw_max, coupled_duration)


def _set_camera(unwrapped, camera: str) -> None:
    """Point the viewport camera according to ``CAMERA_PRESETS``."""
    controller = getattr(unwrapped, "viewport_camera_controller", None)
    if controller is None:
        print("[WARN] no viewport camera controller - camera stays at its default pose.")
        return
    preset = CAMERA_PRESETS[camera]
    if preset["origin"] == "asset_root":
        controller.set_view_env_index(0)
        controller.update_view_to_asset_root("robot")
    else:
        controller.update_view_to_world()
    controller.update_view_location(eye=preset["eye"], lookat=preset["lookat"])
    print(f"[INFO] viewport camera -> {camera} (origin={preset['origin']})")


def _inject_command(
    unwrapped,
    vx: float,
    vy: float,
    wz: float,
    height: float,
    front_arm_ratio: float,
) -> None:
    """Write the scripted command straight into the command buffers.

    ``_get_observations`` refreshes ``self.command`` from
    ``command_generator.command`` every step, so writing the generator buffer is
    enough; the height command is a plain per-env buffer written at reset.
    """
    generator = unwrapped.command_generator
    generator.vel_command_b[:, 0] = vx
    generator.vel_command_b[:, 1] = vy
    generator.vel_command_b[:, 2] = wz
    unwrapped.command[:, 0] = vx
    unwrapped.command[:, 1] = vy
    unwrapped.command[:, 2] = wz
    unwrapped.height_cmd[:] = height
    unwrapped.height_goal_cmd[:] = height
    # Refresh the height-dependent safe upper angle first, then place the arm
    # goal between its configured lower limit (ratio=0, fully raised) and that
    # upper bound (ratio=1, furthest down).  The issued target approaches this
    # goal through the environment's same rate limiter used by training and on
    # hardware; the policy observation therefore sees the smooth target.
    unwrapped._update_front_arm_commands(unwrapped.robot._ALL_INDICES, resample=False)
    ratio = max(0.0, min(1.0, float(front_arm_ratio)))
    lower = torch.full_like(
        unwrapped.front_arm_command_upper,
        math.radians(float(unwrapped.cfg.front_arm_command_lower_deg)),
    )
    scripted_goal = lower + ratio * (unwrapped.front_arm_command_upper - lower)
    unwrapped.front_arm_goal_cmd.copy_(scripted_goal)
    unwrapped._rate_limit_front_arm_target()


def _refresh_script_command_in_policy_obs(obs, unwrapped) -> None:
    """Patch the newest history frame so the policy sees this step's command.

    ``env.step`` constructs the observation returned for the *next* action.
    Since the play script injects a new command immediately before the current
    action, its command-related fields would otherwise lag by one policy step.
    Also update the stored newest frame so subsequent history contains the
    commands actually seen by the policy, rather than the pre-injection values.
    """
    policy_obs = obs if isinstance(obs, torch.Tensor) else obs.get("policy")
    single_dim = int(unwrapped.cfg.num_single_obs)
    if (
        policy_obs is None
        or policy_obs.ndim != 2
        or policy_obs.shape[-1] < single_dim
        or single_dim < 5
    ):
        raise RuntimeError(
            f"Unexpected scripted policy observation shape "
            f"{None if policy_obs is None else tuple(policy_obs.shape)} "
            f"for single-frame dimension {single_dim}."
        )
    newest = policy_obs[:, -single_dim:]
    newest[:, 0:3].copy_(
        unwrapped._scale_obs_component(
            "command", unwrapped._clip_obs_component("command", unwrapped.command)
        )
    )
    newest[:, 3:4].copy_(
        unwrapped._scale_obs_component(
            "height_cmd",
            unwrapped._clip_obs_component("height_cmd", unwrapped.height_cmd.unsqueeze(-1)),
        )
    )
    front_arm_target_obs = unwrapped.front_arm_target_cmd.unsqueeze(-1)
    newest[:, -1:].copy_(
        unwrapped._scale_obs_component(
            "front_arm_target",
            unwrapped._clip_obs_component("front_arm_target", front_arm_target_obs),
        )
    )
    if bool(unwrapped.cfg.use_frame_stack):
        unwrapped._policy_obs_history[:, -1].copy_(newest)


def _run_dynamic_script(
    env,
    unwrapped,
    policy,
    *,
    duration: float,
    height_center: float,
    height_amp: float,
    vel_max: float,
    yaw_max: float,
    coupled_duration: float | None = None,
    video_wrapper=None,
    video_name: str | None = None,
    recorder: PlayTimeSeriesRecorder | None = None,
) -> None:
    """Play the whole dynamic command sequence once, printing per-phase metrics."""
    # Freeze the random command generator: no resampling, no standing-env mask.
    generator = unwrapped.command_generator
    generator.cfg.resampling_time_range = (1.0e9, 1.0e9)
    generator.cfg.rel_standing_envs = 0.0
    generator.cfg.heading_command = False
    generator.is_standing_env[:] = False
    generator.is_heading_env[:] = False

    phases = _script_phases(duration, height_center, height_amp, vel_max, yaw_max, coupled_duration)
    step_dt = unwrapped.step_dt
    phase_steps = [max(1, int(round(phase_duration / step_dt))) for _, phase_duration, _ in phases]

    print(
        f"[INFO] dynamic command script: {len(phases)} sub-phases, "
        f"basic duration {duration:.1f} s, coupled duration {phases[-1][1]:.1f} s "
        f"= {sum(phase_duration for _, phase_duration, _ in phases):.1f} s "
        f"({sum(phase_steps)} steps)"
    )
    print(
        f"[INFO]   height {height_center - height_amp:.3f}..{height_center + height_amp:.3f} m, "
        f"|vx| <= {vel_max:.2f} m/s, |wz| <= {yaw_max:.2f} rad/s"
    )
    print(
        f"[INFO]   front arm ratio 0=fully raised "
        f"({unwrapped.cfg.front_arm_command_lower_deg:.2f} deg), "
        "1=height-dependent furthest-down bound"
    )
    # Reset once per camera pass, then run all phases continuously.
    with torch.inference_mode():
        obs, _ = env.reset()

    if video_wrapper is not None:
        # NOTE: gymnasium's RecordVideo stops any ongoing recording inside
        # ``reset()``, so the recording must be started *after* the reset.
        prefix = video_name or "mongo-script"
        video_wrapper.name_prefix = prefix
        video_wrapper.start_recording(prefix)

    summary = []
    previous_policy_actions = None
    for (name, phase_duration, profile), steps_per_phase in zip(phases, phase_steps):
        print(f"[INFO] phase {name}: {phase_duration:.2f} s ({steps_per_phase} steps)", flush=True)
        # accumulate |command| (the signed mean of a full sine cycle is 0) plus
        # the absolute tracking error of each channel
        acc = {
            "vx": 0.0,
            "vx_err": 0.0,
            "vx_err_p95": 0.0,
            "vx_bias": 0.0,
            "wz": 0.0,
            "wz_err": 0.0,
            "wz_err_p95": 0.0,
            "wz_bias": 0.0,
            "h": 0.0,
            "h_err": 0.0,
            "h_bias": 0.0,
            "pitch_abs_deg": 0.0,
            "pitch_abs_deg_p95": 0.0,
            "pitch_rate_abs": 0.0,
            "action_delta_rms": 0.0,
            "action_delta_rms_p95": 0.0,
            "filtered_action_delta_rms": 0.0,
            "issued_target_delta_rms": 0.0,
            "front_wheel_force": 0.0,
            "front_wheel_clearance": 0.0,
            "front_arm_position_err_deg": 0.0,
            "front_arm_target_deg": 0.0,
        }
        for step in range(steps_per_phase):
            vx, vy, wz, height, front_arm_ratio = profile(step * step_dt)
            with torch.inference_mode():
                # NOTE: the command buffers become *inference* tensors after the
                # first policy step, so they can only be written in place inside
                # an inference context.
                _inject_command(
                    unwrapped, vx, vy, wz, height, front_arm_ratio
                )
                _refresh_script_command_in_policy_obs(obs, unwrapped)
                front_arm_target_sample = unwrapped.front_arm_target_cmd.clone()
                actions = policy(obs)
                command_sample = recorder.capture_command() if recorder is not None else None
                obs, _, _, _ = env.step(actions)
                if recorder is not None:
                    recorder.record(actions, phase=name, command_sample=command_sample)

                data = unwrapped.robot.data
                actual_vx = data.root_lin_vel_b[:, 0]
                actual_wz = data.root_ang_vel_b[:, 2]
                actual_h = data.root_pos_w[:, 2]
                if previous_policy_actions is None:
                    action_delta_per_env = torch.zeros(actions.shape[0], device=actions.device)
                else:
                    action_delta_per_env = torch.sqrt(
                        torch.mean(torch.square(actions - previous_policy_actions), dim=-1)
                    )
                previous_policy_actions = actions.clone()
                log = unwrapped.extras.get("log", {})
                if "Metrics/pitch_deg" not in log:
                    log = _play_metrics(unwrapped)
                pitch_abs_deg_per_env = torch.abs(log["Metrics/pitch_deg"])
                front_wheel_force = float(log["Metrics/front_wheel_force"].mean())
                front_wheel_clearance = float(
                    log["Metrics/front_wheel_clearance"].mean()
                )
                filtered_action_delta = float(
                    log["Metrics/filtered_action_delta_rms"].mean()
                )
                issued_target_delta = float(
                    log["Metrics/issued_target_delta_rms"].mean()
                )
                front_arm_target_deg = float(torch.rad2deg(front_arm_target_sample).mean())
                front_arm_position_err_deg = float(
                    torch.rad2deg(
                        data.joint_pos[:, unwrapped._front_arm_joint_idx]
                        - front_arm_target_sample
                    )
                    .abs()
                    .mean()
                )
            acc["vx"] += abs(vx)
            vx_err_per_env = torch.abs(actual_vx - vx)
            acc["vx_err"] += float(vx_err_per_env.mean())
            acc["vx_err_p95"] += float(torch.quantile(vx_err_per_env, 0.95))
            acc["vx_bias"] += float((actual_vx - vx).mean())
            acc["wz"] += abs(wz)
            wz_err_per_env = torch.abs(actual_wz - wz)
            acc["wz_err"] += float(wz_err_per_env.mean())
            acc["wz_err_p95"] += float(torch.quantile(wz_err_per_env, 0.95))
            acc["wz_bias"] += float((actual_wz - wz).mean())
            acc["h"] += height
            acc["h_err"] += float(torch.abs(actual_h - height).mean())
            acc["h_bias"] += float((actual_h - height).mean())
            acc["pitch_abs_deg"] += float(pitch_abs_deg_per_env.mean())
            acc["pitch_abs_deg_p95"] += float(torch.quantile(pitch_abs_deg_per_env, 0.95))
            acc["pitch_rate_abs"] += float(torch.abs(data.root_ang_vel_b[:, 1]).mean())
            acc["action_delta_rms"] += float(action_delta_per_env.mean())
            acc["action_delta_rms_p95"] += float(torch.quantile(action_delta_per_env, 0.95))
            acc["filtered_action_delta_rms"] += filtered_action_delta
            acc["issued_target_delta_rms"] += issued_target_delta
            acc["front_wheel_force"] += front_wheel_force
            acc["front_wheel_clearance"] += front_wheel_clearance
            acc["front_arm_position_err_deg"] += front_arm_position_err_deg
            acc["front_arm_target_deg"] += front_arm_target_deg

        n = max(steps_per_phase, 1)
        row = {key: value / n for key, value in acc.items()}
        summary.append((name, row))
        print(
            f"[phase] {name:24s} mean|cmd|(vx={row['vx']:.3f} wz={row['wz']:.3f} h={row['h']:.3f}) "
            f"| mean|err|(vx={row['vx_err']:.3f} wz={row['wz_err']:.3f} h={row['h_err']:.4f}) "
            f"| bias(vx={row['vx_bias']:+.3f} wz={row['wz_bias']:+.3f} h={row['h_bias']:+.4f}) "
            f"| stability(pitch={row['pitch_abs_deg']:.2f}deg "
            f"pitch_rate={row['pitch_rate_abs']:.3f} raw_d={row['action_delta_rms']:.3f} "
            f"filt_d={row['filtered_action_delta_rms']:.3f} "
            f"issued_d={row['issued_target_delta_rms']:.3f} "
            f"front_F={row['front_wheel_force']:.1f}N "
            f"front_clear={row['front_wheel_clearance']:.3f}m "
            f"arm_err={row['front_arm_position_err_deg']:.2f}deg "
            f"arm={row['front_arm_target_deg']:.1f}deg) "
            f"| p95(vx_err={row['vx_err_p95']:.3f} wz_err={row['wz_err_p95']:.3f} "
            f"pitch={row['pitch_abs_deg_p95']:.2f}deg action_d={row['action_delta_rms_p95']:.3f})",
            flush=True,
        )

    if video_wrapper is not None:
        video_wrapper.stop_recording()

    print("\n[script summary] mean |error| per sub-phase")
    print(
        f"{'phase':26s} {'|vx| cmd':>9} {'vx err':>8} {'|wz| cmd':>9} "
        f"{'wz err':>8} {'h err (m)':>10} {'|pitch|':>9} {'raw d':>8} "
        f"{'filt d':>8} {'issued d':>9} {'front F':>9} {'clear m':>9} "
        f"{'arm err':>9} {'arm deg':>9}"
    )
    for name, row in summary:
        print(
            f"{name:26s} {row['vx']:9.3f} {row['vx_err']:8.3f} {row['wz']:9.3f} "
            f"{row['wz_err']:8.3f} {row['h_err']:10.5f} {row['pitch_abs_deg']:9.3f} "
            f"{row['action_delta_rms']:8.3f} {row['filtered_action_delta_rms']:8.3f} "
            f"{row['issued_target_delta_rms']:9.3f} {row['front_wheel_force']:9.2f} "
            f"{row['front_wheel_clearance']:9.4f} "
            f"{row['front_arm_position_err_deg']:9.3f} "
            f"{row['front_arm_target_deg']:9.2f}"
        )


def main():
    """Play with a plain PPO RSL-RL agent."""
    if args_cli.checkpoint is None:
        raise SystemExit("[ERROR] --checkpoint is required for play.")

    scripted = args_cli.script != "none"
    if scripted and args_cli.num_envs is None:
        # the scripted test is a single-robot demo; it must be decided before
        # ``parse_env_cfg`` so the scene is built with one environment
        args_cli.num_envs = 1
        print("[INFO] --script forces num_envs=1")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    physics_resolution = resolve_external_force_iterations(
        env_cfg.sim.physx, args_cli.checkpoint, args_cli.external_force_iterations,
    )
    velocity_resolution = resolve_joint_velocity_source(env_cfg, args_cli.checkpoint)
    motor_speed_resolution = resolve_motor_speed_limits(env_cfg, args_cli.checkpoint, args_cli.motor_speed_limits)
    print(f"[INFO] Joint velocity feedback: {velocity_resolution}")
    print(f"[INFO] Motor speed protection: {motor_speed_resolution}")
    print(
        "[INFO] Play TGS external-force iterations: "
        f"{physics_resolution['current_source_value']} -> {physics_resolution['effective_value']} "
        f"({physics_resolution['source']})"
    )
    if physics_resolution['source'] == 'legacy_snapshot_compatibility':
        print(
            "[WARN] Training snapshot omits enable_external_forces_every_iteration. "
            "Using False for legacy compatibility; the original engine runtime value is unknown. "
            "Use --external_force_iterations on/off for a controlled comparison."
        )
    elif physics_resolution['source'] == 'current_config_no_snapshot':
        print("[WARN] No params/env.yaml beside the checkpoint; playback uses current source physics.")
    env_cfg.play = True
    env_cfg.play_keep_done_reset = True


    if args_cli.front_arm_upper_margin_deg < 0.0:
        raise SystemExit("[ERROR] --front_arm_upper_margin_deg must be non-negative.")
    env_cfg.front_arm_command_upper_margin_deg += args_cli.front_arm_upper_margin_deg
    if args_cli.front_arm_upper_margin_deg > 0.0:
        print(
            "[INFO] additional/total front-arm furthest-down safety margin: "
            f"{args_cli.front_arm_upper_margin_deg:.2f}/"
            f"{env_cfg.front_arm_command_upper_margin_deg:.2f} deg"
        )

    if scripted:
        # The scripted test drives the commands itself: keep the robot alive
        # (no terminations / time-outs) so a single continuous run is recorded,
        # and make sure the command buffers are not resampled behind our back.
        env_cfg.play_keep_done_reset = False
        env_cfg.episode_length_s = 1.0e6
        env_cfg.height_range = (args_cli.script_height_center, args_cli.script_height_center)
    # Opt-in single-robot close-up.  The default (no flag) keeps the world camera
    # defined in ``MongoFlatEnvCfg.viewer`` so recordings show several robots.
    if scripted:
        preset = CAMERA_PRESETS["tracking" if args_cli.camera == "tracking" else "global"]
        env_cfg.viewer.origin_type = preset["origin"]
        if preset["origin"] == "asset_root":
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = preset["eye"]
        env_cfg.viewer.lookat = preset["lookat"]
    elif args_cli.viewer_robot:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-1.5, -1.1, 0.8)
        env_cfg.viewer.lookat = (0.35, 0.0, 0.28)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # Keep reset-time and interval randomization reproducible during evaluation.
    # Training already copies this value; play previously left env_cfg.seed as
    # None even though the agent config has a seed.
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    resume_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    plot_root = None
    if not args_cli.no_plot_export:
        plot_root = os.path.abspath(args_cli.plot_dir) if args_cli.plot_dir else os.path.join(
            log_dir, "play_plots", timestamp,
        )
    audit_root = plot_root or os.path.join(log_dir, "play_audits", timestamp)
    os.makedirs(audit_root, exist_ok=True)
    dump_yaml(os.path.join(audit_root, "effective_env.yaml"), env_cfg)
    dump_yaml(os.path.join(audit_root, "effective_agent.yaml"), agent_cfg)
    with open(os.path.join(audit_root, "play_config.json"), "w", encoding="utf-8") as stream:
        json.dump({"checkpoint": resume_path, "arguments": vars(args_cli),
                   "physics_resolution": physics_resolution,
                   "velocity_resolution": velocity_resolution,
                   "motor_speed_resolution": motor_speed_resolution,
                   "config_note": "Other fields still come from current task source, not a full training YAML restore."},
                  stream, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved effective playback settings to: {audit_root}")

    # create the environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_wrapper = None
    if args_cli.video:
        if scripted:
            # the script starts/stops each recording itself (one per camera pass)
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: False,
                "video_length": 0,
                "disable_logger": True,
            }
        else:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        video_wrapper = env

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create the runner and load the checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _load_checkpoint_for_play(runner, resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the policy for deployment
    runner.alg.policy.eval()
    export_model_dir = os.path.abspath(args_cli.export_dir) if args_cli.export_dir else os.path.join(
        os.path.dirname(resume_path), "exported",
    )
    normalizer = getattr(runner, "obs_normalizer", None)
    export_policy_as_jit(
        runner.alg.policy, normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        runner.alg.policy, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
    )
    # The network exports raw actions.  Export the exact optional low-pass +
    # slew path used by training/Play as a second TorchScript module.  With
    # both features disabled this module is intentionally an identity map.
    action_processor = MongoActionProcessor(
        action_dim=env.unwrapped.cfg.action_space,
        leg_action_dim=env.unwrapped._leg_action_dim,
        leg_current_weight=env.unwrapped.leg_action_low_pass_curr_weight,
        wheel_current_weight=env.unwrapped.wheel_action_low_pass_curr_weight,
        leg_max_delta=env.unwrapped.leg_action_max_delta,
        wheel_max_delta=env.unwrapped.wheel_action_max_delta,
        use_low_pass=env.unwrapped.use_action_low_pass_filter,
        use_slew_rate_limit=env.unwrapped.use_action_slew_rate_limit,
        batch_size=1,
    )
    torch.jit.script(action_processor).save(
        os.path.join(export_model_dir, "action_processor.pt")
    )
    action_delay_enabled = bool(env.unwrapped.use_act_delay)
    configured_action_delay_steps = {
        key: list(value) for key, value in env.unwrapped.cfg.act_delay_cfg.items()
    }
    effective_action_delay_steps = {
        key: (value if action_delay_enabled else [0, 0])
        for key, value in configured_action_delay_steps.items()
    }
    deployment_config = {
        "policy_rate_hz": 1.0 / float(env.unwrapped.step_dt),
        "servo_rate_hz": 1.0 / float(env.unwrapped.cfg.sim.dt),
        "policy_decimation": int(env.unwrapped.cfg.decimation),
        "motor_profile": getattr(env.unwrapped.cfg, "motor_profile", None),
        "motor_speed_guard": {
            name: dict(enabled=bool(getattr(act, 'speed_limit_enabled', False)),
                       taper_start_fraction=float(getattr(act, 'speed_taper_start_fraction', 0.9)),
                       velocity_limit=act.velocity_limit, velocity_source='native')
            for name, act in env.unwrapped.cfg.robot_cfg.actuators.items()
            if hasattr(act, 'speed_limit_enabled')
        },
        "action_order": list(env.unwrapped._policy_leg_names)
        + list(env.unwrapped._wheel_names),
        "low_pass_enabled": bool(env.unwrapped.use_action_low_pass_filter),
        "leg_current_weight": float(env.unwrapped.leg_action_low_pass_curr_weight),
        "wheel_current_weight": float(env.unwrapped.wheel_action_low_pass_curr_weight),
        "slew_rate_limit_enabled": bool(env.unwrapped.use_action_slew_rate_limit),
        "leg_max_delta_per_policy_step": float(env.unwrapped.leg_action_max_delta),
        "wheel_max_delta_per_policy_step": float(env.unwrapped.wheel_action_max_delta),
        "leg_action_scale_rad": float(env.unwrapped.leg_action_scale),
        "wheel_action_scale_rad_s": float(env.unwrapped.wheel_action_scale),
        "actor_previous_action_semantics": "conditioned_policy_action_before_joint_target_clipping",
        "actor_joint_position_semantics": "absolute_radians",
        "joint_velocity_source": env.unwrapped.cfg.joint_velocity_source,
        "sensor_sample_phase": "before_last_physics_substep",
        "pd_velocity_source": env.unwrapped.cfg.joint_velocity_source,
        "actor_front_arm_target_semantics": "absolute_radians",
        "mute_wheel_pos_obs": bool(env.unwrapped.cfg.mute_wheel_pos_obs),
        "actor_single_frame_dim": int(env.unwrapped.cfg.num_single_obs),
        "actor_history_frames": (
            int(env.unwrapped.cfg.num_obs_hist) if env.unwrapped.cfg.use_frame_stack else 1
        ),
        "actor_input_dim": int(env.unwrapped.cfg.observation_space),
        "actor_history_order": "oldest_to_newest",
        "actor_history_reset": "repeat_first_valid_frame",
        "action_delay_enabled": action_delay_enabled,
        "effective_action_delay_steps": effective_action_delay_steps,
        "configured_action_delay_steps_dormant": configured_action_delay_steps,
        "physics_step_ms": float(env.unwrapped.cfg.sim.dt) * 1000.0,
        "effective_action_delay_ms": (
            float(effective_action_delay_steps["leg_actions"][0])
            * float(env.unwrapped.cfg.sim.dt) * 1000.0
        ),
        "deployment_note": (
            "Run raw policy output through action_processor.pt and call reset() at startup. "
            "Match the recorded conditioning flags and action scales. "
            "Simulation action delay is not included in action_processor.pt."
        ),
    }
    with open(
        os.path.join(export_model_dir, "deployment_config.json"),
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(deployment_config, config_file, indent=2, ensure_ascii=False)
    print(f"[INFO] Exported policy to: {export_model_dir}")
    from mongo_tasks.direct.mongo.flat.physics_snapshot import runtime_snapshot
    with open(os.path.join(export_model_dir, "physics_env0.json"), "w", encoding="utf-8") as stream:
        json.dump(runtime_snapshot(env.unwrapped), stream, indent=2)

    reset_snapshots = PlayResetSnapshots(env.unwrapped, os.path.join(audit_root, 'reset_snapshots'),
                                        args_cli.plot_env_index)
    inference_policy = policy

    def policy(obs):
        reset_snapshots.capture_policy_input(obs)
        return inference_policy(obs)

    if scripted:
        # ---------------------------------------------------------------
        # scripted dynamic-command test (optionally recorded from two cameras)
        # ---------------------------------------------------------------
        cameras = ("global", "tracking") if args_cli.camera == "both" else (args_cli.camera,)
        for camera in cameras:
            reset_snapshots.context = f'dynamic_{camera}'
            _set_camera(env.unwrapped, camera)
            recorder = (
                PlayTimeSeriesRecorder(env.unwrapped, args_cli.plot_env_index)
                if plot_root is not None
                else None
            )
            _run_dynamic_script(
                env,
                env.unwrapped,
                policy,
                duration=args_cli.script_duration,
                height_center=args_cli.script_height_center,
                height_amp=args_cli.script_height_amp,
                vel_max=args_cli.script_vel_max,
                yaw_max=args_cli.script_yaw_max,
                coupled_duration=args_cli.script_coupled_duration,
                video_wrapper=video_wrapper,
                video_name=f"mongo-script-{camera}",
                recorder=recorder,
            )
            if recorder is not None:
                recorder.save(os.path.join(plot_root, camera))
        env.close()
        return

    # reset the environment
    obs, _ = env.reset()
    recorder = (
        PlayTimeSeriesRecorder(env.unwrapped, args_cli.plot_env_index)
        if plot_root is not None
        else None
    )

    timestep = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                command_sample = recorder.capture_command() if recorder is not None else None
                obs, _, _, _ = env.step(actions)
                if recorder is not None:
                    recorder.record(actions, command_sample=command_sample)
            timestep += 1
            if args_cli.max_steps and timestep >= args_cli.max_steps:
                break
            if args_cli.real_time:
                sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[INFO] Play interrupted; exporting collected time-series data.")
    finally:
        if recorder is not None:
            recorder.save(plot_root)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
