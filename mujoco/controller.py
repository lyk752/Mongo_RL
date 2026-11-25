"""Mongo actor observation, history, action mapping and explicit PD controller."""
import json
import math
from pathlib import Path
import re

import mujoco
import numpy as np
import yaml

from build_model import NAMES, DEFAULT_DEG, default_actuators, motor_for_joint, speed_limited_effort
from feedback import encoder_velocity
from physics import configure, resolve


class ArchiveLoader(yaml.SafeLoader):
    """Accept Isaac's tuple tags without enabling Python object construction."""


ArchiveLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                              lambda loader, node: loader.construct_sequence(node))

ArchiveLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice',
                              lambda loader, node: {'slice': loader.construct_sequence(node)})

def front_upper(height, cfg):
    argument = (height * 1000 - cfg.get('front_arm_command_height_offset_mm', 145.3)) / cfg.get('front_arm_command_length_mm', 240.)
    angle = math.degrees(math.asin(float(np.clip(argument, -1, 1))))
    angle -= cfg.get('front_arm_command_angle_offset_deg', 27.) + cfg.get('front_arm_command_upper_margin_deg', 8.)
    return math.radians(float(np.clip(angle, cfg.get('front_arm_command_lower_deg', -15.11),
                                     cfg.get('front_arm_command_upper_limit_deg', float('inf')))))


def load_config(path=None):
    if path:
        with Path(path).open() as stream:
            cfg = yaml.load(stream, Loader=ArchiveLoader)
    else:
        cfg = json.loads(Path(__file__).with_name('config.json').read_text())
        cfg['robot_cfg'] = {'actuators': default_actuators()}
    if not isinstance(cfg, dict):
        raise ValueError('Environment config must be a mapping')
    if cfg.get('joint_velocity_source', 'native') not in ('native', 'position_delta'):
        raise ValueError('Unsupported joint_velocity_source')
    for flag in ('use_act_delay', 'use_obs_delay', 'use_self_act_noise', 'use_self_obs_noise'):
        if cfg.get(flag, False):
            raise ValueError(f'{flag}=true is not supported by this deterministic validator')
    for key in ('self_obs_noise_cfg', 'self_act_noise_cfg', 'observation_noise_model', 'action_noise_model'):
        if cfg.get(key):
            raise ValueError(f'{key} is not supported by this deterministic validator')
    if cfg.get('leg_action_encoding', 'raw') != 'raw' or not cfg.get('use_wheel_vel_control', True):
        raise ValueError('Expected raw leg actions and wheel velocity control')
    if cfg.get('num_single_obs', 31) != 31 or cfg.get('num_obs_hist', 5) != 5:
        raise ValueError('Expected 31 observations x 5 history frames')
    if not cfg.get('use_frame_stack', True):
        raise ValueError('This actor requires frame stacking')
    return cfg


def check_deployment(path, cfg):
    meta = json.loads(Path(path).read_text())
    groups = cfg.get('robot_cfg', cfg.get('robot', {})).get('actuators', {})
    for name, recorded in meta.get('motor_speed_guard', {}).items():
        current = groups.get(name, {})
        for field, default in [('enabled', False), ('taper_start_fraction', 0.9)]:
            cfg_field = 'speed_limit_enabled' if field == 'enabled' else 'speed_taper_start_fraction'
            if recorded[field] != current.get(cfg_field, default):
                raise ValueError(f'Deployment motor speed guard {name}.{field} differs from env YAML')
        if recorded.get('velocity_source', 'native') != 'native':
            raise ValueError('Motor speed protection requires native physical velocity')
        for joint in current.get('joint_names_expr', []):
            if joint in NAMES and not np.isclose(resolve(recorded['velocity_limit'], joint, 0.),
                                                  resolve(current.get('velocity_limit'), joint, 0.)):
                raise ValueError(f'Deployment motor velocity limit for {joint} differs from env YAML')
    if meta.get('joint_velocity_source', 'native') != cfg.get('joint_velocity_source', 'native'):
        raise ValueError('Policy and config have different joint_velocity_source semantics')
    if meta.get('pd_velocity_source', cfg.get('joint_velocity_source', 'native')) != cfg.get('joint_velocity_source', 'native'):
        raise ValueError('Policy and controller have different PD velocity feedback')
    if cfg.get('joint_velocity_source') == 'position_delta' and meta.get('sensor_sample_phase') != 'before_last_physics_substep':
        raise ValueError('Encoder feedback requires before_last_physics_substep metadata')
    expected = {'action_order': list(NAMES[:4] + NAMES[5:]),
                'actor_input_dim': 155, 'actor_single_frame_dim': 31,
                'actor_history_frames': 5, 'actor_history_order': 'oldest_to_newest',
                'actor_joint_position_semantics': 'absolute_radians',
                'actor_front_arm_target_semantics': 'absolute_radians',
                'actor_previous_action_semantics': 'conditioned_policy_action_before_joint_target_clipping',
                'actor_history_reset': 'repeat_first_valid_frame',
                'action_delay_enabled': False}
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            raise ValueError(f'Incompatible deployment metadata {key}: {meta[key]!r}')
    for key, source, default in [('leg_action_scale_rad', 'leg_action_scale', .5),
                                 ('wheel_action_scale_rad_s', 'wheel_vel_action_scale', 10.),
                                 ('mute_wheel_pos_obs', 'mute_wheel_pos_obs', True),
                                 ('low_pass_enabled', 'use_action_low_pass_filter', False),
                                 ('slew_rate_limit_enabled', 'use_action_slew_rate_limit', False)]:
        if key in meta and meta[key] != cfg.get(source, default):
            raise ValueError(f'{key} differs between deployment JSON and env YAML')
    hz = 1 / (cfg.get('sim', {}).get('dt', .005) * cfg.get('decimation', 4))
    if not np.isclose(meta.get('policy_rate_hz', hz), hz):
        raise ValueError('Deployment policy rate differs from env YAML')
    physics_dt = cfg.get('sim', {}).get('dt', .005)
    for key, expected_value in [('physics_step_ms', physics_dt * 1000.),
                                 ('servo_rate_hz', 1. / physics_dt),
                                 ('policy_decimation', cfg.get('decimation', 4))]:
        if key in meta and not np.isclose(meta[key], expected_value):
            raise ValueError(f'Deployment {key} differs from env YAML')


class Controller:
    def __init__(self, model, cfg, physics_snapshot=None):
        self.cfg = cfg
        self.physics = configure(model, cfg, physics_snapshot)
        self.feedback_source = cfg.get('joint_velocity_source', 'native')
        self.physics_dt = model.opt.timestep
        self.gravity_w = model.opt.gravity.copy()
        gravity_norm = np.linalg.norm(self.gravity_w)
        if gravity_norm > 0:
            self.gravity_w /= gravity_norm
        self.qadr = np.array([model.joint(n).qposadr[0] for n in NAMES])
        self.vadr = np.array([model.joint(n).dofadr[0] for n in NAMES])
        self.aids = np.array([model.actuator(n).id for n in NAMES])
        self.default = np.deg2rad([DEFAULT_DEG[n] for n in NAMES])
        self.kp = np.array([motor_for_joint(n)['stiffness'] for n in NAMES])
        self.kd = np.array([motor_for_joint(n)['damping'] for n in NAMES])
        self.effort = np.array([motor_for_joint(n)['effort_limit_nm'] for n in NAMES])
        self.motor_velocity_limit = np.array([motor_for_joint(n)['velocity_limit_rad_s'] for n in NAMES])
        # Missing fields in a historical archive mean legacy fixed-effort PD.
        self.speed_limit_enabled = np.zeros(7, dtype=bool)
        self.speed_taper_start_fraction = np.full(7, 0.9)
        robot = cfg.get('robot_cfg', cfg.get('robot', {}))
        initial = robot.get('init_state', {}).get('joint_pos', {})
        for i, name in enumerate(NAMES):
            if name in initial:
                self.default[i] = initial[name]
        for group in robot.get('actuators', {}).values():
            for i, name in enumerate(NAMES):
                if not any(re.fullmatch(expr, name) for expr in group.get('joint_names_expr', [])):
                    continue
                self.speed_limit_enabled[i] = group.get('speed_limit_enabled', False)
                self.speed_taper_start_fraction[i] = group.get('speed_taper_start_fraction', 0.9)
                for key, dest in [('stiffness', self.kp), ('damping', self.kd), ('effort_limit', self.effort),
                                  ('velocity_limit', self.motor_velocity_limit)]:
                    value = group.get(key)
                    if value is not None:
                        dest[i] = resolve(value, name, dest[i])
                armature = group.get('armature')
                if armature is not None:
                    if physics_snapshot is None:
                        model.dof_armature[self.vadr[i]] = resolve(armature, name, model.dof_armature[self.vadr[i]])
        if self.physics['runtime']:
            for i, name in enumerate(NAMES):
                row = self.physics['joints'][name]
                self.kp[i], self.kd[i], self.effort[i] = row['kp'], row['kd'], row['effort_limit']
                self.motor_velocity_limit[i] = row.get('velocity_limit', self.motor_velocity_limit[i])
                self.speed_limit_enabled[i] = row.get('speed_limit_enabled', False)
                self.speed_taper_start_fraction[i] = row.get('speed_taper_start_fraction', 0.9)
        if (not np.all(np.isfinite(self.motor_velocity_limit) & (self.motor_velocity_limit > 0))
                or not np.all(np.isfinite(self.speed_taper_start_fraction)
                              & (self.speed_taper_start_fraction >= 0) & (self.speed_taper_start_fraction < 1))):
            raise ValueError('Invalid motor speed guard limits or taper fraction.')
        model.actuator_ctrlrange[self.aids] = np.column_stack((-self.effort, self.effort))
        self.dt = model.opt.timestep * cfg.get('decimation', 4)
        self.reset()

    def reset(self):
        self.previous = np.zeros(6)
        self.history = None
        self.front = float(self.default[4])
        self.position = self.default[:5].copy()
        self.velocity = np.zeros(7)
        self.encoder_previous = None
        self.feedback_velocity = np.zeros(7)
        self.sensor = None
        self.reset_observation = None

    def sample_feedback(self, data):
        position = data.qpos[self.qadr].copy()
        self.feedback_velocity = (data.qvel[self.vadr].copy() if self.encoder_previous is None
                                  else encoder_velocity(position, self.encoder_previous, self.physics_dt))
        self.encoder_previous = position
        # mj_step integrates qpos after updating xmat; use the current root
        # quaternion so IMU and encoders share the same pre-substep timestamp.
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        self.sensor = (position, self.feedback_velocity.copy(), data.qvel[3:6].copy(),
                       rotation.reshape(3, 3).T @ self.gravity_w)

    def set_command(self, height, front_fraction, snap=False):
        lower = math.radians(self.cfg.get('front_arm_command_lower_deg', -15.11))
        target = lower + np.clip(front_fraction, 0, 1) * (front_upper(height, self.cfg) - lower)
        delta = math.radians(self.cfg.get('front_arm_command_max_rate_deg_s', 30)) * self.dt
        self.front = float(target if snap else self.front + np.clip(target - self.front, -delta, delta))

    def _component(self, name, values):
        limit = self.cfg.get('obs_input_clip_cfg', {}).get(name, 100.)
        low, high = limit if isinstance(limit, (list, tuple)) else (-limit, limit)
        values = np.clip(np.atleast_1d(values), low, high)
        if self.cfg.get('obs_input_scale_enabled', True) and 'policy' in self.cfg.get('obs_input_scale_streams', ['policy']):
            scale = self.cfg.get('obs_input_scale_cfg', {}).get(name, 1.)
            if isinstance(scale, dict):
                scale = [scale.get(k, 1.) for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')]
            values = values * scale
        return values

    def observe(self, data, command, height):
        rotation = data.xmat[1].reshape(3, 3)
        # Free joint rotational velocity is local; gravity is world -> body.
        gravity = rotation.T @ self.gravity_w
        pos = data.qpos[self.qadr].copy()
        if self.cfg.get('mute_wheel_pos_obs', True):
            pos[5:] = 0.
        vel = data.qvel[self.vadr]
        angular = data.qvel[3:6]
        if self.feedback_source == 'position_delta' and self.sensor is not None:
            pos, vel, angular, gravity = (v.copy() for v in self.sensor)
            if self.cfg.get('mute_wheel_pos_obs', True):
                pos[5:] = 0.
        parts = [('command', command), ('height_cmd', height), ('root_ang_vel_b', angular),
                 ('projected_gravity_b', gravity), ('joint_pos', pos),
                 ('joint_vel_leg', vel[:5]), ('joint_vel_wheel', vel[5:]),
                 ('actions', self.previous), ('front_arm_target', self.front)]
        frame = np.concatenate([self._component(k, v) for k, v in parts]).astype(np.float32)
        if self.reset_observation is not None:
            # The initial physical/reset sensor state is already archived.
            # Patch only commands, just like _refresh_script_command_in_policy_obs.
            reset_frame = self.reset_observation.copy()
            reset_frame[:4], reset_frame[-1] = frame[:4], frame[-1]
            frame = reset_frame
            self.reset_observation = None
        if frame.shape != (31,) or not np.isfinite(frame).all():
            raise FloatingPointError('Invalid actor observation')
        if self.history is None:
            self.history = np.tile(frame, (5, 1))
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame
        return self.history.reshape(1, 155).copy()

    def action(self, raw):
        raw = np.asarray(raw, dtype=float).reshape(6)
        if not np.isfinite(raw).all():
            raise FloatingPointError('Nonfinite policy output')
        cfg = self.cfg
        conditioned = raw.copy()
        if cfg.get('use_action_low_pass_filter', False):
            weight = np.array([cfg.get('leg_action_low_pass_curr_weight', .3)] * 4 +
                              [cfg.get('wheel_action_low_pass_curr_weight', .3)] * 2)
            conditioned = (1 - weight) * self.previous + weight * conditioned
        if cfg.get('use_action_slew_rate_limit', False):
            delta = np.array([cfg.get('leg_action_max_delta', .1)] * 4 + [cfg.get('wheel_action_max_delta', .2)] * 2)
            conditioned = self.previous + np.clip(conditioned - self.previous, -delta, delta)
        self.previous = conditioned.copy()
        self.position[:4] = np.clip(self.default[:4] + cfg.get('leg_action_scale', .5) * conditioned[:4],
                                   cfg.get('leg_action_clip_lower', [-3.08923278] * 2 + [-.46408305] * 2),
                                   cfg.get('leg_action_clip_upper', [.257087] * 2 + [3.08923278] * 2))
        self.position[4] = self.front
        vmax = cfg.get('max_wheel_vel', 100.)
        self.velocity[5:] = np.clip(conditioned[4:] * cfg.get('wheel_vel_action_scale', 10.), -vmax, vmax)

    def apply(self, data):
        if self.feedback_source == 'position_delta':
            self.sample_feedback(data)
        error = np.zeros(7)
        error[:5] = self.position - data.qpos[self.qadr[:5]]
        velocity = self.feedback_velocity if self.feedback_source == 'position_delta' else data.qvel[self.vadr]
        torque = self.kp * error + self.kd * (self.velocity - velocity)
        bounded = speed_limited_effort(torque, data.qvel[self.vadr], self.effort,
                                       self.motor_velocity_limit, self.speed_taper_start_fraction, np)
        data.ctrl[self.aids] = np.where(self.speed_limit_enabled, bounded,
                                        np.clip(torque, -self.effort, self.effort))
