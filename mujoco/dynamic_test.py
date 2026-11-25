#!/usr/bin/env python3
"""Replay Isaac Play's 15 dynamic phases and export the same CSV/PNG channels."""
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from build_model import ROOT, NAMES
from sim2sim import parser as sim_parser, run
from reset_state import root_com_velocity

sys.path.insert(0, str(ROOT / 'scripts'))
from playback import script_phases, TimeSeriesPlots, torque_differences


def parser():
    p = sim_parser()
    p.description = __doc__
    p.add_argument('--script_duration', '--script-duration', type=float, default=8.)
    p.add_argument('--script_coupled_duration', '--script-coupled-duration', type=float, default=None)
    p.add_argument('--script_height_center', '--script-height-center', type=float, default=.37)
    p.add_argument('--script_height_amp', '--script-height-amp', type=float, default=.08)
    p.add_argument('--script_vel_max', '--script-vel-max', type=float, default=.8)
    p.add_argument('--script_yaw_max', '--script-yaw-max', type=float, default=2.)
    return p


class DynamicCommands:
    """Sample exactly as Play: round each phase length, then use local step * dt."""
    quit_requested = False

    def __init__(self, args):
        values = [args.script_duration, args.script_height_center, args.script_height_amp,
                  args.script_vel_max, args.script_yaw_max]
        if args.script_coupled_duration is not None:
            values.append(args.script_coupled_duration)
        if (not np.isfinite(values).all() or args.script_duration <= 0
                or args.script_height_amp < 0 or args.script_height_center <= args.script_height_amp
                or min(args.script_vel_max, args.script_yaw_max) < 0
                or (args.script_coupled_duration is not None and args.script_coupled_duration <= 0)):
            raise ValueError('Require finite parameters, positive durations/heights and nonnegative amplitudes.')
        self.phases = script_phases(*[args.script_duration, args.script_height_center,
            args.script_height_amp, args.script_vel_max, args.script_yaw_max, args.script_coupled_duration])

    def configure(self, cfg, args):
        self.dt = cfg['sim']['dt'] * cfg['decimation']
        self.phase_steps = [max(1, int(round(duration / self.dt))) for _, duration, _ in self.phases]
        self.num_steps = sum(self.phase_steps)
        self.index = self.local_step = 0
        self.phase = self.phases[0][0]
        # Initial arm target is raised. There is one slew update before each
        # inference, including phase boundaries, just as in Play injection.
        args.height = args.script_height_center
        args.front_fraction = 0.
        args.duration = self.num_steps * self.dt
        print(f'Dynamic script: 15 phases, {self.num_steps} steps, {args.duration:g} s; dt={self.dt:g} s')

    def key_callback(self, key):
        if key == 256:
            self.quit_requested = True

    def sample(self):
        while self.local_step >= self.phase_steps[self.index]:
            self.index += 1
            self.local_step = 0
        self.phase, _, profile = self.phases[self.index]
        vx, vy, wz, height, fraction = profile(self.local_step * self.dt)
        self.local_step += 1
        # Height is injected directly; do not apply keyboard/training height slew.
        return np.array([vx, vy, wz]), height, fraction

    def manifest(self):
        start = 0
        rows = []
        for (name, duration, _), steps in zip(self.phases, self.phase_steps):
            rows.append(dict(phase=name, requested_duration_s=duration, steps=steps,
                             start_step=start, start_time_s=start*self.dt, duration_s=steps*self.dt))
            start += steps
        return rows


class DynamicRecorder(TimeSeriesPlots):
    def __init__(self):
        self.rows = []
        self.output = None

    def configure(self, model, cfg, ctrl, output):
        self.model, self.ctrl, self.output = model, ctrl, output
        self.dt = ctrl.dt
        self.joint_names, self.coordinates = [], []
        for jid in range(model.njnt):
            kind = model.jnt_type[jid]
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = model.joint(jid).name
            for axis in range(3 if kind == mujoco.mjtJoint.mjJNT_BALL else 1):
                self.joint_names.append(f'{name}:{axis}' if kind == mujoco.mjtJoint.mjJNT_BALL else name)
                self.coordinates.append((jid, axis))
        self.action_names = list(NAMES[:4] + NAMES[5:])
        self.actuated_joint_names = list(NAMES)
        self.front_body = model.body('front_wheel').id
        self.previous_torque = self.before_previous_torque = None

    def begin_step(self, command, height, raw, phase):
        self.command, self.height = np.array(command), float(height)
        self.raw, self.phase = np.array(raw).reshape(6), phase
        self.front_target = self.ctrl.front
        self.front_peak = 0.
        self.acceleration = np.zeros(self.model.nv)

    def before_substep(self, data):
        self.previous_velocity = data.qvel.copy()

    def after_substep(self, data):
        self.acceleration = (data.qvel - self.previous_velocity) / self.model.opt.timestep
        net = np.zeros(3)
        force = np.zeros(6)
        for index, contact in enumerate(data.contact):
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.front_body not in (b1, b2):
                continue
            mujoco.mj_contactForce(self.model, data, index, force)
            world = contact.frame.reshape(3,3).T @ force[:3]
            net += world if b2 == self.front_body else -world
        self.front_peak = max(self.front_peak, float(np.linalg.norm(net)))

    @staticmethod
    def euler(quat):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        r = rotation.reshape(3,3)
        return np.array([np.arctan2(r[2,1], r[2,2]), np.arcsin(np.clip(-r[2,0],-1,1)),
                         np.arctan2(r[1,0],r[0,0])])

    def record(self, data):
        ctrl, model = self.ctrl, self.model
        rotation = data.xmat[model.body('wlbody').id].reshape(3,3)
        velocity = root_com_velocity(model, data)
        rpy = self.euler(data.qpos[3:7])
        actual_front = np.rad2deg(data.qpos[ctrl.qadr[4]])
        row = dict(step=len(self.rows), time_s=len(self.rows)*self.dt, phase=self.phase,
            command_vx_m_s=self.command[0], command_vy_m_s=self.command[1], command_wz_rad_s=self.command[2],
            command_height_m=self.height, command_height_goal_m=self.height,
            command_front_arm_deg=np.rad2deg(self.front_target), actual_vx_m_s=velocity[0],
            actual_vy_m_s=velocity[1], actual_wz_rad_s=data.qvel[5], actual_height_m=data.qpos[2],
            actual_front_arm_deg=actual_front, front_arm_position_error_deg=actual_front-np.rad2deg(self.front_target),
            front_wheel_contact_force_n=self.front_peak,
            front_wheel_clearance_m=data.xpos[self.front_body,2]-.0303,
            roll_deg=np.rad2deg(rpy[0]), pitch_deg=np.rad2deg(rpy[1]))
        for name, (jid, axis) in zip(self.joint_names, self.coordinates):
            qadr, vadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]+axis
            position = (self.euler(data.qpos[qadr:qadr+4])[axis]
                        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL else data.qpos[qadr])
            row[f'joint_position_rad/{name}'] = position
            row[f'joint_velocity_rad_s/{name}'] = data.qvel[vadr]
            row[f'joint_acceleration_rad_s2/{name}'] = self.acceleration[vadr]
        for name, value in zip(self.action_names, self.raw):
            row[f'policy_action/{name}'] = value
        torque = data.ctrl[ctrl.aids].copy()
        delta, second = torque_differences(torque, ctrl.effort, self.previous_torque,
                                           self.before_previous_torque, np)
        for name, value, d1, d2 in zip(NAMES, torque, delta, second):
            row[f'applied_torque_nm/{name}'] = value
            row[f'effort_limit_nm/{name}'] = float(ctrl.effort[list(NAMES).index(name)])
            row[f'normalized_torque_delta/{name}'] = d1
            row[f'normalized_torque_second_diff/{name}'] = d2
        self.before_previous_torque, self.previous_torque = self.previous_torque, torque
        self.rows.append(row)

    def finish(self, source, args):
        if self.output is None:
            return
        manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            phases=source.manifest(), total_planned_steps=source.num_steps, recorded_steps=len(self.rows),
            complete=len(self.rows)==source.num_steps,
            time_semantics='Play-compatible time_s labels command at start of step; measured state is post-step.',
            contact_force_semantics='Peak norm of net front-wheel contact force over the policy interval.',
            ball_position_semantics='Spherical-joint position channels are XYZ Euler angles; angular velocity components are MuJoCo local DOFs.')
        (self.output/'script_config.json').write_text(json.dumps(manifest,indent=2))
        self.save(str(self.output))
        metrics = []
        for phase, _, _ in source.phases:
            rows = [r for r in self.rows if r['phase'] == phase]
            if not rows:
                continue
            result = dict(phase=phase,samples=len(rows))
            for command, actual, name in [('command_vx_m_s','actual_vx_m_s','vx'),
                    ('command_wz_rad_s','actual_wz_rad_s','wz'),('command_height_m','actual_height_m','height'),
                    ('command_front_arm_deg','actual_front_arm_deg','front_arm')]:
                error = np.array([r[actual]-r[command] for r in rows])
                result.update({f'{name}_mae':float(abs(error).mean()),f'{name}_bias':float(error.mean()),
                               f'{name}_p95_abs':float(np.percentile(abs(error),95))})
            metrics.append(result)
        if metrics:
            with (self.output/'phase_metrics.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)


def main():
    p = parser()
    args = p.parse_args()
    try:
        source = DynamicCommands(args)
    except ValueError as exc:
        p.error(str(exc))
    import matplotlib  # Fail before simulation if plotting dependency is missing.
    recorder = DynamicRecorder()
    try:
        return run(args, source, recorder)
    finally:
        recorder.finish(source,args)


if __name__ == '__main__':
    raise SystemExit(main())
