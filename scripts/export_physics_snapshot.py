#!/usr/bin/env python3
"""Export actual Isaac physics and verify encoder/PD feedback at every substep."""
import argparse
import json
from pathlib import Path

import mongo_bootstrap
mongo_bootstrap.ensure_source_packages()
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory')
    parser.add_argument('--nominal', action='store_true', help='disable domain/reset randomization')
    parser.add_argument('--feedback', choices=['native', 'position_delta'], default='position_delta')
    parser.add_argument('--steps', type=int, default=50, help='50 Hz zero-action verification steps')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.io import dump_yaml
        from mongo_tasks.direct.mongo.flat.physics_snapshot import runtime_snapshot
        from mongo_world.motor_limits import speed_limited_effort
        from isaaclab.utils.types import ArticulationActions
        import mongo_tasks  # noqa: F401 - task registration

        cfg = parse_env_cfg('Mongo-Flat-Play-v0', device=args.device, num_envs=1)
        cfg.seed = 42
        cfg.joint_velocity_source = args.feedback
        cfg.play = True
        cfg.play_keep_done_reset = False
        if args.nominal:
            cfg.reset_randomization_enabled = False
            for name in vars(cfg.events):
                if not name.startswith('_'):
                    setattr(cfg.events, name, None)
        env = gym.make('Mongo-Flat-Play-v0', cfg=cfg)
        u = env.unwrapped
        with torch.inference_mode():
            env.reset(seed=42)
            snapshot = runtime_snapshot(u)
            # Exercise the real instantiated actuator classes with synthetic
            # speeds, independent of whether this rollout reaches overspeed.
            guard_checks = 0
            for actuator in u.robot.actuators.values():
                if not getattr(actuator.cfg, 'speed_limit_enabled', False):
                    continue
                for ratio, expected_drive in [(0.,1.), (.95,.5), (1.,0.), (1.2,0.)]:
                    for sign in (-1.,1.):
                        velocity = sign * ratio * actuator.velocity_limit
                        for braking in (False,True):
                            effort = torch.full_like(velocity, (-sign if braking else sign)*1e6)
                            control = ArticulationActions(joint_positions=torch.zeros_like(velocity),
                                joint_velocities=velocity.clone(), joint_efforts=effort)
                            result = actuator.compute(control, torch.zeros_like(velocity), velocity)
                            expected = actuator.effort_limit * (-sign if braking else sign*expected_drive)
                            torch.testing.assert_close(result.joint_efforts, expected, atol=1e-4, rtol=1e-4)
                            guard_checks += velocity.numel()
            (args.output / 'physics.json').write_text(json.dumps(snapshot, indent=2))
            dump_yaml(str(args.output / 'env.yaml'), cfg)
            traces = {k: [] for k in ('time', 'q', 'native_dq', 'feedback_dq', 'torque', 'expected_torque')}
            original = u.robot._apply_actuator_model

            def checked_actuators():
                original()
                data = u.robot.data
                q = data.joint_pos[:, u._actuate_idx]
                feedback = (u._encoder_velocity if args.feedback == 'position_delta'
                            else data.joint_vel[:, u._actuate_idx])
                desired_velocity = torch.zeros_like(feedback)
                desired_velocity[:, 5:] = u._issued_targets[:, 4:] * u.wheel_action_scale
                gains = [snapshot['joints'][n] for n in u._legs_act_names + u._wheel_names]
                kp = torch.tensor([r['kp'] for r in gains], device=u.device)
                kd = torch.tensor([r['kd'] for r in gains], device=u.device)
                limit = torch.tensor([r['effort_limit'] for r in gains], device=u.device)
                expected = torch.clamp(kp * (data.joint_pos_target[:, u._actuate_idx] - q)
                                       + kd * (desired_velocity - feedback), -limit, limit)
                speed_limit = torch.tensor([r['velocity_limit'] for r in gains], device=u.device)
                fractions = torch.tensor([r['speed_taper_start_fraction'] for r in gains], device=u.device)
                enabled = torch.tensor([r['speed_limit_enabled'] for r in gains], device=u.device)
                expected = torch.where(enabled, speed_limited_effort(expected,
                    data.joint_vel[:, u._actuate_idx], limit, speed_limit, fractions, torch), expected)
                values = (len(traces['time']) * cfg.sim.dt, q[0],
                          data.joint_vel[0, u._actuate_idx], feedback[0],
                          data.applied_torque[0, u._actuate_idx], expected[0])
                for key, value in zip(traces, values):
                    traces[key].append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)

            u.robot._apply_actuator_model = checked_actuators
            for _ in range(args.steps):
                env.step(torch.zeros(1, cfg.action_space, device=u.device))
            traces = {k: np.asarray(v) for k, v in traces.items()}
            np.savez_compressed(args.output / 'physics_trace.npz', **traces)
            delta = np.diff(traces['q'], axis=0)
            delta[:, 5:] = (delta[:, 5:] + np.pi) % (2 * np.pi) - np.pi
            fd_error = traces['feedback_dq'][1:] - delta / cfg.sim.dt
            report = dict(physics_ticks=len(traces['time']), joint_velocity_source=args.feedback,
                          motor_speed_guard_checks=guard_checks,
                          max_pd_torque_error_nm=float(np.max(abs(traces['torque'] - traces['expected_torque']))),
                          max_encoder_difference_rad_s=float(np.max(abs(fd_error))),
                          max_native_encoder_difference_rad_s=float(np.max(abs(traces['native_dq'][1:] - delta / cfg.sim.dt))),
                          purpose='Interface/physics smoke test, not locomotion performance validation')
            report['passed'] = (report['max_pd_torque_error_nm'] < 1e-4
                                and (args.feedback == 'native' or report['max_encoder_difference_rad_s'] < 1e-3))
            (args.output / 'verification.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
        env.close()
        if not report['passed']:
            raise RuntimeError('Encoder/PD verification failed; inspect physics_trace.npz')
    finally:
        app.close()


if __name__ == '__main__':
    main()
