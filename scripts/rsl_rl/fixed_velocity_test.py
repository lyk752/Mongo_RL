"""Batched fixed-height/arm/velocity evaluation; does not train or overwrite exports."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import mongo_bootstrap
mongo_bootstrap.ensure_source_packages()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--replicates', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--heights', type=float, nargs='+', default=[.29, .355, .42])
parser.add_argument('--speeds', type=float, nargs='+', default=[-.8, -.6, -.4, -.2, 0., .2, .4, .6, .8])
parser.add_argument('--prepare-s', type=float, default=6.)
parser.add_argument('--ramp-s', type=float, default=1.)
parser.add_argument('--settle-s', type=float, default=3.)
parser.add_argument('--measure-s', type=float, default=10.)
parser.add_argument('--solver', choices=['training', 'tgs12_1', 'pgs'], default='training')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.replicates < 1 or min(args.prepare_s, args.ramp_s, args.settle_s, args.measure_s) <= 0:
    parser.error('Replicates and durations must be positive.')
OUT = Path(args.output).resolve()
if OUT.exists() and any(OUT.iterdir()):
    parser.error('Output directory must be empty; choose a fresh directory.')
checkpoint_path = Path(args.checkpoint).resolve()
if not checkpoint_path.is_file():
    parser.error('Checkpoint is missing.')
OUT.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
app = launcher.app

import hashlib
import itertools
import json
import math
import time
import numpy as np
import torch
import gymnasium as gym
import yaml
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi
import mongo_tasks
from mongo_tasks.direct.mongo.flat.link_clearance import (
    REAR_LINK_BODY_NAMES, compute_link_clearance_m,
    load_collision_hull_vertices, pad_collision_vertices,
)
from play_config import resolve_motor_speed_limits


def array(value):
    return value.detach().cpu().numpy().copy()


def refresh_policy_command(obs, u):
    """Refresh the newest actor frame and stored history after an override."""
    newest = obs['policy'][:, -int(u.cfg.num_single_obs):]
    for key, values, sl in [
        ('command', u.command, slice(0, 3)),
        ('height_cmd', u.height_cmd[:, None], slice(3, 4)),
        ('front_arm_target', u.front_arm_target_cmd[:, None], slice(-1, None)),
    ]:
        newest[:, sl] = u._scale_obs_component(key, u._clip_obs_component(key, values))
    if u.cfg.use_frame_stack:
        u._policy_obs_history[:, -1].copy_(newest)


def main():
    cases = [dict(env_id=i, height_m=h, arm=arm, vx_m_s=v, replicate=r)
             for i, (h, arm, v, r) in enumerate(itertools.product(
                 args.heights, ['up', 'down'], args.speeds, range(args.replicates)))]
    cfg = parse_env_cfg('Mongo-Flat-Play-v0', device=args.device, num_envs=len(cases))
    saved = yaml.load((checkpoint_path.parent / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    cfg.seed = args.seed
    cfg.play = True
    cfg.play_keep_done_reset = True
    cfg.episode_length_s = args.prepare_s + args.ramp_s + args.settle_s + args.measure_s + 30.
    cfg.sim.dt = float(saved['sim']['dt'])
    cfg.decimation = int(saved['decimation'])
    cfg.sim.render_interval = cfg.decimation
    cfg.contact_sensor.update_period = cfg.sim.dt
    cfg.contact_sensor.history_length = cfg.decimation
    cfg.motor_profile = saved.get('motor_profile', 'legacy_training_snapshot')
    resolve_motor_speed_limits(cfg, checkpoint_path)
    cfg.sim.physx.solver_type = int(saved['sim']['physx']['solver_type'])
    cfg.sim.physx.enable_external_forces_every_iteration = (
        saved['sim']['physx']['enable_external_forces_every_iteration'] == 'true')
    props = saved['robot_cfg']['spawn']
    cfg.robot_cfg.spawn.rigid_props.max_angular_velocity = float(props['rigid_props']['max_angular_velocity'])
    cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = int(props['articulation_props']['solver_position_iteration_count'])
    cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = int(props['articulation_props']['solver_velocity_iteration_count'])
    if args.solver == 'tgs12_1':
        cfg.sim.physx.solver_type = 1
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        cfg.sim.physx.enable_external_forces_every_iteration = True
    elif args.solver == 'pgs':
        cfg.sim.physx.solver_type = 0
    # Assert that the current motor defaults match the historical training plant.
    for group, act in cfg.robot_cfg.actuators.items():
        old = saved['robot_cfg']['actuators'][group]
        for field in ('stiffness', 'damping', 'effort_limit', 'velocity_limit', 'armature'):
            now = getattr(act, field)
            before = old[field]
            if isinstance(now, dict):
                assert all(math.isclose(float(val), float(before[k])) for k, val in now.items()), (group, field)
            elif now is not None:
                assert math.isclose(float(now), float(before)), (group, field)
    dump_yaml(str(OUT / 'effective_env.yaml'), cfg)
    policy_path = checkpoint_path.parent / 'exported/policy.pt'
    policy = torch.jit.load(str(policy_path), map_location=cfg.sim.device).eval()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert all(k in checkpoint['model_state_dict'] and torch.equal(v.cpu(), checkpoint['model_state_dict'][k])
               for k, v in policy.state_dict().items()), 'Export does not match requested checkpoint.'
    assert int(cfg.observation_space) == 155 and int(cfg.state_space) == 77
    env = gym.make('Mongo-Flat-Play-v0', cfg=cfg)
    try:
        obs, _ = env.reset(seed=args.seed)
        u = env.unwrapped
        # The scheduler is controlled entirely by this protocol. Contacts,
        # terminations and numerical-safety checks remain enabled.
        u._step_height_command = lambda: None
        u._step_front_arm_command = lambda: None
        u.command_generator.compute = lambda dt: None
        device = u.device
        heights = torch.tensor([c['height_m'] for c in cases], device=device)
        speeds = torch.tensor([c['vx_m_s'] for c in cases], device=device)
        arm_ratio = torch.tensor([c['arm'] == 'down' for c in cases], device=device)
        ids = u._policy_leg_idx + u._wheel_idx
        joint_names = [u.robot.joint_names[i] for i in ids]
        # Surface clearance is evaluation-only; it is not part of the reward.
        clearance_names = list(REAR_LINK_BODY_NAMES)
        clearance_ids, resolved = u.robot.find_bodies(clearance_names, preserve_order=True)
        assert list(resolved) == clearance_names
        hulls = load_collision_hull_vertices(cfg.robot_cfg.spawn.usd_path, clearance_names)
        clearance_vertices = pad_collision_vertices(hulls, device)
        contact_names = clearance_names + list(cfg.wheel_body_names)
        contact_ids, resolved = u.contact_sensor.find_bodies(contact_names, preserve_order=True)
        assert list(resolved) == contact_names
        view = u.robot.root_physx_view
        actuator_parameters = {}
        for name, act in u.robot.actuators.items():
            actuator_parameters[name] = dict(stiffness=array(act.stiffness).tolist(),
                damping=array(act.damping).tolist(), effort_limit=array(act.effort_limit).tolist())
        metadata = dict(checkpoint=str(checkpoint_path),
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            export_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(), args=vars(args),
            cases=cases, dt=u.step_dt, joint_names=joint_names, contact_names=contact_names,
            solver=dict(type=cfg.sim.physx.solver_type,
                position_iterations=cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count,
                velocity_iterations=cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count,
                external_forces_every_iteration=cfg.sim.physx.enable_external_forces_every_iteration),
            actuator_parameters=actuator_parameters,
            masses=array(view.get_masses()).tolist(),
            inertias=array(view.get_inertias()).tolist(),
            coms=array(view.get_coms()).tolist(),
            materials=array(view.get_material_properties()).tolist(),
            friction=array(view.get_dof_friction_properties()).tolist(),
            joint_armatures=array(view.get_dof_armatures()).tolist(),
            native_joint_stiffness=array(view.get_dof_stiffnesses()).tolist(),
            native_joint_damping=array(view.get_dof_dampings()).tolist(),
            joint_limits=array(view.get_dof_limits()).tolist(),
            joint_velocity_limits=array(view.get_dof_max_velocities()).tolist(),
            randomized_initial_root_state=array(u.robot.data.root_state_w).tolist(),
            randomized_initial_joint_pos=array(u.robot.data.joint_pos).tolist(),
            randomized_initial_joint_vel=array(u.robot.data.joint_vel).tolist(),
            note='Five randomized environments per case by default, not five independent training seeds. '
                 'Each case has its own random plant; forward/reverse are not parameter-paired. '
                 'Failed environments auto-reset under ordinary done rules and are excluded thereafter. '
                 'All current source event randomization is retained; effective_env.yaml is authoritative.')
        (OUT / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        records = {k: [] for k in ['command', 'body', 'q', 'dq', 'torque', 'action',
                                    'target', 'virtual_leg_angle', 'clearance', 'contact_force',
                                    'done', 'failed', 'height_command', 'arm_command']}
        elapsed = []
        failed = torch.zeros(len(cases), device=device, dtype=torch.bool)
        total = args.prepare_s + args.ramp_s + args.settle_s + args.measure_s
        measure_start = args.prepare_s + args.ramp_s + args.settle_s
        start = time.monotonic()
        with torch.inference_mode():
            for step in range(round(total / u.step_dt)):
                if not app.is_running():
                    raise RuntimeError('Simulator closed before protocol completion.')
                t = step * u.step_dt
                factor = min(1., max(0., (t - args.prepare_s) / args.ramp_s))
                u.command_generator.vel_command_b.zero_()
                u.command_generator.vel_command_b[:, 0] = speeds * factor
                u.command.copy_(u.command_generator.command)
                u.height_goal_cmd.copy_(heights)
                u.height_cmd.add_((heights - u.height_cmd).clamp(
                    -cfg.height_command_max_rate_m_s * u.step_dt,
                    cfg.height_command_max_rate_m_s * u.step_dt))
                u.front_arm_command_upper.copy_(u._compute_front_arm_command_upper(u.height_cmd))
                lower = math.radians(float(cfg.front_arm_command_lower_deg))
                u.front_arm_goal_cmd.copy_(lower + arm_ratio * (u.front_arm_command_upper - lower))
                u._rate_limit_front_arm_target()
                refresh_policy_command(obs, u)
                actions = policy(obs['policy'])
                obs, reward, terminated, timeout, info = env.step(actions)
                done = terminated | timeout
                failed |= done
                data = u.robot.data
                angles = euler_xyz_from_quat(data.root_quat_w)
                wheels = quat_apply_inverse(
                    data.root_quat_w[:, None].expand(-1, len(u._wheel_body_idx), -1),
                    data.body_link_pos_w[:, u._wheel_body_idx] - data.root_pos_w[:, None])
                virtual_angle = torch.atan2(-wheels[..., 0], -wheels[..., 2])
                clearance = compute_link_clearance_m(
                    data.body_link_pos_w[:, clearance_ids],
                    data.body_link_quat_w[:, clearance_ids], clearance_vertices,
                    0.0)
                force = u.contact_sensor.data.net_forces_w_history[:, :, contact_ids].norm(dim=-1).amax(dim=1)
                targets = torch.cat((data.joint_pos_target[:, u._policy_leg_idx],
                                     data.joint_vel_target[:, u._wheel_idx]), dim=1)
                body = torch.stack((data.root_lin_vel_b[:, 0], data.root_lin_vel_b[:, 1],
                    data.root_ang_vel_b[:, 2], data.root_pos_w[:, 2],
                    wrap_to_pi(angles[0]), wrap_to_pi(angles[1])), dim=1)
                values = dict(command=u.command, body=body, q=data.joint_pos[:, ids], dq=data.joint_vel[:, ids],
                    torque=data.applied_torque[:, ids], action=actions, target=targets,
                    virtual_leg_angle=virtual_angle, clearance=clearance, contact_force=force,
                    done=done, failed=failed, height_command=u.height_cmd, arm_command=u.front_arm_target_cmd)
                for key, value in values.items():
                    records[key].append(array(value))
                elapsed.append(t + u.step_dt)
                if t >= measure_start:
                    alive = ~failed
                    assert torch.allclose(u.height_cmd[alive], heights[alive], atol=1e-6)
                    assert torch.allclose(u.front_arm_target_cmd[alive], u.front_arm_goal_cmd[alive], atol=1e-6)
                if step % 100 == 0:
                    print(f'[FixedVelocity] t={t:.2f}/{total:.2f} s, failures={int(failed.sum())}, '
                          f'wall={time.monotonic()-start:.1f} s', flush=True)
        np.savez_compressed(OUT / 'tracking.npz', time_s=np.array(elapsed),
                            **{k: np.stack(v) for k, v in records.items()})
        metadata['wall_seconds'] = time.monotonic() - start
        metadata['failed_environments'] = int(failed.sum())
        metadata['complete'] = True
        metadata['measure_start_s'] = measure_start
        metadata['body_channels'] = ['vx_m_s', 'vy_m_s', 'wz_rad_s', 'height_m', 'roll_rad', 'pitch_rad']
        (OUT / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        print('[FixedVelocity] COMPLETE', OUT, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
