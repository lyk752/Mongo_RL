"""GPU integration check for reward phases; does not train or assess policy quality."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mongo_bootstrap
mongo_bootstrap.ensure_source_packages()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy', required=True, help='Existing 155-input TorchScript policy for standing.')
parser.add_argument('--output', required=True, help='Fresh JSON result path.')
parser.add_argument('--num_envs', type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output).resolve()
if output.exists():
    parser.error('--output must not already exist.')
if args.num_envs < 2:
    parser.error('At least two environments are required for partial-reset checks.')
launcher = AppLauncher(args)

import gymnasium as gym
import omni.usd
import torch
from isaaclab_tasks.utils import parse_env_cfg
import mongo_tasks  # noqa: F401


def main():
    cfg = parse_env_cfg('Mongo-Flat-v0', device=args.device, num_envs=args.num_envs)
    cfg.play = True
    cfg.play_keep_done_reset = True
    env = gym.make('Mongo-Flat-v0', cfg=cfg)
    try:
        obs, _ = env.reset(seed=42)
        u = env.unwrapped
        u._step_height_command = u._step_front_arm_command = lambda: None
        u.command_generator.compute = lambda dt: None
        policy = torch.jit.load(args.policy, map_location=u.device).eval()
        assert obs['policy'].shape == (args.num_envs, 155)
        assert obs['critic'].shape == (args.num_envs, 77)
        stage = omni.usd.get_context().get_stage()
        scene = stage.GetPrimAtPath(cfg.sim.physics_prim_path)
        solver = dict(type=scene.GetAttribute('physxScene:solverType').Get(),
                      external_forces_every_iteration=scene.GetAttribute(
                          'physxScene:enableExternalForcesEveryIteration').Get())
        root = stage.GetPrimAtPath(u.robot.root_physx_view.prim_paths[0])
        solver['position_iterations'] = root.GetAttribute('physxArticulation:solverPositionIterationCount').Get()
        solver['velocity_iterations'] = root.GetAttribute('physxArticulation:solverVelocityIterationCount').Get()
        assert solver == dict(type='TGS', external_forces_every_iteration=True,
                              position_iterations=12, velocity_iterations=1), solver
        phases = [('zero_hold', 75, 0.), ('reverse_step', 75, -.6), ('stop_step', 75, 0.),
                  ('height_ramp', 30, 0.), ('arm_ramp', 30, 0.),
                  ('posture_hold', 75, 0.), ('spin', 75, 0.), ('partial_reset', 60, 0.)]
        rows, reset_count, maximum_reward = [], 0, 0.
        with torch.inference_mode():
            for phase, steps, speed in phases:
                anchor_before = u._reward_state.stop_anchor_xy.clone()
                if phase == 'partial_reset':
                    u._reset_idx(torch.tensor([0], device=u.device))
                    assert not u._reward_state.valid[0] and u._reward_state.valid[1:].all()
                    obs = u._get_observations()
                for step in range(steps):
                    u.command.zero_()
                    u.command[:, 0] = speed
                    if phase == 'spin':
                        u.command[:, 2] = 1.
                    u.command_generator.command.copy_(u.command)
                    if phase == 'height_ramp':
                        u.height_cmd.add_(.0006)
                    if phase == 'arm_ramp':
                        u.front_arm_target_cmd.add_(.001)
                    # Keep external commands synchronized with the newest actor frame.
                    newest = obs['policy'][:, -cfg.num_single_obs:]
                    for key, values, sl in [
                        ('command', u.command, slice(0, 3)),
                        ('height_cmd', u.height_cmd[:, None], slice(3, 4)),
                        ('front_arm_target', u.front_arm_target_cmd[:, None], slice(-1, None)),
                    ]:
                        newest[:, sl] = u._scale_obs_component(key, u._clip_obs_component(key, values))
                    u._policy_obs_history[:, -1].copy_(newest)
                    obs, reward, done, timeout, info = env.step(policy(obs['policy']))
                    assert torch.isfinite(reward).all()
                    assert all(torch.isfinite(v).all() for v in obs.values())
                    assert not u._numerical_safety_reset_buf.any()
                    reset_count += int((done | timeout).sum())
                    maximum_reward = max(maximum_reward, float(reward.abs().max()))
                    metrics = {k: float(v) for k, v in info['log'].items() if k.startswith('Metrics/')}
                    if step in (0, 20, 35, steps - 1):
                        rows.append(dict(phase=phase, step=step, **metrics))
                    if phase == 'reverse_step':
                        assert (u._last_reward_terms['stand_still_body_lin_vel'] == 0.).all()
                        assert (u._last_reward_terms['stand_still_body_ang_vel'] == 0.).all()
                    state = u._reward_state
                    if phase in ('reverse_step', 'spin'):
                        assert not state.stop_anchor_valid.any()
                        assert (u._last_reward_terms['stand_still_position'] == 0.).all()
                    if phase == 'spin':
                        assert (u._last_reward_terms['stand_still_wheel_vel'] == 0.).all()
                        assert (u._last_reward_terms['stand_still_body_ang_vel'] == 0.).all()
                        scale = cfg.stand_still_body_transition_scale + (1 - cfg.stand_still_body_transition_scale) * state._gate(state.motion_age)
                        expected = (u.robot.data.root_lin_vel_b[:, :2] / cfg.stand_still_body_lin_vel_scale_m_s).square().sum(1) * scale * cfg.rewards['stand_still_body_lin_vel'] * u.step_dt
                        torch.testing.assert_close(u._last_reward_terms['stand_still_body_lin_vel'], expected)
                    if phase == 'stop_step' and step < 25:
                        assert not state.stop_anchor_valid.any()
                    if phase in ('height_ramp', 'arm_ramp'):
                        torch.testing.assert_close(state.stop_anchor_xy, anchor_before)
                        assert state.stop_anchor_valid.all()

                        assert metrics['Metrics/command_steady_gate'] > .99
                        assert metrics['Metrics/motion_steady_gate'] > .99
                    if step == steps - 1:
                        assert metrics['Metrics/command_steady_gate'] > .99, (phase, metrics)
        assert reset_count == 0, f'Unexpected policy failures during integration check: {reset_count}'
        result = dict(status='PASS', solver=solver, num_envs=args.num_envs,
                      control_steps=sum(p[1] for p in phases), observation_dims=[155, 77],
                      reward_terms=len(cfg.rewards), unexpected_resets=reset_count,
                      max_abs_reward=maximum_reward, samples=rows,
                      policy=str(Path(args.policy).resolve()),
                      note='Existing policy used only to exercise reward code. No new training or behavior acceptance.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(f'PASS: reward phases, native solver {solver}, {args.num_envs} envs; {output}')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        launcher.app.close()
