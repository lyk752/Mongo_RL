"""Check dynamic play metrics without launching Isaac Sim."""

import ast
from contextlib import redirect_stdout
import io
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from offline_support import ROOT, FLAT, env_cfg, isolated_env_class, load_module


def script_scope():
    """Load the actual profiles and scheduler without starting Isaac Sim."""
    path = ROOT / 'scripts/rsl_rl/play.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_script_phases', '_run_dynamic_script')]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    scope = {'math': math, 'torch': torch,
             'script_phases': load_module('shared_playback', ROOT/'scripts/playback.py').script_phases}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, *functions], type_ignores=[])), str(path), 'exec'), scope)
    return scope


class DynamicPlayTests(unittest.TestCase):
    def test_command_refresh_uses_absolute_front_arm_target(self):
        path = ROOT / 'scripts/rsl_rl/play.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_refresh_script_command_in_policy_obs')
        scope = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[function], type_ignores=[])), str(path), 'exec'), scope)
        env = isolated_env_class({'_clip_obs_component', '_scale_obs_component', '_stream_is_scaled',
                                  '_append_observation_history'})()
        env.cfg = env_cfg()
        env.num_envs = 2
        env.cfg.num_single_obs = 31
        constants = load_module('play_obs_constants', FLAT / 'cfg_utils.py')
        env.cfg.obs_input_clip_cfg = constants.MONGO_EXTRA_OBS_CLIP
        env.cfg.obs_input_scale_cfg = constants.MONGO_EXTRA_OBS_SCALE
        env.command = torch.tensor([[.8, 0., -1.], [-.4, 0., 2.]])
        env.height_cmd = torch.tensor([.34, .4])
        env.front_arm_target_cmd = torch.tensor([.3, -.2])
        env.leg_action_scale = .5
        env._front_arm_joint_idx = 4
        env.robot = SimpleNamespace(data=SimpleNamespace(
            default_joint_pos=torch.full((2, 7), .49)))

        for frames in (1, 5):
            for as_dict in (False, True):
                with self.subTest(frames=frames, as_dict=as_dict):
                    policy = torch.full((2, frames * 31), -7.)
                    env.cfg.use_frame_stack = frames > 1
                    env._policy_obs_history = policy.reshape(2, frames, 31).clone()
                    critic = torch.full((2, 77), 9.)
                    obs = {'policy': policy, 'critic': critic} if as_dict else policy
                    scope['_refresh_script_command_in_policy_obs'](obs, env)
                    newest = policy[:, -31:]
                    torch.testing.assert_close(newest[:, :3], env.command)
                    torch.testing.assert_close(newest[:, 3], env.height_cmd * 5)
                    torch.testing.assert_close(newest[:, 30], env.front_arm_target_cmd)
                    torch.testing.assert_close(newest[:, 4:30], torch.full((2, 26), -7.))
                    torch.testing.assert_close(policy[:, :-31], torch.full((2, (frames - 1) * 31), -7.))
                    torch.testing.assert_close(critic, torch.full((2, 77), 9.))
                    if frames > 1:
                        torch.testing.assert_close(env._policy_obs_history, policy.reshape(2, frames, 31))
                        # After the next control step, the injected command must
                        # survive in the previous frame of the rolling history.
                        next_frame = newest.clone()
                        next_frame[:, 0] += .1
                        stacked = env._append_observation_history(
                            next_frame, env._policy_obs_history, torch.ones(2, dtype=torch.bool))
                        torch.testing.assert_close(stacked[:, -62:-31], newest)

    def test_motion_profiles_cover_three_heights_and_both_arm_states(self):
        build = script_scope()['_script_phases']
        phases = build(8., .35, .07, .8, 2.)
        self.assertEqual(len(phases), 15)
        self.assertEqual(len({name for name, _, _ in phases}), 15)
        self.assertEqual([duration for _, duration, _ in phases], [8.] * 14 + [16.])
        self.assertEqual(sum(duration for _, duration, _ in phases), 128.)
        for motion in ('walk', 'spin'):
            selected = [(name, profile) for name, _, profile in phases if f'_{motion}_' in name]
            self.assertEqual(len(selected), 6)
            combinations = set()
            for name, profile in selected:
                vx, vy, wz, height, arm = profile(2.)
                combinations.add((round(height, 2), arm))
                self.assertAlmostEqual(vx, .8 if motion == 'walk' else 0.)
                self.assertAlmostEqual(wz, 2. if motion == 'spin' else 0.)
                self.assertEqual(vy, 0.)
                self.assertEqual(arm, 0. if name.endswith('_up') else 1.)
                # Heights and arm states remain fixed through each motion phase.
                self.assertEqual(profile(6.)[3:], (height, arm))
            self.assertEqual(combinations, {(h, a) for h in (.28, .35, .42) for a in (0., 1.)})
        coupled = phases[-1][2]
        for t in (1., 2., 3.):
            for actual, expected in zip(coupled(t + 8.), coupled(t)):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(build(8., .35, .07, .8, 2., 30.)[-1][1], 30.)
        with self.assertRaises(ValueError):
            build(8., .35, .07, .8, 2., 0.)

    def test_runner_uses_longer_coupled_phase_for_steps_and_recording(self):
        scope = script_scope()
        scope.update({'_inject_command': lambda *args: None,
                      '_refresh_script_command_in_policy_obs': lambda *args: None})
        data = SimpleNamespace(
            root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
            root_pos_w=torch.tensor([[0., 0., .35]]), joint_pos=torch.zeros(1, 1),
        )
        metrics = {f'Metrics/{key}': torch.zeros(1) for key in (
            'pitch_deg', 'front_wheel_force', 'front_wheel_clearance',
            'filtered_action_delta_rms', 'issued_target_delta_rms',
        )}
        unwrapped = SimpleNamespace(
            command_generator=SimpleNamespace(
                cfg=SimpleNamespace(), is_standing_env=torch.ones(1, dtype=torch.bool),
                is_heading_env=torch.ones(1, dtype=torch.bool)),
            step_dt=.02, cfg=SimpleNamespace(front_arm_command_lower_deg=-15.11),
            front_arm_target_cmd=torch.zeros(1), _front_arm_joint_idx=0,
            robot=SimpleNamespace(data=data), extras={'log': metrics},
        )
        env = SimpleNamespace(reset=lambda: (torch.zeros(1, 31), {}),
                              step=Mock(return_value=(torch.zeros(1, 31), None, None, None)))
        recorder, video = Mock(), Mock()
        recorder.capture_command.return_value = {'command_front_arm_deg': 0.}
        output = io.StringIO()
        with redirect_stdout(output):
            scope['_run_dynamic_script'](
                env, unwrapped, lambda obs: torch.zeros(1, 6),
                duration=.04, coupled_duration=.12, height_center=.35, height_amp=.07,
                vel_max=.8, yaw_max=2., recorder=recorder, video_wrapper=video,
            )
        self.assertEqual(env.step.call_count, 34)  # 14 * 2 basic steps + 6 coupled steps.
        self.assertEqual(recorder.record.call_count, 34)
        recorded_phases = [call.kwargs['phase'] for call in recorder.record.call_args_list]
        self.assertEqual(recorded_phases.count('5_coupled_arm_up'), 6)
        self.assertEqual(len(set(recorded_phases)), 15)
        self.assertIn('(34 steps)', output.getvalue())
        video.start_recording.assert_called_once()
        video.stop_recording.assert_called_once()

    def test_metrics_from_simulator_state_without_training_log(self):
        path = ROOT / 'scripts/rsl_rl/play.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_play_metrics')
        scope = {
            'torch': torch,
            'euler_xyz_from_quat': lambda q: (torch.zeros(2), torch.tensor([.1, -.2]), torch.zeros(2)),
            'wrap_to_pi': lambda x: x,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[function], type_ignores=[])), str(path), 'exec'), scope)
        unwrapped = SimpleNamespace(
            robot=SimpleNamespace(data=SimpleNamespace(
                root_quat_w=torch.zeros(2, 4),
                body_pos_w=torch.tensor([[[0., 0., .04]], [[0., 0., 1.08]]]),
            )),
            contact_sensor=SimpleNamespace(data=SimpleNamespace(
                net_forces_w_history=torch.tensor([
                    [[[3., 4., 0.]], [[0., 0., 2.]]],
                    [[[0., 0., 6.]], [[0., 0., 1.]]],
                ]),
            )),
            _front_wheel_contact_idx=0, _front_wheel_body_idx=0,
            scene=SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [0., 0., 1.]])),
            cfg=SimpleNamespace(front_wheel_radius=.03, decimation=4, action_space=6),
            _actions=torch.ones(2, 6), _previous_actions=torch.zeros(2, 6),
            _issued_target_rate_accum=torch.tensor([24., 96.]),
        )
        metrics = scope['_play_metrics'](unwrapped)
        torch.testing.assert_close(metrics['Metrics/front_wheel_force'], torch.tensor([5., 6.]))
        torch.testing.assert_close(metrics['Metrics/front_wheel_clearance'], torch.tensor([.01, .05]))
        torch.testing.assert_close(metrics['Metrics/issued_target_delta_rms'], torch.tensor([1., 2.]))
        torch.testing.assert_close(metrics['Metrics/filtered_action_delta_rms'], torch.ones(2))

    def test_checkpoint_loading_requires_matching_network(self):
        path = ROOT / 'scripts/rsl_rl/play.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_load_checkpoint_for_play')
        scope = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[function], type_ignores=[])), str(path), 'exec'), scope)
        policy = torch.nn.Sequential(torch.nn.Linear(155, 6))
        runner = SimpleNamespace(device='cpu', alg=SimpleNamespace(policy=policy))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            expected = {k: v.clone() for k, v in policy.state_dict().items()}
            torch.save({'model_state_dict': expected}, checkpoint)
            with torch.no_grad():
                for parameter in policy.parameters():
                    parameter.zero_()
            scope['_load_checkpoint_for_play'](runner, str(checkpoint))
            for key, value in policy.state_dict().items():
                torch.testing.assert_close(value, expected[key])
            self.assertFalse(policy.training)
            torch.save({'model_state_dict': torch.nn.Sequential(torch.nn.Linear(31, 6)).state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, '155/77'):
                scope['_load_checkpoint_for_play'](runner, str(checkpoint))

    def test_front_arm_metrics_with_and_without_plot_export(self):
        path = ROOT / 'scripts/rsl_rl/play.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_run_dynamic_script')
        scope = {
            'torch': torch,
            '_script_phases': lambda *args: [('test', .02, lambda t: (0., 0., 0., .35, 0.))],
            '_inject_command': lambda *args: None,
            '_refresh_script_command_in_policy_obs': lambda *args: None,
        }
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[future, function], type_ignores=[])), str(path), 'exec'), scope)

        for export_plots in (False, True):
            with self.subTest(export_plots=export_plots):
                targets = torch.tensor([.2, .4])
                data = SimpleNamespace(
                    root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
                    root_pos_w=torch.tensor([[0., 0., .35]] * 2),
                    joint_pos=targets[:, None].clone(),
                )
                metrics = {f'Metrics/{key}': torch.zeros(2) for key in (
                    'pitch_deg', 'front_wheel_force', 'front_wheel_clearance',
                    'filtered_action_delta_rms', 'issued_target_delta_rms',
                )}
                unwrapped = SimpleNamespace(
                    command_generator=SimpleNamespace(
                        cfg=SimpleNamespace(), is_standing_env=torch.ones(2, dtype=torch.bool),
                        is_heading_env=torch.ones(2, dtype=torch.bool)),
                    step_dt=.02, cfg=SimpleNamespace(front_arm_command_lower_deg=-15.11),
                    front_arm_target_cmd=targets, _front_arm_joint_idx=0,
                    robot=SimpleNamespace(data=data), extras={'log': metrics},
                )

                def step(actions):
                    # A step can change the next target. Metrics must use the
                    # target captured before it, independently for each robot.
                    targets.fill_(.9)
                    return torch.zeros(2, 31), None, None, None

                env = SimpleNamespace(reset=lambda: (torch.zeros(2, 31), {}), step=step)
                recorder = Mock() if export_plots else None
                if recorder is not None:
                    recorder.capture_command.return_value = {'command_front_arm_deg': 11.459}
                output = io.StringIO()
                with redirect_stdout(output):
                    scope['_run_dynamic_script'](
                        env, unwrapped, lambda obs: torch.zeros(2, 6),
                        duration=.02, height_center=.35, height_amp=.07,
                        vel_max=.8, yaw_max=2., recorder=recorder,
                    )
                self.assertIn('arm_err=0.00deg', output.getvalue())
                self.assertIn('arm=17.2deg', output.getvalue())
                if recorder is not None:
                    recorder.record.assert_called_once()


if __name__ == '__main__':
    unittest.main()
