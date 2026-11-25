import math
import unittest
from types import SimpleNamespace

import torch

from offline_support import FLAT, env_cfg, isolated_env_class, load_module

rewards = load_module('flat_rewards', FLAT / 'rewards.py')


def sample_inputs(count=4):
    return dict(
        command=torch.zeros(count, 3), root_lin_vel_b=torch.zeros(count, 3),
        root_ang_vel_b=torch.zeros(count, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
        pitch=torch.zeros(count), height=torch.full((count,), .34), height_cmd=torch.full((count,), .34),
        wheel_pos_b=torch.tensor([[[0., .1, -.21], [0., -.1, -.21]]]).repeat(count, 1, 1),
        joint_acc=torch.zeros(count, 7), joint_vel=torch.zeros(count, 7), joint_pos=torch.zeros(count, 7),
        applied_torque=torch.zeros(count, 7), leg_idx=[0, 1, 2, 3], wheel_idx=[5, 6],
        actions=torch.zeros(count, 6), previous_actions=torch.zeros(count, 6),
        before_previous_actions=torch.zeros(count, 6), terminated=torch.zeros(count, dtype=torch.bool),
        undesired_contact=torch.zeros(count), wheel_contact_peaks=torch.full((count, 2), 50.),
        stand_still_position_error_xy=torch.zeros(count, 2),
        stand_still_position_gate=torch.zeros(count),
        front_wheel_contact_peak=torch.zeros(count),
        command_steady_gate=torch.ones(count), motion_steady_gate=torch.ones(count),
        virtual_leg_pitch_rate=torch.zeros(count, 2),
    )


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = env_cfg()
        self.inputs = sample_inputs()

    def terms(self):
        return rewards.compute_flat_reward_terms(self.cfg, **self.inputs)

    def test_upright_tracking_and_termination_scale(self):
        self.inputs['terminated'][1] = True
        terms = self.terms()
        self.assertEqual(set(terms), set(self.cfg.rewards))
        weighted = rewards.weight_reward_terms(terms, self.cfg.rewards, .02)
        total = torch.stack(list(weighted.values()), dim=-1).sum(-1)
        torch.testing.assert_close(total, torch.tensor([.11, -3.89, .11, .11]))
        self.assertEqual(weighted['termination'][1].item(), -4.)

    def test_saturation_applies_only_to_exponential_kernels(self):
        self.inputs['command'][:, 0] = 3.
        self.inputs['command'][:, 2] = 3.
        self.inputs['height'][:] = .64
        terms = self.terms()
        torch.testing.assert_close(terms['track_lin_vel_xy'], torch.full((4,), math.exp(-1 / .5)))
        torch.testing.assert_close(terms['track_lin_vel_xy_square'], torch.full((4,), (3 * .25)**2))
        torch.testing.assert_close(terms['track_ang_vel_z_square'], torch.full((4,), (3 * .5)**2))
        torch.testing.assert_close(terms['track_height_square'], torch.full((4,), 9.))

    def test_front_arm_does_not_create_policy_cost(self):
        baseline = self.terms()
        for key in ('joint_acc', 'joint_vel', 'applied_torque', 'joint_pos'):
            self.inputs[key][:, 4] = 1000.
        for name, value in self.terms().items():
            torch.testing.assert_close(value, baseline[name])

    def test_still_deadzone_and_sampled_action_history(self):
        self.inputs['command'][:, 0] = .09
        self.inputs['root_lin_vel_b'][:, :2] = torch.tensor([.2, -.3])
        self.inputs['actions'][:] = 1.
        terms = self.terms()
        torch.testing.assert_close(terms['stand_still_lin_vel'], torch.full((4,), .5))
        torch.testing.assert_close(terms['action_rate'], torch.full((4,), 6.))
        torch.testing.assert_close(terms['action_smoothness_leg'], torch.full((4,), 4.))
        torch.testing.assert_close(terms['action_smoothness_wheel'], torch.full((4,), 2.))

    def test_numerical_reset_discards_all_rewards(self):
        terms = self.terms()
        terms['leg_joint_acc'][0] = float('nan')
        weighted = rewards.weight_reward_terms(terms, self.cfg.rewards, .02, torch.tensor([False, True, False, False]))
        for value in weighted.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value[1].item(), 0.)
        with self.assertRaises(KeyError):
            rewards.weight_reward_terms(terms, {'typo': 1.}, .02)

    def test_standstill_wheel_cost_requires_exact_zero_raw_command(self):
        inputs = sample_inputs(9)
        inputs['command'] = torch.tensor([
            [0., 0., 0.], [.09, 0., 0.], [-.09, 0., 0.],
            [0., 0., .09], [0., 0., -.09], [1e-8, 0., 0.],
            [0., 0., 1e-8], [0., 1e-8, 0.], [0., 0., 1.],
        ])
        inputs['joint_vel'][:, inputs['wheel_idx']] = torch.tensor([1.3, -2.3])
        expected = torch.tensor([5., 0., 0., 0., 0., 0., 0., 0., 0.])
        # Toggling the tracking deadzone must not change this strict gate.
        for enabled in (True, False):
            self.cfg.stand_still_deadzone_enabled = enabled
            terms = rewards.compute_flat_reward_terms(self.cfg, **inputs)
            torch.testing.assert_close(terms['stand_still_wheel_vel'], expected)

    def test_standstill_wheel_cost_tolerance_and_time_scale(self):
        self.inputs['joint_vel'][:, self.inputs['wheel_idx']] = torch.tensor([
            [0., 0.], [.2, -.2], [.3, -.3], [1.3, -2.3],
        ])
        terms = self.terms()
        torch.testing.assert_close(terms['stand_still_wheel_vel'], torch.tensor([0., 0., 0., 5.]))
        weighted = rewards.weight_reward_terms(terms, self.cfg.rewards, .02)
        torch.testing.assert_close(weighted['stand_still_wheel_vel'], torch.tensor([0., 0., 0., -.01]))
        self.cfg.stand_still_wheel_vel_deadzone_rad_s = 0.
        terms = self.terms()
        torch.testing.assert_close(terms['stand_still_wheel_vel'], torch.tensor([0., .08, .18, 6.98]))


class FrontArmInterfaceTests(unittest.TestCase):
    def _make_observation_env(self, stacked=False):
        names = {'_update_obs', '_get_observations', '_clip_obs_component',
                 '_scale_obs_component', '_stream_is_scaled', '_append_observation_history'}
        cls = isolated_env_class(names)
        env = cls()
        env.cfg = env_cfg()
        env.cfg.num_single_obs, env.cfg.num_single_privileged_obs = 31, 77
        if not stacked:
            env.cfg.use_frame_stack = False
            env.cfg.num_obs_hist = 1
            env.cfg.observation_space = 31
        env.num_envs = 2
        env._policy_obs_history = torch.zeros(2, env.cfg.num_obs_hist, 31)
        env._obs_history_valid = torch.zeros(2, dtype=torch.bool)
        constants = load_module('obs_constants', FLAT / 'cfg_utils.py')
        env.cfg.obs_input_clip_cfg = constants.MONGO_EXTRA_OBS_CLIP
        env.cfg.obs_input_scale_cfg = constants.MONGO_EXTRA_OBS_SCALE
        env.cfg.privileged_extra_obs_enabled = True
        env._check_reset_ground_clearance = lambda: None
        env._get_observed_height = lambda: torch.tensor([.34, .4])
        env._get_privileged_extra_obs = lambda: torch.zeros(2, 42)
        env._privileged_extra_obs_enabled = True
        env.command_generator = SimpleNamespace(command=torch.zeros(2, 3))
        env._controlled_leg_count = 5
        env._actuate_idx = list(range(7))
        env._legs_act_idx, env._wheel_idx = list(range(5)), [5, 6]
        env._front_arm_joint_idx = 4
        # Nonzero asset defaults distinguish absolute angles from offsets.
        env.default_joint_pos = torch.tensor([[-.05, -.05, .52, .52, .49, 0., 0.]]).repeat(2, 1)
        env.joint_pos = torch.tensor([
            [-.1, -.2, .6, .7, .2, 10., 11.],
            [-.3, -.4, .8, .9, -.1, 12., 13.],
        ])
        env.joint_vel = torch.zeros(2, 7)
        env.joint_vel[:, 4] = .7
        env._encoder_velocity = env.joint_vel.clone()  # seeded reset feedback
        env.height_cmd = torch.tensor([.34, .4])
        env.leg_action_scale = .5
        env.front_arm_target_cmd = torch.tensor([.3, .4])
        env._actions = torch.full((2, 6), .5)
        env._issued_targets = torch.full((2, 6), 9.)
        env.robot = SimpleNamespace(data=SimpleNamespace(
            root_ang_vel_b=torch.zeros(2, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
            root_lin_vel_b=torch.ones(2, 3), default_joint_pos=env.default_joint_pos))
        env._obs_buffers_valid, env.use_self_obs_noise, env._obs_dim_checked = False, False, False
        return env

    def test_observation_layout_and_clean_critic(self):
        env = self._make_observation_env()
        obs = env._get_observations()
        self.assertEqual(obs['policy'].shape, (2, 31))
        self.assertEqual(obs['critic'].shape, (2, 77))
        torch.testing.assert_close(obs['policy'][:, 10:15], env.joint_pos[:, :5])
        torch.testing.assert_close(obs['policy'][:, 21], torch.full((2,), .07))
        torch.testing.assert_close(obs['policy'][:, 30], env.front_arm_target_cmd)
        torch.testing.assert_close(obs['critic'][:, :31], obs['policy'])
        torch.testing.assert_close(obs['policy'][:, 24:30], env._actions)
        self.assertEqual(obs['policy'][:, 15:17].abs().sum().item(), 0.)
        reset_obs = obs['policy'].clone()
        # Regular sensor sampling must agree with the immediate-reset path.
        env.use_obs_delay = False
        env._update_obs()
        torch.testing.assert_close(env.obs_joint_pos, env.joint_pos)
        obs = env._get_observations()
        torch.testing.assert_close(obs['policy'], reset_obs)
        # Delayed actor sensors change, while critic still sees current true state.
        sampled_angles = []

        def delay_angles(angles):
            sampled_angles.append(angles.clone())
            return angles + .1

        env.use_obs_delay = True
        env.obs_delay = {'joint_pos': SimpleNamespace(compute=delay_angles)}
        env._update_obs()
        torch.testing.assert_close(sampled_angles[0], env.joint_pos)
        env.obs_joint_vel = torch.full((2, 7), 3.)
        env.obs_root_ang_vel_b = torch.ones(2, 3)
        obs = env._get_observations()
        torch.testing.assert_close(obs['policy'][:, 10:15], env.joint_pos[:, :5] + .1)
        torch.testing.assert_close(obs['critic'][:, 10:15], env.joint_pos[:, :5])
        torch.testing.assert_close(obs['policy'][:, 30], env.front_arm_target_cmd)
        torch.testing.assert_close(obs['critic'][:, 30], env.front_arm_target_cmd)
        self.assertEqual(obs['policy'][:, 15:17].abs().sum().item(), 0.)
        self.assertEqual(obs['critic'][:, 15:17].abs().sum().item(), 0.)

    def test_five_frame_order_and_partial_reset_with_single_frame_critic(self):
        env = self._make_observation_env(stacked=True)
        self.assertTrue(env.cfg.use_frame_stack)
        self.assertEqual(env.cfg.num_obs_hist, 5)
        self.assertEqual(env.cfg.num_privileged_obs_hist, 1)
        self.assertEqual((env.cfg.observation_space, env.cfg.state_space), (155, 77))
        obs = env._get_observations()
        self.assertEqual(obs['policy'].shape, (2, 155))
        self.assertEqual(obs['critic'].shape, (2, 77))
        current = obs['critic'][:, :31]
        frames = [current.clone() for _ in range(5)]
        torch.testing.assert_close(obs['policy'], torch.cat(frames, dim=1))
        initial_obs = obs['policy'].clone()
        first_returned_obs = obs['policy']
        for step in range(1, 7):
            env.joint_pos[:, 0] += .01 * step
            env.command_generator.command[:, 0] = .1 * step
            obs = env._get_observations()
            frames = frames[1:] + [obs['critic'][:, :31].clone()]
            torch.testing.assert_close(obs['policy'], torch.cat(frames, dim=1))
            self.assertEqual(obs['critic'].shape, (2, 77))
        # Previously returned observations must not alias the rolling buffer.
        torch.testing.assert_close(first_returned_obs, initial_obs)
        # Reset only robot 0, as _reset_idx does; robot 1 keeps its history.
        env._policy_obs_history[0] = 0.
        env._obs_history_valid[0] = False
        env.joint_pos[0, 0] = -.7
        obs = env._get_observations()
        newest = obs['critic'][:, :31]
        torch.testing.assert_close(obs['policy'][0], newest[0].repeat(5))
        torch.testing.assert_close(obs['policy'][1], torch.cat(
            [frame[1] for frame in frames[1:]] + [newest[1]]))

    def test_height_commands_stay_within_requested_range(self):
        env = isolated_env_class({'_sample_height_command'})()
        env.cfg, env.device = env_cfg(), 'cpu'
        self.assertEqual(env.cfg.height_range, [.29, .45])
        self.assertEqual(env.cfg.reset_root_height_range, (.29, .45))
        samples = env._sample_height_command(torch.arange(4096))
        self.assertTrue(((samples >= .29) & (samples <= .45)).all())
        self.assertLess(samples.min().item(), .30)
        self.assertGreater(samples.max().item(), .44)

    def test_height_conditioned_sampling_and_rate_limit(self):
        names = {'_compute_front_arm_command_upper', '_update_front_arm_commands', '_rate_limit_front_arm_target'}
        env = isolated_env_class(names)()
        env.cfg, env.step_dt = env_cfg(), .02
        env.height_cmd = torch.tensor([.29, .34, .42])
        env.front_arm_goal_cmd = torch.zeros(3)
        env.front_arm_target_cmd = torch.zeros(3)
        env.front_arm_command_upper = torch.zeros(3)
        ids = torch.arange(3)
        env._update_front_arm_commands(ids, snap_to_target=True)
        lower = math.radians(-15.11)
        expected = (torch.asin(((env.height_cmd * 1000 - 145.3) / 240).clamp(-1, 1))
                    - math.radians(35)).clamp(min=math.radians(-15.11), max=math.radians(50.))
        torch.testing.assert_close(env.front_arm_command_upper, expected)
        self.assertTrue((env.front_arm_target_cmd >= lower).all())
        self.assertTrue((env.front_arm_target_cmd <= expected).all())
        old = env.front_arm_target_cmd.clone()
        env.front_arm_goal_cmd += 1.
        env._rate_limit_front_arm_target()
        torch.testing.assert_close(env.front_arm_target_cmd - old, torch.full((3,), math.radians(30) * .02))


if __name__ == '__main__':
    unittest.main()
