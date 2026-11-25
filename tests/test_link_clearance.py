"""Check evaluation-only surface geometry and environment reward logging."""
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from offline_support import ROOT, FLAT, env_cfg, isolated_env_class, load_module
from test_baseline import rewards, sample_inputs

geometry = load_module('link_clearance', FLAT / 'link_clearance.py')


class LinkClearanceTests(unittest.TestCase):
    def test_rotated_surface_matches_direct_vertex_transform(self):
        rng = np.random.default_rng(42)
        points = [rng.normal(size=(7, 3)) * .1, rng.normal(size=(11, 3)) * .2]
        rotations = Rotation.random(10, random_state=rng)
        xyzw = rotations.as_quat().reshape(5, 2, 4)
        quat = torch.tensor(xyzw[..., [3, 0, 1, 2]], dtype=torch.float32)
        positions = rng.normal(size=(5, 2, 3))
        matrices = rotations.as_matrix().reshape(5, 2, 3, 3)
        expected = np.array([
            [(points[b] @ matrices[n, b].T + positions[n, b])[:, 2].min() - .15
             for b in range(2)] for n in range(5)
        ])
        packed = geometry.pad_collision_vertices(points, 'cpu')
        actual = geometry.compute_link_clearance_m(torch.tensor(positions).float(), quat, packed, .15)
        torch.testing.assert_close(actual, torch.tensor(expected).float())
        # Horizontal environment spacing must have no effect.
        moved = torch.tensor(positions).float()
        moved[..., :2] += 1000.
        torch.testing.assert_close(geometry.compute_link_clearance_m(moved, quat, packed, .15), actual)

    def test_environment_reward_without_clearance_and_logs(self):
        env = isolated_env_class({'_get_rewards', '_get_wheel_contact_force_peaks'})()
        scope = env._get_rewards.__func__.__globals__
        scope.update(
            compute_flat_reward_terms=rewards.compute_flat_reward_terms,
            weight_reward_terms=rewards.weight_reward_terms,
            quat_apply_inverse=lambda q, v: v,  # identity root orientation below
            euler_xyz_from_quat=lambda q: (torch.zeros(4), torch.zeros(4), torch.zeros(4)),
            wrap_to_pi=lambda x: x,
        )
        cfg, inputs = env_cfg(), sample_inputs()
        env.cfg, env.num_envs, env.device, env.step_dt = cfg, 4, 'cpu', .02
        env._reward_state = load_module('reward_state', FLAT / 'reward_state.py').CommandRewardState(cfg, 4, 'cpu')
        env._wheel_body_idx, env._wheel_contact_idx = [12, 13], [0, 1]
        env._undesired_contact_idx = [2]
        env._front_wheel_contact_idx = 1
        positions = torch.zeros(4, 14, 3)
        positions[..., 2] = torch.tensor([.11, .13, .04, .13])[:, None]
        quats = torch.zeros(4, 14, 4)
        quats[..., 0] = 1.
        env.robot = SimpleNamespace(data=SimpleNamespace(
            body_link_pos_w=positions, body_link_quat_w=quats,
            body_pos_w=positions, root_pos_w=torch.zeros(4, 3),
            root_quat_w=quats[:, 0], root_lin_vel_b=inputs['root_lin_vel_b'],
            root_ang_vel_b=inputs['root_ang_vel_b'], projected_gravity_b=inputs['projected_gravity_b'],
            joint_acc=inputs['joint_acc'], applied_torque=inputs['applied_torque'],
        ))
        env.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=torch.zeros(4, 4, 3, 3)))
        env.contact_sensor.data.net_forces_w_history[0, 3, 1, :2] = torch.tensor([150., 200.])
        env.command, env.height_cmd = inputs['command'], inputs['height_cmd']
        env.joint_vel, env.joint_pos = inputs['joint_vel'], inputs['joint_pos']
        env._policy_leg_idx, env._wheel_idx = inputs['leg_idx'], inputs['wheel_idx']
        env._actions, env._previous_actions = inputs['actions'], inputs['previous_actions']
        env._before_previous_actions = inputs['before_previous_actions']
        env.reset_terminated = inputs['terminated']
        env._numerical_safety_reset_buf = torch.tensor([False, False, True, False])
        env._episode_sums = {key: torch.zeros(4) for key in cfg.rewards}
        env.extras = {}
        env._get_observed_height = lambda: inputs['height']
        env._front_arm_joint_idx, env.front_arm_target_cmd = 4, torch.zeros(4)
        env._step_height_command = env._step_front_arm_command = lambda: None
        env.command_generator = SimpleNamespace(compute=lambda dt: None)
        total = env._get_rewards()
        self.assertTrue(torch.isfinite(total).all())
        torch.testing.assert_close(env._last_reward_terms['front_wheel_impact'], torch.tensor([-.08, 0., 0., 0.]))
        self.assertEqual(env.extras['log']['Metrics/front_wheel_contact_peak_max_n'].item(), 250.)
        self.assertNotIn('link_clearance', env._last_reward_terms)
        self.assertFalse(any('link_clearance' in key for key in env.extras['log']))
        expected = sum(env._last_reward_terms.values())
        torch.testing.assert_close(total, expected)
        self.assertEqual(total[2].item(), 0.)  # Numerical-safety reset stays masked.
        old_log = env.extras['log']
        old_log['Episode/old_reset_count'] = torch.tensor(1.)
        env._get_rewards()
        self.assertIsNot(old_log, env.extras['log'])
        self.assertNotIn('Episode/old_reset_count', env.extras['log'])


@unittest.skipUnless(importlib.util.find_spec('pxr'), 'USD Python libraries are unavailable outside Isaac Sim')
class CollisionGeometryTests(unittest.TestCase):
    def test_default_usd_hulls_cover_all_configured_rear_links(self):
        vertices = geometry.load_collision_hull_vertices(
            str(ROOT / 'model/usd/Mongo0909_caster.usda'), geometry.REAR_LINK_BODY_NAMES)
        self.assertEqual(len(vertices), 12)
        for points in vertices:
            self.assertTrue(np.isfinite(points).all())
        # A large calf collision-surface offset makes COM/origin proxies unsafe.
        self.assertLess(vertices[2][:, 2].min(), -.19)

    def test_usd_instance_scale_and_relative_transform(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'robot.usda'
            stage = Usd.Stage.CreateNew(str(file))
            root = UsdGeom.Xform.Define(stage, '/Robot')
            stage.SetDefaultPrim(root.GetPrim())
            body = UsdGeom.Xform.Define(stage, '/Robot/body')
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            body.AddTranslateOp().Set(Gf.Vec3d(4., 3., 2.))
            body.AddRotateYOp().Set(40.)
            prototype = UsdGeom.Xform.Define(stage, '/Shape')
            UsdPhysics.CollisionAPI.Apply(prototype.GetPrim())
            mesh = UsdGeom.Mesh.Define(stage, '/Shape/Mesh')
            mesh.CreatePointsAttr([(-1., -1., -1.), (1., -1., -1.), (0., 1., -1.), (0., 0., 1.)])
            mesh.AddTranslateOp().Set(Gf.Vec3d(.1, .2, .3))
            mesh.AddScaleOp().Set(Gf.Vec3f(.02, .03, .04))
            collision = UsdGeom.Xform.Define(stage, '/Robot/body/collisions')
            collision.GetPrim().GetReferences().AddInternalReference('/Shape')
            collision.GetPrim().SetInstanceable(True)
            stage.GetRootLayer().Save()
            points = geometry.load_collision_hull_vertices(str(file), ['body'])[0]
            np.testing.assert_allclose(points.min(0), [.08, .17, .26], atol=1e-6)
            np.testing.assert_allclose(points.max(0), [.12, .23, .34], atol=1e-6)
            with self.assertRaises(ValueError):
                geometry.load_collision_hull_vertices(str(file), ['missing'])


if __name__ == '__main__':
    unittest.main()
