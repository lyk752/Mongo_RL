"""Contract checks run inside the independent MuJoCo environment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_model import DEFAULT_DEG, ROOT
from controller import Controller, check_deployment, load_config
from sim2sim import make_model, parser


class Sim2SimTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        # Existing actor-contract tests exercise the legacy/native interface;
        # dedicated tests below cover the new encoder feedback and sample phase.
        self.cfg['joint_velocity_source'] = 'native'
        self.args = parser().parse_args(['--zero-action', '--headless'])
        self.model = make_model(self.args, self.cfg)
        self.data = mujoco.MjData(self.model)
        self.ctrl = Controller(self.model, self.cfg)
        for name, angle in DEFAULT_DEG.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = np.deg2rad(angle)
        mujoco.mj_forward(self.model, self.data)

    def test_actor_frame_order_absolute_angles_and_history(self):
        self.data.qvel[self.ctrl.vadr] = np.arange(7)
        self.data.qvel[3:6] = [.2, -.4, .6]
        self.ctrl.front = .1
        expected = np.r_[.2, -.1, .3, 1.7, .1, -.2, .3, 0, 0, -1,
                         np.deg2rad([-3, -3, 30, 30, 28]), 0, 0,
                         np.arange(7) * .1, np.zeros(6), .1]
        obs = self.ctrl.observe(self.data, [.2, -.1, .3], .34).reshape(5, 31)
        np.testing.assert_allclose(obs, np.tile(expected, (5, 1)), atol=1e-7)
        previous = obs[-1].copy()
        self.ctrl.action(np.arange(6) / 10)
        obs = self.ctrl.observe(self.data, [.5, 0, 0], .34).reshape(5, 31)
        np.testing.assert_array_equal(obs[-2], previous)
        np.testing.assert_allclose(obs[-1, 24:30], np.arange(6) / 10)
        self.ctrl.reset()
        obs = self.ctrl.observe(self.data, [0, 0, 0], .34).reshape(5, 31)
        np.testing.assert_array_equal(obs[0], obs[-1])
        np.testing.assert_array_equal(obs[:, 24:30], 0)

    def test_gravity_and_local_angular_velocity(self):
        # 90 degree world-y rotation; gravity points along local +x.
        self.data.qpos[3:7] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
        self.data.qvel[3:6] = [.1, .2, .3]
        mujoco.mj_forward(self.model, self.data)
        obs = self.ctrl.observe(self.data, [0, 0, 0], .34)[0, -31:]
        np.testing.assert_allclose(obs[7:10], [1, 0, 0], atol=1e-7)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, 1, velocity, 1)
        np.testing.assert_allclose(velocity[:3], self.data.qvel[3:6], atol=1e-7)

    def test_clipping_previous_action_and_torque_saturation(self):
        self.ctrl.action([100, -100, 100, -100, 100, -100])
        np.testing.assert_allclose(self.ctrl.position[:4], [.257087, -3.08923278, 3.08923278, -.46408305])
        vmax = self.cfg['max_wheel_vel']
        np.testing.assert_allclose(self.ctrl.velocity[5:], [vmax, -vmax])
        np.testing.assert_array_equal(self.ctrl.previous, [100, -100, 100, -100, 100, -100])
        self.ctrl.apply(self.data)
        self.assertTrue(np.all(np.abs(self.data.ctrl) <= self.ctrl.effort))
        with self.assertRaises(FloatingPointError):
            self.ctrl.action([np.nan] * 6)

    def test_archived_robot_cfg_overrides_motor_and_default_pose(self):
        self.cfg['robot_cfg'] = {
            'init_state': {'joint_pos': {'left_hip_joint': .1}},
            'actuators': {'legs': {'joint_names_expr': ['left_hip_joint'],
                                  'stiffness': 31., 'damping': 3.,
                                  'effort_limit': 12., 'armature': .02}}}
        ctrl = Controller(self.model, self.cfg)
        self.assertEqual(ctrl.default[0], .1)
        self.assertEqual(ctrl.kp[0], 31.)
        self.assertEqual(ctrl.kd[0], 3.)
        self.assertEqual(ctrl.effort[0], 12.)
        self.assertEqual(self.model.dof_armature[ctrl.vadr[0]], .02)

    def test_closed_chain_and_no_robot_self_contact(self):
        self.assertEqual((self.model.nu, self.model.neq), (7, 4))
        self.assertEqual(self.model.joint('front_podarthrum_joint').type, mujoco.mjtJoint.mjJNT_BALL)
        self.assertLess(np.max(np.abs(self.data.efc_pos[:12])), 2e-5)
        for _ in range(40):
            self.ctrl.action(np.zeros(6))
            self.ctrl.apply(self.data)
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        for contact in self.data.contact:
            self.assertIn(self.model.geom('floor').id, (contact.geom1, contact.geom2))

    def test_metadata_rejects_incompatible_history_and_yaml_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'deployment.json'
            path.write_text(json.dumps({'actor_history_order': 'newest_to_oldest'}))
            with self.assertRaises(ValueError):
                check_deployment(path, self.cfg)
            path = Path(folder) / 'bad.yaml'
            path.write_text('!!python/object/apply:os.system ["false"]')
            with self.assertRaises(Exception):
                load_config(path)

    def test_passive_and_caster_physics_and_solver_vs_actuator_damping(self):
        cfg = dict(self.cfg, robot_cfg={'actuators': {
            'passive': {'joint_names_expr': ['.*_knee_joint'], 'armature': .002,
                        'damping': .7, 'effort_limit': 0., 'viscous_friction': .012},
            'caster': {'joint_names_expr': ['front_podarthrum_joint.*'], 'armature': .0001,
                       'friction': .015, 'dynamic_friction': .010, 'viscous_friction': .001}}})
        ctrl = Controller(self.model, cfg)
        knee = self.model.joint('left_knee_joint').dofadr[0]
        self.assertEqual(self.model.dof_armature[knee], .002)
        self.assertEqual(self.model.dof_damping[knee], .012)
        caster = self.model.joint('front_podarthrum_joint').dofadr[0]
        np.testing.assert_allclose(self.model.dof_armature[caster:caster+3], .0001)
        np.testing.assert_allclose(self.model.dof_frictionloss[caster:caster+3], .015)
        self.assertEqual(ctrl.physics['joints']['front_podarthrum_joint:0']['dynamic_friction'], .010)

    def test_encoder_pd_and_observation_share_pre_substep_sample(self):
        cfg = dict(self.cfg, joint_velocity_source='position_delta')
        ctrl = Controller(self.model, cfg)
        ctrl.apply(self.data)  # seed at reset
        self.data.qpos[ctrl.qadr] += np.array([.01] * 5 + [.02, -.02])
        self.data.qvel[ctrl.vadr] = 99.  # different native signal must not leak
        ctrl.apply(self.data)
        expected = np.array([.01] * 5 + [.02, -.02]) / self.model.opt.timestep
        np.testing.assert_allclose(ctrl.feedback_velocity, expected)
        from build_model import speed_limited_effort
        np.testing.assert_allclose(self.data.ctrl[ctrl.aids], speed_limited_effort(
            ctrl.kp * np.r_[ctrl.position - self.data.qpos[ctrl.qadr[:5]], 0., 0.] - ctrl.kd * expected,
            self.data.qvel[ctrl.vadr], ctrl.effort, ctrl.motor_velocity_limit,
            ctrl.speed_taper_start_fraction, np))
        self.data.qpos[ctrl.qadr] += .001  # simulate last physics substep
        obs = ctrl.observe(self.data, [0, 0, 0], .34)[0, -31:]
        np.testing.assert_allclose(obs[17:24], expected * .1, atol=1e-7)
        ctrl.reset()
        self.data.qvel[:] = 0.
        ctrl.apply(self.data)
        np.testing.assert_array_equal(ctrl.feedback_velocity, 0.)

    def test_runtime_snapshot_preserves_mass_inertia_and_anisotropic_ball_parameters(self):
        ctrl = self.ctrl
        joints = {name: dict(row) for name, row in ctrl.physics['joints'].items()}
        for i, name in enumerate(ctrl_names_for_test()):
            joints[name].update(kp=float(ctrl.kp[i]), kd=float(ctrl.kd[i]), effort_limit=float(ctrl.effort[i]))
        for axis in range(3):
            joints[f'front_podarthrum_joint:{axis}']['viscous_friction'] = .001 * (axis+1)
        snapshot = dict(schema='mongo_physics_v1', physics_dt=self.model.opt.timestep,
                        decimation=self.cfg['decimation'],
                        gravity=[0, 0, -9.81], joints=joints,
                        bodies={'wlbody': dict(mass=9., com=[.01, .02, .03],
                                              inertia_body=[[.05, .002, 0], [.002, .06, 0], [0, 0, .07]])})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'physics.json'
            path.write_text(json.dumps(snapshot))
            Controller(self.model, self.cfg, path)
            mujoco.mj_saveModel(self.model, str(Path(folder)/'model.mjb'), None)
            restored = mujoco.MjModel.from_binary_path(str(Path(folder)/'model.mjb'))
            bid = restored.body('wlbody').id
            self.assertEqual(restored.body_mass[bid], 9.)
            np.testing.assert_allclose(restored.body_ipos[bid], [.01, .02, .03])
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, restored.body_iquat[bid])
            rotation = rotation.reshape(3, 3)
            np.testing.assert_allclose(rotation @ np.diag(restored.body_inertia[bid]) @ rotation.T,
                                       snapshot['bodies']['wlbody']['inertia_body'], atol=1e-12)
            adr = restored.joint('front_podarthrum_joint').dofadr[0]
            np.testing.assert_allclose(restored.dof_damping[adr:adr+3], [.001, .002, .003])

    def test_encoder_imu_uses_current_quaternion_instead_of_stale_xmat(self):
        ctrl = Controller(self.model, dict(self.cfg, joint_velocity_source='position_delta'))
        self.data.qpos[3:7] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
        # Deliberately do not call mj_forward: this is mj_step's post-integration phase.
        ctrl.apply(self.data)
        obs = ctrl.observe(self.data, [0, 0, 0], .34)[0, -31:]
        np.testing.assert_allclose(obs[7:10], [1, 0, 0], atol=1e-7)

    def test_24v_defaults_and_legacy_yaml_override_all_active_motors(self):
        from build_model import NAMES, default_actuators
        self.assertAlmostEqual(self.model.opt.timestep, .002)
        self.assertEqual(self.cfg['decimation'], 10)
        np.testing.assert_allclose(self.ctrl.effort, [25]*4 + [4, 2.42, 2.42])
        np.testing.assert_allclose(self.model.dof_armature[self.ctrl.vadr], [.120528]*4 + [.0085, 0, 0])
        legacy = dict(self.cfg, sim={'dt': .005}, decimation=4, max_wheel_vel=100.,
                      robot_cfg={'actuators': default_actuators()})
        for group, torque, speed, armature in [('legs_act',35.,30.,.015795),
                                                ('front_arm',15.,25.,.015795), ('wheel',10.,80.,.0001)]:
            legacy['robot_cfg']['actuators'][group].update(
                effort_limit=torque, velocity_limit=speed, armature=armature)
        model = make_model(self.args, legacy)
        ctrl = Controller(model, legacy)
        self.assertAlmostEqual(ctrl.dt, .02)
        np.testing.assert_allclose(ctrl.effort, [35]*4 + [15,10,10])
        np.testing.assert_allclose(ctrl.motor_velocity_limit, [30]*4 + [25,80,80])
        np.testing.assert_allclose(model.dof_armature[ctrl.vadr], [.015795]*5 + [.0001]*2)

    def test_same_policy_rate_cannot_hide_wrong_servo_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'deployment.json'
            path.write_text(json.dumps({'physics_step_ms': 5., 'policy_rate_hz': 50.}))
            with self.assertRaisesRegex(ValueError, 'physics_step_ms'):
                check_deployment(path, self.cfg)

    def test_actual_physical_speed_cuts_drive_and_preserves_braking(self):
        ctrl = self.ctrl
        self.assertTrue(ctrl.speed_limit_enabled.all())
        for sign in (-1.,1.):
            self.data.qvel[ctrl.vadr] = sign * ctrl.motor_velocity_limit * 1.1
            original_velocity = self.data.qvel.copy()
            ctrl.position = self.data.qpos[ctrl.qadr[:5]] + sign*1e4
            ctrl.velocity[:] = sign*1e4
            ctrl.apply(self.data)
            np.testing.assert_array_equal(self.data.ctrl[ctrl.aids], 0.)
            np.testing.assert_array_equal(self.data.qvel, original_velocity)
            ctrl.position = self.data.qpos[ctrl.qadr[:5]] - sign*1e4
            ctrl.velocity[:] = -sign*1e4
            ctrl.apply(self.data)
            np.testing.assert_allclose(self.data.ctrl[ctrl.aids], -sign*ctrl.effort)

    def test_old_archive_has_no_speed_guard_and_bad_export_is_rejected(self):
        from build_model import default_actuators
        legacy = dict(self.cfg, robot_cfg={'actuators':default_actuators()})
        for group in legacy['robot_cfg']['actuators'].values():
            group.pop('speed_limit_enabled')
            group.pop('speed_taper_start_fraction')
        ctrl = Controller(self.model, legacy)
        self.assertFalse(ctrl.speed_limit_enabled.any())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'deploy.json'
            path.write_text(json.dumps({'motor_speed_guard':{'legs_act':{
                'enabled':True,'taper_start_fraction':.9,'velocity_limit':4.712389}}}))
            with self.assertRaisesRegex(ValueError,'motor speed guard'):
                check_deployment(path,legacy)

    def test_free_motor_spinup_converges_to_speed_limit_in_physics(self):
        from build_model import MOTOR_PROFILE, speed_limited_effort
        for motor in MOTOR_PROFILE['motors'].values():
            xml = '''<mujoco><option timestep="0.002" gravity="0 0 0"/>
                <worldbody><body><joint name="rotor" type="hinge" axis="0 0 1" armature="ARMATURE"/>
                <inertial pos="0 0 0" mass="1" diaginertia="0.006 0.006 0.006"/>
                </body></worldbody><actuator><motor joint="rotor"/></actuator></mujoco>'''
            model = mujoco.MjModel.from_xml_string(xml.replace('ARMATURE',str(motor['armature_kg_m2'])))
            data = mujoco.MjData(model)
            limit, effort = motor['velocity_limit_rad_s'], motor['effort_limit_nm']
            speeds = []
            for _ in range(1500):
                data.ctrl[0] = speed_limited_effort(np.array(1e6), data.qvel[0],
                                                    effort, limit, .9, np)
                mujoco.mj_step(model,data)
                speeds.append(data.qvel[0])
            self.assertLessEqual(max(speeds),limit*(1+1e-6))
            self.assertAlmostEqual(data.qvel[0],limit,delta=limit*1e-4)


def ctrl_names_for_test():
    from build_model import NAMES
    return NAMES


if __name__ == '__main__':
    unittest.main()
