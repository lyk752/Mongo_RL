"""Validate the actual Isaac actuator expressions without starting Isaac Sim."""
import ast
import json
import math
from types import SimpleNamespace
import unittest

from offline_support import ROOT, env_cfg


class MotorProfileTests(unittest.TestCase):
    def test_datasheet_conversion_and_actuator_wiring(self):
        source = ROOT/'source/mongo_world/mongo_world/assets/mongo.py'
        tree = ast.parse(source.read_text())
        scope = dict(json=json, math=math, _repo_root=lambda: ROOT,
                     SpeedLimitedPDActuatorCfg=lambda **kw: SimpleNamespace(**kw),
                     IdealPDActuatorCfg=lambda **kw: SimpleNamespace(**kw))
        asset = None
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                continue
            if node.targets[0].id == 'MONGO0909_CFG':
                asset = node.value
                continue
            try:
                scope[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(source), 'eval'), scope)
            except NameError:
                pass  # USD resolution is not needed for checking actuator expressions.
        actuators = next(k.value for k in asset.keywords if k.arg == 'actuators')
        groups = eval(compile(ast.Expression(actuators), str(source), 'eval'), scope)
        for name, limit, rpm, inertia in [('legs_act',25.,45,930e-7*36**2),
                                          ('front_arm',4.,320,850e-7*10**2), ('wheel',2.42,710,0.)]:
            group = groups[name]
            self.assertTrue(group.speed_limit_enabled)
            self.assertEqual(group.speed_taper_start_fraction, .9)
            self.assertTrue(all(v == limit for v in group.effort_limit.values()))
            for v in group.velocity_limit.values():
                self.assertAlmostEqual(v, rpm*math.pi/30)
            self.assertAlmostEqual(group.armature, inertia)
        self.assertEqual(groups['legs_inact'].armature, .0001)
        self.assertEqual(groups['front_caster'].armature, .0001)

    def test_deployment_timing_and_physical_delay_units(self):
        profile = json.loads((ROOT/'model/motor_config_24v.json').read_text())
        cfg = env_cfg()
        self.assertEqual(cfg.motor_profile, profile['profile'])
        self.assertEqual(cfg.decimation, 10)
        self.assertAlmostEqual(profile['physics_dt_s'] * cfg.decimation, .02)
        self.assertAlmostEqual(cfg.max_wheel_vel, 490*math.pi/30)
        self.assertEqual(cfg.max_wheel_torque, 2.42)
        for value in cfg.act_delay_cfg.values():
            self.assertEqual([v*profile['physics_dt_s'] for v in value], [.01,.01])
        self.assertFalse(cfg.use_act_delay)
        self.assertFalse(cfg.use_obs_delay)


if __name__ == '__main__':
    unittest.main()
