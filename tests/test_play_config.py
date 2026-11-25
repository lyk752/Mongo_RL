"""Checkpoint physics provenance must prevent silent legacy/new solver mismatches."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('play_config', ROOT/'scripts/rsl_rl/play_config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
resolve = module.resolve_external_force_iterations


class PlayPhysicsConfigTests(unittest.TestCase):
    def test_speed_guard_preserves_archive_and_allows_explicit_transfer(self):
        for recorded, expected in [('', False), ('speed_limit_enabled: true\n      speed_taper_start_fraction: 0.8', True)]:
            with tempfile.TemporaryDirectory() as folder:
                path = self.checkpoint(folder, 'robot_cfg:\n  actuators:\n    legs_act:\n      '+recorded+'\n')
                act = SimpleNamespace(speed_limit_enabled=True, speed_taper_start_fraction=.9)
                cfg = SimpleNamespace(robot_cfg=SimpleNamespace(actuators={'legs_act':act}))
                module.resolve_motor_speed_limits(cfg,path)
                self.assertEqual(act.speed_limit_enabled,expected)
                self.assertEqual(act.speed_taper_start_fraction,.8 if expected else .9)
                module.resolve_motor_speed_limits(cfg,path,'on')
                self.assertTrue(act.speed_limit_enabled)
    def test_feedback_legacy_and_recorded_semantics(self):
        for text, expected in [('sim: {}\n', 'native'),
                               ('joint_velocity_source: position_delta\n', 'position_delta')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as folder:
                path = self.checkpoint(folder, text)
                cfg = SimpleNamespace(joint_velocity_source='position_delta')
                result = module.resolve_joint_velocity_source(cfg, path)
                self.assertEqual(cfg.joint_velocity_source, expected)
                self.assertEqual(result['effective_value'], expected)

    def checkpoint(self, folder, yaml_text=None):
        root = Path(folder)
        if yaml_text is not None:
            (root/'params').mkdir()
            (root/'params/env.yaml').write_text(yaml_text)
        return root/'model_11000.pt'

    def test_legacy_archive_does_not_inherit_new_source_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.checkpoint(folder, 'sim:\n  physx:\n    solver_type: 1\n')
            cfg = SimpleNamespace(enable_external_forces_every_iteration=True)
            result = resolve(cfg, path)
            self.assertFalse(cfg.enable_external_forces_every_iteration)
            self.assertEqual(result['source'], 'legacy_snapshot_compatibility')
            self.assertIsNone(result['recorded_value'])

    def test_recorded_boolean_is_authoritative(self):
        for value in ('true', 'false'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                path = self.checkpoint(folder, f'sim:\n  physx:\n    enable_external_forces_every_iteration: {value}\n')
                cfg = SimpleNamespace(enable_external_forces_every_iteration=value!='true')
                result = resolve(cfg, path)
                self.assertEqual(cfg.enable_external_forces_every_iteration, value=='true')
                self.assertEqual(result['source'], 'training_snapshot')

    def test_explicit_override_and_missing_snapshot_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.checkpoint(folder)
            cfg = SimpleNamespace(enable_external_forces_every_iteration=True)
            self.assertEqual(resolve(cfg, path)['source'], 'current_config_no_snapshot')
            self.assertTrue(cfg.enable_external_forces_every_iteration)
            self.assertEqual(resolve(cfg, path, 'off')['source'], 'explicit_cli')
            self.assertFalse(cfg.enable_external_forces_every_iteration)
            resolve(cfg, path, 'on')
            self.assertTrue(cfg.enable_external_forces_every_iteration)

    def test_unsupported_or_malformed_setting_is_not_silently_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.checkpoint(folder, 'sim:\n  physx:\n    enable_external_forces_every_iteration: definitely\n')
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                resolve(SimpleNamespace(), path)
            (Path(folder)/'params/env.yaml').unlink()
            with self.assertRaisesRegex(ValueError, 'cannot enable'):
                resolve(SimpleNamespace(), path, 'on')
            self.assertFalse(resolve(SimpleNamespace(), path, 'off')['supported'])


if __name__ == '__main__':
    unittest.main()
