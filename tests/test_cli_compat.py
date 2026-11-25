"""Regression coverage for Isaac Lab releases without rsl_rl.utils."""
import argparse
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from offline_support import ROOT, load_module

cli = load_module('baseline_cli', ROOT / 'scripts/rsl_rl/cli_args.py')


class CliCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.package = ModuleType('isaaclab_rl.rsl_rl')
        self.package.__path__ = []
        parent = ModuleType('isaaclab_rl')
        parent.__path__ = []
        parent.rsl_rl = self.package
        self.modules = {
            'isaaclab_rl': parent,
            'isaaclab_rl.rsl_rl': self.package,
            'isaaclab_rl.rsl_rl.utils': None,
        }
        self.cfg = SimpleNamespace(
            algorithm=SimpleNamespace(share_cnn_encoders=False),
            experiment_name='baseline', seed=42, run_name='', clip_actions=None,
            logger='tensorboard',
        )

    def test_old_lab_without_utils_still_applies_cli_options(self):
        parser = argparse.ArgumentParser()
        cli.add_rsl_rl_args(parser)
        args = parser.parse_args(['--experiment_name', 'compat', '--run_name', 'old_lab', '--clip_actions', '5'])
        with patch.dict(sys.modules, self.modules), patch('importlib.metadata.version', return_value='3.1.0'):
            result = cli.update_rsl_rl_cfg(self.cfg, args)
        self.assertIs(result, self.cfg)
        self.assertEqual(result.experiment_name, 'compat')
        self.assertEqual(result.run_name, 'old_lab')
        self.assertEqual(result.clip_actions, 5.)
        self.assertFalse(hasattr(result.algorithm, 'share_cnn_encoders'))

    def test_exported_helper_is_used(self):
        migrated = SimpleNamespace()
        helper = Mock(return_value=migrated)
        self.package.handle_deprecated_rsl_rl_cfg = helper
        with patch.dict(sys.modules, self.modules):
            self.assertIs(cli._adapt_rsl_rl_cfg(self.cfg, '3.1.0'), migrated)
        helper.assert_called_once_with(self.cfg, '3.1.0')

    def test_helper_available_only_in_utils_is_used(self):
        utils = ModuleType('isaaclab_rl.rsl_rl.utils')
        utils.handle_deprecated_rsl_rl_cfg = Mock(return_value=self.cfg)
        self.modules[utils.__name__] = utils
        with patch.dict(sys.modules, self.modules):
            self.assertIs(cli._adapt_rsl_rl_cfg(self.cfg, '3.1.0'), self.cfg)
        utils.handle_deprecated_rsl_rl_cfg.assert_called_once_with(self.cfg, '3.1.0')

    def test_utils_without_helper_preserves_legacy_cfg(self):
        self.modules['isaaclab_rl.rsl_rl.utils'] = ModuleType('isaaclab_rl.rsl_rl.utils')
        with patch.dict(sys.modules, self.modules):
            self.assertIs(cli._adapt_rsl_rl_cfg(self.cfg, '3.1.0'), self.cfg)

    def test_missing_internal_dependency_is_not_silenced(self):
        def import_module(name):
            if name == 'isaaclab_rl.rsl_rl':
                return self.package
            raise ModuleNotFoundError("No module named 'internal_dependency'", name='internal_dependency')

        with patch('importlib.import_module', side_effect=import_module):
            with self.assertRaises(ModuleNotFoundError) as caught:
                cli._adapt_rsl_rl_cfg(self.cfg, '3.1.0')
        self.assertEqual(caught.exception.name, 'internal_dependency')


if __name__ == '__main__':
    unittest.main()
