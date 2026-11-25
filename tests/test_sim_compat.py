"""PhysX constructor compatibility without starting Isaac Sim."""

from dataclasses import dataclass
import unittest

from offline_support import FLAT, load_module

compat = load_module('sim_compat', FLAT / 'sim_compat.py')


class SimCompatibilityTests(unittest.TestCase):
    def test_supported_version_enables_external_forces(self):
        @dataclass
        class PhysxCfg:
            solver_type: int = 0
            enable_external_forces_every_iteration: bool = False

        cfg = PhysxCfg(solver_type=1, **compat.external_force_iteration_kwargs(PhysxCfg))
        self.assertEqual(cfg.solver_type, 1)
        self.assertTrue(cfg.enable_external_forces_every_iteration)

    def test_older_version_constructs_with_warning(self):
        @dataclass
        class PhysxCfg:
            solver_type: int = 0
            gpu_max_rigid_patch_count: int = 0
            gpu_max_rigid_contact_count: int = 0

        with self.assertWarnsRegex(RuntimeWarning, 'physics may differ'):
            cfg = PhysxCfg(
                solver_type=1,
                gpu_max_rigid_patch_count=2**20,
                gpu_max_rigid_contact_count=2**23,
                **compat.external_force_iteration_kwargs(PhysxCfg),
            )
        self.assertEqual(cfg.solver_type, 1)
        self.assertEqual(cfg.gpu_max_rigid_patch_count, 2**20)
        self.assertEqual(cfg.gpu_max_rigid_contact_count, 2**23)


if __name__ == '__main__':
    unittest.main()
