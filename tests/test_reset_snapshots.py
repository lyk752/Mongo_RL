"""Record actual motor limits and refresh reset snapshots after observations."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from offline_support import ROOT, FLAT, load_module

physics=load_module('reset_test_physics',FLAT/'physics_snapshot.py')
playback=load_module('reset_test_playback',ROOT/'scripts/playback.py')
logger_module=load_module('reset_test_logger',ROOT/'scripts/rsl_rl/reset_snapshots.py')


class ResetSnapshotTests(unittest.TestCase):
    def test_limits_use_actuators_in_requested_order_not_solver_limits(self):
        robot=SimpleNamespace(data=SimpleNamespace(joint_effort_limits=torch.full((1,2),1e9)),
            actuators={'leg':SimpleNamespace(joint_names=['hip'],effort_limit=torch.tensor([[25.]])),
                       'wheel':SimpleNamespace(joint_names=['wheel'],effort_limit=torch.tensor([[2.42]]))})
        limits=physics.actuator_effort_limits(robot,['wheel','hip'])
        torch.testing.assert_close(limits,torch.tensor([2.42,25.]))
        first,second=playback.torque_differences(torch.tensor([2.42,25.]),limits,
            torch.tensor([-2.42,0.]),torch.tensor([2.42,-25.]),torch)
        torch.testing.assert_close(first,torch.tensor([2.,1.]))
        torch.testing.assert_close(second,torch.tensor([4.,0.]))

    def test_every_tracked_reset_saves_new_physics_and_ready_history(self):
        class Env:
            num_envs=2
            common_step_counter=0
            generation=0
            history_ready=False
            def _reset_idx(self,ids):
                self.generation+=1;self.history_ready=False
            def _get_observations(self):
                self.history_ready=True
                return {'policy':torch.full((2,155),float(self.generation))}
        env=Env()
        def snapshot(e,index,include_state=False):
            self.assertTrue(e.history_ready)
            self.assertTrue(include_state)
            return {'generation':e.generation,'env_index':index}
        with tempfile.TemporaryDirectory() as folder:
            logger=logger_module.PlayResetSnapshots(env,folder,0,snapshot)
            logger.context='dynamic_global'
            class DetachableDict(dict):
                def detach(self):return self
            env._reset_idx(torch.tensor([0]));obs=env._get_observations();logger.capture_policy_input(DetachableDict(obs))
            env._reset_idx(torch.tensor([1]));env._get_observations()
            self.assertEqual(len(logger.entries),1)
            env.common_step_counter=800;logger.context='dynamic_tracking'
            env._reset_idx(torch.tensor([0]));env._get_observations()
            first=json.loads((Path(folder)/'reset_000001.json').read_text())
            second=json.loads((Path(folder)/'reset_000002.json').read_text())
            self.assertEqual(first['generation'],1)
            self.assertEqual(second['generation'],3)
            self.assertEqual(second['reset_metadata']['context'],'dynamic_tracking')
            self.assertEqual(second['reset_metadata']['common_step_counter'],800)
            np.testing.assert_array_equal(np.load(Path(folder)/'reset_000001.policy_input.npy'),np.ones(155))


if __name__=='__main__':unittest.main()
