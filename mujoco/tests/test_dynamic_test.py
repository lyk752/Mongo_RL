"""Dynamic schedules must match Play samples and reach inference without a tick lag."""
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller import Controller, load_config
from dynamic_test import DynamicCommands, DynamicRecorder, parser, script_phases
from sim2sim import run


class DynamicTests(unittest.TestCase):
    def test_every_sample_uses_play_phase_time_and_rounding(self):
        args=parser().parse_args(['--script_duration','.055','--script_coupled_duration','.095',
                                 '--script_height_center','.38','--script_height_amp','.07',
                                 '--script_vel_max','1.5','--script_yaw_max','4'])
        source=DynamicCommands(args)
        with redirect_stdout(io.StringIO()):source.configure(load_config(),args)
        self.assertEqual(source.phase_steps,[3]*14+[5])
        self.assertEqual(source.num_steps,47)
        expected=script_phases(.055,.38,.07,1.5,4.,.095)
        for (name,_,profile),steps in zip(expected,source.phase_steps):
            for step in range(steps):
                command,height,arm=source.sample()
                np.testing.assert_allclose([*command,height,arm],profile(step*.02),atol=0,rtol=0)
                self.assertEqual(source.phase,name)
        self.assertEqual(args.height,.38)
        self.assertEqual(args.duration,47*.02)

    def test_default_protocol_and_invalid_ranges(self):
        args=parser().parse_args([]);source=DynamicCommands(args)
        with redirect_stdout(io.StringIO()):source.configure(load_config(),args)
        self.assertEqual(source.num_steps,6400)
        self.assertEqual(args.duration,128.)
        for flag,value in [('--script_duration','nan'),('--script_height_amp','-.1'),
                           ('--script_height_center','.01'),('--script_vel_max','-1'),
                           ('--script_coupled_duration','0')]:
            with self.subTest(flag=flag),self.assertRaises(ValueError):
                DynamicCommands(parser().parse_args([flag,value]))

    def test_commands_and_slew_targets_reach_policy_and_csv_same_tick(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            args=parser().parse_args(['--zero-action','--headless','--script_duration','.04',
                                     '--script_coupled_duration','.06','--output',str(Path(folder)/'run')])
            source=DynamicCommands(args);recorder=DynamicRecorder();observations=[]
            original=Controller.observe
            def capture(ctrl,data,command,height):
                observations.append((command.copy(),height,ctrl.front))
                return original(ctrl,data,command,height)
            with patch.object(Controller,'observe',capture):
                code=run(args,source,recorder)
            self.assertEqual(code,0)
            self.assertEqual(len(recorder.rows),31)
            self.assertEqual(len(recorder.joint_names),18)
            self.assertEqual(len({r['phase'] for r in recorder.rows}),15)
            for row,(command,height,front) in zip(recorder.rows,observations):
                np.testing.assert_allclose([row[f'command_{x}_m_s'] for x in ('vx','vy')],command[:2])
                self.assertEqual(row['command_wz_rad_s'],command[2])
                self.assertEqual(row['command_height_m'],height)
                self.assertAlmostEqual(row['command_front_arm_deg'],np.rad2deg(front))
                self.assertLess(abs(row['command_front_arm_deg']-(-15.11)),20)
            # Height steps are injected directly; no keyboard height-rate limiter.
            values=[r['command_height_m'] for r in recorder.rows]
            self.assertGreater(max(abs(np.diff(values))),.05)
            angles=[r['command_front_arm_deg'] for r in recorder.rows]
            self.assertLessEqual(max(abs(np.diff(angles))),.6+1e-8)
            # Rendering is shared with Play; mock only costly PNG generation here.
            with patch.object(DynamicRecorder,'save') as save:
                recorder.finish(source,args)
                save.assert_called_once_with(str(args.output))
            manifest=json.loads((args.output/'script_config.json').read_text())
            self.assertTrue(manifest['complete'])
            with (args.output/'phase_metrics.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),15)


if __name__=='__main__':unittest.main()
