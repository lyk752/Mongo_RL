"""Keyboard events must reach the current actor observation and logged command."""
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import Controller, load_config
from keyboard_control import KeyboardCommands, parser
from sim2sim import run


class KeyboardTests(unittest.TestCase):
    def source(self, extra=()):
        args = parser().parse_args(['--zero-action', '--headless', *extra])
        source = KeyboardCommands(args)
        source.configure(load_config(), args)
        return source

    def test_latched_commands_limits_stop_and_exit(self):
        with redirect_stdout(io.StringIO()):
            source = self.source()
            for key in 'WWA':
                source.key_callback(ord(key))
            command, height, fraction = source.sample()
            np.testing.assert_allclose(command, [.2, 0, .2])
            np.testing.assert_array_equal(source.sample()[0], command)
            for _ in range(100):
                for key in 'WAIU':
                    source.key_callback(ord(key))
            command, height, fraction = source.sample()
            np.testing.assert_allclose(command, [1.5, 0, 4.])
            self.assertAlmostEqual(source.height_goal, .45)
            self.assertEqual(fraction, 1.)
            source.key_callback(ord('X'))
            np.testing.assert_array_equal(source.sample()[0], [0, 0, 0])
            source.key_callback(256)
            source.sample()
            self.assertTrue(source.quit_requested)

    def test_height_rate_and_archive_bounds(self):
        with redirect_stdout(io.StringIO()):
            source = self.source()
            source.key_callback(ord('I'))
            source.sample()
            self.assertAlmostEqual(source.height_goal, .35)
            self.assertAlmostEqual(source.height, .3406)
            cfg = load_config()
            cfg['commands'] = {'ranges': {'lin_vel_x': [.15, .3], 'ang_vel_z': [0., 0.]}}
            source.configure(cfg, source.args)
            for key in 'SSDA':
                source.key_callback(ord(key))
            np.testing.assert_array_equal(source.sample()[0], [0, 0, 0])

    def test_commands_reach_actor_and_csv_on_same_step(self):
        class ScriptedKeys(KeyboardCommands):
            count = 0

            def sample(self):
                self.count += 1
                for key in ('WIU' if self.count == 2 else 'X' if self.count == 4 else ''):
                    self.key_callback(ord(key))
                return super().sample()

        observed = []
        original = Controller.observe

        def capture(ctrl, data, command, height):
            result = original(ctrl, data, command, height)
            observed.append((command.copy(), height, ctrl.front))
            return result

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = parser().parse_args(['--zero-action', '--headless', '--duration', '.12',
                                       '--output', str(Path(folder) / 'run')])
            with patch.object(Controller, 'observe', capture):
                self.assertEqual(run(args, ScriptedKeys(args)), 0)
            with (args.output / 'tracking.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(rows[0]['command_vx']), 0.)
            self.assertAlmostEqual(float(rows[1]['command_vx']), .1)
            self.assertAlmostEqual(float(rows[3]['command_vx']), 0.)
            for row, (command, height, front) in zip(rows, observed):
                self.assertAlmostEqual(float(row['command_vx']), command[0])
                self.assertAlmostEqual(float(row['command_height']), height)
                self.assertAlmostEqual(float(row['front_target_rad']), front)
            # One front-arm slew step per policy tick, including a changed command.
            self.assertGreater(observed[1][2], observed[0][2])
            self.assertLessEqual(observed[1][2] - observed[0][2], np.deg2rad(.6) + 1e-9)


if __name__ == '__main__':
    unittest.main()
