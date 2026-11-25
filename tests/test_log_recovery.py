"""Exercise actual missing-log handling without starting a simulator."""
import errno
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from offline_support import ROOT, load_module

recovery = load_module('log_recovery',ROOT/'scripts/rsl_rl/log_recovery.py')


class LogRecoveryTests(unittest.TestCase):
    def test_directory_removed_during_learning_saves_checkpoint_elsewhere(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); log=base/'logs/run';log.mkdir(parents=True)
            event=log/'events.out.tfevents.test'
            def learn(**kwargs):
                event.write_bytes(b'initial')
                event.unlink();log.rmdir()
                runner.current_learning_iteration=7
                with event.open('ab') as stream:stream.write(b'next')
            def save(path):
                self.assertTrue(runner.disable_logs)
                Path(path).write_bytes(b'model-and-optimizer')
            runner=SimpleNamespace(current_learning_iteration=5,disable_logs=False,learn=learn,save=save)
            def dump(path,cfg):
                p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('snapshot')
            with self.assertRaises(recovery.TrainingLogDirectoryLost):
                recovery.learn_with_log_recovery(runner,iterations=10,log_dir=log,
                    recovery_root=base/'recovery',env_cfg=SimpleNamespace(),
                    agent_cfg=object(),dump_yaml=dump)
            folder=next((base/'recovery').iterdir())
            self.assertEqual((folder/'emergency.pt').read_bytes(),b'model-and-optimizer')
            meta=json.loads((folder/'recovery.json').read_text())
            self.assertEqual(meta['remaining_iterations'],7)
            self.assertTrue((folder/'params/env.yaml').exists())
            self.assertFalse(log.exists())
            self.assertFalse(runner.disable_logs)

    def test_unrelated_missing_file_is_not_intercepted(self):
        def learn(**kw):raise FileNotFoundError(errno.ENOENT,'missing','/assets/robot.usd')
        runner=SimpleNamespace(current_learning_iteration=0,learn=learn)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'recovery'
            with self.assertRaises(FileNotFoundError):
                recovery.learn_with_log_recovery(runner,iterations=2,log_dir='/logs/run',
                    recovery_root=root,env_cfg=object(),agent_cfg=object(),dump_yaml=None)
            self.assertFalse(root.exists())

    def test_normal_training_is_unchanged(self):
        calls=[]
        runner=SimpleNamespace(current_learning_iteration=0,learn=lambda **kw:calls.append(kw))
        recovery.learn_with_log_recovery(runner,iterations=2,log_dir='/logs/run',
            recovery_root='/unused',env_cfg=None,agent_cfg=None,dump_yaml=None)
        self.assertEqual(calls,[dict(num_learning_iterations=2,init_at_random_ep_len=True)])
