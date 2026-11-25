"""Preserve training state if a run's log directory disappears mid-training."""
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile


class TrainingLogDirectoryLost(RuntimeError):
    """Training stopped with a checkpoint saved outside the missing log tree."""


def learn_with_log_recovery(runner, *, iterations, log_dir, recovery_root,
                            env_cfg, agent_cfg, dump_yaml):
    start_iteration = runner.current_learning_iteration
    try:
        return runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    except FileNotFoundError as error:
        # Do not disguise missing assets, datasets, or other unrelated failures.
        if error.filename is None:
            raise
        missing = Path(os.fsdecode(error.filename)).absolute()
        original = Path(log_dir).absolute()
        if not missing.is_relative_to(original):
            raise
        root = Path(recovery_root)
        root.mkdir(parents=True, exist_ok=True)
        recovery = Path(tempfile.mkdtemp(
            prefix=datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + original.name + '_', dir=root))
        checkpoint = recovery / 'emergency.pt'
        iteration = runner.current_learning_iteration
        completed = max(0, iteration-start_iteration+1)
        metadata = {
            'original_log_dir': str(original), 'missing_path': str(missing),
            'checkpoint': str(checkpoint), 'iteration': iteration,
            'requested_iterations': iterations, 'remaining_iterations': max(0,iterations-completed),
            'note': 'Model/optimizer saved; simulator and RNG state are not restored.',
        }
        # Emergency save must not call an external logger that may also be broken.
        previous_disable = runner.disable_logs
        try:
            runner.disable_logs = True
            runner.save(str(checkpoint))
        except Exception as save_error:
            raise RuntimeError(
                f'Log path disappeared: {missing}. Emergency save failed: {save_error}'
            ) from error
        finally:
            runner.disable_logs = previous_disable
        config_error = None
        try:
            dump_yaml(str(recovery/'params/env.yaml'),env_cfg)
            dump_yaml(str(recovery/'params/agent.yaml'),agent_cfg)
        except Exception as config_error_value:
            config_error = str(config_error_value)
        metadata['config_save_error'] = config_error
        (recovery/'recovery.json').write_text(json.dumps(metadata,indent=2)+'\n')
        message = (
            f'Training log path disappeared: {missing}. Current model and optimizer saved to '
            f'{checkpoint}. Preserve the original training options, replace --checkpoint with this '
            f'file, add --resume_training, and use a NEW log directory. '
            f'Remaining iterations: {metadata["remaining_iterations"]}. '
        )
        if config_error:
            message += f'Checkpoint saved, but config snapshot failed: {config_error}. '
        print('[RECOVERY] '+message,flush=True)
        raise TrainingLogDirectoryLost(message) from error
