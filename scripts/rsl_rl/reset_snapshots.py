"""Save tracked-environment physics/state after every completed reset observation."""
import json
from pathlib import Path

import numpy as np


class PlayResetSnapshots:
    def __init__(self, env, directory, env_index=0, snapshot_fn=None):
        if not 0 <= env_index < env.num_envs:
            raise ValueError('Reset snapshot environment index is out of range')
        if snapshot_fn is None:
            from mongo_tasks.direct.mongo.flat.physics_snapshot import runtime_snapshot
            snapshot_fn = runtime_snapshot
        self.env, self.env_index, self.snapshot_fn = env, env_index, snapshot_fn
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries, self.pending = [], False
        self.awaiting_policy_input = False
        self.context = 'rollout'
        original_reset, original_obs = env._reset_idx, env._get_observations

        def reset(env_ids):
            result = original_reset(env_ids)
            if env_ids is None or self.env_index in env_ids:
                self.pending = True
            return result

        def observations():
            result = original_obs()
            if self.pending:
                self.save()
                self.pending = False
            return result

        env._reset_idx, env._get_observations = reset, observations

    def save(self):
        number = len(self.entries) + 1
        path = self.directory / f'reset_{number:06d}.json'
        snapshot = self.snapshot_fn(self.env, self.env_index, include_state=True)
        snapshot['reset_metadata'] = dict(number=number, context=self.context,
            common_step_counter=int(self.env.common_step_counter))
        path.write_text(json.dumps(snapshot, indent=2))
        self.entries.append(dict(number=number, context=self.context, snapshot=path.name,
                                 common_step_counter=int(self.env.common_step_counter)))
        self.awaiting_policy_input = True
        self.write_index()
        print(f'[INFO] Reset physics and initial state: {path}', flush=True)

    def write_index(self):
        (self.directory / 'index.json').write_text(json.dumps(
            dict(env_index=self.env_index, resets=self.entries), indent=2))

    def capture_policy_input(self, obs):
        if not self.awaiting_policy_input:
            return
        value = obs['policy'] if hasattr(obs, 'keys') else obs
        array = np.asarray(value[self.env_index].detach().cpu().numpy(), dtype=np.float32)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError('Reset policy input must be one finite actor vector')
        path = self.directory / f'reset_{len(self.entries):06d}.policy_input.npy'
        np.save(path, array)
        self.entries[-1]['first_policy_input'] = path.name
        self.write_index()
        self.awaiting_policy_input = False
