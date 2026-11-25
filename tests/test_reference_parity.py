"""Compare against actual upstream AST equations without importing Isaac Sim.

These checks run when the original repos are available beside this checkout.
The baseline itself has no runtime dependency on either original repo.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from offline_support import ROOT, env_cfg, literal_class, runner_cfg
from test_baseline import rewards, sample_inputs

REFERENCE = ROOT.parent / 'wheeled-legged_RL'
BASE = REFERENCE / 'source/agent_tasks/agent_tasks/direct/wheelbipe'


@unittest.skipUnless(REFERENCE.is_dir(), 'Original reference checkout is unavailable')
class ReferenceParityTests(unittest.TestCase):
    def test_network_ppo_and_reward_config(self):
        ref_agent = BASE / 'agents/rsl_rl_ppo_cfg.py'
        tree = ast.parse(ref_agent.read_text())
        def call_kwargs(class_name, field):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            node = next(n for n in cls.body if isinstance(n, ast.Assign) and n.targets[0].id == field)
            return {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
        cfg = runner_cfg()
        for key, value in call_kwargs('WheelbipeV14FlatPPORunnerCfg', 'policy').items():
            self.assertEqual(cfg['policy'][key], value)
        for key, value in call_kwargs('WheelbipeV13FlatPPORunnerCfg', 'algorithm').items():
            if key == 'schedule':
                self.assertEqual(cfg['algorithm'][key], 'fixed')
                continue  # Intentional stability override after adaptive-LR regression.
            self.assertEqual(cfg['algorithm'][key], value)
        ref_cls = next(n for n in ast.parse((BASE / 'wheelbipe_V14/env_cfg.py').read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == 'WheelbipeV14FlatEnvCfg')
        node = next(n for n in ref_cls.body if isinstance(n, ast.Assign) and n.targets[0].id == 'rewards')
        # Original weights/formulas match at full steady-state scaling.
        weights = dict(env_cfg().rewards)
        self.assertEqual(weights.pop('stand_still_wheel_vel'), -0.10)
        self.assertEqual(weights.pop('stand_still_body_lin_vel'), -0.5)
        self.assertEqual(weights.pop('stand_still_body_ang_vel'), -0.1)
        self.assertEqual(weights.pop('virtual_leg_pitch_rate'), -0.1)
        self.assertEqual(weights.pop('front_wheel_impact'), -1.0)
        self.assertEqual(weights.pop('stand_still_position'), 0.0)
        self.assertEqual(weights['track_lin_vel_xy'], 1.5)
        weights['track_lin_vel_xy'] = 1.0  # Intentional Mongo tracking priority override.
        self.assertEqual(weights, {k.arg: ast.literal_eval(k.value) for k in node.value.keywords})
        reference_values = self.reference_cfg()
        for name, value in vars(env_cfg()).items():
            if name in reference_values and (name.endswith('_sigma') or name.endswith('_constraint') or name.startswith('orientation_')):
                self.assertEqual(value, reference_values[name], name)

    @staticmethod
    def reference_cfg():
        values = {}
        for filename, name in [
            ('wheelbipe25_v3/env_cfg.py', 'Wheelbipe25v3BaseEnvCfg'),
            ('wheelbipe25_v3/env_cfg.py', 'Wheelbipe25v3ReduceSpringFlatEnvCfg'),
            ('wheelbipe25_v3/env_cfg.py', 'Wheelbipe25v3FlatEnvCfg'),
            ('wheelbipe_V13/env_cfg.py', 'WheelbipeV13FlatEnvCfg'),
            ('wheelbipe_V14/env_cfg.py', 'WheelbipeV14FlatEnvCfg'),
        ]:
            values.update(literal_class(BASE / filename, name))
        return values

    def test_all_37_reward_terms_against_upstream_expressions(self):
        source_path = BASE / 'wheelbipe25_v3/env.py'
        tree = ast.parse(source_path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_rewards')
        statements = method.body
        selected = set()
        provided = {'self', 'torch', 'wheel_pos_b', 'height_reward_ref', 'euler_xyz_from_quat', 'wrap_to_pi'}

        # Backward slice each reward expression, retaining upstream if/else
        # blocks and its earlier definition when a term is multiplied by gates.
        def require(name, before):
            if name in provided or name in {'getattr', 'bool', 'float', 'max', 'min', 'len'}:
                return
            for i in range(before - 1, -1, -1):
                stores = {n.id for n in ast.walk(statements[i]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                if name in stores:
                    if i in selected:
                        return
                    selected.add(i)
                    loads = {n.id for n in ast.walk(statements[i]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                    for dependency in loads:
                        # Variables defined inside an if block may also appear
                        # in that block; requiring no earlier definition is safe.
                        require(dependency, i)
                    return

        keys = [key for key in env_cfg().rewards
                if key not in ('stand_still_wheel_vel',
                               'stand_still_body_lin_vel', 'stand_still_body_ang_vel',
                               'virtual_leg_pitch_rate', 'front_wheel_impact', 'stand_still_position')]
        # Limit selection to definitions before locals() gathers reward terms.
        limit = next(i for i, n in enumerate(statements) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'reward_terms' for t in n.targets))
        for key in keys:
            require('rew_' + key, limit)
        result = ast.parse('def reference(self, wheel_pos_b, height_reward_ref):\n    pass').body[0]
        result.body = [statements[i] for i in sorted(selected)] + [ast.Return(value=ast.Dict(
            keys=[ast.Constant(k) for k in keys], values=[ast.Name(id='rew_' + k, ctx=ast.Load()) for k in keys]))]

        torch.manual_seed(42)
        inputs = sample_inputs(256)
        for key in ('command', 'root_lin_vel_b', 'root_ang_vel_b', 'projected_gravity_b',
                    'pitch', 'wheel_pos_b', 'joint_acc', 'joint_vel', 'joint_pos',
                    'applied_torque', 'actions', 'previous_actions', 'before_previous_actions'):
            inputs[key] = torch.randn_like(inputs[key])
        inputs['command'][:50, [0, 2]] = .05
        inputs['height'] = .34 + torch.randn(256) * .15
        inputs['terminated'] = torch.rand(256) < .1
        inputs['undesired_contact'] = (torch.rand(256) < .2).float()
        inputs['wheel_contact_peaks'] = torch.rand(256, 2) * 10
        cfg = SimpleNamespace(**self.reference_cfg())
        data = SimpleNamespace(**{k: inputs[k] for k in ('root_lin_vel_b', 'root_ang_vel_b', 'projected_gravity_b',
                                                      'joint_acc', 'applied_torque')})
        data.root_quat_w = torch.zeros(256, 4)
        fake = SimpleNamespace(
            cfg=cfg, robot=SimpleNamespace(data=data), num_envs=256, device='cpu',
            joint_pos=inputs['joint_pos'], joint_vel=inputs['joint_vel'], command=inputs['command'],
            reset_terminated=inputs['terminated'], _actuate_idx=[0, 1, 2, 3, 5, 6],
            _legs_act_idx=[0, 1, 2, 3], _wheel_idx=[5, 6],
            _left_right_leg_joint_pair_idx=([0, 2], [1, 3]),
            _actions=inputs['actions'], _previous_actions=inputs['previous_actions'],
            _before_previous_actions=inputs['before_previous_actions'],
            _previous_applied_torque=torch.zeros(256, 7), _before_previous_applied_torque=torch.zeros(256, 7),
            _get_policy_action_slices=lambda: (slice(0, 4), slice(4, 6)),
            _get_height_reward_target_height=lambda: inputs['height_cmd'],
            _get_vel_orientation_x_gate_enabled=lambda: False,
            _get_vel_orientation_y_gate_enabled=lambda: False,
            _get_vel_height_gate_enabled=lambda: False,
            _undesired_contact_link_idx=[2],
            _get_wheel_contact_force_peaks=lambda _: inputs['wheel_contact_peaks'],
        )
        forces = torch.zeros(256, 3, 3, 3)
        forces[:, 0, 2, 0] = inputs['undesired_contact'] * 100
        fake.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
        scope = {'torch': torch,
                 'wrap_to_pi': lambda x: (x + torch.pi).remainder(2 * torch.pi) - torch.pi,
                 'euler_xyz_from_quat': lambda _: (torch.zeros(256), inputs['pitch'], torch.zeros(256))}
        air_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_wheel_air_spin_reward')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, result, air_method], type_ignores=[])), str(source_path), 'exec'), scope)
        fake._get_wheel_air_spin_reward = lambda *args: scope['_get_wheel_air_spin_reward'](fake, *args)
        expected = scope['reference'](fake, inputs['wheel_pos_b'], inputs['height'])
        actual = rewards.compute_flat_reward_terms(env_cfg(), **inputs)
        for key in keys:
            torch.testing.assert_close(actual[key], expected[key].float(), msg=lambda msg: f'{key}: {msg}')


if __name__ == '__main__':
    unittest.main()
