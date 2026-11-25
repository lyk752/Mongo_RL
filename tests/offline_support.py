"""Load torch-only modules and literal configs without starting Isaac Sim."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FLAT = ROOT / 'source/mongo_tasks/mongo_tasks/direct/mongo/flat'
AGENT = FLAT.parent / 'agents/rsl_rl_ppo_cfg.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_class(path, name):
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    result = {}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            names, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            names, value = [node.target], node.value
        else:
            continue
        for target in names:
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
    return result


def runner_cfg():
    result = literal_class(AGENT, 'MongoFlatPPORunnerCfg')
    cls = next(n for n in ast.parse(AGENT.read_text()).body if isinstance(n, ast.ClassDef))
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            key = node.targets[0].id
            result[key] = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
            result[key]['class_name'] = 'ActorCritic' if key == 'policy' else 'PPO'
    return result


def env_cfg():
    path = FLAT / 'env_cfg.py'
    values = literal_class(path, 'MongoFlatEnvCfg')
    constants = vars(load_module('offline_cfg_constants', FLAT / 'cfg_utils.py'))
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'MongoFlatEnvCfg')
    for node in cls.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if isinstance(value, ast.Name) and value.id in constants:
            for target in targets:
                values[target.id] = constants[value.id]
    cfg = SimpleNamespace(**values)
    post_init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
    scope = constants.copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[post_init], type_ignores=[])), str(path), 'exec'), scope)
    scope['__post_init__'](cfg)
    return cfg


def isolated_env_class(method_names):
    """Execute actual interface methods with a tensor robot double in CPU tests."""
    method_names = set(method_names)
    import math
    import torch
    source = ast.parse((FLAT / 'env.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'MongoFlatEnv')
    cls.bases = []
    cls.decorator_list = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    scope = {'torch': torch, 'math': math,
             'encoder_velocity': load_module('velocity_feedback', FLAT / 'velocity_feedback.py').encoder_velocity,
             'condition_policy_actions': load_module('actions', FLAT / 'action_pipeline.py').condition_policy_actions}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(FLAT / 'env.py'), 'exec'), scope)
    return scope['MongoFlatEnv']
