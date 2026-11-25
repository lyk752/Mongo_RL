#!/usr/bin/env python3
"""Check a saved MuJoCo MJB against the originating Isaac runtime snapshot."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from build_model import NAMES


def check(snapshot_path, run_path):
    source = json.loads(snapshot_path.read_text())
    model = mujoco.MjModel.from_binary_path(str(run_path / 'model.mjb'))
    resolved = json.loads((run_path / 'resolved_physics.json').read_text())
    errors = {k: [] for k in ['mass_kg', 'com_m', 'inertia_kg_m2', 'armature_kg_m2',
                              'static_friction_nm', 'viscous_friction_nm_s_rad',
                              'kp', 'kd', 'effort_limit_nm', 'motor_velocity_limit_rad_s',
                              'speed_limit_enabled', 'speed_taper_start_fraction', 'contact_static_mu']}
    for name, row in source['bodies'].items():
        bid = model.body(name).id
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[bid])
        rotation = rotation.reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[bid]) @ rotation.T
        errors['mass_kg'].append(abs(model.body_mass[bid] - row['mass']))
        errors['com_m'].append(float(np.max(abs(model.body_ipos[bid] - row['com']))))
        errors['inertia_kg_m2'].append(float(np.max(abs(inertia - row['inertia_body']))))
    for name, row in source['joints'].items():
        joint, axis = name.rsplit(':', 1) if ':' in name else (name, '0')
        dof = model.joint(joint).dofadr[0] + int(axis)
        for values, source_key, error_key in [
            (model.dof_armature, 'armature', 'armature_kg_m2'),
            (model.dof_frictionloss, 'static_friction', 'static_friction_nm'),
            (model.dof_damping, 'viscous_friction', 'viscous_friction_nm_s_rad')]:
            errors[error_key].append(abs(values[dof] - row[source_key]))
    for i, name in enumerate(NAMES):
        for key in ('speed_limit_enabled', 'speed_taper_start_fraction'):
            if key in source['joints'][name]:
                errors[key].append(abs(float(resolved[key][i]) - float(source['joints'][name][key])))
        for key in ('kp', 'kd'):
            errors[key].append(abs(resolved[key][i] - source['joints'][name][key]))
        effort = model.actuator_ctrlrange[model.actuator(name).id, 1]
        errors['effort_limit_nm'].append(abs(effort - source['joints'][name]['effort_limit']))
        if 'velocity_limit' in source['joints'][name] and 'motor_velocity_limit_rad_s' in resolved:
            errors['motor_velocity_limit_rad_s'].append(abs(
                resolved['motor_velocity_limit_rad_s'][i] - source['joints'][name]['velocity_limit']))
    for pair in range(model.npair):
        geom = model.pair_geom1[pair]
        if model.geom_bodyid[geom] == 0:
            geom = model.pair_geom2[pair]
        body = model.body(model.geom_bodyid[geom]).name
        mu = np.mean(source.get('materials', {}).get(body, [[1., 1., 0.]]), axis=0)[0]
        mu *= source.get('ground_material', {}).get('static_friction', 1.)
        errors['contact_static_mu'].append(float(np.max(abs(model.pair_friction[pair, :2] - mu))))
    report = dict(max_absolute_errors={k: max(v, default=0.) for k, v in errors.items()},
                  body_count=len(source['bodies']), dof_count=len(source['joints']),
                  scope='Mapped numeric parameters only; dynamic friction, restitution, collision shapes and solver trajectories are not equivalent.')
    report['passed'] = max(report['max_absolute_errors'].values()) < 1e-6
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.snapshot, args.run)
    (args.run / 'parameter_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
