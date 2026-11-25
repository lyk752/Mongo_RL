"""Resolve nominal actuator physics and optional measured Isaac runtime snapshots.

MuJoCo has one dry-friction effort, not separate static/dynamic efforts. We
use the static effort for frictionloss and report the dynamic effort separately.
Contact laws and articulation solvers remain engine-specific.
"""
import json
import re

import mujoco
import numpy as np


def resolve(value, name, default):
    if value is None:
        return default
    if isinstance(value, dict):
        matches = [v for expr, v in value.items() if re.fullmatch(expr, name)]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous parameter patterns for {name}')
        return matches[0] if matches else default
    return value


def configure(model, cfg, snapshot_path=None):
    robot = cfg.get('robot_cfg', cfg.get('robot', {}))
    model.opt.gravity[:] = cfg.get('sim', {}).get('gravity', [0, 0, -9.81])
    rows = {}
    for jid in range(1, model.njnt):
        name = model.joint(jid).name
        count = 3 if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL else 1
        for axis in range(count):
            label = f'{name}:{axis}' if count == 3 else name
            dof = model.jnt_dofadr[jid] + axis
            row = dict(armature=float(model.dof_armature[dof]),
                       static_friction=float(model.dof_frictionloss[dof]),
                       dynamic_friction=.01 if count == 3 else float(model.dof_frictionloss[dof]),
                       viscous_friction=float(model.dof_damping[dof]))
            for group in robot.get('actuators', {}).values():
                if any(re.fullmatch(expr, label) for expr in group.get('joint_names_expr', [])):
                    for src, dest in [('armature', 'armature'), ('friction', 'static_friction'),
                                      ('dynamic_friction', 'dynamic_friction'),
                                      ('viscous_friction', 'viscous_friction')]:
                        row[dest] = float(resolve(group.get(src), label, row[dest]))
            rows[label] = row
    snapshot = None
    if snapshot_path:
        snapshot = json.loads(snapshot_path.read_text())
        if snapshot.get('schema') != 'mongo_physics_v1':
            raise ValueError('Expected a mongo_physics_v1 runtime snapshot')
        if not np.isclose(snapshot['physics_dt'], model.opt.timestep):
            raise ValueError('Snapshot physics_dt differs from environment config')
        if snapshot['decimation'] != cfg.get('decimation', 4):
            raise ValueError('Snapshot decimation differs from environment config')
        if set(snapshot['joints']) != set(rows):
            raise ValueError('Runtime snapshot joint names differ from the MuJoCo model')
        rows = snapshot['joints']
        model.opt.gravity[:] = snapshot['gravity']
        for name, body in snapshot['bodies'].items():
            bid = model.body(name).id
            model.body_mass[bid] = body['mass']
            model.body_ipos[bid] = body['com']
            inertia = np.array(body['inertia_body']).reshape(3, 3)
            values, axes = np.linalg.eigh((inertia + inertia.T) / 2)
            if body['mass'] <= 0 or np.any(values <= 0):
                raise ValueError(f'Invalid runtime mass/inertia for {name}')
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            model.body_inertia[bid] = values
            mujoco.mju_mat2Quat(model.body_iquat[bid], axes.ravel())
    for label, row in rows.items():
        name, axis = label.rsplit(':', 1) if ':' in label else (label, '0')
        dof = model.joint(name).dofadr[0] + int(axis)
        for key in ('armature', 'static_friction', 'dynamic_friction', 'viscous_friction'):
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'Invalid {key} for {label}')
        model.dof_armature[dof] = row['armature']
        model.dof_frictionloss[dof] = row['static_friction']
        # This is solver viscous friction; passive actuator damping with an
        # effort limit of zero must NOT be copied here.
        model.dof_damping[dof] = row['viscous_friction']
    ground = (snapshot.get('ground_material', {}) if snapshot else
              cfg.get('terrain', {}).get('physics_material', cfg.get('sim', {}).get('physics_material', {})))
    ground_mu = float(ground.get('static_friction', 1.))
    mode = ground.get('friction_combine_mode', 'multiply')
    if mode != 'multiply':
        raise ValueError(f'Only multiply contact combination is supported, got {mode}')
    contact_rows = []
    for pair in range(model.npair):
        geom = model.pair_geom1[pair]
        if model.geom_bodyid[geom] == 0:
            geom = model.pair_geom2[pair]
        body = model.body(model.geom_bodyid[geom]).name
        materials = np.asarray(snapshot.get('materials', {}).get(body, [[1., 1., 0.]]) if snapshot
                               else [[1., 1., 0.]])
        material = np.mean(materials, axis=0)
        mu = ground_mu * float(material[0])
        model.pair_friction[pair, :2] = mu
        contact_rows.append(dict(body=body, effective_static_mu=mu,
                                 source_material_mean=material.tolist()))
    return dict(schema='mongo_resolved_physics_v1', joints=rows,
                snapshot=str(snapshot_path) if snapshot_path else None,
                contacts=contact_rows,
                bodies={model.body(i).name: dict(mass=float(model.body_mass[i]),
                        com=model.body_ipos[i].tolist(), inertia=model.body_inertia[i].tolist(),
                        inertia_quat_wxyz=model.body_iquat[i].tolist()) for i in range(1, model.nbody)},
                contact_mapping='body-mean static_mu * ground static_mu via explicit pairs; dynamic_mu/restitution recorded only; analytic tyre/caster',
                dry_friction_mapping='static_effort -> frictionloss; dynamic_effort recorded only',
                runtime=snapshot)
