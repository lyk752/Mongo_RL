"""Restore one Isaac reset without replacing the controller's reference pose."""
import json

import mujoco
import numpy as np


def restore_reset(model, data, ctrl, path):
    payload = json.loads(path.read_text())
    state = payload.get('initial_state', {})
    if state.get('schema') != 'mongo_reset_state_v1':
        raise ValueError('--reset-snapshot requires a snapshot containing mongo_reset_state_v1 initial_state')
    vectors = ('root_pos', 'root_quat_wxyz', 'root_link_lin_vel_w', 'root_ang_vel_w', 'previous_actions')
    for name in vectors:
        value = np.asarray(state[name], dtype=float)
        size = 4 if name == 'root_quat_wxyz' else 6 if name == 'previous_actions' else 3
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid reset vector {name}')
    quat = np.asarray(state['root_quat_wxyz'])
    if not np.isclose(np.linalg.norm(quat), 1., atol=1e-5):
        raise ValueError('Invalid reset root quaternion')
    data.qpos[:3], data.qpos[3:7] = state['root_pos'], quat
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3,3)
    data.qvel[:3] = state['root_link_lin_vel_w']
    data.qvel[3:6] = rotation.T @ state['root_ang_vel_w']
    for jid in range(model.njnt):
        kind, name = model.jnt_type[jid], model.joint(jid).name
        qadr, vadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if kind == mujoco.mjtJoint.mjJNT_BALL:
            angles = np.array([state['joint_pos'][f'{name}:{axis}'] for axis in range(3)])
            if not np.allclose(angles, 0., atol=1e-7):
                raise ValueError('Nonzero Isaac spherical reset angles require an explicit joint-frame conversion')
            data.qpos[qadr:qadr+4] = [1,0,0,0]
            data.qvel[vadr:vadr+3] = [state['joint_vel'][f'{name}:{axis}'] for axis in range(3)]
        else:
            data.qpos[qadr] = state['joint_pos'][name]
            data.qvel[vadr] = state['joint_vel'][name]
    history = np.asarray(state['policy_history'], dtype=np.float32)
    if history.shape != (5,31) or not np.isfinite(history).all():
        raise ValueError('Reset requires finite 5 x 31 actor history')
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError('Nonfinite reset joint state')
    ctrl.reset()
    ctrl.front = float(state['front_arm_target_rad'])
    ctrl.position[:] = data.qpos[ctrl.qadr[:5]]
    ctrl.velocity[5:] = data.qvel[ctrl.vadr[5:]]
    ctrl.previous = np.asarray(state['previous_actions'], dtype=float)
    ctrl.history = history.copy()
    mujoco.mj_forward(model, data)
    # Preserve the first complete actor frame exactly; only current commands
    # are patched before first inference, matching Isaac's script injection.
    ctrl.reset_observation = history[-1].copy()
    joint_position_errors, joint_velocity_errors = [], []
    for jid in range(model.njnt):
        name, kind = model.joint(jid).name, model.jnt_type[jid]
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if kind == mujoco.mjtJoint.mjJNT_BALL:
            for axis in range(3):
                joint_velocity_errors.append(abs(data.qvel[model.jnt_dofadr[jid]+axis]-state['joint_vel'][f'{name}:{axis}']))
        else:
            joint_position_errors.append(abs(data.qpos[model.jnt_qposadr[jid]]-state['joint_pos'][name]))
            joint_velocity_errors.append(abs(data.qvel[model.jnt_dofadr[jid]]-state['joint_vel'][name]))
    return dict(source=str(path), root_pos=data.qpos[:3].tolist(), root_quat_wxyz=data.qpos[3:7].tolist(),
                root_link_lin_vel_w=data.qvel[:3].tolist(), root_ang_vel_w=(rotation@data.qvel[3:6]).tolist(),
                joint_pos={model.joint(j).name:float(data.qpos[model.jnt_qposadr[j]])
                           for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE},
                front_arm_target_rad=ctrl.front, actor_history=history.tolist(),
                root_position_max_error=float(np.max(abs(data.qpos[:3]-state['root_pos']))),
                root_quaternion_max_error=float(np.max(abs(data.qpos[3:7]-quat))),
                root_link_velocity_max_error=float(np.max(abs(data.qvel[:3]-state['root_link_lin_vel_w']))),
                root_angular_velocity_max_error=float(np.max(abs(rotation@data.qvel[3:6]-state['root_ang_vel_w']))),
                joint_position_max_error=float(max(joint_position_errors, default=0.)),
                joint_velocity_max_error=float(max(joint_velocity_errors, default=0.)))


def root_com_velocity(model, data):
    """Root COM velocity in body axes, matching Isaac root_lin_vel_b."""
    root = model.body('wlbody').id
    rotation = data.xmat[root].reshape(3,3)
    return rotation.T @ data.qvel[:3] + np.cross(data.qvel[3:6], model.body_ipos[root])
