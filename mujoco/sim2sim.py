#!/usr/bin/env python3
"""Run a Mongo ONNX actor in MuJoCo, with reproducible CSV and JSON diagnostics."""
import argparse
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import time
import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from build_model import DEFAULT_DEG, ROOT
from controller import Controller, check_deployment, load_config
from reset_state import restore_reset, root_com_velocity


def default_policy_path():
    selection = ROOT / 'mujoco/policy.json'
    return (selection.parent / json.loads(selection.read_text())['policy']).resolve()


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, help='ONNX actor; defaults to the frozen policy selected by mujoco/policy.json')
    p.add_argument('--env-config', type=Path, help='training params/env.yaml; inferred from policy directory')
    p.add_argument('--deployment-config', type=Path, help='defaults to deployment_config.json beside policy')
    p.add_argument('--physics-snapshot', type=Path, help='mongo_physics_v1 JSON exported from an Isaac runtime')
    p.add_argument('--reset-snapshot', type=Path, help='restore both physics and initial state/history from one Play reset snapshot')
    p.add_argument('--zero-action', action='store_true', help='PD/model smoke test without an actor')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--duration', type=positive, default=20.)
    p.add_argument('--vx', type=float, default=0.)
    p.add_argument('--vy', type=float, default=0.)
    p.add_argument('--wz', type=float, default=0.)
    p.add_argument('--height', type=positive, default=.34)
    p.add_argument('--front-fraction', type=float, default=0., help='0: raised; 1: height-dependent upper bound')
    p.add_argument('--output', type=Path, help='new output directory (existing files are never overwritten)')
    return p


def model_xml(args, cfg):
    path = ROOT / 'mujoco/models/mongo.xml'
    tree = ET.parse(path)
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(ROOT / 'model/meshes'))
    root.find('option').set('timestep', str(cfg.get('sim', {}).get('dt', .005)))
    # Explicit pairs allow the training material's multiply combine rule. The
    # engines' native default combination rules are different.
    for body in root.find('worldbody').iter('body'):
        for i, geom in enumerate(body.findall('geom')):
            if geom.get('name') is None:
                geom.set('name', f"{body.get('name')}_geom_{i}")
    contact = ET.SubElement(root, 'contact')
    geoms = list(root.find('worldbody').iter('geom'))
    for robot_geom in geoms:
        if robot_geom.get('contype') != '1':
            continue
        for ground_geom in geoms:
            if ground_geom.get('contype') == '2':
                ET.SubElement(contact, 'pair', geom1=robot_geom.get('name'),
                              geom2=ground_geom.get('name'), condim='3',
                              friction='1 1 .005 .0001 .0001', solref='.01 1')
    return ET.tostring(root, encoding='unicode')


def make_model(args, cfg):
    return mujoco.MjModel.from_xml_string(model_xml(args, cfg))


@contextmanager
def interactive_viewer(model, data, **options):
    from mujoco import viewer as mj_viewer
    threads_before = set(threading.enumerate())
    viewer = mj_viewer.launch_passive(model, data, **options)
    # launch_passive creates a daemon render thread. Handle.close() only signals
    # exit; wait for its cleanup before Python/GLFW teardown at process exit.
    viewer_threads = [thread for thread in threading.enumerate()
                      if thread not in threads_before]
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in viewer_threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError('MuJoCo viewer thread did not finish closing')


def run(args, command_source=None, recorder=None):
    if args.reset_snapshot:
        if args.physics_snapshot and args.physics_snapshot.resolve() != args.reset_snapshot.resolve():
            raise ValueError('--reset-snapshot and --physics-snapshot must refer to the same snapshot')
        args.physics_snapshot = args.reset_snapshot
    if not args.zero_action and args.policy is None:
        args.policy = default_policy_path()
        print(f'Using default policy: {args.policy}', flush=True)
    if args.zero_action and args.policy is not None:
        raise ValueError('--policy and --zero-action are mutually exclusive')
    if not 0 <= args.front_fraction <= 1:
        raise ValueError('front-fraction must be in [0,1]')
    if not np.isfinite([args.vx, args.vy, args.wz]).all():
        raise ValueError('Commands must be finite')
    if args.policy and args.env_config is None:
        candidate = args.policy.resolve().parent.parent / 'params/env.yaml'
        if not candidate.is_file():
            raise ValueError('Cannot infer training config; supply --env-config')
        args.env_config = candidate
    cfg = load_config(args.env_config)
    if command_source is not None:
        command_source.configure(cfg, args)
    decimation = int(cfg.get('decimation', 4))
    if decimation <= 0 or decimation != cfg.get('decimation', 4):
        raise ValueError('decimation must be a positive integer')
    session = None
    if args.policy:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(args.policy), sess_options=options, providers=['CPUExecutionProvider'])
        inp, out = session.get_inputs(), session.get_outputs()
        if len(inp) != 1 or len(out) != 1 or inp[0].shape[-1] != 155 or out[0].shape[-1] != 6:
            raise ValueError('Expected a feedforward actor with [1,155] -> [1,6]')
        if args.deployment_config is None:
            candidate = args.policy.with_name('deployment_config.json')
            if candidate.is_file():
                args.deployment_config = candidate
    if args.deployment_config:
        check_deployment(args.deployment_config, cfg)
    model = make_model(args, cfg)
    data = mujoco.MjData(model)
    ctrl = Controller(model, cfg, args.physics_snapshot)
    # Refresh composite inertias after runtime snapshot overrides.
    mujoco.mj_setConst(model, data)
    initial_state = cfg.get('robot_cfg', cfg.get('robot', {})).get('init_state', {})
    data.qpos[:3] = initial_state.get('pos', data.qpos[:3])
    data.qpos[3:7] = initial_state.get('rot', data.qpos[3:7])
    for name, degrees in DEFAULT_DEG.items():
        data.qpos[model.joint(name).qposadr[0]] = np.deg2rad(degrees)
    initial = cfg.get('robot_cfg', cfg.get('robot', {})).get('init_state', {}).get('joint_pos', {})
    for name, angle in initial.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            data.qpos[model.jnt_qposadr[jid]] = angle
    # Start from a quiet, deterministic assembly rather than training randomization.
    data.qpos[ctrl.qadr] = ctrl.default
    mujoco.mj_forward(model, data)
    ctrl.set_command(args.height, args.front_fraction, snap=True)
    initial_state_check = restore_reset(model, data, ctrl, args.reset_snapshot) if args.reset_snapshot else None
    dt = model.opt.timestep * decimation
    steps = max(1, int(np.ceil(args.duration / dt)))
    if command_source is not None and hasattr(command_source, 'num_steps'):
        steps = command_source.num_steps
    output = args.output or ROOT / 'mujoco/outputs' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    if initial_state_check is not None:
        (output/'initial_state_check.json').write_text(json.dumps(initial_state_check, indent=2))
    if recorder is not None:
        recorder.configure(model, cfg, ctrl, output)
    (output / 'source_model.xml').write_text(model_xml(args, cfg))
    # MJCF permits only scalar ball-joint friction/damping, whereas runtime
    # snapshots may have three independently randomized values. MJB preserves
    # those exact per-axis parameters even when XML cannot express them.
    mujoco.mj_saveModel(model, str(output / 'model.mjb'), None)
    xml_representable = True
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL:
            start = model.jnt_dofadr[jid]
            for values in (model.dof_armature, model.dof_damping, model.dof_frictionloss):
                xml_representable &= bool(np.all(values[start:start+3] == values[start]))
    if xml_representable:
        mujoco.mj_saveLastXML(str(output / 'model.xml'), model)
    resolved_physics = dict(ctrl.physics, kp=ctrl.kp.tolist(), kd=ctrl.kd.tolist(),
                            effort_limit=ctrl.effort.tolist(),
                            motor_velocity_limit_rad_s=ctrl.motor_velocity_limit.tolist(),
                            speed_limit_enabled=ctrl.speed_limit_enabled.tolist(),
                            speed_taper_start_fraction=ctrl.speed_taper_start_fraction.tolist(),
                            motor_velocity_limit_semantics='directional effort taper using physical speed when enabled; external overspeed remains possible',
                            joint_velocity_source=ctrl.feedback_source,
                            sensor_sample_phase=('before_last_physics_substep' if ctrl.feedback_source == 'position_delta'
                                                 else 'current_state_legacy_mujoco'),
                            physics_dt=model.opt.timestep, decimation=decimation,
                            xml_representable=xml_representable,
                            gravity=model.opt.gravity.tolist())
    (output / 'resolved_physics.json').write_text(json.dumps(resolved_physics, indent=2))
    (output / 'resolved_env.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    command = np.array([args.vx, args.vy, args.wz])
    height = args.height
    rows = []
    reason = 'duration_completed'
    viewer_options = {} if command_source is None else {'key_callback': command_source.key_callback}
    context = nullcontext(None) if args.headless else interactive_viewer(model, data, **viewer_options)
    fields = ['time_s', 'command_vx', 'command_vy', 'command_wz', 'command_height', 'front_target_rad',
              'x', 'y', 'z', 'relative_height', 'vx', 'vy', 'wz', 'roll_deg', 'pitch_deg',
              'front_actual_rad', 'loop_error_m', 'max_abs_torque_nm']
    fields += [f'action_{i}' for i in range(6)]
    fields += [f'q/{n}' for n in ctrl_names()] + [f'dq/{n}' for n in ctrl_names()] + [f'torque/{n}' for n in ctrl_names()]
    fields += [f'feedback_dq/{n}' for n in ctrl_names()]
    with (output / 'tracking.csv').open('w', newline='') as stream, context as viewer:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(steps):
            wall_start = time.monotonic()
            if viewer is not None and not viewer.is_running():
                reason = 'viewer_closed'
                break
            if command_source is not None:
                command, height, fraction = command_source.sample()
                if command_source.quit_requested:
                    reason = 'user_exit'
                    break
                ctrl.set_command(height, fraction)
            obs = ctrl.observe(data, command, height)
            if step == 0:
                np.save(output/'initial_policy_observation.npy', obs[0])
            raw = np.zeros(6) if session is None else session.run(None, {session.get_inputs()[0].name: obs})[0][0]
            ctrl.action(raw)
            if recorder is not None:
                recorder.begin_step(command, height, raw, getattr(command_source, 'phase', 'policy_rollout'))
            for _ in range(decimation):
                if recorder is not None:
                    recorder.before_substep(data)
                ctrl.apply(data)
                mujoco.mj_step(model, data)
                if recorder is not None:
                    recorder.after_substep(data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    reason = 'nonfinite_state'
                    break
            mujoco.mj_forward(model, data)
            rotation = data.xmat[1].reshape(3, 3)
            velocity = root_com_velocity(model, data)
            angular = data.qvel[3:6]
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            pitch = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
            # Equality anchors are explicit sites, independently measurable.
            loop = max(np.linalg.norm(data.site_xpos[model.site(f'{side}_{i}_a').id] -
                                      data.site_xpos[model.site(f'{side}_{i}_b').id])
                       for side in ('left', 'right') for i in range(2))
            x, y, z = data.qpos[:3]
            relative_height = z
            row = dict(zip(fields[:18], [data.time, *command, height, ctrl.front, x, y, z,
                                        relative_height, velocity[0], velocity[1], angular[2],
                                        np.rad2deg(roll), np.rad2deg(pitch), data.qpos[ctrl.qadr[4]], loop,
                                        np.abs(data.ctrl).max()]))
            row.update({f'action_{i}': a for i, a in enumerate(ctrl.previous)})
            for name, q, dq, torque in zip(ctrl_names(), data.qpos[ctrl.qadr], data.qvel[ctrl.vadr], data.ctrl[ctrl.aids]):
                row.update({f'q/{name}': q, f'dq/{name}': dq, f'torque/{name}': torque})
            feedback = ctrl.feedback_velocity if ctrl.feedback_source == 'position_delta' else data.qvel[ctrl.vadr]
            row.update({f'feedback_dq/{n}': v for n, v in zip(ctrl_names(), feedback)})
            writer.writerow(row)
            rows.append(row)
            if recorder is not None:
                recorder.record(data)
            if reason == 'nonfinite_state':
                break
            if sum(w.number for w in data.warning):
                reason = 'mujoco_warning'
                break
            if relative_height < .15 or abs(roll) > np.deg2rad(60) or abs(pitch) > np.deg2rad(60):
                reason = 'fall'
                break
            if loop > .02:
                reason = 'loop_error'
                break
            if command_source is None:
                ctrl.set_command(args.height, args.front_fraction)
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                time.sleep(max(0., dt - (time.monotonic() - wall_start)))
    def mean_error(actual, goal):
        return float(np.mean([abs(r[actual] - r[goal]) for r in rows])) if rows else None
    report = {'status': reason, 'steps': len(rows), 'simulated_seconds': float(data.time),
              'requested_seconds': args.duration, 'mujoco_version': mujoco.__version__,
              'model_xml_sha256': hashlib.sha256((output / 'model.xml').read_bytes()).hexdigest() if (output / 'model.xml').is_file() else None,
              'source_model_xml_sha256': hashlib.sha256((output / 'source_model.xml').read_bytes()).hexdigest(),
              'model_mjb_sha256': hashlib.sha256((output / 'model.mjb').read_bytes()).hexdigest(),
              'joint_velocity_source': ctrl.feedback_source,
              'body_velocity_reference': 'root_COM_in_body_axes',
              'physics_snapshot_sha256': hashlib.sha256(args.physics_snapshot.read_bytes()).hexdigest() if args.physics_snapshot else None,
              'policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest() if args.policy else None,
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'vx_mae_m_s': mean_error('vx', 'command_vx'), 'wz_mae_rad_s': mean_error('wz', 'command_wz'),
              'height_mae_m': mean_error('relative_height', 'command_height'),
              'front_mae_rad': mean_error('front_actual_rad', 'front_target_rad'),
              'max_loop_error_m': max((r['loop_error_m'] for r in rows), default=None),
              'max_abs_pitch_deg': max((abs(r['pitch_deg']) for r in rows), default=None),
              'displacement_x_m': float(data.qpos[0]),
              'warning_counts': [int(w.number) for w in data.warning]}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Output: {output}')
    return 0 if reason in ('duration_completed', 'viewer_closed', 'user_exit') else 2


def ctrl_names():
    from build_model import NAMES
    return NAMES


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
