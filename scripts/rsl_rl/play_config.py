"""Resolve playback physics against the checkpoint's recorded training settings."""
from pathlib import Path

import yaml


def resolve_motor_speed_limits(env_cfg, checkpoint_path, selection='auto'):
    """Do not silently add speed protection to an archived actor's plant."""
    if selection not in ('auto', 'on', 'off'):
        raise ValueError(f'Unknown motor-speed selection: {selection}')
    path = Path(checkpoint_path).resolve().parent / 'params/env.yaml'
    saved = yaml.load(path.read_text(), Loader=yaml.BaseLoader) if path.is_file() else None
    if saved is not None and selection == 'auto':
        env_cfg.motor_profile = saved.get('motor_profile', 'legacy_training_snapshot')
    groups = {}
    for name, actuator in env_cfg.robot_cfg.actuators.items():
        if not hasattr(actuator, 'speed_limit_enabled'):
            continue
        if selection != 'auto':
            enabled, source = selection == 'on', 'explicit_cli'
        elif saved is not None:
            old = saved.get('robot_cfg', {}).get('actuators', {}).get(name) or {}
            raw = old.get('speed_limit_enabled', 'false')
            if raw not in ('true', 'false'):
                raise ValueError(f'Invalid speed_limit_enabled for {name}: {raw}')
            enabled, source = raw == 'true', 'training_snapshot'
            actuator.speed_taper_start_fraction = float(old.get('speed_taper_start_fraction', 0.9))
        else:
            enabled, source = actuator.speed_limit_enabled, 'current_config_no_snapshot'
        actuator.speed_limit_enabled = enabled
        groups[name] = dict(enabled=enabled, taper_start_fraction=actuator.speed_taper_start_fraction,
                            source=source)
    return groups


def resolve_joint_velocity_source(env_cfg, checkpoint_path):
    """Never change an existing actor's velocity semantics through source edits."""
    path = Path(checkpoint_path).resolve().parent / 'params/env.yaml'
    before = env_cfg.joint_velocity_source
    if path.is_file():
        saved = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        value = saved.get('joint_velocity_source', 'native')
        source = 'training_snapshot' if 'joint_velocity_source' in saved else 'legacy_snapshot_compatibility'
    else:
        value, source = before, 'current_config_no_snapshot'
    if value not in ('native', 'position_delta'):
        raise ValueError(f'Invalid joint_velocity_source in {path}: {value!r}')
    env_cfg.joint_velocity_source = value
    return dict(effective_value=value, current_source_value=before, source=source)


def resolve_external_force_iterations(physx_cfg, checkpoint_path, selection='auto'):
    """Return provenance while choosing the TGS external-force integration flag.

    Legacy training archives omit this option. Use False for their compatibility
    playback, rather than inheriting the new training source's opt-in True.
    This is a compatibility choice, not proof of the old engine's runtime state.
    """
    if selection not in ('auto', 'on', 'off'):
        raise ValueError(f'Unknown external-force selection: {selection}')
    path = Path(checkpoint_path).resolve().parent / 'params/env.yaml'
    field = 'enable_external_forces_every_iteration'
    before = getattr(physx_cfg, field, None)
    recorded = None
    if path.is_file():
        # BaseLoader reads Isaac's Python-tagged YAML as data, without creating
        # Python objects. Only this scalar is required here.
        saved = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        raw = saved.get('sim', {}).get('physx', {}).get(field)
        if raw is not None:
            if raw not in ('true', 'false'):
                raise ValueError(f'Invalid {field} in {path}: {raw!r}')
            recorded = raw == 'true'
    if selection != 'auto':
        value, source = selection == 'on', 'explicit_cli'
    elif recorded is not None:
        value, source = recorded, 'training_snapshot'
    elif path.is_file():
        value, source = False, 'legacy_snapshot_compatibility'
    else:
        value, source = before, 'current_config_no_snapshot'
    supported = hasattr(physx_cfg, field)
    if supported:
        setattr(physx_cfg, field, value)
    elif value is True:
        raise ValueError(f'This Isaac Lab version cannot enable {field}; use a matching version.')
    return dict(field=field, selection=selection, source=source,
                training_snapshot=str(path) if path.is_file() else None,
                recorded_value=recorded, current_source_value=before,
                effective_value=value, supported=supported)
