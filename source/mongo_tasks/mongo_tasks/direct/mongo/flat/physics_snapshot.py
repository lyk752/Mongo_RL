"""Export one instantiated robot, after startup/reset randomization, as SI data."""


def actuator_effort_limits(robot, joint_names, env_index=0):
    """Actual explicit-motor limits in requested order, never solver max forces."""
    import torch
    limits = {name: actuator.effort_limit[env_index, j]
              for actuator in robot.actuators.values()
              for j, name in enumerate(actuator.joint_names)}
    return torch.stack([limits[name] for name in joint_names]).clamp_min(1e-6)


def reset_state(env, index=0):
    """Portable link-frame state plus the actor's just-reset history."""
    data = env.robot.data
    values = lambda x: x[index].detach().cpu().tolist()
    return dict(schema='mongo_reset_state_v1',
                root_pos=values(data.root_link_pos_w - env.scene.env_origins),
                root_quat_wxyz=values(data.root_link_quat_w),
                root_link_lin_vel_w=values(data.root_link_lin_vel_w),
                root_ang_vel_w=values(data.root_link_ang_vel_w),
                root_com_lin_vel_b=values(data.root_lin_vel_b),
                joint_pos=dict(zip(env.robot.joint_names, values(data.joint_pos))),
                joint_vel=dict(zip(env.robot.joint_names, values(data.joint_vel))),
                front_arm_target_rad=float(env.front_arm_target_cmd[index]),
                height_command_m=float(env.height_cmd[index]),
                velocity_command=values(env.command), previous_actions=values(env._actions),
                policy_history=values(env._policy_obs_history),
                position_reference='root link origin, relative to environment origin',
                linear_velocity_reference='root link origin in world coordinates')


def runtime_snapshot(env, env_index=0, include_state=False):
    robot = env.robot
    view = robot.root_physx_view
    index = env_index
    array = lambda tensor: tensor.detach().cpu().numpy()
    armature = array(view.get_dof_armatures())[index]
    friction = array(view.get_dof_friction_properties())[index]
    joints = {}
    for i, name in enumerate(robot.joint_names):
        joints[name] = dict(armature=float(armature[i]),
                            static_friction=float(friction[i, 0]),
                            dynamic_friction=float(friction[i, 1]),
                            viscous_friction=float(friction[i, 2]))
    for actuator in robot.actuators.values():
        for j, name in enumerate(actuator.joint_names):
            joints[name]['speed_limit_enabled'] = getattr(actuator.cfg, 'speed_limit_enabled', False)
            joints[name]['speed_taper_start_fraction'] = getattr(actuator.cfg, 'speed_taper_start_fraction', 0.9)
            for field, output in [('stiffness', 'kp'), ('damping', 'kd'), ('effort_limit', 'effort_limit'),
                                  ('velocity_limit', 'velocity_limit')]:
                joints[name][output] = float(array(getattr(actuator, field))[index, j])
    masses = array(view.get_masses())[index]
    coms = array(view.get_coms())[index]
    inertias = array(view.get_inertias())[index]
    bodies = {name: dict(mass=float(masses[i]), com=coms[i, :3].tolist(),
                         inertia_body=inertias[i].reshape(3, 3).T.tolist())
              for i, name in enumerate(robot.body_names)}
    materials = {}
    for name, path in zip(robot.body_names, view.link_paths[index]):
        link = robot._physics_sim_view.create_rigid_body_view(path)
        materials[name] = array(link.get_material_properties())[0].tolist()
    ground = env.cfg.terrain.physics_material
    snapshot = dict(schema='mongo_physics_v1', env_index=index,
                physics_dt=env.cfg.sim.dt, decimation=env.cfg.decimation,
                motor_profile=getattr(env.cfg, 'motor_profile', None),
                gravity=list(env.cfg.sim.gravity), joints=joints, bodies=bodies,
                materials=materials,
                ground_material=dict(static_friction=ground.static_friction,
                                     dynamic_friction=ground.dynamic_friction,
                                     restitution=ground.restitution,
                                     friction_combine_mode=ground.friction_combine_mode),
                joint_velocity_source=env.cfg.joint_velocity_source,
                external_forces_every_iteration=getattr(
                    env.cfg.sim.physx, 'enable_external_forces_every_iteration', None),
                note='One runtime realization; no contact/material or collision-geometry equivalence implied.')
    if include_state:
        snapshot['initial_state'] = reset_state(env, index)
    return snapshot
