"""Build the deployable MJCF from the vendored Mongo model; no Isaac imports."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source/mongo_world'))
from mongo_world.motor_limits import speed_limited_effort
NAMES = ('left_hip_joint', 'right_hip_joint', 'left_parallel_1_joint',
         'right_parallel_1_joint', 'front_hip_joint',
         'left_podarthrum_joint', 'right_podarthrum_joint')
MOTOR_PROFILE = json.loads((ROOT / 'model/motor_config_24v.json').read_text())


def default_actuators():
    """Isaac-compatible nominal groups, shared with zero-action validation."""
    groups = {}
    for group, names in [('legs_act', NAMES[:4]), ('front_arm', NAMES[4:5]), ('wheel', NAMES[5:])]:
        motor = MOTOR_PROFILE['motors'][group]
        groups[group] = dict(joint_names_expr=list(names), stiffness=motor['stiffness'],
                             speed_limit_enabled=True,
                             speed_taper_start_fraction=MOTOR_PROFILE['speed_taper_start_fraction'],
                             damping=motor['damping'], effort_limit=motor['effort_limit_nm'],
                             velocity_limit=motor['velocity_limit_rad_s'], armature=motor['armature_kg_m2'])
    return groups


def motor_for_joint(name):
    group = 'front_arm' if name == 'front_hip_joint' else 'wheel' if name in NAMES[5:] else 'legs_act'
    return MOTOR_PROFILE['motors'][group]
DEFAULT_DEG = dict(zip(NAMES, [-3, -3, 30, 30, 28, 0, 0]))
for side, angles in [('left', [19.899820, -19.905178, 6.810596, -19.912850]),
                     ('right', [19.907970, -19.901633, 6.815199, -19.920997])]:
    DEFAULT_DEG.update({f'{side}_{suffix}_joint': a for suffix, a in
                        zip(('knee', 'parallel_2', 'parallel_3', 'parallel_4'), angles)})


def build(path=None):
    path = Path(path or ROOT / 'mujoco/models/mongo.xml')
    root = ET.parse(ROOT / 'model/wl0901.xml').getroot()
    root.find('compiler').set('meshdir', '../../model/meshes')
    ET.SubElement(root, 'option', timestep=str(MOTOR_PROFILE['physics_dt_s']), gravity='0 0 -9.81',
                  integrator='Euler', solver='Newton', iterations='100', tolerance='1e-10')
    defaults = ET.SubElement(root, 'default')
    ET.SubElement(defaults, 'geom', friction='1 0.005 0.0001', solref='0.01 1')
    world = root.find('worldbody')
    base = world.find('body')
    base.set('pos', '0 0 0.34')
    ET.SubElement(base, 'freejoint', name='root')
    for geom in base.iter('geom'):
        geom.set('contype', '1')
        geom.set('conaffinity', '2')  # robot/terrain only, no self collisions
    for joint in base.iter('joint'):
        name = joint.get('name')
        joint.attrib.pop('actuatorfrcrange', None)
        if name in NAMES:
            joint.set('armature', str(motor_for_joint(name)['armature_kg_m2']))
        else:
            joint.set('armature', '0.0001')
        if name == 'front_podarthrum_joint':
            joint.attrib.pop('axis', None)
            joint.set('type', 'ball')
            joint.set('damping', '0.001')
            joint.set('frictionloss', '0.015')
    # Use analytic wheel contacts rather than convex hulls of the mesh tyres.
    for side, offset in [('left', .104298), ('right', -.104298)]:
        body = base.find(f".//body[@name='{side}_wheel']")
        body.find('geom').set('contype', '0')
        body.find('geom').set('conaffinity', '0')
        ET.SubElement(body, 'geom', name=f'{side}_tyre', type='cylinder',
                      size='0.13 0.025', pos=f'0 {offset} 0', quat='0.70710678 0.70710678 0 0',
                      contype='1', conaffinity='2', rgba='.12 .12 .12 1')
    caster = base.find(".//body[@name='front_wheel']")
    caster.find('geom').set('contype', '0')
    caster.find('geom').set('conaffinity', '0')
    ET.SubElement(caster, 'geom', name='caster_contact', type='sphere', size='.0303',
                  contype='1', conaffinity='2', rgba='.15 .15 .15 1')
    # Explicit attachment sites avoid connect's implicit qpos=0 anchor inference.
    equality = root.find('equality')
    equality.clear()
    for side in ('left', 'right'):
        for index, (b1, p1, b2, p2) in enumerate([
            ('thigh', '-0.098 0 0', 'parallel_3', '-0.098 0 0.06501'),
            ('calf', '-0.04368 0 0.04947', 'parallel_4', '-0.11699 0 0')]):
            for label, body, pos in [('a', b1, p1), ('b', b2, p2)]:
                ET.SubElement(base.find(f".//body[@name='{side}_{body}']"), 'site',
                              name=f'{side}_{index}_{label}', pos=pos, size='.003')
            ET.SubElement(equality, 'connect', name=f'{side}_loop_{index}',
                          site1=f'{side}_{index}_a', site2=f'{side}_{index}_b', solref='.01 1', solimp='.999 .999 .001')
    actuator = ET.SubElement(root, 'actuator')
    for name in NAMES:
        limit = motor_for_joint(name)['effort_limit_nm']
        ET.SubElement(actuator, 'motor', name=name, joint=name,
                      ctrllimited='true', ctrlrange=f'-{limit} {limit}')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='20 20 .1',
                  contype='2', conaffinity='1', rgba='.6 .65 .7 1')
    ET.SubElement(world, 'light', pos='0 0 4', directional='true', dir='0 0 -1')
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


if __name__ == '__main__':
    print(build())
