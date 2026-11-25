"""A paired reset restores pose, velocity, arm and history without target jumps."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller import Controller,load_config
from sim2sim import make_model,parser
from reset_state import restore_reset,root_com_velocity


class ResetStateTests(unittest.TestCase):
    def test_restore_all_state_and_preserve_reference_pose(self):
        cfg=load_config();model=make_model(parser().parse_args(['--zero-action']),cfg)
        data=mujoco.MjData(model);ctrl=Controller(model,cfg);default=ctrl.default.copy()
        joint_pos={};joint_vel={}
        for jid in range(model.njnt):
            kind=model.jnt_type[jid];name=model.joint(jid).name
            if kind==mujoco.mjtJoint.mjJNT_FREE:continue
            if kind==mujoco.mjtJoint.mjJNT_BALL:
                for axis in range(3):joint_pos[f'{name}:{axis}']=0.;joint_vel[f'{name}:{axis}']=.01*axis
            else:joint_pos[name]=.01*jid;joint_vel[name]=.02*jid
        history=np.tile(np.arange(31,dtype=float)/100,(5,1))
        state=dict(schema='mongo_reset_state_v1',root_pos=[.1,-.2,.38],root_quat_wxyz=[np.sqrt(.5),0,0,np.sqrt(.5)],
            root_link_lin_vel_w=[.2,.1,-.03],root_ang_vel_w=[.1,.2,.3],joint_pos=joint_pos,joint_vel=joint_vel,
            front_arm_target_rad=.17,previous_actions=[0]*6,policy_history=history.tolist())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reset.json';path.write_text(json.dumps({'initial_state':state}))
            report=restore_reset(model,data,ctrl,path)
        np.testing.assert_allclose(data.qpos[:3],state['root_pos'])
        self.assertLess(report['root_angular_velocity_max_error'],1e-12)
        self.assertEqual(report['joint_position_max_error'],0.)
        self.assertEqual(report['joint_velocity_max_error'],0.)
        np.testing.assert_array_equal(ctrl.default,default)
        self.assertEqual(ctrl.front,.17)
        ctrl.set_command(.38,0.)
        obs=ctrl.observe(data,[.2,0,.3],.38).reshape(5,31)
        np.testing.assert_allclose(obs[:-1],history[:-1],atol=1e-7)
        np.testing.assert_allclose(obs[-1,4:-1],history[-1,4:-1],atol=1e-7)
        self.assertAlmostEqual(obs[-1,-1],ctrl.front,places=7)
        # COM point agrees with the engine's own body-velocity query.
        actual=np.empty(6);body=model.body('wlbody').id
        # mjOBJ_BODY local axes are principal inertia axes, not link axes.
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,actual,0)
        np.testing.assert_allclose(root_com_velocity(model,data),data.xmat[body].reshape(3,3).T@actual[3:],atol=1e-12)


if __name__=='__main__':unittest.main()
