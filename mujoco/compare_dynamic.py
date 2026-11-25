#!/usr/bin/env python3
"""Verify paired reset provenance and compare Isaac/MuJoCo dynamic CSV traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from check_physics_snapshot import check as check_physics


def read(path):
    with (path/'tracking_data.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty tracking data: {path}')
    return {k:np.asarray([r[k] for r in rows],dtype=str if k=='phase' else float) for k in rows[0]}


def compare(isaac, mujoco, snapshot, output):
    a,b=read(isaac),read(mujoco)
    output.mkdir(parents=True,exist_ok=False)
    if len(a['step']) != len(b['step']):
        raise ValueError('Trace lengths differ; compare only completed identical protocols')
    physical=check_physics(snapshot,mujoco)
    initial=json.loads((mujoco/'initial_state_check.json').read_text())
    source_input=np.load(snapshot.with_suffix('.policy_input.npy'))
    replay_input=np.load(mujoco/'initial_policy_observation.npy')
    keys=['command_vx_m_s','command_vy_m_s','command_wz_rad_s','command_height_m','command_front_arm_deg']
    command_errors={key:float(abs(a[key]-b[key]).max()) for key in keys}
    normalization={}
    motors=[key.split('/',1)[1] for key in a if key.startswith('applied_torque_nm/')]
    for name,data in [('isaac',a),('mujoco',b)]:
        errors=[]
        for motor in motors:
            torque=data['applied_torque_nm/'+motor];limit=data['effort_limit_nm/'+motor]
            expected=np.r_[0.,np.diff(torque)]/np.maximum(limit,1e-6)
            errors.append(float(abs(expected-data['normalized_torque_delta/'+motor]).max()))
        normalization[name]=max(errors)
    summary=dict(samples=len(a['step']),phase_names_identical=bool(np.array_equal(a['phase'],b['phase'])),
        timestamp_max_error_s=float(abs(a['time_s']-b['time_s']).max()),command_max_errors=command_errors,
        first_policy_input_max_error=float(abs(source_input-replay_input).max()),
        initial_state_max_errors={k:v for k,v in initial.items() if k.endswith('_max_error')},
        torque_normalization_max_errors=normalization,physics_parameters=physical,
        reset_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        sources=dict(isaac=str(isaac),mujoco=str(mujoco),reset_snapshot=str(snapshot)))
    run=json.loads((mujoco/'summary.json').read_text())
    summary['reset_snapshot_hash_matches']=run['physics_snapshot_sha256']==summary['reset_snapshot_sha256']
    summary['alignment_passed']=(summary['phase_names_identical'] and summary['reset_snapshot_hash_matches']
        and summary['timestamp_max_error_s']<1e-8 and summary['first_policy_input_max_error']<2e-6
        and max(summary['initial_state_max_errors'].values(),default=0.)<1e-6
        and max(normalization.values())<1e-5 and physical['passed']
        and all(value<(1e-3 if key=='command_front_arm_deg' else 2e-6) for key,value in command_errors.items()))
    metrics=[]
    channels=[('command_vx_m_s','actual_vx_m_s','vx_m_s',1.),
        ('command_wz_rad_s','actual_wz_rad_s','wz_rad_s',1.),
        ('command_height_m','actual_height_m','height_mm',1000.),
        ('command_front_arm_deg','actual_front_arm_deg','front_arm_deg',1.)]
    for engine,data in [('isaac',a),('mujoco',b)]:
        for phase in ['ALL',*dict.fromkeys(data['phase'])]:
            mask=np.ones(len(data['step']),bool) if phase=='ALL' else data['phase']==phase
            row=dict(engine=engine,phase=phase,samples=int(mask.sum()))
            for cmd,actual,label,scale in channels:
                error=(data[actual][mask]-data[cmd][mask])*scale
                row.update({label+'_mae':float(abs(error).mean()),label+'_bias':float(error.mean()),
                            label+'_p95_abs':float(np.percentile(abs(error),95))})
            metrics.append(row)
    with (output/'phase_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    summary['overall_metrics']=[r for r in metrics if r['phase']=='ALL']
    (output/'comparison.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(13,13),sharex=True)
    panels=[*[(cmd,actual,label,scale) for cmd,actual,label,scale in channels],(None,'pitch_deg','Pitch (deg)',1.)]
    for ax,(cmd,actual,label,scale) in zip(axes,panels):
        if cmd:ax.plot(a['time_s'],a[cmd]*scale,'k--',label='Command',lw=.9)
        ax.plot(a['time_s'],a[actual]*scale,label='Isaac',lw=.85)
        ax.plot(b['time_s'],b[actual]*scale,label='MuJoCo paired reset',lw=.85)
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Command time (s)');fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--isaac',type=Path,required=True,help='one Play camera pass directory')
    p.add_argument('--mujoco',type=Path,required=True,help='dynamic_test.py output directory')
    p.add_argument('--reset-snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='new comparison directory')
    args=p.parse_args()
    result=compare(args.isaac,args.mujoco,args.reset_snapshot,args.output)
    print(json.dumps(result,indent=2))
    return 0 if result['alignment_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
