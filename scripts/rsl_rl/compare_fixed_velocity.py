"""Audit physical/initial-state matching before comparing fixed-speed solvers."""
from pathlib import Path
import argparse
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mongo_fixed_velocity_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def save_csv(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def flatten(value, path=''):
    if isinstance(value,dict):
        return {k:v for key,item in value.items() for k,v in flatten(item,path+'.'+key if path else key).items()}
    if isinstance(value,list):
        return {k:v for i,item in enumerate(value) for k,v in flatten(item,f'{path}[{i}]').items()}
    return {path:value}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--original')
    parser.add_argument('--candidates',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    directories=[Path(args.baseline).resolve(),*[Path(p).resolve() for p in args.candidates]]
    metadata=[json.loads((p/'metadata.json').read_text()) for p in directories]
    base=metadata[0]
    audit=[]
    fields=['masses','friction','joint_armatures','randomized_initial_root_state','randomized_initial_joint_pos',
        'randomized_initial_joint_vel','inertias','coms','materials','native_joint_stiffness',
        'native_joint_damping','joint_limits','joint_velocity_limits']

    def compare_meta(reference, candidate, label, require_full=True):
        for key in ['checkpoint_sha256','export_sha256','cases','joint_names','contact_names','dt']:
            audit.append(dict(candidate=label,field=key,matched=reference[key]==candidate[key],max_abs_difference=0.))
        for key in fields:
            if not require_full and key not in reference:
                continue
            a,b=np.asarray(reference[key]),np.asarray(candidate[key])
            diff=float(np.max(np.abs(a-b))) if a.shape==b.shape else float('inf')
            audit.append(dict(candidate=label,field=key,
                matched=bool(a.shape==b.shape and np.allclose(a,b,rtol=0,atol=1e-7)),max_abs_difference=diff))
        ra,ca=flatten(reference['actuator_parameters']),flatten(candidate['actuator_parameters'])
        assert ra.keys()==ca.keys()
        difference=max(abs(float(ra[k])-float(ca[k])) for k in ra)
        audit.append(dict(candidate=label,field='all_actuator_gains_and_effort_limits',
                          matched=difference<=1e-7,max_abs_difference=difference))

    if args.original:
        old=json.loads((Path(args.original)/'metadata.json').read_text())
        compare_meta(old,base,'original_to_audit_baseline',False)
        replay=[]
        with np.load(Path(args.original)/'tracking.npz') as a, np.load(directories[0]/'tracking.npz') as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                x,y=a[key],b[key]
                equal=bool(np.array_equal(x,y))
                diff=0. if equal else float(np.max(np.abs(x.astype(float)-y.astype(float))))
                replay.append(dict(array=key,bitwise_equal=equal,max_abs_difference=diff))
        (out/'baseline_replay_validation.json').write_text(json.dumps(replay,indent=2)+'\n')
    for path,candidate in zip(directories[1:],metadata[1:]):
        compare_meta(base,candidate,path.name)
    save_csv(out/'parameter_matching.csv',audit)
    if not all(row['matched'] for row in audit):
        raise AssertionError('Physical/initial-state matching failed; see parameter_matching.csv.')
    config_differences={}
    baseline_config=flatten(yaml.load((directories[0]/'effective_env.yaml').read_text(),Loader=yaml.BaseLoader))
    allowed={'sim.physx.solver_type','sim.physx.enable_external_forces_every_iteration',
             'robot_cfg.spawn.articulation_props.solver_velocity_iteration_count'}
    for path in directories[1:]:
        config=flatten(yaml.load((path/'effective_env.yaml').read_text(),Loader=yaml.BaseLoader))
        changes={key:dict(baseline=baseline_config.get(key),candidate=config.get(key))
                 for key in set(baseline_config)|set(config) if baseline_config.get(key)!=config.get(key)}
        config_differences[path.name]=changes
        assert set(changes)<=allowed,(path,changes)
    (out/'config_differences.json').write_text(json.dumps(config_differences,indent=2)+'\n')

    records=[list(csv.DictReader((p/'case_metrics.csv').open())) for p in directories]
    maps=[{int(r['env_id']):r for r in rows} for rows in records]
    assert all(set(m)==set(maps[0]) for m in maps)
    common=[i for i in maps[0] if all(int(m[i]['failed'])==0 and int(m[i]['measurement_samples'])==500 for m in maps)]
    metrics=['hip_common_rms_deg','vx_mae_m_s','height_mae_mm','virtual_leg_common_rms_deg',
        'hip_last_half_rms_deg','min_link_clearance_mm','rear_contact_over3n_pct','rear_contact_peak_n',
        'wheel_torque_peak_nm','pitch_peak_deg']
    joint_names=base['joint_names'][:4]
    rows=[]
    for label,path,m in zip(['baseline',*[p.name for p in directories[1:]]],directories,maps):
        for i in common:
            r=m[i]
            row=dict(solver=label,env_id=i,height_m=float(r['height_m']),arm=r['arm'],vx_m_s=float(r['vx_m_s']))
            row.update({k:float(r[k]) for k in metrics})
            row['joint_feedback_bias_abs_mean_rad_s']=float(np.mean([
                abs(float(r[name+'_dq_mean_rad_s'])-float(r[name+'_net_q_rate_rad_s'])) for name in joint_names]))
            for name in joint_names:
                row[name+'_feedback_bias_rad_s']=float(r[name+'_dq_mean_rad_s'])-float(r[name+'_net_q_rate_rad_s'])
            rows.append(row)
    save_csv(out/'paired_case_metrics.csv',rows)
    grouped=[]
    labels=list(dict.fromkeys(r['solver'] for r in rows))
    for label in labels:
        for height in sorted(set(r['height_m'] for r in rows)):
            for speed in sorted(set(r['vx_m_s'] for r in rows)):
                selected=[r for r in rows if r['solver']==label and r['height_m']==height and r['vx_m_s']==speed]
                if selected:
                    row=dict(solver=label,height_m=height,vx_m_s=speed,paired_trials=len(selected))
                    row.update({key:float(np.mean([r[key] for r in selected]))
                                for key in [*metrics,'joint_feedback_bias_abs_mean_rad_s']})
                    grouped.append(row)
    save_csv(out/'paired_group_metrics.csv',grouped)
    summary=dict(parameter_matching_passed=True,config_differences=config_differences,
        common_surviving_cases=len(common),total_cases=len(maps[0]),
        runs={label:dict(directory=str(path),solver=meta['solver'],
            failures=sum(int(r['failed']) for r in rs),completed=meta['complete'])
            for label,path,meta,rs in zip(labels,directories,metadata,records)},
        groups=grouped)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharex=True,constrained_layout=True)
    for col,height in enumerate(sorted(set(r['height_m'] for r in rows))):
        for label in labels:
            group=[r for r in grouped if r['solver']==label and r['height_m']==height]
            x=[r['vx_m_s'] for r in group]
            for ax,key in zip(axes[:,col],['hip_common_rms_deg','vx_mae_m_s','joint_feedback_bias_abs_mean_rad_s']):
                ax.plot(x,[r[key] for r in group],'-o',label=label.replace('2026-10-20_model12500_',''))
                ax.grid(alpha=.2)
        axes[0,col].set_title(f'height {height:.3f} m')
        axes[-1,col].set_xlabel('fixed vx (m/s)')
        for ax in axes[:,col]:ax.legend(fontsize=7)
    axes[0,0].set_ylabel('common hip residual RMS (deg)')
    axes[1,0].set_ylabel('vx MAE (m/s)')
    axes[2,0].set_ylabel('feedback mean-rate bias (rad/s)')
    fig.savefig(out/'solver_comparison.png',dpi=150);plt.close(fig)
    example=max((i for i in common if float(maps[0][i]['vx_m_s'])<0),
                key=lambda i:float(maps[0][i]['hip_common_rms_deg']))
    summary['example_env_id']=example
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(3,len(directories),figsize=(16,9),sharex=True,sharey='row',constrained_layout=True)
    trace=[]
    for col,(path,label) in enumerate(zip(directories,labels)):
        with np.load(path/'tracking.npz') as archive:
            time=archive['time_s'];mask=time>base['measure_start_s']+1e-6
            q=archive['q'][mask,example,:2].mean(1)
            target=archive['target'][mask,example,:2].mean(1)
            body=archive['body'][mask,example]
        axes[0,col].plot(time[mask],body[:,0])
        axes[0,col].axhline(float(maps[0][example]['vx_m_s']),ls='--',c='black')
        axes[1,col].plot(time[mask],np.degrees(q))
        axes[2,col].plot(time[mask],np.degrees(target))
        axes[0,col].set_title(label.replace('2026-10-20_model12500_',''))
        for ax in axes[:,col]:ax.grid(alpha=.2)
        axes[-1,col].set_xlabel('time (s)')
        for t,v,angle,goal in zip(time[mask],body[:,0],q,target):
            trace.append(dict(solver=label,env_id=example,time_s=float(t),vx=float(v),
                              common_hip_deg=float(np.degrees(angle)),target_deg=float(np.degrees(goal))))
    for ax,label in zip(axes[:,0],['vx (m/s)','common hip (deg)','issued target (deg)']):
        ax.set_ylabel(label)
    fig.suptitle(f'Exactly matched plant and initial state: environment {example}')
    fig.savefig(out/'paired_example.png',dpi=150);plt.close(fig)
    save_csv(out/'paired_example.csv',trace)
    print('Parameter/config matching passed. Paired surviving cases:',len(common))
    for r in grouped:
        if r['height_m']==.355 and r['vx_m_s']==-.8:
            print('Mid-height -0.8:',r)


if __name__=='__main__':
    main()
