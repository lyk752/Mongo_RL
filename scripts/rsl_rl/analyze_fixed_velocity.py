"""Analyze the fixed_velocity_test.py NPZ without starting Isaac Sim."""
from pathlib import Path
import argparse
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mongo_fixed_velocity_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend, welch


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    out = Path(args.directory).resolve()
    meta = json.loads((out/'metadata.json').read_text())
    with np.load(out/'tracking.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    assert meta.get('complete'), 'Protocol did not complete.'
    dt = meta['dt']
    time = data['time_s']
    measure = time > meta['measure_start_s'] + 1e-6
    names = meta['joint_names']
    hip = [names.index(s+'_hip_joint') for s in ['left','right']]
    wheel = [names.index(s+'_podarthrum_joint') for s in ['left','right']]
    parallel = [names.index(s+'_parallel_1_joint') for s in ['left','right']]
    rows = []
    for case in meta['cases']:
        i=case['env_id']
        failed=bool(data['failed'][:,i].any())
        mask=measure & ~data['failed'][:,i]
        row=dict(case,failed=int(failed),measurement_samples=int(mask.sum()))
        if mask.sum()<10:
            rows.append(row);continue
        q=data['q'][mask,i]
        dq=data['dq'][mask,i]
        body=data['body'][mask,i]
        common=q[:,hip].mean(1)
        residual=detrend(common)
        freq,power=welch(residual,fs=1/dt,nperseg=min(256,len(residual)))
        # Ignore the zero bin and report the short-record estimate, not a
        # precision identification of a mechanical natural frequency.
        peak=float(freq[1:][np.argmax(power[1:])])
        phi=data['virtual_leg_angle'][mask,i].mean(1)
        target=data['target'][mask,i][:,hip].mean(1)
        torque=data['torque'][mask,i][:,hip]
        clearance=data['clearance'][mask,i].min(1)
        half=len(residual)//2
        row.update(
            vx_mae_m_s=float(abs(body[:,0]-case['vx_m_s']).mean()),
            vx_mean_m_s=float(body[:,0].mean()),
            height_mae_mm=float(abs(body[:,3]-case['height_m']).mean()*1000),
            pitch_peak_deg=float(np.degrees(abs(body[:,5]).max())),
            hip_common_rms_deg=float(np.degrees(np.sqrt(np.mean(residual**2)))),
            hip_common_p95_p05_deg=float(np.degrees(np.percentile(common,95)-np.percentile(common,5))),
            hip_common_peak_hz=peak,
            hip_first_half_rms_deg=float(np.degrees(np.sqrt(np.mean(detrend(common[:half])**2)))),
            hip_last_half_rms_deg=float(np.degrees(np.sqrt(np.mean(detrend(common[half:])**2)))),
            target_common_rms_deg=float(np.degrees(np.sqrt(np.mean(detrend(target)**2)))),
            virtual_leg_common_rms_deg=float(np.degrees(np.sqrt(np.mean(detrend(phi)**2)))),
            hip_pair_corr=float(np.corrcoef(detrend(q[:,hip],axis=0).T)[0,1]),
            hip_torque_rms_nm=float(np.sqrt(np.mean(torque**2))),
            wheel_torque_peak_nm=float(abs(data['torque'][mask,i][:,wheel]).max()),
            wheel_speed_rms_rad_s=float(np.sqrt(np.mean(dq[:,wheel]**2))),
            min_link_clearance_mm=float(clearance.min()*1000),
            below70mm_pct=float((clearance<.07).mean()*100),
            rear_contact_over3n_pct=float((data['contact_force'][mask,i,:12]>3).any(1).mean()*100),
            rear_contact_peak_n=float(data['contact_force'][mask,i,:12].max()),
            actual_command_max_error_m_s=float(abs(data['command'][mask,i,0]-case['vx_m_s']).max()),
            height_command_range_mm=float(np.ptp(data['height_command'][mask,i])*1000),
            arm_command_range_deg=float(np.degrees(np.ptp(data['arm_command'][mask,i]))),
        )
        for j,name in enumerate(names[:4]):
            # The mean-feedback / net-angle-rate comparison is insensitive to
            # high-frequency aliasing; it is a diagnostic, not causal proof.
            row[f'{name}_dq_mean_rad_s']=float(dq[:,j].mean())
            row[f'{name}_net_q_rate_rad_s']=float((q[-1,j]-q[0,j])/((len(q)-1)*dt))
        rows.append(row)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    rows=[{k:row.get(k,'') for k in keys} for row in rows]
    save_csv(out/'case_metrics.csv',rows)
    grouped=[]
    groupkeys=sorted(set((r['height_m'],r['arm'],r['vx_m_s']) for r in rows))
    fields=['vx_mae_m_s','height_mae_mm','hip_common_rms_deg','hip_common_p95_p05_deg',
        'hip_common_peak_hz','hip_first_half_rms_deg','hip_last_half_rms_deg','target_common_rms_deg',
        'virtual_leg_common_rms_deg','hip_pair_corr','hip_torque_rms_nm','wheel_torque_peak_nm',
        'wheel_speed_rms_rad_s','min_link_clearance_mm','below70mm_pct','rear_contact_over3n_pct']
    for height,arm,speed in groupkeys:
        group=[r for r in rows if (r['height_m'],r['arm'],r['vx_m_s'])==(height,arm,speed)]
        valid=[r for r in group if not r['failed'] and r['measurement_samples']>0]
        g=dict(height_m=height,arm=arm,vx_m_s=speed,trials=len(group),failures=sum(r['failed'] for r in group))
        for field in fields:
            values=[r[field] for r in valid]
            g[field+'_mean']=float(np.mean(values)) if values else None
        grouped.append(g)
    save_csv(out/'group_metrics.csv',grouped)
    good=[r for r in rows if not r['failed'] and r['measurement_samples']>0]
    worst=max((r for r in good if r['vx_m_s']<0),key=lambda r:r['hip_common_rms_deg'])
    summary=dict(total_trials=len(rows),failed_trials=sum(r['failed'] for r in rows),
        measurement_samples_per_trial=int(measure.sum()),measurement_duration_s=float(measure.sum()*dt),
        max_command_error_m_s=max(r['actual_command_max_error_m_s'] for r in good),
        max_height_command_range_mm=max(r['height_command_range_mm'] for r in good),
        max_arm_command_range_deg=max(r['arm_command_range_deg'] for r in good),
        worst_reverse_case=worst,
        minimum_measured_link_clearance_mm=min(r['min_link_clearance_mm'] for r in good),
        groups=grouped,
        note='RMS is the linearly detrended common-mode (mean of left/right hips) over 10 s. '
             'Each group has independent randomized environment samples, not matched forward/reverse plants.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,constrained_layout=True)
    for col,height in enumerate(sorted(set(r['height_m'] for r in grouped))):
        for arm in ['up','down']:
            g=[r for r in grouped if r['height_m']==height and r['arm']==arm]
            axes[0,col].plot([r['vx_m_s'] for r in g],[r['hip_common_rms_deg_mean'] for r in g],'-o',label=arm)
            axes[1,col].plot([r['vx_m_s'] for r in g],[r['vx_mae_m_s_mean'] for r in g],'-o',label=arm)
        axes[0,col].set_title(f'height {height:.3f} m')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.axvline(0,c='gray',ls='--');ax.legend()
        axes[1,col].set_xlabel('fixed vx command (m/s)')
    axes[0,0].set_ylabel('common hip residual RMS (deg)')
    axes[1,0].set_ylabel('vx MAE (m/s)')
    fig.savefig(out/'fixed_speed_summary.png',dpi=150);plt.close(fig)

    # Show a common randomized replicate at +/- the worst reverse speed.
    cases_to_plot=[r for r in good if r['height_m']==worst['height_m'] and r['arm']==worst['arm']
                   and r['replicate']==worst['replicate']
                   and r['vx_m_s'] in [worst['vx_m_s'],-worst['vx_m_s']]]
    fig,axes=plt.subplots(4,len(cases_to_plot),figsize=(14,11),sharex=True,squeeze=False,constrained_layout=True)
    for col,case in enumerate(cases_to_plot):
        i=case['env_id'];mask=measure;t=time[mask]
        axes[0,col].plot(t,data['body'][mask,i,0],label='actual vx')
        axes[0,col].axhline(case['vx_m_s'],ls='--',c='black',label='command')
        for side,j in zip(['left','right'],hip):
            axes[1,col].plot(t,np.degrees(data['q'][mask,i,j]),label=side+' hip')
            axes[2,col].plot(t,np.degrees(data['target'][mask,i,j]),label=side+' target')
            axes[3,col].plot(t,data['torque'][mask,i,j],label=side+' torque')
        axes[0,col].set_title(f"h={case['height_m']}, arm={case['arm']}, vx={case['vx_m_s']}, rep={case['replicate']}")
        for ax,label in zip(axes[:,col],['vx (m/s)','hip angle (deg)','issued target (deg)','hip torque (Nm)']):
            ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(loc='upper right')
        axes[-1,col].set_xlabel('time (s)')
    fig.savefig(out/'reverse_forward_timeseries.png',dpi=150);plt.close(fig)
    # Export the plotted trials as portable CSV as well as the full batched NPZ.
    trace=[]
    for case in cases_to_plot:
        i=case['env_id']
        for k,t in enumerate(time):
            trace.append(dict(env_id=i,time_s=float(t),vx_command=float(data['command'][k,i,0]),
                vx_actual=float(data['body'][k,i,0]),height=float(data['body'][k,i,3]),
                left_hip_q=float(data['q'][k,i,hip[0]]),right_hip_q=float(data['q'][k,i,hip[1]]),
                left_hip_target=float(data['target'][k,i,hip[0]]),right_hip_target=float(data['target'][k,i,hip[1]])))
    save_csv(out/'example_tracking.csv',trace)
    print('Trials / failures:',summary['total_trials'],summary['failed_trials'])
    print('Worst reverse case:',worst)
    print('Constant-command checks:',summary['max_command_error_m_s'],summary['max_height_command_range_mm'],summary['max_arm_command_range_deg'])


if __name__=='__main__':
    main()
