#!/usr/bin/env python3
"""Generate portable flat sim2sim evaluation plots and cross-engine comparison tables."""
import argparse
import csv
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mongo_mujoco_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_flat import save_csv


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def trace(path):
    rows=read(path)
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--isaac-reference',type=Path)
    args=p.parse_args(); out=args.directory
    rows=read(out/'case_metrics.csv')
    heights=sorted({float(r['height']) for r in rows})
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for col,h in enumerate(heights):
        for arm,label in [(0,'arm up'),(1,'arm down')]:
            for kind,ax,command,actual in [('linear',axes[0,col],'vx','vx_mean'),('turn',axes[1,col],'wz','wz_mean')]:
                group=sorted([r for r in rows if r['kind']==kind and float(r['height'])==h and float(r['arm'])==arm],key=lambda r:float(r[command]))
                ax.plot([float(r[command]) for r in group],[float(r[actual]) for r in group],'-o',label=label)
        for ax,limit in [(axes[0,col],1.5),(axes[1,col],4)]:
            ax.plot([-limit,limit],[-limit,limit],'k--',alpha=.4,label='ideal')
            ax.grid(alpha=.2);ax.legend();ax.set_title(f'height={h:.3f} m')
        axes[0,col].set_xlabel('command vx (m/s)');axes[0,col].set_ylabel('mean vx, t>=5s (m/s)')
        axes[1,col].set_xlabel('command wz (rad/s)');axes[1,col].set_ylabel('mean wz, t>=5s (rad/s)')
    fig.savefig(out/'tracking_sweep.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for col,h in enumerate(heights):
        for r in rows:
            if r['kind']=='long_stand' and float(r['height'])==h:
                d=trace(out/r['case']/'tracking.csv');mask=d['time_s']>=5
                x,y=d['x'][mask],d['y'][mask]
                axes[col].plot(d['time_s'][mask]-5,np.hypot(x-x[0],y-y[0]),label='arm down' if float(r['arm']) else 'arm up')
        axes[col].set_title(f'zero command, h={h:.3f} m');axes[col].set_xlabel('time since 5s (s)')
        axes[col].set_ylabel('distance from position at 5s (m)');axes[col].grid(alpha=.2);axes[col].legend()
    fig.savefig(out/'stationary_drift.png',dpi=170);plt.close(fig)
    d=trace(out/'dynamic/tracking.csv')
    fig,axes=plt.subplots(5,1,figsize=(15,12),sharex=True,constrained_layout=True)
    for ax,actual,command,label,scale in zip(axes[:4],['vx','wz','z','front_actual_rad'],
        ['command_vx','command_wz','command_height','front_target_rad'],
        ['vx (m/s)','wz (rad/s)','height (m)','front arm (deg)'],[1,1,1,180/np.pi]):
        ax.plot(d['time_s'],d[actual]*scale,label='actual',lw=.9)
        ax.plot(d['time_s'],d[command]*scale,'--',label='issued command',lw=.9)
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(loc='upper right')
    for name in ['roll_deg','pitch_deg']:axes[-1].plot(d['time_s'],d[name],label=name,lw=.9)
    axes[-1].set_xlabel('simulation time (s)');axes[-1].set_ylabel('body angles (deg)');axes[-1].legend();axes[-1].grid(alpha=.2)
    fig.savefig(out/'dynamic_tracking.png',dpi=170);plt.close(fig)
    # Recompute phase masks using rounded timestamps to avoid floating-point overlap.
    start=0.;phases=[]
    for segment in json.loads((out/'dynamic/schedule.json').read_text()):
        t=np.round(d['time_s'],6);end=start+segment['duration'];mask=(t>start)&(t<=end)
        if mask.any():
            phases.append(dict(name=segment['name'],start=start,end=end,samples=int(mask.sum()),
                vx_mae=float(abs(d['vx'][mask]-d['command_vx'][mask]).mean()),
                wz_mae=float(abs(d['wz'][mask]-d['command_wz'][mask]).mean()),
                height_mm_mae=float(abs(d['z'][mask]-d['command_height'][mask]).mean()*1000)))
        start=end
    assert sum(r['samples'] for r in phases)==len(d['time_s']), 'Dynamic phase samples must be disjoint and exhaustive'
    save_csv(out/'dynamic_phases.csv',phases)
    if args.isaac_reference:
        reference=args.isaac_reference
        meta=json.loads((reference/'metadata.json').read_text())
        assert meta['complete']
        data=np.load(reference/'tracking.npz')
        mask=data['time_s']>meta['measure_start_s']+1e-6
        comparisons=[]
        for row in rows:
            if row['kind']!='linear':continue
            candidates=[c for c in meta['cases'] if c['height_m']==float(row['height']) and c['vx_m_s']==float(row['vx']) and c['arm']==('down' if float(row['arm']) else 'up')]
            if not candidates:continue
            valid=[c for c in candidates if not data['failed'][:,c['env_id']].any()]
            vals=[]
            for c in valid:
                body=data['body'][mask,c['env_id']]
                vals.append([np.mean(abs(body[:,0]-c['vx_m_s'])),body[:,0].mean(),np.mean(abs(body[:,3]-c['height_m']))*1000])
            mean=np.mean(vals,axis=0) if vals else [None]*3
            comparisons.append(dict(height=float(row['height']),arm=float(row['arm']),vx=float(row['vx']),
                isaac_trials=len(candidates),isaac_failures=len(candidates)-len(valid),
                isaac_vx_mae=mean[0],isaac_vx_mean=mean[1],isaac_height_mm_mae=mean[2],
                mujoco_vx_mae=float(row['vx_mae']),mujoco_vx_mean=float(row['vx_mean']),mujoco_height_mm_mae=float(row['height_mm_mae'])))
        save_csv(out/'isaac_comparison.csv',comparisons)
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for ax,h in zip(axes,heights):
            for arm,color in [(0,'tab:blue'),(1,'tab:orange')]:
                g=sorted([r for r in comparisons if r['height']==h and r['arm']==arm],key=lambda r:r['vx'])
                label='down' if arm else 'up'
                ax.plot([r['vx'] for r in g],[r['mujoco_vx_mean'] for r in g],'-o',color=color,label='MuJoCo '+label)
                ax.plot([r['vx'] for r in g],[r['isaac_vx_mean'] for r in g],'--x',color=color,label='Isaac '+label)
            ax.plot([-.8,.8],[-.8,.8],'k:',alpha=.4);ax.set_title(f'h={h:.3f} m');ax.grid(alpha=.2);ax.legend()
            ax.set_xlabel('command vx (m/s)');ax.set_ylabel('steady mean vx (m/s)')
        fig.savefig(out/'isaac_comparison.png',dpi=170);plt.close(fig)
    print('Plots and tables:',out)


if __name__=='__main__':main()
