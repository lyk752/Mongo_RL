#!/usr/bin/env python3
"""Evaluate a frozen Mongo actor over flat command/height/arm sweeps and transitions."""
import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime
from itertools import product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from build_model import ROOT, NAMES
from sim2sim import default_policy_path, parser as sim_parser, run


class Schedule:
    def __init__(self, segments, dt=.02):
        self.segments, self.dt, self.step = segments, dt, 0
        self.quit_requested = False

    def configure(self, cfg, args):
        self.dt = cfg['sim']['dt'] * cfg['decimation']

    def key_callback(self, key):
        pass

    def sample(self):
        t = self.step * self.dt
        self.step += 1
        previous = self.segments[0]['target']
        for segment in self.segments:
            if t < segment['duration']:
                alpha = min(t / segment['duration'], 1.) if segment.get('ramp') else 1.
                target = np.array(previous) + alpha * (np.array(segment['target']) - previous)
                return np.array([target[0], 0., target[1]]), float(target[2]), float(target[3])
            t -= segment['duration']
            previous = segment['target']
        return np.array([previous[0], 0., previous[1]]), previous[2], previous[3]


def dynamic_segments():
    segments = [dict(name='settle', duration=5., target=[0., 0., .355, 0.])]
    for name, target, ramp in [
        ('forward', [.8, 0, .355, 0], 2), ('stop_forward', [0, 0, .355, 0], 2),
        ('reverse', [-.8, 0, .355, 0], 2), ('stop_reverse', [0, 0, .355, 0], 2),
        ('turn_left', [0, 2, .355, 0], 2), ('turn_right', [0, -2, .355, 0], 2),
        ('stop_turn', [0, 0, .355, 0], 2), ('high', [0, 0, .45, 0], 4),
        ('low', [0, 0, .29, 0], 6), ('arm_down', [0, 0, .29, 1], 2),
        ('arm_up', [0, 0, .29, 0], 2)]:
        segments += [dict(name=name+'_ramp', duration=float(ramp), target=target, ramp=True),
                     dict(name=name+'_hold', duration=8., target=target)]
    return segments


def metrics(path, case, settle=5.):
    with (path/'tracking.csv').open() as f:
        rows = list(csv.DictReader(f))
    data = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}
    summary = json.loads((path/'summary.json').read_text())
    result = dict(case, status=summary['status'], seconds=summary['simulated_seconds'],
                  peak_pitch_deg=summary['max_abs_pitch_deg'], loop_error_mm=summary['max_loop_error_m']*1000)
    if not rows:
        return result
    mask = data['time_s'] >= settle
    # Keep failures visible and avoid comparing a short surviving prefix to full trials.
    result['steady_samples'] = int(mask.sum())
    if mask.sum() < 50:
        return result
    for actual, target, label, scale in [('vx','command_vx','vx',1), ('wz','command_wz','wz',1),
                                         ('z','command_height','height_mm',1000),
                                         ('front_actual_rad','front_target_rad','arm_deg',180/np.pi)]:
        err = (data[actual][mask] - data[target][mask]) * scale
        result[label+'_mae'] = float(np.abs(err).mean())
        result[label+'_bias'] = float(err.mean())
        result[label+'_p95_abs'] = float(np.percentile(abs(err),95))
    result['vx_mean'] = float(data['vx'][mask].mean())
    result['wz_mean'] = float(data['wz'][mask].mean())
    result['roll_rms_deg'] = float(np.sqrt(np.mean(data['roll_deg'][mask]**2)))
    result['pitch_rms_deg'] = float(np.sqrt(np.mean(data['pitch_deg'][mask]**2)))
    result['peak_roll_deg'] = float(abs(data['roll_deg']).max())
    dx, dy = data['x'][mask]-data['x'][mask][0], data['y'][mask]-data['y'][mask][0]
    result['steady_max_drift_m'] = float(np.hypot(dx,dy).max())
    result['steady_net_displacement_m'] = float(np.hypot(dx[-1],dy[-1]))
    for name, limit in zip(NAMES, [35]*4+[15,10,10]):
        result['saturation_pct/'+name] = float((abs(data['torque/'+name][mask]) >= limit*.99).mean()*100)
    result['hip_oscillation_std_deg'] = float(np.rad2deg(np.std(
        (data['q/left_hip_joint'][mask]+data['q/right_hip_joint'][mask])/2)))
    return result


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,default=default_policy_path(),help='ONNX actor; defaults to mujoco/policy.json selection')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=args.output or ROOT/'mujoco/outputs'/('flat_evaluation_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S'))
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for height, arm in product([.29,.355,.45],[0.,1.]):
        commands=[(v,0.,'linear') for v in [-1.5,-.8,-.4,0.,.4,.8,1.5]]
        commands += [(0.,w,'turn') for w in [-4.,-2.,-1.,1.,2.,4.]]
        commands += [(v,w,'coupled') for v,w in product([-.8,.8],[-2.,2.])]
        for vx,wz,kind in commands:
            cases.append(dict(kind=kind,height=height,arm=arm,vx=vx,wz=wz,duration=20.))
        cases.append(dict(kind='long_stand',height=height,arm=arm,vx=0.,wz=0.,duration=60.))
    rows=[]
    for i,case in enumerate(cases):
        name=f"{i:03d}_{case['kind']}_h{case['height']}_a{int(case['arm'])}_v{case['vx']}_w{case['wz']}"
        dest=output/name
        options=sim_parser().parse_args(['--policy',str(args.policy),'--headless','--duration',str(case['duration']),
            '--height',str(case['height']),'--front-fraction',str(case['arm']), '--vx',str(case['vx']),
            '--wz',str(case['wz']),'--output',str(dest)])
        with (output/'runner.log').open('a') as log, redirect_stdout(log):
            run(options)
        row=metrics(dest,dict(case,case=name))
        rows.append(row)
        save_csv(output/'case_metrics.csv',rows)
        if (i+1)%18==0:
            print(f"{i+1}/{len(cases)} cases; failures={sum(r['status']!='duration_completed' for r in rows)}",flush=True)
    segments=dynamic_segments()
    dest=output/'dynamic'
    options=sim_parser().parse_args(['--policy',str(args.policy),'--headless', '--height','.355',
        '--duration',str(sum(s['duration'] for s in segments)),'--output',str(dest)])
    with (output/'runner.log').open('a') as log, redirect_stdout(log):
        run(options,Schedule(segments))
    (dest/'schedule.json').write_text(json.dumps(segments,indent=2))
    start=0.; phases=[]
    with (dest/'tracking.csv').open() as f:
        data=list(csv.DictReader(f))
    for segment in segments:
        selected=[r for r in data if start < round(float(r['time_s']), 6) <= start+segment['duration']]
        if selected:
            phases.append(dict(name=segment['name'],start=start,end=start+segment['duration'],samples=len(selected),
                vx_mae=float(np.mean([abs(float(r['vx'])-float(r['command_vx'])) for r in selected])),
                wz_mae=float(np.mean([abs(float(r['wz'])-float(r['command_wz'])) for r in selected])),
                height_mm_mae=float(np.mean([abs(float(r['z'])-float(r['command_height']))*1000 for r in selected]))))
        start+=segment['duration']
    save_csv(output/'dynamic_phases.csv',phases)
    summary=dict(total_cases=len(rows),completed=sum(r['status']=='duration_completed' for r in rows),
        failures=[r for r in rows if r['status']!='duration_completed'],
        dynamic=json.loads((dest/'summary.json').read_text()),
        note='Single deterministic initial assembly per case. Steady metrics start at 5 s. No reset, noise or plant randomization.')
    (output/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
    print('Output:',output,flush=True)


if __name__=='__main__':
    main()
