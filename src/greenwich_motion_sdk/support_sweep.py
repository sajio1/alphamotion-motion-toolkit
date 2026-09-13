"""Re-score cached fixed-support material-patch velocities without inference/FK.

Published speed thresholds inform sensitivity analysis, not a dynamics certificate
or an exact reproduction of the cited contact/landmark definitions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .physics_audit import intervals


def sweep(report_path, output, speeds=(7.5,10.,15.,20.,25.), durations=(.1,), provenance=None):
    started=time.monotonic();report_path=Path(report_path)
    payload=report_path.read_bytes();report=json.loads(payload)
    if 'fixed_support_config' not in report:raise ValueError('Requires a fixed-support audit and its cached CSVs')
    if any(not np.isfinite(v) or v<=0 for v in (*speeds,*durations)):
        raise ValueError('Thresholds and durations must be finite and positive')
    cache=[];rows=report['results'];ids=[r['clip_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate clip IDs')
    total_foot_frames=0;total_foot_seconds=0.;evaluated_seconds=0.;reference_seconds=0.
    for row in rows:
        fps=row['fps'];feet={};total_foot_frames+=len(row['feet'])*row['frames']
        total_foot_seconds+=len(row['feet'])*row['frames']/fps
        for side,foot in row['feet'].items():
            path=report_path.parent/f'{row["clip_id"]}__{side}__fixed.csv'
            data=np.genfromtxt(path,delimiter=',',names=True)
            velocity=np.atleast_1d(data['fixed_patch_speed_cm_s'])
            valid=np.atleast_1d(data['evaluated']).astype(bool)
            times=np.atleast_1d(data['time_s'])
            if len(velocity)!=row['frames'] or not np.allclose(times,np.arange(row['frames'])/fps):
                raise ValueError(f'Cached timeline mismatch: {path}')
            if not np.array_equal(valid,np.isfinite(velocity)) or valid.sum()!=foot['fixed_patch_evaluated_frames']:
                raise ValueError(f'Cached support mask mismatch: {path}')
            feet[side]=(velocity,valid)
            evaluated_seconds+=valid.sum()/fps;reference_seconds+=foot['fixed_source_support_frames']/fps
        cache.append((row,feet))
    result=[];evaluated_frames=sum(valid.sum() for _,feet in cache for _,valid in feet.values())
    for duration in sorted(set(durations)):
        for speed in sorted(set(speeds)):
            clips=[];exceed_frames=0;exceed_seconds=0.;sustained_seconds=0.
            for row,feet in cache:
                events={};instant=False;has_support=False
                for side,(velocity,valid) in feet.items():
                    mask=valid&(np.nan_to_num(velocity)>speed)
                    exceed_frames+=int(mask.sum());exceed_seconds+=mask.sum()/row['fps']
                    instant|=bool(mask.any());has_support|=bool(valid.any())
                    events[side]=intervals(mask,row['fps'],duration)
                    sustained_seconds+=sum(e['duration_s'] for e in events[side])
                status='drift_flag' if any(events.values()) else ('no_drift_flag' if has_support else 'no_confirmed_fixed_support')
                clips.append({'clip_id':row['clip_id'],'status':status,'any_exceedance':instant,'events':events})
            counts=Counter(c['status'] for c in clips)
            result.append({'speed_cm_s':speed,'minimum_duration_s':duration,
                'counts':{key:counts[key] for key in ('drift_flag','no_drift_flag','no_confirmed_fixed_support')},
                'any_exceedance_clip_count':sum(c['any_exceedance'] for c in clips),
                'exceedance_evaluated_frame_ratio':exceed_frames/evaluated_frames if evaluated_frames else None,
                'exceedance_evaluated_time_ratio':exceed_seconds/evaluated_seconds if evaluated_seconds else None,
                'sustained_exceedance_foot_seconds':sustained_seconds,'clips':clips})
    drift=[];drift_clips={}
    for row in rows:
        values=[e['patch_max_median_displacement_cm'] for f in row['feet'].values() for e in f['fixed_patch_support_episodes']]
        drift.extend(values);drift_clips[row['clip_id']]=max(values,default=None)
    summary={'schema':'greenwich.fixed-support.sensitivity.v1',
        'input_report_sha256':hashlib.sha256(payload).hexdigest(),
        'selection_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
        'clip_count':len(rows),'fixed_support_config':report['fixed_support_config'],
        'evaluated_foot_frames':int(evaluated_frames),'all_foot_frames':total_foot_frames,
        'evaluated_foot_seconds':evaluated_seconds,'reference_stationary_foot_seconds':reference_seconds,
        'all_foot_seconds':total_foot_seconds,'evaluated_foot_frame_ratio':evaluated_frames/total_foot_frames,
        'episode_drift_cm':{'median':float(np.median(drift)),'p95':float(np.percentile(drift,95)),'max':max(drift)} if drift else None,
        'cumulative_drift_descriptive_counts':{
            str(limit):{'episodes_above':sum(v>limit for v in drift),
                        'clips_above':sum(v is not None and v>limit for v in drift_clips.values())}
            for limit in (1.,2.,5.)},
        'provenance':json.loads(Path(provenance).read_text()) if provenance else None,
        'interpretation':'Speed sensitivity on identical cached source-stationary persistent material patches. Excluded frames are unknown. Published landmark-based metrics are not reproduced. Duration and drift cutoffs are local engineering settings, not standards.',
        'sweeps':result,'elapsed_s':time.monotonic()-started}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'support_sweep.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Fixed-support threshold comparison','',summary['interpretation'],'',
        '| Speed cm/s | Duration s | Flagged clips | No flag | Unknown | Exceeding evaluated frames |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in result:
        c=r['counts'];ratio=r['exceedance_evaluated_frame_ratio']
        lines.append(f'| {r["speed_cm_s"]:g} | {r["minimum_duration_s"]:g} | {c["drift_flag"]} | {c["no_drift_flag"]} | {c["no_confirmed_fixed_support"]} | {ratio:.2%} |')
    lines+=['',f'Evaluated {evaluated_seconds:.2f} foot-seconds; coverage {summary["evaluated_foot_frame_ratio"]:.2%} of all foot frames.','']
    if summary['provenance']:
        for source in summary['provenance']['sources']:
            lines.append(f'- [{source["title"]}]({source["url"]}): {source["detail"]}')
    (output/'support_sweep.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'clip_count':len(rows),'elapsed_s':summary['elapsed_s'],
        'coverage':summary['evaluated_foot_frame_ratio'],
        'sweeps':[{k:v for k,v in r.items() if k!='clips'} for r in result]}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--speeds',default='7.5,10,15,20,25');p.add_argument('--durations',default='0.1')
    p.add_argument('--provenance',type=Path)
    a=p.parse_args()
    sweep(a.report,a.output,tuple(map(float,a.speeds.split(','))),tuple(map(float,a.durations.split(','))),a.provenance)


if __name__=='__main__':main()
