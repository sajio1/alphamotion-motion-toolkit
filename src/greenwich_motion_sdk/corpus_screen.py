"""Reproducible release sampling with streamed exact-source sole/slip checks.

Memory-only execution never extracts BVHs, renders, runs inference or writes data.
The screening math is the existing locomotion_audit v2 implementation.
"""
import argparse
from contextlib import ExitStack
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time
import zipfile
from ._bvh import load_bvh_text
from .locomotion_audit import audit


def sample_release(index, archive_root, count=1000, seed=20260912, robot='h2'):
    unique={}
    for line in Path(index).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row['robot_id']==robot:unique.setdefault(row['clip_id'],row)
    if not 1<=count<=len(unique):raise ValueError('Sample count exceeds unique release captures')
    chosen=random.Random(seed).sample(sorted(unique),count)
    rows=[]
    for stem in chosen:
        row=dict(unique[stem]);row['archive']=str(Path(archive_root)/Path(row['archive']).name)
        row['source_archive_member']=row['source']['archive_member']
        rows.append(row)
    return rows,len(unique)


def source_records(rows, source_archive, cache_directory=None,prepare_compact=False):
    """Yield selected captures from one sequential gzip scan, without extraction."""
    wanted={row['source_archive_member']:row for row in rows}
    if len(wanted)!=len(rows):raise ValueError('Ambiguous source capture mapping')
    if cache_directory is not None:
        cache_directory=Path(cache_directory);cache_directory.mkdir(parents=True,exist_ok=True)
        for member,row in list(wanted.items()):
            cached=cache_directory/(row['clip_id']+'.bvh')
            if cached.is_file():
                with zipfile.ZipFile(row['archive']) as package:payload=package.read(row['member'])
                if hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Artifact checksum mismatch')
                row['source_bvh']=str(cached.resolve());row['_source_bvh']=load_bvh_text(cached.read_text())
                wanted.pop(member);yield row;row.pop('_source_bvh',None)
    if not wanted:return
    total=Path(source_archive).stat().st_size;started=time.monotonic();last=started
    with ExitStack() as stack:
        packages={}
        raw=stack.enter_context(Path(source_archive).open('rb'))
        try:
            from isal import igzip
            compressed=stack.enter_context(igzip.IGzipFile(fileobj=raw,mode='rb'))
            source=stack.enter_context(tarfile.open(fileobj=compressed,mode='r|',bufsize=1024*1024))
            print('Source decompression backend: ISA-L',flush=True)
        except ImportError:
            source=stack.enter_context(tarfile.open(fileobj=raw,mode='r|gz',bufsize=1024*1024))
            print('Source decompression backend: stdlib',flush=True)
        for member in source:
            if member.name in wanted and member.isfile():
                row=wanted.pop(member.name)
                if row['archive'] not in packages:
                    packages[row['archive']]=stack.enter_context(zipfile.ZipFile(row['archive']))
                payload=packages[row['archive']].read(row['member'])
                if hashlib.sha256(payload).hexdigest()!=row['sha256']:
                    raise ValueError(f'Artifact checksum mismatch: {row["clip_id"]}')
                with source.extractfile(member) as capture:
                    text=capture.read().decode('utf-8')
                if cache_directory is not None:
                    cached=cache_directory/(row['clip_id']+'.bvh')
                    partial=cached.with_suffix('.bvh.partial');partial.write_text(text,encoding='utf-8')
                    partial.replace(cached);row['source_bvh']=str(cached.resolve())
                target_fps=None
                if prepare_compact:
                    from io import BytesIO
                    from .compact import load_compact
                    row['_compact_motion']=load_compact(BytesIO(payload),robot=row['robot_id'])
                    target_fps=row['_compact_motion']['fps']
                row['_source_bvh']=load_bvh_text(text,target_fps=target_fps)
                del payload,text
                yield row
                row.pop('_source_bvh',None)
                row.pop('_compact_motion',None)
                del row
            if time.monotonic()-last>=15:
                print(json.dumps({'archive_scan_fraction':raw.tell()/total,'remaining_sources':len(wanted),
                    'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
                last=time.monotonic()
            if not wanted:break
    if wanted:raise FileNotFoundError(f'{len(wanted)} exact source captures missing from archive')


def screen(index, archive_root, source_archive, robots, descriptor, output=None,
           count=1000, seed=20260912, robot_id='h2', slip_cm_s=15.,
           penetration_cm=.5, bad_duration_s=.1, threads=4, support_config=None,metric_profile='legacy',source_cache=None):
    import torch
    torch.set_num_threads(threads)
    roster=json.loads(Path(robots).read_text(encoding='utf-8-sig'))
    matches=[r for r in roster if r['name']==robot_id]
    if len(matches)!=1:raise ValueError('Expected one requested robot in roster')
    rows,population=sample_release(index,archive_root,count,seed,robot_id)
    selection_sha=hashlib.sha256('\n'.join(sorted(r['clip_id'] for r in rows)).encode()).hexdigest()
    print(json.dumps({'sample_count':count,'unique_release_population':population,'seed':seed,
        'selection_sha256':selection_sha,'memory_only':output is None}),flush=True)
    started=time.monotonic()
    cache=source_cache if source_cache is not None else (Path(output)/'source' if output is not None else None)
    report=audit({'clips':source_records(rows,source_archive,cache)},matches[0],descriptor,output,
                 slip_cm_s,penetration_cm,bad_duration_s,progress_every=10,support_config=support_config,kimodo_metrics=metric_profile=='kimodo')
    results=report['results']
    if len(results)!=count:raise ValueError('Incomplete sample evaluation')
    categories={r['clip_id']:r['source']['category'] for r in rows}
    breakdown={}
    for result in results:
        bucket=breakdown.setdefault(categories[result['clip_id']],Counter())
        bucket['total']+=1;bucket[result['slip_status']]+=1
    summary={'schema':'greenwich.corpus-sole-screen.summary.v1','count':count,
        'seed':seed,'selection_sha256':selection_sha,'unique_release_population':population,
        'sampling':'uniform random unique source clip IDs, without replacement',
        'thresholds':report['thresholds'],'slip_counts':dict(Counter(r['slip_status'] for r in results)),
        'penetration_counts':dict(Counter(r['penetration_status'] for r in results)),
        'slip_fail_percent':100*sum(r['slip_status']=='fail' for r in results)/count,
        'slip_fail_clip_ids':[r['clip_id'] for r in results if r['slip_status']=='fail'],
        'categories':{name:dict(value) for name,value in breakdown.items()},
        'native_joint_fk_max_error_cm':max(r['native_joint_fk_max_error_cm'] for r in results),
        'elapsed_s':time.monotonic()-started,'output_written':output is not None,
        'interpretation':'Current v2 numeric slip screen; advisory, not a human acceptance or dynamics-success rate.'}
    if output is not None:
        (Path(output)/'sample.json').write_text(json.dumps({'count':count,'seed':seed,'clips':rows},indent=2),encoding='utf-8')
        (Path(output)/'sample_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if support_config is not None:
        import numpy as np
        feet=[f for r in results for f in r['feet'].values()]
        episodes=[e for f in feet for e in f['fixed_patch_support_episodes']]
        drift=[e['patch_max_median_displacement_cm'] for e in episodes]
        summary.update({'fixed_support_counts':dict(Counter(r['fixed_support_status'] for r in results)),
            'fixed_support_flag_clip_ids':[r['clip_id'] for r in results if r['fixed_support_status']=='drift_flag'],
            'fixed_support_config':report['fixed_support_config'],
            'fixed_support_episodes':len(episodes),
            'fixed_support_evaluated_foot_seconds':sum(f['fixed_patch_evaluated_frames']/r['fps'] for r in results for f in r['feet'].values()),
            'reference_fixed_support_foot_seconds':sum(f['fixed_source_support_frames']/r['fps'] for r in results for f in r['feet'].values()),
            'episode_drift_cm':{'median':float(np.median(drift)),'p95':float(np.percentile(drift,95)),'max':max(drift)} if drift else None,
            'interpretation':report['fixed_support_interpretation']+' Legacy slip_counts are a comparator, not the fixed-support result.'})
        if output is not None:
            (Path(output)/'sample_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Print compact completion statistics; failed IDs remain in the returned
    # report and optional summary file, avoiding megabytes of console output.
    if metric_profile=='kimodo':
        import numpy as np
        aggregate={}
        for actor in ('robot','source'):
            values=[r['kimodo'][actor] for r in results]
            contact=sum(v['contact_toe_frame_pairs'] for v in values)
            skating=sum(v['skating_toe_frame_pairs'] for v in values)
            aggregate[actor]={'per_motion_mean_foot_skate_ratio':float(np.mean([v['foot_skate_ratio'] for v in values])),
                'per_motion_mean_foot_skate_from_height_m_s':float(np.mean([v['foot_skate_from_height'] for v in values])),
                'contact_weighted_foot_skate_ratio':skating/(contact+1e-6),
                'contact_toe_frame_pairs':contact,'skating_toe_frame_pairs':skating,
                'clips_with_skating_frames':sum(v['skating_toe_frame_pairs']>0 for v in values),
                'clips_without_near_ground_toe_pairs':sum(v['contact_toe_frame_pairs']==0 for v in values)}
        summary={k:summary[k] for k in ('count','seed','selection_sha256','unique_release_population','sampling','elapsed_s')}
        summary.update({'schema':'greenwich.kimodo-toe-screen.summary.v1','protocol':report['kimodo_protocol'],
            'aggregate':aggregate,'results':[{ 'clip_id':r['clip_id'],**r['kimodo']} for r in results],
            'interpretation':'NVIDIA toe-height metrics with robot landmark adapter. A skating frame is not a clip rejection. Source may have intentional sliding; source comparison is retained.'})
        if output is not None:(Path(output)/'sample_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('slip_fail_clip_ids','results')}),flush=True)
    return report,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','archive-root','source-archive','robots','descriptor'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--count',type=int,default=1000);p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--robot',default='h2');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--slip-cm-s',type=float,default=15.)
    p.add_argument('--penetration-cm',type=float,default=.5)
    p.add_argument('--bad-duration-s',type=float,default=.1)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--support-config',type=Path)
    p.add_argument('--metric-profile',choices=('legacy','kimodo'),default='legacy')
    p.add_argument('--source-cache',type=Path)
    a=p.parse_args()
    if a.threads<1 or any(v<=0 for v in (a.slip_cm_s,a.penetration_cm,a.bad_duration_s)):
        p.error('Thread count and thresholds must be positive')
    if a.prepare_only:
        if a.output is None:p.error('prepare-only requires output')
        rows,population=sample_release(a.index,a.archive_root,a.count,a.seed,a.robot)
        a.output.mkdir(parents=True,exist_ok=True)
        for number,row in enumerate(source_records(rows,a.source_archive,a.output/'source'),1):
            print(f'Prepared exact source {number}/{a.count}: {row["clip_id"]}',flush=True)
        (a.output/'sample.json').write_text(json.dumps({'schema':'greenwich.review.sample.v1',
            'count':a.count,'seed':a.seed,'population':population,'sampling':'uniform random unique released captures',
            'clips':rows},indent=2),encoding='utf-8')
        return
    from .fixed_support import SupportConfig
    config=SupportConfig.load(a.support_config) if a.support_config else None
    screen(a.index,a.archive_root,a.source_archive,a.robots,a.descriptor,a.output,a.count,a.seed,
           a.robot,a.slip_cm_s,a.penetration_cm,a.bad_duration_s,a.threads,config,a.metric_profile,a.source_cache)


if __name__=='__main__':main()
