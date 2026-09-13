"""Resumable full-release fixed-support screening with streamed exact sources.

Only drift_flag is flagged. Unconfirmed support is accepted by review policy,
while measurement coverage remains explicit. No new generation or video.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
import time
import numpy as np
from .corpus_screen import sample_release,source_records
from .fixed_support import SupportConfig
from .locomotion_audit import audit


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def load_ledger(path):
    """Recover only a torn final append; interior corruption is an error."""
    rows={}
    if not path.exists():return rows
    with path.open('rb+') as stream:
        while True:
            offset=stream.tell();line=stream.readline()
            if not line:break
            try:row=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if stream.read():raise ValueError('Interior ledger corruption')
                stream.truncate(offset);break
            if row['clip_id'] in rows:raise ValueError('Duplicate ledger result')
            rows[row['clip_id']]=row
    return rows


def run(index,archive_root,source_archive,robots,descriptor,output,support_config,
        slip_cm_s=15.,duration_s=.1,penetration_cm=.5,robot_id='h2',batch_size=64,threads=4,limit=None,
        sample_count=None,sample_seed=20260912,reuse_directory=None):
    import torch
    torch.set_num_threads(threads)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=SupportConfig.load(support_config)
    if any(not np.isfinite(v) or v<=0 for v in (slip_cm_s,duration_s,penetration_cm)) or batch_size<1 or threads<1:
        raise ValueError('Thresholds, threads and batch size must be positive')
    roster=json.loads(Path(robots).read_text(encoding='utf-8-sig'))
    matches=[r for r in roster if r['name']==robot_id]
    if len(matches)!=1:raise ValueError('Expected exactly one requested robot')
    # Count without assuming a hardcoded release size; every unique source ID.
    _,population=sample_release(index,archive_root,1,robot=robot_id)
    rows,_=sample_release(index,archive_root,sample_count or population,seed=sample_seed,robot=robot_id)
    rows.sort(key=lambda r:r['source_archive_member'])
    if limit is not None:rows=rows[:limit]
    signature={'schema':'greenwich.corpus-fixed-support.run.v1','robot':robot_id,
        'index_sha256':hashlib.sha256(Path(index).read_bytes()).hexdigest(),
        'descriptor_sha256':hashlib.sha256(Path(descriptor).read_bytes()).hexdigest(),
        'robot_xml_sha256':hashlib.sha256(Path(matches[0]['xml']).read_bytes()).hexdigest(),
        'population':population,'count':len(rows),'support_config':asdict(config),
        'slip_cm_s':slip_cm_s,'duration_s':duration_s,'penetration_cm':penetration_cm,
        'accept_unconfirmed_support':True,'criterion':'fixed_support_status only; penetration is a separate diagnostic',
        'selection_sha256':hashlib.sha256('\n'.join(sorted(r['clip_id'] for r in rows)).encode()).hexdigest(),
        'source_archive':str(Path(source_archive).resolve())}
    if sample_count is not None:
        signature.update({'sampling':'simple random unique clip IDs without replacement',
            'sample_seed':sample_seed,'sample_count':sample_count,'alpha':.05})
    run_path=output/'run.json'
    if run_path.exists() and json.loads(run_path.read_text())!=signature:
        raise ValueError('Output belongs to a different configuration; choose another output')
    atomic_json(run_path,signature)
    ledger=output/'results.jsonl';done=load_ledger(ledger)
    if not set(done)<=set(r['clip_id'] for r in rows):raise ValueError('Ledger contains captures outside selection')
    lookup={r['clip_id']:r for r in rows};cache=output/'speed-cache';cache.mkdir(exist_ok=True)
    if reuse_directory is not None:
        previous=Path(reuse_directory);old_signature=json.loads((previous/'run.json').read_text())
        keys=('robot','index_sha256','descriptor_sha256','robot_xml_sha256','support_config','slip_cm_s','duration_s','penetration_cm','accept_unconfirmed_support')
        if any(signature[key]!=old_signature[key] for key in keys):raise ValueError('Reuse configuration mismatch')
        prior=load_ledger(previous/'results.jsonl');reuse=[r for key,r in prior.items() if key in lookup and key not in done]
        with ledger.open('a',encoding='utf-8') as stream:
            for row in reuse:
                source_cache=previous/'speed-cache'/(row['clip_id']+'.npz');target_cache=cache/source_cache.name
                if not source_cache.exists():raise ValueError('Missing reusable speed cache')
                if not target_cache.exists():
                    try:os.link(source_cache,target_cache)
                    except OSError:shutil.copy2(source_cache,target_cache)
                stream.write(json.dumps(row,separators=(',',':'))+'\n');done[row['clip_id']]=row
            stream.flush();os.fsync(stream.fileno())
        print(json.dumps({'randomly_selected_cached_results_reused':len(reuse)}),flush=True)
    atomic_json(output/'selection.json',{'sampling':signature.get('sampling','full population'),'clips':rows})
    pending=[r for r in rows if r['clip_id'] not in done]
    initial=len(done);started=time.monotonic()
    counts=Counter(r['review_status'] for r in done.values());measurements=Counter(r['fixed_support_status'] for r in done.values())
    pending_arrays={}
    def progress(state='running'):
        elapsed=time.monotonic()-started;new=len(done)-initial
        value={'state':state,'completed':len(done),'total':len(rows),'remaining':len(rows)-len(done),
            'counts':dict(counts),'measurement_counts':dict(measurements),
            'elapsed_this_run_s':elapsed,'eta_s':elapsed/new*(len(rows)-len(done)) if new else None,
            'last_clip_id':next(reversed(done)) if done else None,'thresholds':{'slip_cm_s':slip_cm_s,'duration_s':duration_s},
            'accepted_unconfirmed_support':measurements['no_confirmed_fixed_support']}
        atomic_json(output/'progress.json',value)
        return value
    def series_sink(clip_id,role,fps,speed,valid):
        arrays=pending_arrays.setdefault(clip_id,{'fps':np.asarray(fps)})
        arrays[role+'__speed_cm_s']=speed;arrays[role+'__evaluated']=valid
        if len(arrays)==5:
            target=cache/(clip_id+'.npz');partial=target.with_suffix('.npz.partial')
            with partial.open('wb') as stream:np.savez_compressed(stream,**arrays)
            partial.replace(target);pending_arrays.pop(clip_id)
    with ledger.open('a',encoding='utf-8',buffering=1) as sink:
        def save(result):
            if not (cache/(result['clip_id']+'.npz')).exists():raise ValueError('Missing completed speed cache')
            status=result['fixed_support_status']
            row={**result,'review_status':'flagged' if status=='drift_flag' else 'good',
                'accepted_without_confirmed_support':status=='no_confirmed_fixed_support',
                'category':lookup[result['clip_id']]['source']['category']}
            sink.write(json.dumps(row,separators=(',',':'))+'\n');sink.flush()
            done[row['clip_id']]=row;counts[row['review_status']]+=1;measurements[status]+=1
            if (len(done)-initial)%25==0:
                os.fsync(sink.fileno());print(json.dumps(progress()),flush=True)
        print(json.dumps(progress()),flush=True)
        records=iter(source_records(pending,source_archive,prepare_compact=True))
        try:
            while True:
                # Copy rows so streamed generator cleanup cannot discard BVH
                # arrays from earlier records in the current bounded batch.
                batch=list(itertools.islice((dict(r) for r in records),batch_size))
                if not batch:break
                audit({'clips':batch},matches[0],descriptor,None,slip_cm_s,penetration_cm,duration_s,
                    support_config=config,progress_every=batch_size+1,result_sink=save,fixed_series_sink=series_sink)
                del batch
            os.fsync(sink.fileno())
        except BaseException as error:
            value=progress('interrupted' if isinstance(error,KeyboardInterrupt) else 'error')
            value['error']=str(error);atomic_json(output/'progress.json',value);raise
        finally:records.close()
    if len(done)!=len(rows):raise ValueError('Incomplete full corpus screen')
    value=progress('complete');breakdown={}
    for r in done.values():
        c=breakdown.setdefault(r['category'],Counter());c['total']+=1;c[r['review_status']]+=1
    summary={**signature,**value,'categories':{k:dict(v) for k,v in breakdown.items()},
        'flagged_clip_ids':[r['clip_id'] for r in done.values() if r['review_status']=='flagged'],
        'good_fraction':counts['good']/len(rows),'numeric_measurement_labels_preserved':True}
    atomic_json(output/'summary.json',summary)
    if sample_count is not None:
        from .sampling_statistics import proportion_interval
        stats={'sampling':signature['sampling'],'seed':sample_seed,
            'selection_sha256':signature['selection_sha256'],
            'good':proportion_interval(counts['good'],len(rows),population),
            'flagged':proportion_interval(counts['flagged'],len(rows),population),
            'accepted_unconfirmed_support':proportion_interval(measurements['no_confirmed_fixed_support'],len(rows),population),
            'categories':{},
            'scope':'These estimates apply only to the finite release and the fixed-support 15 cm/s, 0.1 s review rule. Unconfirmed support counts good by user policy. Not a physical success rate or generalization guarantee.',
            'aql_note':'AQL acceptance sampling requires chosen AQL/RQL and producer/consumer risk; no lot acceptance decision is inferred from this estimation sample.',
            'references':['https://www.itl.nist.gov/div898/handbook/ppc/section3/ppc333.htm',
                          'https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc23.htm']}
        all_rows,_=sample_release(index,archive_root,population,robot=robot_id)
        category_population=Counter(r['source']['category'] for r in all_rows)
        for category,N in category_population.items():
            group=[r for r in done.values() if r['category']==category]
            stats['categories'][category]={'population':N,'sample_count':len(group),
                'good':proportion_interval(sum(r['review_status']=='good' for r in group),len(group),N) if group else None,
                'warning':'Category interval is unadjusted for multiple comparisons; small category samples are imprecise.'}
        atomic_json(output/'statistics.json',stats)
        from .sampling_statistics import report_markdown
        (output/'statistics.md').write_text(report_markdown(stats),encoding='utf-8')
    lines=['# Full fixed-support screen','',f'Completed {len(done)}/{len(rows)} captures.',
        f'Good: {counts["good"]}; flagged: {counts["flagged"]}; accepted without confirmed support: {measurements["no_confirmed_fixed_support"]}.',
        f'Criterion: speed > {slip_cm_s:g} cm/s for at least {duration_s:g} s on confirmed persistent sole patches.',
        'This is the requested review acceptance policy, not a dynamics certificate.','']
    if sample_count is not None:
        good=stats['good'];flagged=stats['flagged']
        lines+=['## Random-sample statistics','',
            f'SRS without replacement: {len(rows)} of {population}, seed {sample_seed}.',
            f'Good estimate: {good["estimated_proportion"]:.2%}; exact 95% CI {good["confidence_interval"][0]:.2%}–{good["confidence_interval"][1]:.2%}.',
            f'Flagged estimate: {flagged["estimated_proportion"]:.2%}; exact 95% CI {flagged["confidence_interval"][0]:.2%}–{flagged["confidence_interval"][1]:.2%}.',
            f'Projected good count: {good["estimated_population_count"]:.0f}, interval {good["population_count_interval"]}.',
            f'Worst-case planning precision: ±{good["worst_case_planning_margin"]:.2%} at 95% confidence (approximation).',
            stats['scope'],'',
            'Intervals cover finite-population clip counts; they do not incorporate misclassification by the screening rule.']
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(value),flush=True);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('index','archive-root','source-archive','robots','descriptor','output','support-config'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--slip-cm-s',type=float,default=15.);p.add_argument('--duration-s',type=float,default=.1)
    p.add_argument('--penetration-cm',type=float,default=.5);p.add_argument('--robot',default='h2')
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--limit',type=int)
    p.add_argument('--sample-count',type=int);p.add_argument('--sample-seed',type=int,default=20260912)
    p.add_argument('--reuse-directory',type=Path)
    a=p.parse_args()
    run(a.index,a.archive_root,a.source_archive,a.robots,a.descriptor,a.output,a.support_config,
        a.slip_cm_s,a.duration_s,a.penetration_cm,a.robot,a.batch_size,a.threads,a.limit,a.sample_count,a.sample_seed,a.reuse_directory)


if __name__=='__main__':main()
