"""Deterministic diverse review subsets, referencing original archives in place."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import zipfile
from .compact import load_compact

def action_family(stem):
    """Filename heuristic for review diversity; not authoritative task metadata."""
    stem=re.sub(r'__A\d+.*$','',stem.lower(),flags=re.I)
    stem=re.sub(r'_(?:r|l|\d+)(?=_|$)','',stem)
    return stem.strip('_')

def category(stem):
    text=stem.lower()
    rules=[('jump',r'jump|hop|leap'),('run',r'jog|run|sprint'),
           ('low_body',r'crawl|prone|crouch|squat|kneel|sit|heels|toe'),
           ('object_interaction',r'grab|pick|carry|throw|item|box|lift|beer|bottle|light|sip|dog|crutch'),
           ('combat',r'punch|kick|box|combat|fight|sword|attack|defend'),
           ('dance',r'dance|dancing|latino|hiphop|vouge|vogue'),
           ('walk',r'walk|step|turn'),('gesture',r'wave|point|laugh|victory|idle|smoke|hand')]
    return next((name for name,pattern in rules if re.search(pattern,text)),'other')

def build_sample(index, archive_root, output, *, count=100,seed=20260912,max_per_action=1):
    if count<1 or max_per_action<1:raise ValueError('Count and per-action limit must be positive')
    rows=[json.loads(line) for line in Path(index).read_text(encoding='utf-8').splitlines() if line.strip()]
    rng=random.Random(seed);rng.shuffle(rows)
    buckets=defaultdict(list)
    for row in rows:
        seconds=row['duration_s'];duration='short' if seconds<5 else ('medium' if seconds<10 else 'long')
        buckets[(category(row['clip_id']),duration)].append(row)
    keys=sorted(buckets);chosen=[];families=Counter()
    while len(chosen)<count:
        advanced=False
        for key in keys:
            while buckets[key]:
                row=buckets[key].pop();family=action_family(row['clip_id'])
                if families[family]>=max_per_action:continue
                chosen.append({**row,'review_category':key[0],'action_family':family})
                families[family]+=1;advanced=True;break
            if len(chosen)==count:break
        if not advanced:raise ValueError(f'Only {len(chosen)} clips satisfy action diversity; relax --max-per-action')
    clips=[]
    for row in chosen:
        archive=Path(archive_root)/Path(row['archive']).name
        with zipfile.ZipFile(archive) as z:payload=z.read(row['member'])
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Artifact checksum mismatch')
        motion=load_compact(archive,member=row['member'],robot=row['robot_id'])
        if len(motion['q'])!=row['frames'] or motion['fps']!=row['fps']:raise ValueError('Index/frame mismatch')
        clips.append({**row,'archive':str(archive.resolve()),
                      'label':f"{row['clip_id']} · {row['duration_s']:.1f}s",
                      'integrity_verified':True})
    result={'schema':'greenwich.review.sample.v1','count':len(clips),'seed':seed,
            'sampling':'round-robin filename-category/duration buckets; filename family cap',
            'max_per_action':max_per_action,'action_families':len(families),
            'total_duration_s':sum(x['duration_s'] for x in clips),
            'categories':dict(Counter(x['review_category'] for x in clips)),
            'archives':dict(Counter(Path(x['archive']).name for x in clips)),
            'motion_quality':'not_evaluated','clips':clips}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


def build_explicit_sample(index, archive_root, output, clips):
    """Validate exact requested release captures without regenerating motion."""
    if not clips or len(set(clips)) != len(clips):
        raise ValueError('Explicit clip IDs must be nonempty and unique')
    rows=[json.loads(line) for line in Path(index).read_text(encoding='utf-8').splitlines() if line.strip()]
    selected=[]
    for clip in clips:
        matches=[r for r in rows if r['clip_id']==clip]
        if len(matches)!=1: raise ValueError(f'Expected one release artifact for {clip}, found {len(matches)}')
        row=matches[0]
        archive=Path(archive_root)/Path(row['archive']).name
        with zipfile.ZipFile(archive) as z: payload=z.read(row['member'])
        if hashlib.sha256(payload).hexdigest()!=row['sha256']: raise ValueError(f'Checksum mismatch: {clip}')
        motion=load_compact(archive,member=row['member'],robot=row['robot_id'])
        if len(motion['q'])!=row['frames'] or motion['fps']!=row['fps']: raise ValueError('Timeline mismatch')
        selected.append({**row,'archive':str(archive.resolve()),'source_clip_id':clip,
            'variant':'cloud100','iterations':100,'review_category':category(clip),
            'label':row.get('source',{}).get('semantic_label',action_family(clip)),
            'integrity_verified':True})
    result={'schema':'greenwich.review.sample.v1','count':len(selected),
        'sampling':'explicit semantic selection for demonstration; not a random benchmark',
        'motion_quality':'not_evaluated','clips':selected}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260912);p.add_argument('--max-per-action',type=int,default=1)
    p.add_argument('--clips',nargs='+',help='Exact release clip IDs; bypass random sampling')
    a=p.parse_args()
    result=build_explicit_sample(a.index,a.archive_root,a.output,a.clips) if a.clips else build_sample(a.index,a.archive_root,a.output,count=a.count,seed=a.seed,max_per_action=a.max_per_action)
    print(json.dumps({k:v for k,v in result.items() if k!='clips'},ensure_ascii=False))
if __name__=='__main__':main()
