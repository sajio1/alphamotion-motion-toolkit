"""Resolve exact SOMA capture paths and cache only selected source BVHs."""
import argparse,json,shutil,tarfile,time
from pathlib import Path
import pandas as pd

def select_cached(index, archive_root, clips, cached_roots, output):
    """Select exact published artifacts and existing source BVHs for previews.

    No model generation, archive scan or change to any trajectory is involved.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    wanted=set(clips);selected={}
    for line in Path(index).read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['clip_id'] in wanted:
            if row['clip_id'] in selected and row['sha256']!=selected[row['clip_id']]['sha256']:raise ValueError('Ambiguous artifact')
            selected[row['clip_id']]=row
    if set(selected)!=wanted:raise ValueError(f'Missing published clips: {wanted-set(selected)}')
    cached={}
    for root in cached_roots:
        for path in Path(root).rglob('*.bvh'):cached.setdefault(path.stem,[]).append(path)
    rows=[]
    for clip in clips:
        row=dict(selected[clip]);matches=cached.get(clip,[])
        if len(matches)!=1:raise ValueError(f'Expected one exact cached source BVH for {clip}: {len(matches)}')
        if Path(row['source']['archive_member']).stem!=clip:raise ValueError('Source metadata identity mismatch')
        row.update(archive=str((Path(archive_root)/row['archive']).resolve()),source_bvh=str(matches[0].resolve()),
            source_clip_id=clip,variant='cloud100',iterations=100)
        rows.append(row)
    target=output/'selected-previews.json'
    target.write_text(json.dumps({'clips':rows,'motion_modified':False},indent=2),encoding='utf-8')
    print(f'Selected {len(rows)} published previews: {target}',flush=True);return target

def prepare(manifest, metadata, archive, output, cached_roots=()):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    data=json.loads(Path(manifest).read_text(encoding='utf-8'))
    frame=pd.read_parquet(metadata)
    paths=frame['move_soma_uniform_path'].astype(str)
    rows=data['clips']; wanted={}
    cached={p.stem:p for root in cached_roots for p in Path(root).rglob('*.bvh')}
    for row in rows:
        matches=paths[paths.str.endswith('/'+row['clip_id']+'.bvh')].tolist()
        if len(matches)!=1: raise ValueError(f'Ambiguous source capture for {row["clip_id"]}: {len(matches)}')
        row['source_archive_member']=matches[0]
        target=output/(row['clip_id']+'.bvh'); row['source_bvh']=str(target.resolve())
        if not target.is_file() and row['clip_id'] in cached:
            shutil.copyfile(cached[row['clip_id']],target)
        if not target.is_file(): wanted[matches[0]]=target
    started=time.monotonic(); last=started; total=Path(archive).stat().st_size
    if wanted:
        with open(archive,'rb') as raw,tarfile.open(fileobj=raw,mode='r|gz') as source:
            for member in source:
                if member.name in wanted and member.isfile():
                    target=wanted.pop(member.name)
                    stream=source.extractfile(member)
                    with stream,target.with_suffix('.partial').open('wb') as dst: shutil.copyfileobj(stream,dst,1024*1024)
                    target.with_suffix('.partial').replace(target)
                    print(f'Cached {target.name}; remaining {len(wanted)}',flush=True)
                if time.monotonic()-last>=15:
                    progress={'archive_read_bytes':raw.tell(),'archive_bytes':total,'remaining':len(wanted),
                              'elapsed_s':time.monotonic()-started}
                    (output/'progress.json').write_text(json.dumps(progress),encoding='utf-8')
                    print(f'Archive scan {raw.tell()/total:.1%}, remaining {len(wanted)}',flush=True)
                    last=time.monotonic()
                if not wanted: break
        if wanted: raise FileNotFoundError(f'Missing source paths: {list(wanted)}')
    data['source_metadata']=str(Path(metadata).resolve())
    data['source_archive']=str(Path(archive).resolve())
    result=output/'source_manifest.json'
    result.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(f'Source manifest ready: {result}',flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','metadata','archive','output','index','archive-root'):p.add_argument('--'+key,type=Path)
    p.add_argument('--cached-roots',nargs='*',type=Path,default=[])
    p.add_argument('--select-cached',action='store_true');p.add_argument('--clips',nargs='+')
    a=p.parse_args()
    if a.select_cached:
        if not all((a.index,a.archive_root,a.clips,a.cached_roots,a.output)):p.error('Cached selection requires index, archive root, clips, cached roots and output')
        select_cached(a.index,a.archive_root,a.clips,a.cached_roots,a.output)
    else:
        if not all((a.manifest,a.metadata,a.archive,a.output)):p.error('Source extraction requires manifest, metadata, archive and output')
        prepare(a.manifest,a.metadata,a.archive,a.output,a.cached_roots)
if __name__=='__main__': main()
