"""Rank cached published previews by anatomical hand/forearm direction agreement.

This is a candidate-selection heuristic, not a calibrated hand-pose benchmark.
The robot distal hand ray is estimated from its hand visual mesh; manual source
comparison remains required. No motion, model output or contact is modified.
"""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from .compact import load_compact,native_visual_motion
from ._bvh import load_bvh

def angle(a,b):
    a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-9)
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-9)
    return np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1)))

def geometry_slot(desc,name):
    slot=desc['joint_names'].index(name)
    ids=[i for i,g in enumerate(desc['geoms']) if any(j.get('slot') and j['slot'][0]==slot for j in desc['bodies'][g['body']]['joints'])]
    if not ids:raise ValueError(f'Missing native visual {name}')
    return ids

def body_origin(desc,i,p,r):
    g=desc['geoms'][i];br=r[:,i]@np.asarray(g['rotation']).reshape(3,3).T
    return p[:,i]-np.einsum('tij,j->ti',br,g['position'])

def rank(index,archive_root,descriptor,cached_roots,output,seconds=8,side='left'):
    desc=json.loads(gzip.decompress(Path(descriptor).read_bytes()))
    hand_ids=geometry_slot(desc,side+'_wrist_yaw_link');elbow=geometry_slot(desc,side+'_elbow_link')[0]
    # Prefer the distal hand mesh over the small wrist motor sleeve.
    def extent(i):
        g=desc['geoms'][i];v=np.asarray(desc['meshes'][g['mesh']]['vertices']).reshape(-1,3)
        return np.linalg.norm(np.ptp(v,axis=0))
    hand=max(hand_ids,key=extent);g=desc['geoms'][hand]
    vertices=np.asarray(desc['meshes'][g['mesh']]['vertices']).reshape(-1,3)
    local=vertices@np.asarray(g['rotation']).reshape(3,3).T+g['position']
    lengths=np.linalg.norm(local,axis=1);tip=vertices[lengths>=np.quantile(lengths,.9)].mean(0)
    cached={}
    for root in cached_roots:
        for path in Path(root).rglob('*.bvh'):cached.setdefault(path.stem,path)
    results=[];prefix=side.capitalize()
    for line in Path(index).read_text(encoding='utf-8').splitlines():
        row=json.loads(line);clip=row['clip_id']
        if clip not in cached:continue
        m=load_compact(Path(archive_root)/row['archive'],member=row['member'],robot=row['robot_id'])
        n=min(len(m['q']),int(seconds*m['fps']));b=load_bvh(cached[clip])
        stride=round(1/b['dt']/m['fps']);take=np.arange(n)*stride
        if stride<1 or take[-1]>=len(b['positions']) or abs(1/b['dt']/stride-m['fps'])>.02:raise ValueError('Timeline mismatch')
        source=b['positions'][take];names=b['names']
        sw=source[:,names.index(prefix+'Hand')];se=source[:,names.index(prefix+'ForeArm')]
        st=source[:,names.index(prefix+'HandMiddleEnd')];sf=sw-se;sh=st-sw
        p,r=native_visual_motion(desc,m['q'][:n],m['root_t_cm'][:n],m['root_rot6d'][:n])
        rw=body_origin(desc,hand,p,r);re=body_origin(desc,elbow,p,r)
        rt=p[:,hand]+np.einsum('tij,j->ti',r[:,hand],tip);rf=rw-re;rh=rt-rw
        metrics={'hand_direction_p95_deg':float(np.percentile(angle(sh,rh),95)),
            'forearm_direction_p95_deg':float(np.percentile(angle(sf,rf),95)),
            'wrist_bend_difference_p95_deg':float(np.percentile(np.abs(angle(sf,sh)-angle(rf,rh)),95))}
        results.append({'clip_id':clip,'semantic_label':row['source']['semantic_label'],'frames_checked':n,**metrics,
            'score':metrics['hand_direction_p95_deg']+metrics['wrist_bend_difference_p95_deg'],
            'source_bvh':str(cached[clip]),'minimum_sole_height_cm':float(m['sole_height_cm'][:n].min())})
    results.sort(key=lambda r:r['score']);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'candidate-ranking.json').write_text(json.dumps({'heuristic':__doc__,'side':side,'seconds':seconds,'candidates':results},indent=2),encoding='utf-8')
    for row in results[:20]:print(json.dumps({k:v for k,v in row.items() if k not in ('source_bvh',)}),flush=True)
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('index','archive-root','descriptor','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cached-roots',nargs='+',type=Path,required=True);p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--side',choices=('left','right'),default='left');a=p.parse_args()
    rank(a.index,a.archive_root,a.descriptor,a.cached_roots,a.output,a.seconds,a.side)
if __name__=='__main__':main()
