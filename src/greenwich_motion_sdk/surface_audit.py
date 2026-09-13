"""Whole-visual-surface floor audit, extending a precommitted locomotion sample.

Exact mesh minimum heights, conservative source-stationary non-foot patches.
No force, object contact, dynamics or learned non-foot label is inferred.
"""
import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from .compact import native_visual_motion
from .corpus_screen import source_records
from .fixed_support import SupportConfig, measure_fixed_support
from .physics_audit import intervals
from .sampling_statistics import proportion_interval


def source_landmarks(body_name):
    """Semantic native link -> source anatomy; interface mapping, not task tuning."""
    name=body_name.lower();side='Left' if name.startswith('left') else 'Right'
    if 'wrist' in name or 'hand' in name:return [side+'Hand']
    if 'elbow' in name:return [side+'ForeArm']
    if 'shoulder' in name:return [side+'Arm']
    if 'knee' in name:return [side+'Leg']
    if 'hip' in name:return [side+'UpLeg']
    if 'ankle' in name or 'foot' in name:return [] # original foot assay retained
    if 'head' in name:return ['Head']
    if 'pelvis' in name:return ['Hips']
    if 'torso' in name or 'waist' in name:return ['Spine2']
    return []


def mesh_extrema(vertices):
    """Convex hull retains exact extrema for every rigid rotation; no decimation."""
    vertices=np.asarray(vertices,float).reshape(-1,3)
    try:return vertices[ConvexHull(vertices).vertices]
    except QhullError:return vertices


def evaluate(motion,bvh,descriptor,body_names,geometry,config,penetration_cm=.5,duration_s=.1,slip_cm_s=15.):
    positions,rotations=native_visual_motion(descriptor,motion['q'],motion['root_t_cm'],motion['root_rot6d'])
    fps=motion['fps'];T=len(positions);source=bvh['positions'][:T];names=bvh['names']
    if len(source)!=T or abs(1/bvh['dt']-fps)>.02:raise ValueError('Source timestamp mismatch')
    heights=np.empty((T,len(geometry)));surfaces=[];speeds=[];masks=[]
    for i,(vertices,hull) in enumerate(geometry):
        g=descriptor['geoms'][i];body_name=body_names[g['body']];r=rotations[:,i];p=positions[:,i]
        heights[:,i]=(r[:,1]@hull.T).min(1)*100+p[:,1]*100
        penetration=intervals(heights[:,i]<-penetration_cm,fps,duration_s)
        landmarks=source_landmarks(body_name);metric=None;speed=np.full(T,np.nan);mask=np.zeros(T,bool)
        if landmarks and all(n in names for n in landmarks):
            ref=source[:,[names.index(n) for n in landmarks]]
            if (ref[:,:,1].min(1)<config.source_height_cm).any() and (heights[:,i]<=config.contact_band_cm).any():
                # Only low actual material vertices can qualify as a contact patch.
                # This exact material subset retains vertex identities over time.
                low=np.zeros(len(vertices),bool)
                for a in range(0,T,64):
                    y=r[a:a+64,1]@vertices.T+p[a:a+64,1,None]
                    low|=(y<=config.contact_band_cm/100).any(0)
                if low.sum()>=config.minimum_patch_vertices:
                    world=np.einsum('tij,vj->tvi',r,vertices[low])*100+p[:,None]*100
                    metric,speed,mask=measure_fixed_support(ref,world,r,fps,config,slip_cm_s,penetration_cm,duration_s)
        surfaces.append({'surface_id':i,'native_body':body_name,'source_landmarks':landmarks,
            'minimum_height_cm':float(heights[:,i].min()),'max_penetration_cm':float(max(0,-heights[:,i].min())),
            'penetration_events':penetration,'geometry_contact_frames':int(((heights[:,i]<=config.contact_band_cm)&(heights[:,i]>=-penetration_cm)).sum()),
            'fixed_support':metric,'support_status':'unconfirmed' if metric is None or not mask.any() else ('drift_flag' if metric['fixed_patch_slip_events'] else 'no_drift_flag')})
        speeds.append(speed);masks.append(mask)
    return surfaces,heights,np.stack(speeds,1),np.stack(masks,1)


def run(selection,foot_ledger,descriptor_path,xml,source_archive,output,config,penetration_cm=.5,duration_s=.1,slip_cm_s=15.):
    import mujoco as mj
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cache=output/'surface-cache';cache.mkdir(exist_ok=True)
    rows=json.loads(Path(selection).read_text(encoding='utf-8'))['clips']
    foot={r['clip_id']:r for r in map(json.loads,Path(foot_ledger).read_text().splitlines())}
    if set(foot)!=set(r['clip_id'] for r in rows):raise ValueError('Sample / foot ledger mismatch')
    desc=json.loads(gzip.decompress(Path(descriptor_path).read_bytes()))
    model=mj.MjModel.from_xml_path(str(xml));body_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,i) or f'body_{i}' for i in range(model.nbody)]
    if len(body_names)!=len(desc['bodies']):raise ValueError('Native body roster mismatch')
    geometry=[]
    for g in desc['geoms']:
        v=np.asarray(desc['meshes'][g['mesh']]['vertices'],float).reshape(-1,3);geometry.append((v,mesh_extrema(v)))
    signature={'selection_sha256':hashlib.sha256(Path(selection).read_bytes()).hexdigest(),
        'foot_ledger_sha256':hashlib.sha256(Path(foot_ledger).read_bytes()).hexdigest(),
        'descriptor_sha256':hashlib.sha256(Path(descriptor_path).read_bytes()).hexdigest(),
        'xml_sha256':hashlib.sha256(Path(xml).read_bytes()).hexdigest(),
        'support_config':config.__dict__,'slip_cm_s':slip_cm_s,'penetration_cm':penetration_cm,'duration_s':duration_s}
    cfg=output/'run.json'
    if cfg.exists() and json.loads(cfg.read_text())!=signature:raise ValueError('Resume configuration differs')
    cfg.write_text(json.dumps(signature,indent=2));ledger=output/'results.jsonl'
    results=list(map(json.loads,ledger.read_text().splitlines())) if ledger.exists() else []
    done={r['clip_id'] for r in results};pending=[r for r in rows if r['clip_id'] not in done];started=time.monotonic()
    with ledger.open('a',encoding='utf-8') as stream:
        for row in source_records(pending,source_archive,prepare_compact=True):
            surfaces,heights,speeds,masks=evaluate(row['_compact_motion'],row['_source_bvh'],desc,body_names,geometry,config,penetration_cm,duration_s,slip_cm_s)
            pen=any(s['penetration_events'] for s in surfaces);extra_slip=any(s['support_status']=='drift_flag' for s in surfaces)
            prior=foot[row['clip_id']];good=prior['review_status']=='good' and not pen and not extra_slip
            result={'clip_id':row['clip_id'],'category':row['source']['category'],'fps':row['fps'],'frames':row['frames'],
                'foot_slip_flag':prior['fixed_support_status']=='drift_flag','foot_penetration_flag':prior['penetration_status']=='fail',
                'whole_surface_penetration_flag':pen,'nonfoot_fixed_support_slip_flag':extra_slip,'combined_good':good,'surfaces':surfaces}
            partial=cache/(row['clip_id']+'.partial.npz');target=cache/(row['clip_id']+'.npz')
            np.savez_compressed(partial,min_height_cm=heights,patch_speed_cm_s=speeds,patch_evaluated=masks,fps=row['fps'],
                surface_names=np.asarray([s['native_body'] for s in surfaces]),contact=(heights<=config.contact_band_cm)&(heights>=-penetration_cm),penetration=heights < -penetration_cm)
            partial.replace(target);stream.write(json.dumps(result,separators=(',',':'))+'\n');stream.flush();results.append(result)
            if len(results)%25==0:
                progress={'completed':len(results),'total':len(rows),'elapsed_this_run_s':time.monotonic()-started,'good':sum(r['combined_good'] for r in results)}
                (output/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    n=len(results);prior_stats=json.loads(Path(foot_ledger).with_name('statistics.json').read_text());N=prior_stats['good']['population']
    if n!=len(rows):raise ValueError('Incomplete sample')
    stats={'schema':'greenwich.whole-surface-screen.v1','population':N,'sample_count':n,'seed':prior_stats['seed'],
        'good':proportion_interval(sum(r['combined_good'] for r in results),n,N),
        'foot_combined_good':proportion_interval(sum(r['review_status']=='good' and r['penetration_status']=='pass_screen' for r in foot.values()),n,N),
        'whole_surface_penetration_count':sum(r['whole_surface_penetration_flag'] for r in results),
        'nonfoot_support_slip_count':sum(r['nonfoot_fixed_support_slip_flag'] for r in results),
        'surfaces_flagged':dict(Counter(s['native_body'] for r in results for s in r['surfaces'] if s['penetration_events'] or s['support_status']=='drift_flag')),
        'elapsed_this_run_s':time.monotonic()-started,'thresholds':signature,
        'scope':'Random release sample. All visual meshes tested for floor penetration, including hands, knees, pelvis and torso. Non-foot stationary support uses semantic source-joint and persistent material-patch proxies. Unconfirmed support accepted. Object contact, force-bearing contact and dynamics unverified. This is a local engineering screen, not an ETH benchmark reproduction.'}
    (output/'statistics.json').write_text(json.dumps(stats,indent=2));print(json.dumps({k:v for k,v in stats.items() if k not in ('thresholds','surfaces_flagged')}),flush=True)
    return stats


def rescreen(foot_directory,surface_directory,output,slip_cm_s=15.,penetration_cm=.5,duration_s=.1):
    """Reclassify cached numeric arrays, without BVH, meshes, inference or rendering.

    Speeds retain the originally extracted stationary-support masks. Changing
    those support-mask settings requires SurfaceAudit again; reject thresholds
    and persistence duration can be changed here.
    """
    if any(not np.isfinite(v) or v<=0 for v in (slip_cm_s,penetration_cm,duration_s)):raise ValueError('Invalid thresholds')
    foot_directory=Path(foot_directory);surface_directory=Path(surface_directory);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((surface_directory/'statistics.json').read_text());results=[];started=time.monotonic()
    for line in (surface_directory/'results.jsonl').read_text().splitlines():
        row=json.loads(line);stem=row['clip_id'];foot_flags=[]
        with np.load(foot_directory/'speed-cache'/(stem+'.npz'),allow_pickle=False) as cache:
            fps=float(cache['fps'])
            for side in ('left_foot','right_foot'):
                v=cache[side+'__speed_cm_s'];valid=cache[side+'__evaluated']
                foot_flags.extend(intervals(valid&np.isfinite(v)&(v>slip_cm_s),fps,duration_s))
        surface_flags=[];slip_flags=[];supported=[]
        with np.load(surface_directory/'surface-cache'/(stem+'.npz'),allow_pickle=False) as cache:
            if float(cache['fps'])!=fps:raise ValueError('Cache FPS mismatch')
            for i,name in enumerate(cache['surface_names']):
                pen=intervals(cache['min_height_cm'][:,i]<-penetration_cm,fps,duration_s)
                v=cache['patch_speed_cm_s'][:,i];valid=cache['patch_evaluated'][:,i]
                slip=intervals(valid&np.isfinite(v)&(v>slip_cm_s),fps,duration_s)
                if pen:surface_flags.append({'surface_id':i,'native_body':str(name),'events':pen})
                if slip:slip_flags.append({'surface_id':i,'native_body':str(name),'events':slip})
                if valid.any():supported.append(str(name))
        results.append({'clip_id':stem,'category':row['category'],'good':not(foot_flags or surface_flags or slip_flags),
            'foot_slip_events':foot_flags,'surface_penetration':surface_flags,'nonfoot_slip':slip_flags,'nonfoot_confirmed_surfaces':supported})
    n=len(results);stats={'schema':'greenwich.cached-whole-surface-screen.v1','sample_count':n,'population':prior['population'],'seed':prior['seed'],
        'good':proportion_interval(sum(r['good'] for r in results),n,prior['population']),
        'flagged':proportion_interval(sum(not r['good'] for r in results),n,prior['population']),
        'counts':{'foot_slip':sum(bool(r['foot_slip_events']) for r in results),'whole_surface_penetration':sum(bool(r['surface_penetration']) for r in results),'nonfoot_support_slip':sum(bool(r['nonfoot_slip']) for r in results)},
        'thresholds':{'slip_cm_s':slip_cm_s,'penetration_cm':penetration_cm,'minimum_duration_s':duration_s},'elapsed_s':time.monotonic()-started,
        'scope':prior['scope'],'support_masks':'Frozen from extraction configuration in surface/run.json and foot/run.json. Unknown support accepted. Object contact unavailable.'}
    (output/'statistics.json').write_text(json.dumps(stats,indent=2));(output/'results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in results)+'\n')
    print(json.dumps(stats),flush=True);return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('selection','foot-ledger','descriptor','xml','source-archive','output','support-config'):p.add_argument('--'+name,type=Path)
    p.add_argument('--rescreen',action='store_true');p.add_argument('--foot-directory',type=Path);p.add_argument('--surface-directory',type=Path)
    p.add_argument('--slip-cm-s',type=float,default=15.);p.add_argument('--penetration-cm',type=float,default=.5);p.add_argument('--duration-s',type=float,default=.1)
    a=p.parse_args()
    if a.rescreen:
        if not all((a.foot_directory,a.surface_directory,a.output)):p.error('Rescreen requires foot-directory, surface-directory and output')
        rescreen(a.foot_directory,a.surface_directory,a.output,a.slip_cm_s,a.penetration_cm,a.duration_s)
    else:
        if not all((a.selection,a.foot_ledger,a.descriptor,a.xml,a.source_archive,a.output,a.support_config)):p.error('Extraction requires selection, foot ledger, descriptor, XML, source archive, output and support config')
        run(a.selection,a.foot_ledger,a.descriptor,a.xml,a.source_archive,a.output,SupportConfig.load(a.support_config),a.penetration_cm,a.duration_s,a.slip_cm_s)


if __name__=='__main__':main()
