"""Recompute cross-body diagnostics from saved motion; no robot ground truth assumed."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

def distribution(x):
    x=np.asarray(x).reshape(-1)
    x=x[np.isfinite(x)]
    return {'mean':float(x.mean()),'p95':float(np.percentile(x,95)),'max':float(x.max())} if len(x) else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--robots',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo/'src'))
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine.spatial import key_joints
    from alphamotion.engine.nets.rotations import rot6d_to_matrix
    import torch
    roster=json.loads(a.robots.read_text(encoding='utf-8-sig'))
    keys={r['name']:key_joints(build_from_mjcf(r['xml'],r['body'])[0]) for r in roster}
    records=[];flat=[]
    source_names=['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']
    for path in sorted(a.root.glob('*/*/motion.npz')):
        report=json.loads((path.parent/'report.json').read_text())
        with np.load(path,allow_pickle=False) as m:
            ri,rnames,tags=keys[path.parent.name]
            sn=m['source_joint_names'].tolist();si=m['source_role_indices'].tolist() if 'source_role_indices' in m else [sn.index(n) for n in source_names]
            src=m['source_positions_cm'][:,si];target=m['world_position_cm'][:,ri]
            sr=src[:,1:]-src[:,:1];tr=target[:,1:]-target[:,:1]
            absolute=np.linalg.norm(target[:,1:]-src[:,1:],axis=-1)
            relative=np.linalg.norm(tr-sr,axis=-1)
            denom=np.linalg.norm(sr,axis=-1)*np.linalg.norm(tr,axis=-1)
            direction=np.degrees(np.arccos(np.clip(np.sum(sr*tr,axis=-1)/np.maximum(denom,1e-8),-1,1)))
            direction[denom<1e-8]=np.nan
            raw=rot6d_to_matrix(torch.tensor(m['raw_rot6d'])).numpy()
            final=rot6d_to_matrix(torch.tensor(m['rot6d'])).numpy()
            delta=np.swapaxes(raw,-1,-2)@final
            rot_delta=np.degrees(np.arccos(np.clip((np.trace(delta,axis1=-2,axis2=-1)-1)/2,-1,1)))
            fps=float(m['fps']);root=m['root_t']
            row={'source':path.parent.parent.name,'robot':path.parent.name,
                 'representation':report.get('input_representation','soma77'),
                 'frames':len(root),'duration_s':len(root)/fps,
                 'mapping':dict(zip(tags,[dict(source=sn[s],target=t) for s,t in zip(si,rnames)])),
                 'source_position_difference_cm':{},'root_relative_difference_cm':{},
                 'root_to_landmark_direction_difference_deg':{},
                 'native_rotation_correction_deg':distribution(rot_delta),
                 'max_penetration_cm':report['max_penetration_cm'],
                 'support_slip_p95_cm_s':report['source_support_slip_p95_cm_s'],
                 'max_limit_violation_rad':report['max_joint_limit_violation_rad'],
                 'timing_s':report['timing_s'],
                 'refine_s':(report.get('contact_refinement') or {}).get('seconds',0)}
            for j,tag in enumerate(tags[1:]):
                row['source_position_difference_cm'][tag]=distribution(absolute[:,j])
                row['root_relative_difference_cm'][tag]=distribution(relative[:,j])
                row['root_to_landmark_direction_difference_deg'][tag]=distribution(direction[:,j])
            row['model_compute_s']=row['timing_s']['encode_shared']+row['timing_s']['decode']
            row['motion_compute_s']=row['model_compute_s']+row['timing_s']['joint_projection']+row['refine_s']
            records.append(row)
            flat.append(dict(source=row['source'],robot=row['robot'],representation=row['representation'],
                duration_s=row['duration_s'],left_hand_root_relative_p95_cm=float(np.percentile(relative[:,1],95)),
                right_hand_root_relative_p95_cm=float(np.percentile(relative[:,2],95)),
                hand_direction_p95_deg=float(np.percentile(direction[:,1:3],95)),
                max_penetration_cm=row['max_penetration_cm'],support_slip_p95_cm_s=row['support_slip_p95_cm_s'],
                model_compute_s=row['model_compute_s'],motion_compute_s=row['motion_compute_s']))
    contract={
        'no_paired_robot_ground_truth':True,
        'position_differences_include_morphology':'Unscaled centimetres; different limb lengths/root heights contribute. Not pure model reconstruction error.',
        'direction_metric':'Root-to-landmark ray angle; NOT wrist rotation error.',
        'native_rotation_correction':'Final vs raw decoded orientation on SAME robot descriptor; not source wrist tracking accuracy.',
        'horizontal_root':'Source XZ is supplied as guide; not autonomous model locomotion accuracy.',
        'smpl':'smpl22-derived is a SOMA adapter ablation, not native SMPL dataset generalization.',
        'bvh':'Source files use SOMA77 in BVH containers; BVH and SOMA are not independent datasets.',
        'timing':'encode is shared across robots, included per-cell for standalone comparison; excludes shared startup and render.',
        'success_rate':None}
    (a.root/'accuracy.json').write_text(json.dumps({'contract':contract,'results':records},indent=2),encoding='utf-8')
    if flat:
        with (a.root/'accuracy.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({'cells':len(records),'report':str(a.root/'accuracy.json')}))

if __name__=='__main__':main()
