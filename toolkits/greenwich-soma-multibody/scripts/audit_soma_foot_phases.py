"""Audit saved motion: sole pitch during support and simultaneous flight. No inference."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','pipeline','root','robots'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.repo/'src'),str(a.pipeline/'src')]
    import torch
    from scipy.ndimage import minimum_filter1d
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine.spatial import key_joints
    from alphamotion.engine import constraints as c
    from greenwich_umi_proof.sole_geometry import build_foot_sole_model,_neutral_descriptor_fk
    results=[]
    for robot in json.loads(a.robots.read_text(encoding='utf-8-sig')):
        spec,dof,rest,*_=build_from_mjcf(robot['xml'],robot['body'])
        sole=build_foot_sole_model(robot['xml'],spec,rest,['left_foot','right_foot'],key_joints(spec)[0][4:6])
        _,nr=_neutral_descriptor_fk(spec,rest)
        for path in a.root.glob('*/'+robot['name']+'/motion.npz'):
            m=np.load(path);R=c.rot6d_to_matrix(torch.tensor(m['rot6d'])).numpy();w=m['world_position_cm'];support=m['source_contact_proxy']
            stable=minimum_filter1d(support.astype(float),size=2*round(float(m['fps'])*.1)+1,axis=0,mode='constant')>0
            feet={}
            for k,(role,f) in enumerate(sole.feet.items()):
                v=f.vertices_local_cm;forward=(v@nr[f.joint].T)[:,2]
                ends=np.stack([v[forward<=np.quantile(forward,.2)].mean(0),v[forward>=np.quantile(forward,.8)].mean(0)])
                delta=np.einsum('tij,j->ti',R[:,f.joint],ends[1]-ends[0])
                pitch=np.degrees(np.arctan2(delta[:,1],np.linalg.norm(delta[:,[0,2]],axis=1)))
                feet[role]={label:float(np.percentile(pitch[mask],95)) if mask.any() else None for label,mask in [('support_toe_up_p95_deg',support[:,k]),('stable_toe_up_p95_deg',stable[:,k])]}
            heights=m['sole_surface_height_cm'];flight=(heights>3).all(1)
            results.append({'clip':path.parent.parent.name,'robot':robot['name'],'feet':feet,'both_feet_above_3cm_frames':int(flight.sum()),'peak_lower_foot_cm':float(heights.min(1).max()),'root_max_frame_delta_cm':float(np.linalg.norm(np.diff(m['root_t'],axis=0),axis=1).max()),'max_penetration_cm':float(np.maximum(-heights,0).max())})
    (a.root/'foot_phase_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
