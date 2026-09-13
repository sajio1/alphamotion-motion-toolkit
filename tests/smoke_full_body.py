"""Manual GPU integration smoke: SOMA-derived SMPL-format fixture, not a corpus benchmark.

All artifacts are written to the supplied output. External source/model/robots
are required. This fixture tests source format plumbing, not SMPL shape fitting.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from greenwich_motion_sdk import load_motion,Pipeline
from greenwich_motion_sdk._smpl_bind import SMPL_TO_SOMA
from greenwich_motion_sdk._bvh import load_bvh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','bind','toolkit','robots','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--iterations',type=int,default=100)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    clip=load_motion(a.source,format='soma',bind_path=a.bind);bind=load_bvh(a.bind)
    names=SMPL_TO_SOMA;ids=[clip.names.index(n) for n in names]
    parents=[]
    for j in ids:
        parent=clip.parents[j]
        while parent>=0 and parent not in ids:parent=clip.parents[parent]
        parents.append(ids.index(parent) if parent>=0 else -1)
    bp=bind['positions'][0,[bind['names'].index(n) for n in names]]
    offsets=np.zeros_like(bp)
    for j,parent in enumerate(parents):
        if parent>=0:offsets[j]=bp[j]-bp[parent]
    global_R=clip.global_rotation[:,ids];local=global_R.copy()
    for j,parent in enumerate(parents):
        if parent>=0:local[:,j]=global_R[:,parent].swapaxes(-1,-2)@global_R[:,j]
    poses=Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(len(local),-1)
    np.savez_compressed(a.output/'smpl-format-fixture.npz',poses=poses,trans=clip.world_position_cm[:,0],mocap_framerate=1/np.diff(clip.timestamps_s).mean())
    options={'names':names,'parents':parents,'rest_offsets':offsets.tolist(),'pelvis_offset_source_units':[0,0,0],
             'frame':{'basis_to_yup':np.eye(3).tolist(),'cm_per_unit':1.,'ground_y_cm':0.}}
    (a.output/'smpl-profile.json').write_text(json.dumps(options,indent=2),encoding='utf-8')
    source={'id':'smpl-smoke','format':'smpl','path':'smpl-format-fixture.npz','options_file':'smpl-profile.json',
            'label':'SOMA-derived SMPL format integration fixture',
            'roles':['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot'],
            'foot_pairs':[['LeftFoot','LeftToeBase'],['RightFoot','RightToeBase']],
            'foot_landmark_height_cm':[float(bp[[7,10],1].min()),float(bp[[8,11],1].min())]}
    manifest=a.output/'sources.json';manifest.write_text(json.dumps({'schema':'alphamotion.sources.v1','sources':[source]},indent=2),encoding='utf-8')
    print(Pipeline(a.toolkit).convert(manifest,a.output/'results',a.robots,max_seconds=a.seconds,contact_iterations=a.iterations))


if __name__=='__main__':main()
