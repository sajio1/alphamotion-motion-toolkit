"""Evaluate saved full-body SDK outputs without running the model/refiner again."""
import argparse
import json
from pathlib import Path
import numpy as np


def evaluate_outputs(root,robot,descriptor,output,*,slip_cm_s=15.,penetration_cm=.5,bad_duration_s=.1):
    """Shared native-sole screen; necessary checks, not a dynamics certificate.

    Requires noncompact Pipeline.convert outputs with sampled source.npz.
    No pose fitting, trajectory alignment or unknown-to-pass conversion.
    """
    from .locomotion_audit import audit
    if min(slip_cm_s,penetration_cm,bad_duration_s)<=0:raise ValueError('Positive thresholds required')
    root=Path(root);paths=sorted(root.glob('*/'+robot['name']+'/motion.npz'))
    if not paths:raise ValueError('No noncompact conversion outputs found for robot')
    records=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as m:
            motion={'q':m['q'],'root_rot6d':m['rot6d'][:,0],'root_t_cm':m['root_t'],
                    'joint_names':m['joint_names'],'sole_height_cm':m['sole_surface_height_cm'],
                    'model_contact':m['model_contact'],'fps':float(m['fps'])}
            footids=m['source_foot_indices'].tolist()
        records.append({'clip_id':path.parent.parent.name,'source_canonical':str(path.parent.parent/'source.npz'),
                        'source_foot_indices':footids,'_compact_motion':motion})
    return audit({'clips':records},robot,descriptor,output,slip_cm_s,penetration_cm,bad_duration_s)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','robots','descriptor','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--robot',required=True)
    p.add_argument('--slip-cm-s',type=float,default=15.)
    p.add_argument('--penetration-cm',type=float,default=.5)
    p.add_argument('--bad-duration-s',type=float,default=.1)
    a=p.parse_args();roster=json.loads(a.robots.read_text(encoding='utf-8-sig'))
    robot=next(r for r in roster if r['name']==a.robot)
    evaluate_outputs(a.root,robot,a.descriptor,a.output,slip_cm_s=a.slip_cm_s,
                     penetration_cm=a.penetration_cm,bad_duration_s=a.bad_duration_s)


if __name__=='__main__':main()
