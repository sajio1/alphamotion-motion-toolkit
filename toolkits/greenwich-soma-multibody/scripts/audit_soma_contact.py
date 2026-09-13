"""Re-read saved results: every-frame geometry checks and 10 Hz visual review sheets."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    results=[]
    for path in a.root.glob('*/*/motion.npz'):
        m=np.load(path);fps=float(m['fps']);world=m['world_position_cm'];source=m['source_positions_cm'];names=m['joint_names'].tolist()
        r=m['rot6d'];r0=r[:-1];r1=r[1:]
        def mat(x):
            e1=x[...,:3];e1=e1/np.linalg.norm(e1,axis=-1,keepdims=True)
            e2=x[...,3:]-(x[...,3:]*e1).sum(-1,keepdims=True)*e1;e2/=np.linalg.norm(e2,axis=-1,keepdims=True)
            return np.stack([e1,e2,np.cross(e1,e2)],-1)
        rr=mat(r);relative=rr[:-1].swapaxes(-1,-2)@rr[1:]
        angles=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
        heights=m['sole_surface_height_cm'];root=m['root_t'];support=m['source_contact_proxy']
        stats={'robot':path.parent.name,'all_frames_finite':bool(np.isfinite(world).all()),'frames':len(root),
               'root_max_frame_translation_cm':float(np.linalg.norm(np.diff(root,axis=0),axis=-1).max()),
               'root_max_frame_rotation_deg':float(angles[:,0].max()),'joint_global_max_frame_rotation_deg':float(angles.max()),
               'max_penetration_cm':float(np.maximum(-heights,0).max()),
               'swing_clearance_peak_cm':[float(heights[~support[:,j],j].max()) if (~support[:,j]).any() else None for j in range(2)],
               'root_xyz_range_cm':np.ptp(root,axis=0).tolist(),'source_root_xyz_range_cm':np.ptp(m['source_root_cm'],axis=0).tolist(),
               'visual_sampling_hz':10,'visual_review':'pending'}
        cap=cv2.VideoCapture(str(path.parent/'preview.mp4'));tiles=[]
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(root)
        assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<.001
        for f in range(0,len(root),max(1,round(fps/10))):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
            im=cv2.resize(im,(480,270));cv2.rectangle(im,(0,0),(210,25),(255,255,255),-1)
            cv2.putText(im,f'{path.parent.name} frame {f} / {f/fps:.2f}s',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);tiles.append(im)
        cap.release()
        for page,k in enumerate(range(0,len(tiles),25)):
            batch=tiles[k:k+25]
            while len(batch)%5:batch.append(np.zeros_like(batch[0]))
            cv2.imwrite(str(path.parent/f'review_{page}.jpg'),np.concatenate([np.concatenate(batch[j:j+5],axis=1) for j in range(0,len(batch),5)],axis=0))
        (path.parent/'audit.json').write_text(json.dumps(stats,indent=2));results.append(stats)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
