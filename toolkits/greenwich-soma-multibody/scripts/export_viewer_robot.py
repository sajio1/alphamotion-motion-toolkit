"""Export a vendor MJCF's native visual FK for the portable motion viewer."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--xml',type=Path,required=True)
    p.add_argument('--body',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo/'src'))
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.embodiment.mjcf_build import build_merge, AX
    from alphamotion.viz.kinematics import visual_mesh_geom_ids
    spec,_,_,qnames,_=build_from_mjcf(str(a.xml),a.body)
    model=mj.MjModel.from_xml_path(str(a.xml))
    keep,frames=build_merge(model)
    slots={name:[j,s] for j,names in enumerate(qnames) for s,name in enumerate(names)}
    bodies=[]
    def matrix(quat):return Rotation.from_quat(quat[[1,2,3,0]]).as_matrix().reshape(-1).tolist()
    for b in range(model.nbody):
        joints=[]
        for ji in range(int(model.body_jntadr[b]),int(model.body_jntadr[b]+model.body_jntnum[b])):
            if ji < 0:continue
            if model.jnt_type[ji]==mj.mjtJoint.mjJNT_FREE:continue
            name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,ji)
            if model.jnt_type[ji]==mj.mjtJoint.mjJNT_BALL and name not in slots:
                adr=int(model.jnt_qposadr[ji])
                joints.append({'slot':None,'fixed_rotation':matrix(model.qpos0[adr:adr+4]),
                               'position':model.jnt_pos[ji].tolist()})
                continue
            if model.jnt_type[ji]!=mj.mjtJoint.mjJNT_HINGE:raise ValueError(f'Unsupported motor {name}')
            joints.append({'slot':slots.get(name),'angle':float(model.qpos0[model.jnt_qposadr[ji]]),
                           'axis':model.jnt_axis[ji].tolist(),'position':model.jnt_pos[ji].tolist()})
        bodies.append({'parent':int(model.body_parentid[b]) if b else -1,
                       'position':model.body_pos[b].tolist(),'rotation':matrix(model.body_quat[b]),'joints':joints})
    meshes={};geoms=[]
    for gid in visual_mesh_geom_ids(model):
        mid=int(model.geom_dataid[gid]);key=str(mid)
        if key not in meshes:
            va,vn=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
            fa,fn=int(model.mesh_faceadr[mid]),int(model.mesh_facenum[mid])
            meshes[key]={'vertices':model.mesh_vert[va:va+vn].round(7).reshape(-1).tolist(),
                         'faces':model.mesh_face[fa:fa+fn].reshape(-1).tolist()}
        rgba=model.geom_rgba[gid].tolist()
        mat=int(model.geom_matid[gid])
        if mat>=0:rgba=model.mat_rgba[mat].tolist()
        geoms.append({'mesh':key,'body':int(model.geom_bodyid[gid]),'position':model.geom_pos[gid].tolist(),
                      'rotation':matrix(model.geom_quat[gid]),'color':rgba})
    result={'schema':'greenwich.viewer.robot.v1','robot':'h2','joint_names':list(spec.joint_names),
            'root_body':int(keep[0]),'root_frame':int(frames[keep[0]]),
            'zup_to_yup':AX.reshape(-1).tolist(), 'coordinate_provenance':'alphamotion.embodiment.mjcf_build.AX',
            'bodies':bodies,'geoms':geoms,'meshes':meshes}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
    print(json.dumps({'bodies':len(bodies),'geoms':len(geoms),'bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
