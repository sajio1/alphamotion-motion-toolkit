"""Reusable BoneSeed locomotion probe: Prime root height, source XZ, native robot projection.

No standing/crouching seed, no procedural gait or render lift. Model root height
is an initial reference for the explicit contact residual, never fixed.
"""
import argparse,json,sys,time,os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--pipeline',type=Path,required=True)
    p.add_argument('--dataset-workspace',type=Path);p.add_argument('--robots',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,help='SDK SOMA/SMPL/BVH/Canonical sources; bypass legacy dataset selection')
    p.add_argument('--model',choices=['bumblebee','megatron','prime'],default='prime')
    p.add_argument('--run-dir',type=Path,help='External checkpoint override for selected v2 model')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--indices',default='1,7,24,36,37,46,55,67,96,99')
    p.add_argument('--limit',type=int);p.add_argument('--fps',type=float,default=30);p.add_argument('--max-seconds',type=float,default=15)
    p.add_argument('--contact-iters',type=int,default=500)
    p.add_argument('--input-representation',choices=['soma77','smpl22-derived'],default='soma77')
    p.add_argument('--contact-config',type=Path,help='JSON overrides for shared contact/flight weights and phase margins')
    p.add_argument('--skip-preview',action='store_true',help='Do not render per-robot MP4 previews')
    p.add_argument('--compact',action='store_true',help='Save one reconstruction-complete NPZ bundle per source motion')
    p.add_argument('--cache',type=Path,help='Shared AlphaMotion cache for resumable bulk jobs')
    p.add_argument('--chunk',type=int,default=32);p.add_argument('--ffmpeg',required=True);a=p.parse_args()
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(a.repo/'src'),str(a.pipeline/'src'),str(a.pipeline/'scripts')]+([str(a.dataset_workspace/'work')] if a.dataset_workspace else [])
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HOME']=str(a.repo/'.local/huggingface');os.environ['ALPHAMOTION_CACHE']=str(a.cache or a.output/'cache')
    import torch,mujoco as mj,cv2
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import gaussian_filter1d
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
    from greenwich_motion_sdk._bvh import load_bvh
    from greenwich_motion_sdk import load_motion
    from alphamotion.engine.nets.rotations import SkeletonSpec
    from alphamotion.embodiment.dof_features import ball_dof
    from alphamotion.engine.spatial import fk_pos
    from alphamotion.engine.greenwich import Greenwich
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine import constraints as c
    from alphamotion.engine.spatial import key_joints
    from greenwich_umi_proof.sole_geometry import build_foot_sole_model
    from greenwich_umi_proof.interactive_viewer import _sample_greenwich_head,YUP_TO_ZUP
    from soma_contact_refine import refine
    from render_hiw500_transfer_showcase import _configure_model,_set_fixed_camera,_writer,_close_writer
    from render_single_embodiment_humi_demo import _fit_action_camera
    a.output.mkdir(parents=True,exist_ok=True)
    if a.source_manifest:
        from greenwich_motion_sdk.source_batch import read_sources,sample_clip
        rows=read_sources(a.source_manifest)
        if a.input_representation!='soma77':raise ValueError('Manifest inputs already define their source representation')
    else:
        if not a.dataset_workspace:p.error('Provide --source-manifest or --dataset-workspace')
        source_dir=a.dataset_workspace/'outputs/seed-100-contact/source'
        selected=set(map(int,a.indices.split(',')))
        rows=[x for x in json.loads((a.dataset_workspace/'outputs/seed-100-contact/selection.json').read_text()) if x['index'] in selected]
        if len(rows)!=len(selected):raise ValueError('Missing requested selection indices')
    if a.limit:rows=rows[:a.limit]
    (a.output/'selection.json').write_text(json.dumps([{k:v for k,v in row.items() if not k.startswith('_')} for row in rows],indent=2))
    begin=time.perf_counter();gw=Greenwich.load(run_dir=a.run_dir or a.repo/'.local/cache/weights'/a.model,model=a.model,device='cuda')
    if not a.source_manifest:
        bind=load_bvh(source_dir/'soma_base_skel_minimal.bvh')
        assert bind['names'][:2]==['Root','Hips']
        names=bind['names'][1:];parents=bind['parents'][1:]-1
        bindR=bind['global_rot'][0,1:];bindP=bind['positions'][0,1:]
        offsets=np.zeros_like(bindP);offsets[0]=bindP[0]
        for j,parent in enumerate(parents):
            if parent>=0:offsets[j]=bindP[j]-bindP[parent]
        human=SkeletonSpec('bones_seed_soma_uniform_full77',parents,offsets.astype(np.float32),names)
        hd,_=ball_dof(len(names))
    targets=[]
    for robot in json.loads(a.robots.read_text(encoding='utf-8-sig')):
        spec,dof,rest,*_=build_from_mjcf(robot['xml'],robot['body']);keys=key_joints(spec)[0]
        sole=build_foot_sole_model(robot['xml'],spec,rest,['left_foot','right_foot'],keys[4:6])
        targets.append((robot,spec,dof,rest,sole))
    startup=time.perf_counter()-begin;ledger=[]
    for row in rows:
        compact_payload={'schema':np.array('greenwich.soma18.compact.v1'),'stem':np.array(row['stem']),'fps':np.array(a.fps,dtype=np.float32)}
        if a.source_manifest:
            clip=row['_clip'];take,dt_source=sample_clip(clip,a.fps,a.max_seconds)
            from greenwich_motion_sdk.motion import MotionClip
            MotionClip(clip.names,clip.parents,clip.rest_offsets_cm,clip.global_rotation[take],
                       clip.world_position_cm[take],clip.timestamps_s[take]-clip.timestamps_s[take[0]],clip.metadata).save(a.output/row['stem']/'source.npz')
            human=SkeletonSpec('sdk_'+row['source_format'],clip.parents,clip.rest_offsets_cm.astype(np.float32),clip.names)
            hd,_=ball_dof(len(clip.names))
            b={'names':clip.names,'parents':clip.parents,'positions':clip.world_position_cm,'dt':dt_source}
            root=clip.world_position_cm[take,0].copy();source_p=clip.world_position_cm[take].copy()
            footids=row['foot_indices'];bind_clear=np.array(row['foot_landmark_height_cm'])
        else:
            b=load_bvh(source_dir/(row['stem']+'.bvh'))
            clip=load_motion(source_dir/(row['stem']+'.bvh'),format='soma',bind_path=source_dir/'soma_base_skel_minimal.bvh')
            assert b['names']==bind['names']
            assert np.max(np.abs(b['global_rot'][:,0]-np.eye(3)))<1e-8
            assert np.max(np.abs(b['positions'][:,0]))<1e-8
            stride=round(1/b['dt']/a.fps)
            if stride<1 or abs(1/b['dt']/stride-a.fps)>.02:raise ValueError('FPS requires explicit resampling')
            take=np.arange(0,min(len(b['values']),round(a.max_seconds/b['dt'])),stride)
            root=b['positions'][take,1].copy();root[:,[0,2]]-=root[0,[0,2]]
            source_p=b['positions'][take].copy();source_p[:,:,[0,2]]-=b['positions'][0,1,[0,2]]
            footids=[b['names'].index(n) for n in ['LeftFoot','LeftToeBase','RightFoot','RightToeBase']]
            bind_clear=np.array([bind['positions'][0,footids[:2],1].min(),bind['positions'][0,footids[2:],1].min()])
        # Ground stays SOMA Y=0. Never estimate it from the minimum animated foot.
        source_foot=source_p[:,footids];source_clear=np.stack([source_foot[:,:2,1].min(1),source_foot[:,2:,1].min(1)],1)
        velocity=np.linalg.norm(np.diff(source_foot[:,:,[0,2]],axis=0),axis=-1)*a.fps
        # Coarse source contact proxy from ankle/toe landmarks (not sole meshes).
        source_contact=(source_clear<8.)
        source_contact[1:] &= np.stack([velocity[:,:2].min(1),velocity[:,2:].min(1)],1)<25.
        # Remove the anatomical landmark-to-sole offset using source bind pose.
        # This does not move the ground or rescale the source trajectory.
        source_clearance=np.maximum(source_clear-bind_clear,0.)
        gr=clip.global_rotation[take]
        gp=clip.world_position_cm[take]-clip.world_position_cm[take,:1]
        r6=c.matrix_to_rot6d(torch.tensor(gr,dtype=torch.float32))
        fk_error=float(np.linalg.norm(fk_pos(r6.numpy(),human)-gp,axis=-1).max())
        if fk_error>.001:raise ValueError(f'SOMA bind/FK mismatch: {fk_error} cm')
        input_description=('SDK '+row['source_format']+' full-body MotionClip; explicit source geometry, units and floor' if a.source_manifest else 'SOMA 77 anatomical joints directly encoded; identity scene Root removed')
        encode_spec,encode_dof=human,hd
        if a.input_representation=='smpl22-derived':
            from greenwich_motion_sdk._smpl_bind import SomaSmplRestAdapter
            encode_spec,encode_dof,encode_rest=gw.embodiment('human_smpl')
            # v2 offsets are native parent-frame vectors; identity rotations are NOT a T pose.
            # Calibrate each parent frame to the source bind's child-bone directions.
            adapter=SomaSmplRestAdapter(bind,encode_spec)
            adapted=adapter.convert(b,mode='pose_bias')
            r6=c.matrix_to_rot6d(torch.tensor(adapted['global_rotation'][take],dtype=torch.float32))
            bind_fk=fk_pos(c.matrix_to_rot6d(torch.tensor(adapter.alignment[None],dtype=torch.float32)).numpy(),encode_spec)[0]
            if not (bind_fk[15,1]>bind_fk[0,1]>max(bind_fk[7,1],bind_fk[8,1])):
                raise ValueError('SMPL bind is not upright in the common Y-up frame')
            gp=fk_pos(r6.numpy(),encode_spec)
            input_description='SMPL22 derived from same SOMA BVH; v2 native offsets calibrated to source bind directions; NOT native SMPL dataset'
        reach=float(np.linalg.norm(gp,axis=-1).max())
        pose=torch.cat([r6,torch.tensor(gp/reach,dtype=torch.float32)],-1).to('cuda')
        t=time.perf_counter()
        codes=torch.cat([gw.encode(pose[k:k+a.chunk],encode_spec,encode_dof) for k in range(0,len(take),a.chunk)])
        torch.cuda.synchronize();encode_s=time.perf_counter()-t
        for robot,spec,dof,rest,sole in targets:
            folder=a.output/row['stem']/robot['name']
            if not a.compact:folder.mkdir(parents=True,exist_ok=True)
            t=time.perf_counter();out=[gw.decode_full(codes[k:k+a.chunk],spec,dof,contact=True) for k in range(0,len(take),a.chunk)]
            raw=torch.cat([x[0] for x in out]);dp=torch.cat([x[1] for x in out]);contact=torch.cat([x[2] for x in out]);torch.cuda.synchronize();decode_s=time.perf_counter()-t
            t=time.perf_counter();dt=torch.tensor(dof,device='cuda',dtype=torch.float64);rs=torch.tensor(rest,device='cuda',dtype=torch.float64)
            q,_=c.fit_angles(raw.double(),spec,dt,rest=rs,method='global',lm_iters=12,soft_margin=1.,clamp=True)
            rr=c.rot6d_to_matrix(raw.double())[:,0]
            if a.contact_iters:
                # Symmetric short-window SO(3) smoothing; same timestamps, no lag.
                smooth=gaussian_filter1d(rr.detach().cpu().numpy(),sigma=a.fps*.04,axis=0,mode='nearest')
                u,_,v=np.linalg.svd(smooth);correction=np.tile(np.eye(3),(len(smooth),1,1));correction[:,2,2]=np.linalg.det(u@v)
                rr=torch.as_tensor(u@correction@v,device='cuda',dtype=torch.float64)
            rot,pos=c.fk_from_angles(q,spec,dt,rest=rs,root_R=rr)
            root_target=root.copy();root_target[:,1]=dp[:,0,0].detach().cpu().numpy()*100
            world=pos.detach().cpu().numpy()+root_target[:,None]
            torch.cuda.synchronize();projection_s=time.perf_counter()-t
            before=sole.sample(world,rot.detach().cpu().numpy())
            before_penetration=float(max(np.maximum(-v.lowest_y_cm,0).max() for v in before.values()))
            rawpos=fk_pos(raw.detach().cpu().numpy(),spec)+root_target[:,None]
            rawsole=sole.sample(rawpos,c.rot6d_to_matrix(raw).detach().cpu().numpy())
            raw_penetration=float(max(np.maximum(-v.lowest_y_cm,0).max() for v in rawsole.values()))
            refinement=None
            if a.contact_iters:
                q,rot,world_t,root_target,refinement=refine(q,root_target,rr,spec,dt,rs,sole,source_contact,a.fps,a.contact_iters,source_clearance=source_clearance,config=json.loads(a.contact_config.read_text()) if a.contact_config else None)
                world=world_t.detach().cpu().numpy()
            rotation=rot.detach().cpu().numpy();samples=sole.sample(world,rotation)
            heights=np.stack([v.lowest_y_cm for v in samples.values()],1)
            feet=np.stack([v.sole_point_cm for v in samples.values()],1)
            predicted=torch.sigmoid(contact[:,sole.joints]).detach().cpu().numpy()>.8
            edge=predicted[1:]&predicted[:-1]
            speeds=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0),axis=-1)*a.fps
            source_edge=source_contact[1:]&source_contact[:-1]
            lo,hi,limited=c._limits(dt,1.)
            limit_violation=float(torch.where(limited,torch.maximum(lo-q,q-hi).clamp_min(0),0).max().detach())
            if not np.isfinite(world).all() or limit_violation>1e-7:raise ValueError('Nonfinite or infeasible final joint trajectory')
            stats={'source':row['stem'],'robot':robot['name'],'model':a.model,'frames':len(take),'fps':a.fps,
                   'input':input_description,'input_representation':row['source_format'] if a.source_manifest else a.input_representation,
                   'source_joint_count':encode_spec.J,'source_fk_max_error_cm':fk_error,
                   'max_joint_limit_violation_rad':limit_violation,
                   'contact_refinement':refinement,'raw_rotation_fk_penetration_cm':raw_penetration,'before_contact_penetration_cm':before_penetration,
                   'root_height':'Prime decoder pos[:,root,0] metres + explicit contact residual; no render lift',
                   'horizontal_translation':'source XZ guide + contact residual; NOT model-predicted locomotion translation',
                   'ground':'canonical Y=0; explicit source floor; contact corrected in saved motion, no render-only lift',
                   'projection':'native joint limits + minimal contact residual; source support proxy, no standing recovery',
                   'root_height_cm':[float(root_target[:,1].min()),float(root_target[:,1].max())],
                   'source_root_height_cm':[float(root[:,1].min()),float(root[:,1].max())],
                   'max_penetration_cm':float(np.maximum(-heights,0).max()),
                   'sole_height_min_max_cm':np.stack([heights.min(0),heights.max(0)],1).tolist(),
                   'predicted_support_clearance_p95_cm':float(np.percentile(np.abs(heights[predicted]),95)) if predicted.any() else None,
                   'predicted_support_slip_p95_cm_s':float(np.percentile(speeds[edge],95)) if edge.any() else None,
                   'source_support_clearance_p95_cm':float(np.percentile(np.abs(heights[source_contact]),95)) if source_contact.any() else None,
                   'source_support_slip_p95_cm_s':float(np.percentile(speeds[source_edge],95)) if source_edge.any() else None,
                   'source_contact_scope':'ankle/toe proxy; <8 cm and <25 cm/s; used only during source support intervals',
                   'timing_s':{'startup_shared':startup,'encode_shared':encode_s,'decode':decode_s,'joint_projection':projection_s},'visual_review':'pending'}
            final_rot6d=c.matrix_to_rot6d(rot).detach().cpu().numpy()
            if a.compact:
                prefix=robot['name']+'__'
                compact_payload[prefix+'q']=q.detach().cpu().numpy().astype(np.float32)
                compact_payload[prefix+'root_rot6d']=final_rot6d[:,0].astype(np.float32)
                compact_payload[prefix+'root_t_cm']=np.asarray(root_target,dtype=np.float32)
                compact_payload[prefix+'joint_names']=np.asarray(spec.joint_names,dtype=str)
                compact_payload[prefix+'sole_height_cm']=heights.astype(np.float32)
                compact_payload[prefix+'model_contact']=predicted.astype(bool)
                brief={k:stats[k] for k in ('source','robot','frames','fps','max_joint_limit_violation_rad','max_penetration_cm')}
                brief['timing_s']=stats['timing_s'];brief['refine_s']=refinement['seconds'] if refinement else 0.
                brief['artifact']='motions/'+row['stem']+'.npz';ledger.append(brief)
                (a.output/'results.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(brief),flush=True)
                continue
            source_roles=row['source_role_indices'] if a.source_manifest else [b['names'].index(n) for n in ['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']]
            np.savez_compressed(folder/'motion.npz',constraint_joint_indices=np.array(key_joints(spec)[0][2:4]),q=q.detach().cpu().numpy(),rot6d=final_rot6d,raw_rot6d=raw.detach().cpu().numpy(),decoder_position=dp.detach().cpu().numpy(),root_t=root_target,world_position_cm=world,joint_names=np.array(spec.joint_names),fps=a.fps,sole_surface_height_cm=heights,source_root_cm=root,source_positions_cm=source_p,source_joint_names=np.array(b['names']),source_parents=b['parents'],source_contact_proxy=source_contact,model_contact=predicted,source_role_indices=np.array(source_roles),source_timestamps_s=clip.timestamps_s[take]-clip.timestamps_s[take[0]],source_foot_indices=np.array(footids))
            if a.skip_preview:
                stats['timing_s']['render']=0.0
                (folder/'report.json').write_text(json.dumps(stats,indent=2))
                ledger.append(stats)
                (a.output/'results.json').write_text(json.dumps(ledger,indent=2))
                print(json.dumps(stats),flush=True)
                continue
            t=time.perf_counter()
            positions,quats,meshes,*_=_sample_greenwich_head(folder/'motion.npz',robot['xml'],robot['body'],rotation_key='rot6d',world_position_key='world_position_cm',floor_y_cm=0.)
            matrices=Rotation.from_quat(quats.reshape(-1,4),scalar_first=True).as_matrix().reshape(*quats.shape[:-1],3,3)
            # Display separation only; source and robot retain metre scale and floor.
            source_view=source_p@YUP_TO_ZUP.T/100.;source_view[:,:,0]-=1.1
            positions[:,:,0]+=1.1
            robot_view=world@YUP_TO_ZUP.T/100.;robot_view[:,:,0]+=1.1
            scene=mj.MjSpec();scene.add_material(name='stage_mat',rgba=[.84,.80,.70,1])
            scene.worldbody.add_geom(name='stage_floor',type=mj.mjtGeom.mjGEOM_PLANE,size=[100,100,.1],material='stage_mat')
            scene.worldbody.add_light(pos=[3,-4,7],dir=[-.2,.3,-1],diffuse=[.45]*3)
            scene.worldbody.add_camera(name='camera',fovy=42)
            scene.attach(mj.MjSpec.from_file(robot['xml']),prefix='robot_',frame=scene.worldbody.add_frame())
            model=scene.compile();_configure_model(model);data=mj.MjData(model);mj.mj_forward(model,data)
            candidates=[i for i in range(model.ngeom) if model.geom_type[i]==mj.mjtGeom.mjGEOM_MESH]
            gids=[i for i in candidates if model.geom_contype[i]==0 and model.geom_conaffinity[i]==0] or candidates
            if len(gids)!=positions.shape[1]:raise ValueError('Visual native geometry mismatch')
            for i in range(model.ngeom):model.geom_rgba[i,3]=int(i in gids or mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,i)=='stage_floor')
            # Use world-space link centres with a mesh margin; imported source mesh
            # bounds can carry vendor scaling already baked into the native renderer.
            viewpoints=np.concatenate([source_view[:,1:],robot_view],axis=1)
            centres=(source_view[:,1]+robot_view[:,0])/2;centres[:,2]=.85
            centred=viewpoints-centres[:,None]
            low=centred.min(axis=(0,1))-.15;high=centred.max(axis=(0,1))+.15
            at=(low+high)/2;direction=np.array([.7,-1.,.25]);direction/=np.linalg.norm(direction)
            right=np.cross(-direction,[0.,0.,1.]);right/=np.linalg.norm(right)
            up=np.cross(right,-direction)
            corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])-at
            tan_y=np.tan(np.deg2rad(21.));tan_x=tan_y*1920/1080
            distance=np.max(corners@direction+np.maximum(np.abs(corners@right)/tan_x,np.abs(corners@up)/tan_y))*1.1
            cp=at+direction*distance
            model.vis.headlight.ambient[:]=.25;model.vis.headlight.diffuse[:]=.35
            camera=_set_fixed_camera(model,data,mj,'camera',cp+centres[0],at+centres[0]);renderer=mj.Renderer(model,height=1080,width=1920)
            writer=_writer(a.ffmpeg,folder/'preview.mp4',a.fps)
            try:
                for frame in range(len(take)):
                    data.geom_xpos[gids]=positions[frame];data.geom_xmat[gids]=matrices[frame].reshape(-1,9)
                    _set_fixed_camera(model,data,mj,'camera',cp+centres[frame],at+centres[frame])
                    renderer.update_scene(data,camera)
                    for axis in (0,1):
                        for tick in range(-12,13):
                            start=np.array([-12.,-12.,.002]);end=np.array([12.,12.,.002])
                            start[axis]=end[axis]=tick*.5
                            geom=renderer.scene.geoms[renderer.scene.ngeom]
                            mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array([.4,.38,.33,1.]))
                            mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,.002,start,end)
                            renderer.scene.ngeom+=1
                    for j,parent in enumerate(b['parents']):
                        if parent<(0 if a.source_manifest else 1):continue
                        geom=renderer.scene.geoms[renderer.scene.ngeom]
                        mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array([.1,.45,.95,1.]))
                        mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,.018,source_view[frame,parent],source_view[frame,j])
                        renderer.scene.ngeom+=1
                    im=renderer.render().copy()
                    cv2.putText(im,f'SOMA (blue skeleton) + PRIME {robot["name"].upper()} | {row["semantic_type"]}',(25,35),cv2.FONT_HERSHEY_SIMPLEX,.85,(20,20,20),2)
                    cv2.putText(im,f'Same scale / floor / timestamp | contact residual | {frame/a.fps:.2f}s',(25,70),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,30),2)
                    cv2.putText(im,f'ROOT {root_target[frame,1]:.1f} cm | SOLE L/R {heights[frame,0]:.1f}/{heights[frame,1]:.1f} cm',(25,1045),cv2.FONT_HERSHEY_SIMPLEX,.65,(120,30,30),1)
                    writer.stdin.write(im.tobytes())
            finally:_close_writer(writer);renderer.close()
            stats['timing_s']['render']=time.perf_counter()-t
            (folder/'report.json').write_text(json.dumps(stats,indent=2));ledger.append(stats)
            (a.output/'results.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(stats),flush=True)
        if a.compact:
            motions=a.output/'motions';motions.mkdir(exist_ok=True)
            np.savez_compressed(motions/(row['stem']+'.npz'),**compact_payload)


if __name__=='__main__':main()

