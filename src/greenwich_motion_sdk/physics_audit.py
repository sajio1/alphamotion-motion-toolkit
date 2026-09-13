"""Fast offline motion screening. Necessary conditions, never a hardware certificate.

Re-evaluate cached physical arrays with another threshold config without model
inference, rendering, or recomputing native kinematics. All coordinates are Y-up
SI inside this module. Contacts are inferred from actual sole geometry.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, QhullError
from scipy.signal import savgol_filter
from scipy.optimize import nnls


@dataclass
class Thresholds:
    contact_height_m: float = .01
    patch_depth_m: float = .005
    penetration_m: float = .005
    slip_m_s: float = .05
    friction_mu: float = .7
    force_residual_bodyweight: float = .05
    moment_residual_bodyweight_m: float = .03
    derivative_window_s: float = .17
    minimum_bad_duration_s: float = .10
    joint_limit_tolerance_deg: float = .5
    geometry_consistency_m: float = .002
    def validate(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('All thresholds must be finite and positive')


def derivative(x, fps, order=1, window_s=.17):
    n=min(len(x) if len(x)%2 else len(x)-1,max(5,int(round(window_s*fps))|1))
    if n<5:raise ValueError('At least five frames required for derivatives')
    return savgol_filter(x,n,3,deriv=order,delta=1/fps,axis=0,mode='interp')


def intervals(mask, fps, minimum=0.):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return [{'start_frame':int(a),'end_frame_exclusive':int(b),'start_s':a/fps,
             'end_s':b/fps,'duration_s':(b-a)/fps}
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if (b-a)/fps>=minimum]


def wrench_fit(points, com, force, moment, mass, mu):
    """Eight-ray inscribed Coulomb cone; no invented external root wrench."""
    points=np.asarray(points);com=np.asarray(com);force=np.asarray(force);moment=np.asarray(moment)
    weight=mass*9.81
    demand=np.r_[force/weight,moment/weight] # torque normalized by bodyweight * 1 metre
    if len(points)==0:return float(np.linalg.norm(demand[:3])),float(np.linalg.norm(demand[3:])),0.
    theta=np.arange(8)*np.pi/4
    rays=np.column_stack([mu*np.cos(theta),np.ones(8),mu*np.sin(theta)])
    columns=[]
    for point in points:
        columns.append(np.concatenate([rays,np.cross(point-com,rays)],axis=1).T)
    matrix=np.concatenate(columns,axis=1)
    coefficients,_=nnls(matrix,demand,maxiter=5000)
    residual=matrix@coefficients-demand
    return float(np.linalg.norm(residual[:3])),float(np.linalg.norm(residual[3:])),float(coefficients.sum())


def native_inertia_trajectory(robot,spec,qnames,q,rot,root,expected):
    """Reconstruct native link inertias; verify mapping against descriptor FK."""
    import mujoco as mj
    from alphamotion.embodiment.mjcf_build import build_merge
    model=mj.MjModel.from_xml_path(robot['xml']);data=mj.MjData(model)
    keep,frames=build_merge(model);base=keep[0];frame=frames[base]
    free=[j for j in range(model.njnt) if model.jnt_type[j]==mj.mjtJoint.mjJNT_FREE and model.jnt_bodyid[j]==base]
    if len(free)!=1:raise ValueError('Native adapter requires one floating root on descriptor base')
    adr=int(model.jnt_qposadr[free[0]])
    bindings=[]
    for j,names in enumerate(qnames):
        for k,name in enumerate(names):
            jid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name)
            if jid<0:raise ValueError('Missing native joint '+name)
            bindings.append((int(model.jnt_qposadr[jid]),j,k))
    def descendant(i):
        while i and i!=base:i=int(model.body_parentid[i])
        return i==base
    bodies=np.array([i for i in range(1,model.nbody) if descendant(i) and model.body_mass[i]>0])
    mass=model.body_mass[bodies].copy();inertia=model.body_inertia[bodies].copy()
    if mass.sum()<=0 or not np.isfinite(inertia).all():raise ValueError('Missing mass/inertia')
    # Canonical column XYZ -> native column XYZ: (x,y,z)->(z,x,y).
    basis=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    centers=[];inertial_rot=[];mapping_error=[];closure_error=[]
    for t in range(len(q)):
        data.qpos[:]=model.qpos0
        for address,j,k in bindings:data.qpos[address]=q[t,j,k]
        mj.mj_forward(model,data)
        delta=(basis@rot[t,0]@basis.T)@data.xmat[frame].reshape(3,3).T
        native_base=delta@data.xmat[base].reshape(3,3)
        data.qpos[adr:adr+3]=basis@root[t]
        data.qpos[adr+3:adr+7]=Rotation.from_matrix(native_base).as_quat(scalar_first=True)
        mj.mj_forward(model,data)
        centers.append(data.xipos[bodies]@basis)
        inertial_rot.append(np.einsum('ij,bjk->bik',basis.T,data.ximat[bodies].reshape(-1,3,3)))
        positions=data.xpos[np.asarray(keep)]@basis
        mapping_error.append(float(np.linalg.norm(positions-expected[t],axis=1).max()))
        eq=data.efc_type[:data.nefc]==mj.mjtConstraint.mjCNSTR_EQUALITY
        closure_error.append(float(np.abs(data.efc_pos[:data.nefc][eq]).max()) if eq.any() else 0.)
    bound_names={name for names in qnames for name in names}
    unbound=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt)
             if model.jnt_type[j]!=mj.mjtJoint.mjJNT_FREE and mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j) not in bound_names]
    return dict(link_com_m=np.asarray(centers),link_inertial_rotation=np.asarray(inertial_rot),
                link_mass_kg=mass,link_inertia_kg_m2=inertia,native_fk_error_m=np.asarray(mapping_error),
                native_equality_error=np.asarray(closure_error),unmapped_native_joints=np.asarray(unbound,dtype=str))


def extract(source,robot,output,stage='final'):
    import torch
    from alphamotion.engine import constraints as c
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine.spatial import key_joints
    from greenwich_umi_proof.sole_geometry import build_foot_sole_model
    start=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(source,allow_pickle=False) as z:stored={k:z[k] for k in z.files}
    spec,dof,rest,qnames,*_=build_from_mjcf(robot['xml'],robot['body'])
    if not np.array_equal(stored['joint_names'],spec.joint_names):raise ValueError('Robot/trace joint order mismatch')
    q=np.asarray(stored['q'],float);root=np.asarray(stored['root_t'],float)/100;fps=float(stored['fps'])
    if len(q)<5 or fps<=0 or not all(np.isfinite(a).all() for a in [q,root,stored['rot6d']]):raise ValueError('Invalid trace')
    r=c.rot6d_to_matrix(torch.tensor(stored['rot6d'],dtype=torch.float64)).numpy()
    if stage!='final':
        raw=torch.tensor(stored['raw_rot6d'],dtype=torch.float64)
        r=c.rot6d_to_matrix(raw).numpy()
        q,_=c.fit_angles(raw,spec,torch.tensor(dof,dtype=torch.float64),rest=torch.tensor(rest,dtype=torch.float64),method='global',lm_iters=12,soft_margin=1.,clamp=True)
        q=q.numpy()
        root=np.asarray(stored['raw_root_t'],float)/100
        if stage=='projected_model_height':root[:,1]=stored['decoder_position'][:,0,0]
    rr,pp=c.fk_from_angles(torch.tensor(q),spec,torch.tensor(dof,dtype=torch.float64),rest=torch.tensor(rest,dtype=torch.float64),root_R=torch.tensor(r[:,0]))
    pos=pp.numpy()/100+root[:,None];rotation=rr.numpy()
    sole=build_foot_sole_model(robot['xml'],spec,rest,['left_foot','right_foot'],key_joints(spec)[0][4:6])
    arrays=dict(q=q,root_m=root,fps=np.asarray(fps),rotation=rotation,world_position_m=pos,
                serialized_fk_error_m=np.linalg.norm(pos-stored['world_position_cm']/100,axis=-1).max(axis=1) if stage=='final' else np.zeros(len(q)),
                task_start_frame=np.asarray(stored.get('task_start_frame',0)))
    low,high,limited=c._limits_open(torch.tensor(dof,dtype=torch.float64),1.)
    arrays['joint_limit_excess_deg']=np.rad2deg(np.maximum(np.maximum(low.numpy()-q,q-high.numpy()),0)*limited.numpy()).max(axis=(1,2))
    for k,role in enumerate(sole.roles):
        foot=sole.feet[role];arrays[f'foot_{k}_vertices_m']=np.einsum('tij,vj->tvi',rotation[:,foot.joint],foot.vertices_local_cm/100)+pos[:,foot.joint,None]
    arrays.update(native_inertia_trajectory(robot,spec,qnames,q,rotation,root,pos))
    np.savez_compressed(output/'physics_cache.npz',**arrays)
    provenance={'schema':'greenwich-physics-cache-v1','stage':stage,'source':str(Path(source).resolve()),'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'robot':robot,'robot_xml_sha256':hashlib.sha256(Path(robot['xml']).read_bytes()).hexdigest(),'extract_seconds':time.perf_counter()-start,
                'ground':'Y=0, source coordinates preserved','contacts':'foot mesh near-ground vertices, inferred; no grasp/environment support',
                'payload':'not modeled','mass_parameters':'vendor MJCF; not hardware-identified'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return output/'physics_cache.npz'


def evaluate(cache,output,config=None):
    cfg=config or Thresholds();cfg.validate();start=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(cache,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    fps=float(a['fps']);n=len(a['q']);m=a['link_mass_kg'];mass=float(m.sum());p=a['link_com_m'];r=a['link_inertial_rotation']
    com=np.einsum('tbi,b->ti',p,m)/mass;acc=derivative(com,fps,2,cfg.derivative_window_s)
    vel=derivative(p,fps,1,cfg.derivative_window_s);vcom=derivative(com,fps,1,cfg.derivative_window_s)
    omega=np.zeros((n,len(m),3));edge=Rotation.from_matrix((r[1:]@r[:-1].transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(n-1,len(m),3)*fps
    omega[1:-1]=(edge[:-1]+edge[1:])/2;omega[0]=edge[0];omega[-1]=edge[-1]
    inertia=np.einsum('tbij,bj,tbkj->tbik',r,a['link_inertia_kg_m2'],r)
    momentum=(np.einsum('tbij,tbj->tbi',inertia,omega)+m[None,:,None]*np.cross(p-com[:,None],vel-vcom[:,None])).sum(axis=1)
    moment=derivative(momentum,fps,1,cfg.derivative_window_s);force=mass*(acc-np.array([0.,-9.81,0.]))
    feet=[a[f'foot_{k}_vertices_m'] for k in range(2)]
    heights=np.column_stack([v[:,:,1].min(axis=1) for v in feet]);contact=(heights<=cfg.contact_height_m)&(heights>=-cfg.penetration_m)
    slip=np.zeros((n,2));force_error=[];moment_error=[];normal_load=[]
    for k,v in enumerate(feet):
        velocity=np.gradient(v,1/fps,axis=0)
        for t in range(n):
            near=v[t,:,1]<=min(cfg.contact_height_m,heights[t,k]+cfg.patch_depth_m)
            if contact[t,k] and near.any():slip[t,k]=np.percentile(np.linalg.norm(velocity[t,near][:,[0,2]],axis=1),50)
    for t in range(n):
        patches=[]
        for k,v in enumerate(feet):
            if not contact[t,k]:continue
            points=v[t][v[t,:,1]<=min(cfg.contact_height_m,heights[t,k]+cfg.patch_depth_m)].copy()
            points[:,1]=0 # inferred floor contact, tolerance recorded in config
            points=np.unique(np.round(points,6),axis=0)
            try:points=points[ConvexHull(points[:,[0,2]]).vertices]
            except QhullError:pass
            patches.extend(points)
        f,mo,load=wrench_fit(np.asarray(patches),com[t],force[t],moment[t],mass,cfg.friction_mu)
        force_error.append(f);moment_error.append(mo);normal_load.append(load)
    force_error=np.asarray(force_error);moment_error=np.asarray(moment_error)
    # Derivative endpoints are uncertain; do not reject based on those alone.
    valid=np.ones(n,bool);edge_count=max(2,round(cfg.derivative_window_s*fps/2));valid[:edge_count]=False;valid[-edge_count:]=False
    flags={'floor_penetration':heights.min(axis=1)<-cfg.penetration_m,
           'support_slip':(slip>cfg.slip_m_s).any(axis=1)&valid,
           'contact_force_infeasible':(force_error>cfg.force_residual_bodyweight)&valid,
           'contact_moment_infeasible':(moment_error>cfg.moment_residual_bodyweight_m)&valid,
           'joint_limit':a['joint_limit_excess_deg']>cfg.joint_limit_tolerance_deg,
           'coordinate_or_mapping_error':np.maximum(a['native_fk_error_m'],a['serialized_fk_error_m'])>cfg.geometry_consistency_m}
    events={k:intervals(v,fps,cfg.minimum_bad_duration_s) for k,v in flags.items()}
    uncertain=bool(a['unmapped_native_joints'].size) or float(a['native_equality_error'].max())>.001
    failed=any(events.values())
    report={'schema':'greenwich-fast-physics-v1','status':'rejected' if failed else 'inconclusive' if uncertain else 'passes_necessary_screen',
            'dynamically_certified':False,'thresholds':asdict(cfg),'frames':n,'fps':fps,'duration_s':n/fps,
            'evaluation_seconds':time.perf_counter()-start,'mass_kg':mass,'events':events,
            'metrics':{'maximum_penetration_cm':float(max(0,-heights.min())*100),'no_valid_floor_contact_frames':int((~contact.any(axis=1)).sum()),
                       'both_feet_above_contact_band_frames':int((heights>cfg.contact_height_m).all(axis=1).sum()),
                       'root_height_range_cm':[float(a['root_m'][:,1].min()*100),float(a['root_m'][:,1].max()*100)] if 'root_m' in a else None,
                       'maximum_support_slip_cm_s':float(slip.max()*100),'force_residual_p95_bodyweight':float(np.percentile(force_error[valid],95)),
                       'moment_residual_p95_bodyweight_m':float(np.percentile(moment_error[valid],95)),
                       'native_fk_max_error_cm':float(a['native_fk_error_m'].max()*100),'native_equality_max_error':float(a['native_equality_error'].max())},
            'assumptions':['flat ground Y=0','only feet provide external contact; no hand support, payload or adhesion','Coulomb friction coefficient is configured, not measured','vendor masses and inertias'],
            'not_verified':['actuator torque, power and bandwidth','full joint inverse dynamics','self collision','controller tracking and disturbance recovery','payload/grasp feasibility'],
            'native_model_uncertain':uncertain,'unmapped_native_joints':a['unmapped_native_joints'].tolist(),
            'force_test_caveat':'Penetrated feet are excluded from valid contacts; force failure may be downstream of geometric failure, not an independent cause. Unmapped closed-chain links make inertial results approximate.',
            'interpretation':'Necessary centroidal force/moment and geometric checks. No-contact flight is allowed if ballistic. A pass is not proof of physical execution.'}
    np.savez_compressed(output/'physics_timeseries.npz',time_s=np.arange(n)/fps,com_m=com,com_acceleration_m_s2=acc,
                        angular_momentum_kg_m2_s=momentum,required_force_n=force,required_moment_nm=moment,
                        foot_height_m=heights,contact=contact,slip_m_s=slip,force_residual_bodyweight=force_error,
                        moment_residual_bodyweight_m=moment_error,normal_load_bodyweight=normal_load,**flags)
    (output/'physics_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    columns=np.column_stack([np.arange(n)/fps,heights*100,slip*100,force_error,moment_error,*[v.astype(int) for v in flags.values()]])
    np.savetxt(output/'physics_frames.csv',columns,delimiter=',',header='time_s,left_sole_cm,right_sole_cm,left_slip_cm_s,right_slip_cm_s,force_residual_BW,moment_residual_BWm,'+','.join(flags),comments='')
    lines=['# Motion physics screen', '',f"Status: **{report['status']}**. This is not a dynamics certificate.",'',f"Evaluated {n} frames in {report['evaluation_seconds']:.3f} s.",'']
    for key,ranges in events.items():
        lines.append(f'- {key}: '+('; '.join(f"{v['start_s']:.2f}–{v['end_s']:.2f} s" for v in ranges) or 'no sustained violation'))
    lines+=['','## Limits','',*['- '+v for v in report['not_verified']], '', 'Native model uncertain: '+str(uncertain)]
    (output/'PHYSICS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    plot_timeseries(output/'physics_timeseries.npz',output/'physics_plot.png',cfg)
    return report


def plot_timeseries(source,output,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(source) as z:
        t=z['time_s'];fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True)
        for k,side in enumerate(['Left','Right']):ax[0].plot(t,z['foot_height_m'][:,k]*100,label=side)
        ax[0].axhline(0,color='k',lw=.7);ax[0].axhline(-cfg.penetration_m*100,color='r',ls='--');ax[0].set_ylabel('Lowest sole (cm)');ax[0].legend()
        ax[1].plot(t,z['force_residual_bodyweight']);ax[1].axhline(cfg.force_residual_bodyweight,color='r',ls='--');ax[1].set_ylabel('Force residual / BW')
        ax[2].plot(t,z['moment_residual_bodyweight_m']);ax[2].axhline(cfg.moment_residual_bodyweight_m,color='r',ls='--');ax[2].set_ylabel('Moment residual / BW (m)');ax[2].set_xlabel('Video time (s), includes preparation')
        for a in ax:a.grid(alpha=.2)
        fig.suptitle('Fast necessary-condition screen | assumed foot contacts, no payload or actuator check')
        fig.tight_layout();fig.savefig(output,dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--robots',type=Path);p.add_argument('--robot');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',type=Path);p.add_argument('--pipeline',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--stage',choices=['final','projected_model_height','projected_supplied_height'],default='final')
    a=p.parse_args()
    if a.cache:cache=a.cache
    else:
        if not all([a.source,a.robots,a.robot,a.repo,a.pipeline]):p.error('Extraction requires source, robots, robot, repo and pipeline')
        import sys
        sys.path[:0]=[str(a.repo/'src'),str(a.pipeline/'src')]
        robot=next(r for r in json.loads(a.robots.read_text(encoding='utf-8-sig')) if r['name']==a.robot)
        cache=extract(a.source,robot,a.output,a.stage)
    cfg=Thresholds(**json.loads(a.config.read_text())) if a.config else Thresholds()
    report=evaluate(cache,a.output,cfg);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
