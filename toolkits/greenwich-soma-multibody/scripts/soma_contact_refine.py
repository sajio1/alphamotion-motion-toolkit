"""Robot-independent, source-contact-guided residual projection in centimetres.

No gait synthesis or standing pose. Optimises articulated angles and root XYZ
around the decoded motion; source contact is a geometric proxy, not force truth.
"""
import time
import numpy as np


def refine(q, root, root_R, spec, dof, rest, sole, support, fps, iterations=300, source_clearance=None, config=None):
    import torch
    from scipy.spatial import ConvexHull
    from scipy.ndimage import gaussian_filter1d
    from alphamotion.engine import constraints as c
    cfg={'stable_margin_s':.1,'swing_threshold_cm':3.,'stable_patch_weight':20.,'swing_clearance_weight':12.,'flight_root_weight':5.,
         'analytic_vertical_init':0.,'freeze_arm_refinement':0.}
    if config:
        unknown=set(config)-set(cfg)
        if unknown:raise ValueError(f'Unknown contact settings: {unknown}')
        cfg.update(config)
    if any(not np.isfinite(v) or v<0 for v in cfg.values()):raise ValueError('Contact settings must be finite and nonnegative')
    start=time.perf_counter();device=q.device;dtype=q.dtype
    q0=q.detach().clone();r0=torch.as_tensor(root,device=device,dtype=dtype)
    qq=torch.as_tensor(gaussian_filter1d(q0.cpu().numpy(),sigma=fps*.05,axis=0,mode='nearest'),device=device,dtype=dtype)
    arm_names=('shoulder','elbow','wrist','hand')
    frozen_arm=np.asarray([any(token in name.lower() for token in arm_names) for name in spec.joint_names],dtype=bool)
    frozen_arm=torch.as_tensor(frozen_arm,device=device)
    if cfg['freeze_arm_refinement']:
        qq[:,frozen_arm]=q0[:,frozen_arm]
    qq.requires_grad_()
    dr=torch.zeros_like(r0,requires_grad=True)
    mask=torch.as_tensor(support,device=device,dtype=dtype)
    edges=mask[1:]*mask[:-1]
    # Erode support intervals so heel strike and toe-off can roll freely.
    from scipy.ndimage import minimum_filter1d
    stable=torch.as_tensor(minimum_filter1d(np.asarray(support,dtype=float),size=2*round(fps*cfg['stable_margin_s'])+1,axis=0,mode='constant'),device=device,dtype=dtype)
    clearance=torch.as_tensor(np.zeros_like(support,dtype=float) if source_clearance is None else source_clearance,device=device,dtype=dtype)
    swing=(clearance>cfg['swing_threshold_cm']).to(dtype)*(1-mask)
    from greenwich_umi_proof.sole_geometry import _neutral_descriptor_fk
    _,neutral_R=_neutral_descriptor_fk(spec,rest.detach().cpu().numpy())
    patches=[]
    verts=[]
    for f in sole.feet.values():
        v=f.vertices_local_cm
        neutral=v@neutral_R[f.joint].T
        # Actual underside patches, calibrated from each native foot mesh.
        bottom=neutral[:,1]<neutral[:,1].min()+.5
        along=neutral[bottom,2];indices=np.flatnonzero(bottom)
        patch=np.stack([v[indices[along<=np.quantile(along,.25)]].mean(0),v[indices[along>=np.quantile(along,.75)]].mean(0)])
        patches.append(torch.tensor(patch,device=device,dtype=dtype))
        v=v[ConvexHull(v).vertices]
        verts.append(torch.tensor(v,device=device,dtype=dtype))
    def geometry(rot,world):
        clouds=[torch.einsum('tij,vj->tvi',rot[:,f.joint],v)+world[:,f.joint,None] for f,v in zip(sole.feet.values(),verts)]
        heights=torch.stack([x[:,:,1].min(1).values for x in clouds],1)
        points=torch.stack([world[:,f.joint]+torch.einsum('tij,j->ti',rot[:,f.joint],torch.as_tensor(f.sole_point_local_cm,device=device,dtype=dtype)) for f in sole.feet.values()],1)
        patch_y=torch.stack([(torch.einsum('tij,vj->tvi',rot[:,f.joint],p)+world[:,f.joint,None])[:,:,1] for f,p in zip(sole.feet.values(),patches)],1)
        return heights,points,patch_y
    lo,hi,limited=c._limits(dof,1.)
    flight=swing.prod(1)
    with torch.no_grad():
        initial_rot,initial_pos=c.fk_from_angles(qq,spec,dof,rest=rest,root_R=root_R)
        initial_height,_,_=geometry(initial_rot,initial_pos+r0[:,None])
        # A geometry projection removes the large embodiment-dependent vertical
        # offset before gradient refinement. This changes only root Y and uses
        # the native sole mesh; it is not a standing-height preset.
        ground_lift=torch.relu(.05-initial_height.min(1).values)
        if cfg['analytic_vertical_init']:
            dr[:,1].copy_(ground_lift)
        # Lift the whole predicted configuration during flight instead of
        # satisfying airborne feet by folding both legs under a low pelvis.
        flight_lift=(clearance-initial_height).mean(1)
        flight_root_y=r0[:,1]+flight_lift
        dr[:,1].copy_(torch.where(flight.bool(),flight_lift,dr[:,1]))
    optimizer=torch.optim.Adam([{'params':[qq],'lr':.008},{'params':[dr],'lr':.25}])
    history=[]
    for step in range(iterations):
        optimizer.zero_grad()
        rot,pos=c.fk_from_angles(qq,spec,dof,rest=rest,root_R=root_R)
        world=pos+r0[:,None]+dr[:,None];height,points,patch_y=geometry(rot,world)
        penetration=torch.relu(.05-height).square().mean()
        contact=(mask*height.square()).sum()/mask.sum().clamp_min(1)
        slip=((points[1:,:,[0,2]]-points[:-1,:,[0,2]]).square()*edges[:,:,None]).sum()/edges.sum().clamp_min(1)
        residual=qq-q0
        loss=80*penetration+12*contact+30*slip+4*residual.square().mean()+.012*dr.square().mean()
        loss=loss+cfg['stable_patch_weight']*(stable[:,:,None]*patch_y.square()).sum()/stable.sum().clamp_min(1)
        # Source clearance guides airborne feet; no robot root-height preset.
        loss=loss+cfg['swing_clearance_weight']*(swing*(height-clearance).square()).sum()/swing.sum().clamp_min(1)
        loss=loss+cfg['flight_root_weight']*(flight*(world[:,0,1]-flight_root_y).square()).sum()/flight.sum().clamp_min(1)
        # Penalise actual joint acceleration, not merely the correction: a smooth
        # residual alone preserves abrupt framewise codec/IK branch switches.
        rate=fps/30.
        loss=loss+1500*rate**2*(qq[1:]-qq[:-1]).square().mean()+100000*rate**4*(qq[2:]-2*qq[1:-1]+qq[:-2]).square().mean()
        root_actual=r0+dr
        loss=loss+10*rate**4*(root_actual[2:]-2*root_actual[1:-1]+root_actual[:-2]).square().mean()
        loss.backward();optimizer.step()
        with torch.no_grad():
            qq.copy_(torch.where(limited,torch.minimum(torch.maximum(qq,lo),hi),qq))
            if cfg['freeze_arm_refinement']:
                qq[:,frozen_arm]=q0[:,frozen_arm]
        if step%50==0 or step==iterations-1:
            history.append({'iteration':step+1,'loss':float(loss.detach()),'penetration_cm':float(torch.relu(-height).max().detach())})
    with torch.no_grad():
        rot,pos=c.fk_from_angles(qq,spec,dof,rest=rest,root_R=root_R)
        rr=r0+dr;world=pos+rr[:,None]
    return qq.detach(),rot,world,rr.cpu().numpy(),{'algorithm':'contact_residual_v3','config':cfg,'stable_support_frames':stable.sum(0).tolist(),'airborne_foot_frames':swing.sum(0).tolist(),'iterations':iterations,'seconds':time.perf_counter()-start,'history':history,'max_root_adjustment_cm':float(dr.norm(dim=-1).max().detach()),'joint_adjustment_p95_deg':float(torch.quantile((qq-q0).abs().flatten(),.95).detach()*180/np.pi),'frozen_arm_joints':[name for name,flag in zip(spec.joint_names,frozen_arm.cpu().tolist()) if flag]}
