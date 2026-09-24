"""Robot-independent, source-contact-guided residual projection in centimetres.

No gait synthesis or standing pose. Optimises articulated angles and root XYZ
around the decoded motion; source contact is a geometric proxy, not force truth.
"""
import time
import numpy as np


def refine(q, root, root_R, spec, dof, rest, sole, support, fps, iterations=100, source_clearance=None, config=None, flat_support=None, coordination=None, support_surface=None, support_mask=None):
    import torch
    from scipy.spatial import ConvexHull
    from scipy.ndimage import gaussian_filter1d
    from alphamotion.engine import constraints as c
    cfg={'stable_margin_s':.1,'swing_threshold_cm':3.,'stable_patch_weight':20.,'swing_clearance_weight':12.,'flight_root_weight':5.,
         'analytic_vertical_init':0.,'flight_vertical_init':1.,'freeze_arm_refinement':0.,'lower_temporal_reference':0.,'swing_clearance_floor':0.,'ground_terms_scale':1.,
         'penetration_allowance_cm':0.,'root_temporal_reference':0.,'root_acceleration_weight':10.,'root_velocity_weight':0.,'root_reference_weight':.012,'slip_allowance_cm_s':0.,
         'root_input_smoothing_s':0.,'root_correction_smoothing_s':0.,
         'contact_allowance_cm':0.,'root_learning_rate':.25,'contact_height_release_cm':0.,
         'penetration_weight':80.,'contact_weight':12.,'pose_reference_weight':4.,'support_slip_weight':30.,'support_drift_weight':0.,'final_lr_scale':1.,'upper_temporal_scale':1.,'lower_temporal_scale':1.,'release_support':0.,'conservative_support':0.,'contact_patch_slip':0.}
    if config:
        unknown=set(config)-set(cfg)
        if unknown:raise ValueError(f'Unknown contact settings: {unknown}')
        cfg.update(config)
    if any(not np.isfinite(v) or v<0 for v in cfg.values()):raise ValueError('Contact settings must be finite and nonnegative')
    start=time.perf_counter();device=q.device;dtype=q.dtype
    if len(q)<3:raise ValueError('Contact sequence requires at least three frames')
    q0=q.detach().clone();r0=torch.as_tensor(root,device=device,dtype=dtype)
    if cfg['root_input_smoothing_s']:
        r0=torch.as_tensor(gaussian_filter1d(np.asarray(root),sigma=fps*cfg['root_input_smoothing_s'],axis=0,mode='nearest'),device=device,dtype=dtype)
    # One shared ablation knob. Keep original mean normalization so lower-body
    # gradients are identical, rather than renormalizing after reweighting.
    from alphamotion.engine.spatial import key_joints
    from greenwich_motion_sdk.body_coordination import upper_subtree
    keys=key_joints(spec)[0]
    upper=upper_subtree(spec.parents,keys[2:4],keys[4:6])
    temporal_weight=torch.ones((1,len(upper),1),device=device,dtype=dtype)
    temporal_weight[:,torch.as_tensor(~upper,device=device)]=cfg['lower_temporal_scale']
    temporal_weight[:,torch.as_tensor(upper,device=device)]=cfg['upper_temporal_scale']
    qq=torch.as_tensor(gaussian_filter1d(q0.cpu().numpy(),sigma=fps*.05,axis=0,mode='nearest'),device=device,dtype=dtype)
    temporal_reference=torch.zeros_like(qq)
    if cfg['lower_temporal_reference']:
        # Preserve the low-pass native model motion rather than driving every
        # valid leg velocity toward zero. Upper-body objective is unchanged.
        temporal_reference[:,torch.as_tensor(~upper,device=device)]=qq[:,torch.as_tensor(~upper,device=device)]
    arm_names=('shoulder','elbow','wrist','hand')
    frozen_arm=np.asarray([any(token in name.lower() for token in arm_names) for name in spec.joint_names],dtype=bool)
    frozen_arm=torch.as_tensor(frozen_arm,device=device)
    if cfg['freeze_arm_refinement']:
        qq[:,frozen_arm]=q0[:,frozen_arm]
    qq.requires_grad_()
    dr=torch.zeros_like(r0,requires_grad=True)
    def root_correction():
        if not cfg['root_correction_smoothing_s']:return dr
        from greenwich_motion_sdk.contact_phases import smooth_residual
        return smooth_residual(dr,fps*cfg['root_correction_smoothing_s'])
    mask=torch.as_tensor(support,device=device,dtype=dtype)
    edges=mask[1:]*mask[:-1]
    # Erode support intervals so heel strike and toe-off can roll freely.
    from scipy.ndimage import minimum_filter1d
    stable=torch.as_tensor(minimum_filter1d(np.asarray(support,dtype=float),size=2*round(fps*cfg['stable_margin_s'])+1,axis=0,mode='constant'),device=device,dtype=dtype)
    if cfg['release_support']:
        if flat_support is None: raise ValueError('Released support requires calibrated source flat_support')
        stable=stable*torch.as_tensor(flat_support,device=device,dtype=dtype)
        # A rolling foot's sole center moves even when its toe is touching.
        if not cfg['contact_patch_slip']: edges=stable[1:]*stable[:-1]
    clearance=torch.as_tensor(np.zeros_like(support,dtype=float) if source_clearance is None else source_clearance,device=device,dtype=dtype)
    proximity=torch.zeros_like(mask)
    if cfg['contact_height_release_cm']:
        if source_clearance is None:raise ValueError('Vertical proximity requires source clearance')
        from greenwich_motion_sdk.contact_phases import near_ground_confidence
        proximity=torch.as_tensor(near_ground_confidence(source_clearance,cfg['contact_height_release_cm']),device=device,dtype=dtype)*(1-mask)
    swing=(clearance>cfg['swing_threshold_cm']).to(dtype)*(1-mask)
    if support_surface is None:support_surface=sole
    if support_mask is None:support_mask=support
    support_mask=torch.as_tensor(support_mask,device=device,dtype=dtype)
    if support_mask.shape!=(len(q),len(support_surface.feet)):
        raise ValueError('Support labels must match endpoint surface roles')
    support_edges=support_mask[1:]*support_mask[:-1]
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
        return heights,points,patch_y,clouds
    support_verts=[]
    for endpoint in support_surface.feet.values():
        v=endpoint.vertices_local_cm
        if len(v)>=4:
            try:v=v[ConvexHull(v).vertices]
            except Exception:pass
        support_verts.append(torch.as_tensor(v,device=device,dtype=dtype))
    def support_geometry(rot,world):
        clouds=[torch.einsum('tij,vj->tvi',rot[:,endpoint.joint],v)+world[:,endpoint.joint,None]
                for endpoint,v in zip(support_surface.feet.values(),support_verts)]
        heights=torch.stack([cloud[:,:,1].min(1).values for cloud in clouds],1)
        points=torch.stack([world[:,endpoint.joint]+torch.einsum('tij,j->ti',rot[:,endpoint.joint],
            torch.as_tensor(endpoint.sole_point_local_cm,device=device,dtype=dtype))
            for endpoint in support_surface.feet.values()],1)
        return heights,points,clouds
    lo,hi,limited=c._limits(dof,1.)
    hand_support=support_mask[:,2:].amax(1) if support_mask.shape[1]>2 else torch.zeros(len(q),device=device,dtype=dtype)
    flight=swing.prod(1)*(1-hand_support)
    with torch.no_grad():
        initial_rot,initial_pos=c.fk_from_angles(qq,spec,dof,rest=rest,root_R=root_R)
        initial_world=initial_pos+r0[:,None]
        initial_height,_,_,_=geometry(initial_rot,initial_world)
        initial_support_height,_,_=support_geometry(initial_rot,initial_world)
        # A geometry projection removes the large embodiment-dependent vertical
        # offset before gradient refinement. This changes only root Y and uses
        # the native sole mesh; it is not a standing-height preset.
        ground_lift=torch.relu(.05-initial_support_height.min(1).values)
        if cfg['analytic_vertical_init'] and cfg['ground_terms_scale']:
            dr[:,1].copy_(ground_lift)
            # Solve the one shared vertical root degree of freedom from the
            # currently load-bearing semantic endpoints.  Feet govern normal
            # motion; hands govern inversions; mixed transitions use all
            # declared supports instead of an embodiment-specific root preset.
            active=support_mask.sum(1)
            support_shift=-(initial_support_height*support_mask).sum(1)/active.clamp_min(1)
            dr[:,1].copy_(torch.where(active>0,support_shift,dr[:,1]))
        # Lift the whole predicted configuration during flight instead of
        # satisfying airborne feet by folding both legs under a low pelvis.
        flight_lift=(clearance-initial_height).mean(1)
        flight_root_y=r0[:,1]+flight_lift
        if cfg['ground_terms_scale'] and cfg['flight_vertical_init']:
            dr[:,1].copy_(torch.where(flight.bool(),flight_lift,dr[:,1]))
    optimizer=torch.optim.Adam([{'params':[qq],'lr':.008},{'params':[dr],'lr':cfg['root_learning_rate']}])
    if coordination is not None:
        optimizer.add_param_group({'params':[coordination.variable],'lr':coordination.learning_rate})
        if coordination.follow_contact_reference:
            with torch.no_grad():coordination.variable.copy_(qq[:,coordination.mask])
    def combined_angles():
        if coordination is None:return qq
        full=qq.clone();full[:,coordination.mask]=coordination.variable
        return full
    history=[]
    for step in range(iterations):
        # Optional late settling for the moving material-contact set; no extra
        # iterations and no change to the objective or support classification.
        progress=max(0.,2*step/max(iterations-1,1)-1)
        scale=1.+progress*(cfg['final_lr_scale']-1.)
        optimizer.param_groups[0]['lr']=.008*scale
        optimizer.param_groups[1]['lr']=cfg['root_learning_rate']*scale
        optimizer.zero_grad()
        if coordination is not None and coordination.follow_contact_reference:
            # The serial arm solver regularizes relative to contact-refined
            # angles. Follow that evolving reference without back-propagating
            # arm objectives into it; do not regularize against raw IK jitter.
            with torch.no_grad():coordination.reference.copy_(qq.detach())
        full=combined_angles()
        rot,pos=c.fk_from_angles(full,spec,dof,rest=rest,root_R=root_R)
        correction=root_correction()
        world=pos+r0[:,None]+correction[:,None];height,points,patch_y,clouds=geometry(rot,world)
        support_height,support_points,support_clouds=support_geometry(rot,world)
        penetration=torch.relu(.05-cfg['penetration_allowance_cm']-support_height).square().mean()
        contact=(support_mask*torch.relu(support_height.abs()-cfg['contact_allowance_cm']).square()).sum()/support_mask.sum().clamp_min(1)
        # Near-floor rolling is not stationary support. Bound excessive height
        # softly while allowing the source's small toe/heel clearance. No XZ
        # penalty, exact foot trajectory target, or whole-sole flattening here.
        contact=contact+(proximity*torch.relu(height-clearance.clamp_min(0)-cfg['contact_allowance_cm']).square()).sum()/proximity.sum().clamp_min(1)
        slip=((points[1:,:,[0,2]]-points[:-1,:,[0,2]]).square()*edges[:,:,None]).sum()/edges.sum().clamp_min(1)
        if cfg['contact_patch_slip']:
            # Track material points on the actual lowest surface, not the
            # moving sole center during heel/toe roll. Require the same point
            # near the underside on both frames. Stop gradients through the
            # contact-set choice; the optimizer cannot differentiate the mask.
            from greenwich_motion_sdk.contact_phases import material_patch_slip
            slip=material_patch_slip(clouds,height,edges,allowance_cm=cfg['slip_allowance_cm_s']/fps)
        hand_slip=torch.zeros((),device=device,dtype=dtype)
        if support_mask.shape[1]>2:
            hand_edges=support_edges[:,2:]
            hand_delta=support_points[1:,2:,[0,2]]-support_points[:-1,2:,[0,2]]
            hand_slip=(hand_delta.square()*hand_edges[:,:,None]).sum()/hand_edges.sum().clamp_min(1)
        residual=qq-q0
        ground_loss=cfg['penetration_weight']*penetration+cfg['contact_weight']*contact+cfg['support_slip_weight']*(slip+hand_slip)
        if cfg['support_drift_weight']:
            from greenwich_motion_sdk.contact_phases import material_patch_drift
            ground_loss=ground_loss+cfg['support_drift_weight']*material_patch_drift(clouds,height,mask)
        ground_loss=ground_loss+cfg['stable_patch_weight']*(stable[:,:,None]*patch_y.square()).sum()/stable.sum().clamp_min(1)
        # Source clearance guides airborne feet; no robot root-height preset.
        clearance_error=height-clearance
        if cfg['swing_clearance_floor']:
            # Clearance is a lower bound, never a downward target for a foot
            # already safely in swing. Do not copy exact human foot heights.
            clearance_error=torch.relu(-clearance_error)
        ground_loss=ground_loss+cfg['swing_clearance_weight']*(swing*clearance_error.square()).sum()/swing.sum().clamp_min(1)
        ground_loss=ground_loss+cfg['flight_root_weight']*(flight*(world[:,0,1]-flight_root_y).square()).sum()/flight.sum().clamp_min(1)
        loss=cfg['ground_terms_scale']*ground_loss+cfg['pose_reference_weight']*residual.square().mean()+cfg['root_reference_weight']*correction.square().mean()
        # Default: penalise actual motion to suppress framewise branch switches.
        # Opt-in lower reference follows a short low-pass projected model trace.
        rate=fps/30.
        temporal=qq-temporal_reference
        loss=loss+1500*rate**2*((temporal[1:]-temporal[:-1]).square()*temporal_weight).mean()+100000*rate**4*((temporal[2:]-2*temporal[1:-1]+temporal[:-2]).square()*temporal_weight).mean()
        # Regularize the correction, not the model's intended acceleration:
        # smoothing absolute root height flattens take-off and landing motion.
        root_temporal=correction if cfg['root_temporal_reference'] else r0+correction
        loss=loss+cfg['root_acceleration_weight']*rate**4*(root_temporal[2:]-2*root_temporal[1:-1]+root_temporal[:-2]).square().mean()
        loss=loss+cfg['root_velocity_weight']*rate**2*(root_temporal[1:]-root_temporal[:-1]).square().mean()
        arm_gradient=None
        if coordination is not None:
            arm_loss=coordination.objective(full,fk=(rot,pos))
            # Request only arm derivatives. Arm objectives never accumulate
            # gradients into the contact-owned trunk, root, or legs.
            arm_gradient=torch.autograd.grad(arm_loss,coordination.variable,retain_graph=True)[0]
        loss.backward()
        if coordination is not None:coordination.variable.grad=arm_gradient
        optimizer.step()
        with torch.no_grad():
            qq.copy_(torch.where(limited,torch.minimum(torch.maximum(qq,lo),hi),qq))
            if cfg['freeze_arm_refinement']:
                qq[:,frozen_arm]=q0[:,frozen_arm]
            if coordination is not None:
                v=coordination.variable
                v.copy_(torch.where(coordination.limited,v.maximum(coordination.lo).minimum(coordination.hi),v))
        if step%50==0 or step==iterations-1:
            history.append({'iteration':step+1,'loss':float(loss.detach()),'penetration_cm':float(torch.relu(-support_height).max().detach())})
    with torch.no_grad():
        if coordination is not None:qq[:,coordination.mask]=coordination.variable
        rot,pos=c.fk_from_angles(qq,spec,dof,rest=rest,root_R=root_R)
        rr=r0+root_correction();world=pos+rr[:,None]
        # Root Y is a shared rigid translation, so finish with its exact
        # least-squares solution over the source-declared load-bearing
        # endpoints.  This prevents the soft swing/pose objectives from
        # trading a visibly floating hand for a slightly smaller total loss.
        final_support_height,_,_=support_geometry(rot,world)
        hand_active=support_mask[:,2:].amax(1)>0 if support_mask.shape[1]>2 else torch.zeros(len(q),device=device,dtype=torch.bool)
        if support_mask.shape[1]>2:
            # A source hand label is eligible for a hard root projection only
            # once the generated hand surface is actually below the feet.  If
            # the decoded pose has not entered inversion, translating root
            # alone would bury the legs and cannot create a valid handstand.
            hand_is_lowest=final_support_height[:,2:].min(1).values<=final_support_height[:,:2].min(1).values
            hand_active=hand_active&hand_is_lowest
        selected=torch.zeros_like(support_mask)
        selected[:,:2]=support_mask[:,:2]
        if support_mask.shape[1]>2:
            selected[hand_active,:2]=0
            selected[hand_active,2:]=1
        active=selected.sum(1)>0
        floor=torch.full_like(final_support_height,float('inf'))
        lowest=torch.where(selected.bool(),final_support_height,floor).min(1).values
        vertical_projection=torch.where(active,-lowest,torch.zeros_like(lowest))
        # Feather support-family transitions; a one-frame 20 cm root step is
        # never a valid consequence of changing the active endpoint.  Then
        # lift only as much as needed to keep every endpoint surface above the
        # floor during the blend.
        projected=gaussian_filter1d(vertical_projection.detach().cpu().numpy(),sigma=max(fps*.08,.5),mode='nearest')
        vertical_projection=torch.as_tensor(projected,device=device,dtype=dtype)
        projected_min=final_support_height.min(1).values+vertical_projection
        vertical_projection=vertical_projection+torch.relu(-.05-projected_min)
        rr[:,1]+=vertical_projection
        world=pos+rr[:,None]
    actual_root_delta=rr-torch.as_tensor(root,device=device,dtype=dtype)
    return qq.detach(),rot,world,rr.cpu().numpy(),{'algorithm':'support_aware_contact_residual_v4','config':cfg,'support_roles':list(support_surface.roles),'support_frames':support_mask.sum(0).tolist(),'hard_projected_hand_frames':int(hand_active.sum().detach()),'final_vertical_projection_max_cm':float(vertical_projection.abs().max().detach()),'stable_support_frames':stable.sum(0).tolist(),'airborne_foot_frames':swing.sum(0).tolist(),'iterations':iterations,'seconds':time.perf_counter()-start,'history':history,'max_root_adjustment_cm':float(actual_root_delta.norm(dim=-1).max().detach()),'joint_adjustment_p95_deg':float(torch.quantile((qq-q0).abs().flatten(),.95).detach()*180/np.pi),'frozen_arm_joints':[name for name,flag in zip(spec.joint_names,frozen_arm.cpu().tolist()) if flag and cfg['freeze_arm_refinement']]}
