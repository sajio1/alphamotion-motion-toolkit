"""Body-relative upper-body stage; no absolute human EE goals.

By default changes only the arms, preserving saved trunk/head/root/leg motion.
The public SDK calls the arm-only 100-step path after contact refinement.
Joint upper-subtree optimization remains internal historical research code.
"""
import time
import numpy as np


def upper_subtree(parents, wrists, feet):
    def ancestors(j):
        result=set()
        while j>=0: result.add(int(j)); j=int(parents[j])
        return result
    protected=ancestors(feet[0]) | ancestors(feet[1])
    path=(ancestors(wrists[0]) | ancestors(wrists[1]))-protected
    mask=np.array([bool(ancestors(j) & path) for j in range(len(parents))])
    return mask


def torso_frame(left, right, pelvis):
    """Anatomical frame: lateral, up, forward; NumPy or torch tensors."""
    if isinstance(left,np.ndarray):
        norm=lambda v:v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)
        cross=lambda a,b:np.cross(a,b)
        stack=lambda xs:np.stack(xs,-1)
    else:
        import torch
        norm=lambda v:torch.nn.functional.normalize(v,dim=-1,eps=1e-8)
        cross=lambda a,b:torch.linalg.cross(a,b,dim=-1)
        stack=lambda xs:torch.stack(xs,-1)
    up=norm((left+right)/2-pelvis)
    lateral=left-right; lateral=norm(lateral-(lateral*up).sum(-1,keepdims=True)*up)
    return stack([lateral,up,norm(cross(lateral,up))])


def direction_diagnostics(actual, reference):
    """Same-timestamp unit-vector audit; fast source gestures are not glitches.

    Arrays end in XYZ. This is an anatomical direction diagnostic, not EE
    centimetre accuracy, force feasibility, or a naturalness classifier.
    """
    a,b=np.asarray(actual,float),np.asarray(reference,float)
    if a.shape!=b.shape or a.ndim<2 or a.shape[-1]!=3 or not np.isfinite([a,b]).all():
        raise ValueError('Matching finite direction arrays required')
    def unit(v):
        n=np.linalg.norm(v,axis=-1,keepdims=True)
        if np.any(n<1e-8):raise ValueError('Degenerate direction')
        return v/n
    a,b=unit(a),unit(b)
    angle=lambda x,y:np.rad2deg(np.arccos(np.clip((x*y).sum(-1),-1,1)))
    error=angle(a,b)
    da,db=angle(a[1:],a[:-1]),angle(b[1:],b[:-1])
    excess=np.maximum(0,da-db)
    return {'direction_error_p95_deg':float(np.percentile(error,95)),
            'direction_error_max_deg':float(error.max()),
            'actual_step_max_deg':float(da.max()) if da.size else 0.,
            'source_step_max_deg':float(db.max()) if db.size else 0.,
            'extra_step_p95_deg':float(np.percentile(excess,95)) if excess.size else 0.,
            'extra_step_max_deg':float(excess.max()) if excess.size else 0.}


def refine_coordination(source, source_names, q, baseline_R, spec, dof, rest, wrists, feet,
                        *, fps, iterations=100, learning_rate=.03, relation_tolerance=.15, palm_frames=None,
                        optimize_trunk=False, prepare_only=False):
    """Preserve source arm/torso relations at robot scale with optional trunk assistance.

    Source indices are anatomical interface metadata. No clip-name rules, human
    absolute XYZ, posture presets or hand contact/force labels are introduced.
    Near-body wrist relations are soft intervals measured in arm lengths.
    """
    import torch
    from alphamotion.engine import constraints as c
    if fps<=0 or iterations<1 or learning_rate<=0 or relation_tolerance<0:
        raise ValueError('Invalid coordination settings')
    start=time.perf_counter();names=list(source_names);parents=np.asarray(spec.parents)
    def sj(name):
        if name not in names: raise ValueError('Source anatomy mapping required: '+name)
        return names.index(name)
    # Native elbow/knee semantics are adapter metadata, not robot identity.
    def chain_landmarks(distal, term):
        chain=[];j=int(distal)
        while j>=0:chain.append(j);j=int(parents[j])
        matches=[j for j in chain if term in spec.joint_names[j].lower()]
        if not matches: raise ValueError('Native anatomical landmark required: '+term)
        return matches[0],chain
    elbows=[];shoulders=[];knees=[]
    upper=upper_subtree(parents,wrists,feet)
    from .body_coordination_helpers import arm_branches, repair_short_excursions
    exclusive=arm_branches(parents,wrists)
    if not isinstance(optimize_trunk,bool):raise ValueError('optimize_trunk must be boolean')
    if not optimize_trunk:upper=exclusive.copy()
    for w,f in zip(wrists,feet):
        elbow,chain=chain_landmarks(w,'elbow');elbows.append(elbow)
        shoulders.append([j for j in chain if exclusive[j]][-1])
        knees.append(chain_landmarks(f,'knee')[0])
    root=int(np.flatnonzero(parents<0)[0])
    heads=[j for j,n in enumerate(spec.joint_names) if 'head' in n.lower()]
    if heads:
        head=heads[-1];head_anchor_semantics='native_head_landmark'
    else:
        # Some valid humanoids have a fixed/no-head embodiment.  Use the
        # deepest common ancestor of the two wrist chains as the upper-torso
        # anchor instead of inventing a robot-specific joint-name alias.
        def ancestry(j):
            result=[]
            while j>=0:result.append(int(j));j=int(parents[j])
            return result
        left=ancestry(wrists[0]);common=set(left)&set(ancestry(wrists[1]))
        head=next((j for j in left if j in common),root)
        head_anchor_semantics='bilateral_arm_common_ancestor_for_fixed_or_absent_head'
    src=torch.as_tensor(source,device=q.device,dtype=q.dtype)
    if src.shape[0]!=q.shape[0] or not torch.isfinite(src).all():raise ValueError('Invalid source timeline')
    ss=[sj('LeftArm'),sj('RightArm')];se=[sj('LeftForeArm'),sj('RightForeArm')];sw=[sj('LeftHand'),sj('RightHand')]
    # SOMA's Shin is the knee; Leg is the proximal thigh. Never infer this
    # mapping from similarly named SMPL/other skeleton joints.
    sk=[sj('LeftShin'),sj('RightShin')]
    sf=torso_frame(src[:,ss[0]],src[:,ss[1]],src[:,sj('Hips')])
    unit=lambda v:torch.nn.functional.normalize(v,dim=-1)
    source_dirs=torch.stack([unit(src[:,se]-src[:,ss]),unit(src[:,sw]-src[:,se])],2)
    local_dirs=torch.einsum('tji,tsbj->tsbi',sf,source_dirs).detach()
    # The wrist motor frame is not the anatomical palm. Reuse static robot
    # interface calibration; source MCP landmarks define distal/normal axes.
    if palm_frames is None:raise ValueError('Explicit native palm frames required for coordination')
    from alphamotion.embodiment.mjcf_build import AX
    local_palm=[];source_palm=[]
    for side,swrist in zip(('Left','Right'),sw):
        axes=palm_frames[side.lower()+'_wrist']
        distal=np.asarray(axes['distal'],float);normal=np.asarray(axes['normal'],float)
        distal=distal/np.linalg.norm(distal);normal=normal-distal*np.dot(distal,normal);normal=normal/np.linalg.norm(normal)
        if not np.isfinite([distal,normal]).all():raise ValueError('Degenerate robot palm axes')
        local_palm.append(np.stack([np.asarray(AX)@distal,np.asarray(AX)@normal],1))
        d=unit(src[:,sj(side+'HandMiddle1')]-src[:,swrist])
        radial=unit(src[:,sj(side+'HandIndex1')]-src[:,sj(side+'HandPinky1')])
        n=unit(torch.linalg.cross(radial,d,dim=-1))*(-1 if side=='Left' else 1)
        source_palm.append(torch.stack([d,n],-1))
    local_palm=torch.as_tensor(np.stack(local_palm),device=q.device,dtype=q.dtype)
    source_palm=torch.stack(source_palm,1)
    desired_palm=(sf[:,None].transpose(-1,-2)@source_palm).detach()
    source_lengths=(torch.linalg.vector_norm(src[:,se]-src[:,ss],dim=-1)+torch.linalg.vector_norm(src[:,sw]-src[:,se],dim=-1)).median(0).values
    _,bp=c.fk_from_angles(q,spec,dof,rest=rest,root_R=baseline_R[:,root])
    lengths=(torch.linalg.vector_norm(bp[:,elbows]-bp[:,shoulders],dim=-1)+torch.linalg.vector_norm(bp[:,wrists]-bp[:,elbows],dim=-1)).median(0).values.detach()
    def anchors(p,shoulder_ids,knee_ids,root_id,head_id):
        chest=(p[:,shoulder_ids[0]]+p[:,shoulder_ids[1]])/2
        return torch.stack([p[:,head_id],(chest+p[:,root_id])/2,p[:,root_id],p[:,knee_ids[0]],p[:,knee_ids[1]]],1)
    sa=anchors(src,ss,sk,sj('Hips'),sj('Head'))
    relative=(src[:,sw,None]-sa[:,None])/source_lengths[None,:,None,None]
    local_relative=torch.einsum('tji,tsaj->tsai',sf,relative).detach()
    proximity=(1-torch.linalg.vector_norm(relative,dim=-1)/.6).clamp(0,1).square().detach()
    _,active=c._slot_axes(dof);mask=torch.as_tensor(upper,device=q.device)[:,None]&active
    groups=[]
    for wrist in wrists:
        group=np.zeros(len(parents),bool);j=int(wrist)
        while j>=0 and exclusive[j]:group[j]=True;j=int(parents[j])
        groups.append(group)
    seed,events=repair_short_excursions(q.detach().cpu().numpy(),fps,groups)
    seed=torch.as_tensor(seed,device=q.device,dtype=q.dtype)
    variable=torch.nn.Parameter(seed[:,mask].clone());optim=torch.optim.Adam([variable],lr=learning_rate)
    lo,hi,limited=c._limits(dof,1.);rate=fps/30.;history=[]
    def terms(full,diagnostics=False,fk=None):
        R,p=fk if fk is not None else c.fk_from_angles(full,spec,dof,rest=rest,root_R=baseline_R[:,root])
        frame=torso_frame(p[:,shoulders[0]],p[:,shoulders[1]],p[:,root])
        dirs=torch.stack([unit(p[:,elbows]-p[:,shoulders]),unit(p[:,wrists]-p[:,elbows])],2)
        local=torch.einsum('tji,tsbj->tsbi',frame,dirs)
        direction=(1-(local*local_dirs).sum(-1).clamp(-1,1)).mean()
        a=anchors(p,shoulders,knees,root,head)
        rel=torch.einsum('tji,tsaj->tsai',frame,(p[:,wrists,None]-a[:,None])/lengths[None,:,None,None])
        relation=(torch.relu(torch.linalg.vector_norm(rel-local_relative,dim=-1)-relation_tolerance).square()*proximity).sum()/proximity.sum().clamp_min(1)
        torso=(frame-sf).square().mean()
        actual_palm=frame[:,None].transpose(-1,-2)@R[:,wrists]@local_palm
        palm_error=(1-(actual_palm*desired_palm).sum(-2).clamp(-1,1)).mean(0)
        palm=.5*palm_error[:,0].mean()+.25*palm_error[:,1].mean()
        # Penalize variation of the anatomical error, not variation of the
        # gesture. A source whip/punch may be fast without being an IK glitch.
        semantic_error=torch.cat([(local-local_dirs).flatten(1),
                                  (actual_palm-desired_palm).flatten(1),
                                  (frame-sf).flatten(1)],1)
        jitter=(semantic_error[1:]-semantic_error[:-1]).square().mean() if len(q)>1 else semantic_error.sum()*0
        if diagnostics:
            numpy=lambda t:t.detach().cpu().numpy()
            return {'arm_segments':direction_diagnostics(numpy(local),numpy(local_dirs)),
                    'palms':direction_diagnostics(numpy(actual_palm.transpose(-1,-2)),numpy(desired_palm.transpose(-1,-2))),
                    'torso_axes':direction_diagnostics(numpy(frame.transpose(-1,-2)),numpy(sf.transpose(-1,-2)))}
        return direction,relation,torso,palm,jitter
    def objective(full,fk=None):
        direction,relation,torso,palm,jitter=terms(full,fk=fk)
        delta=variable-seed[:,mask]
        # Smooth the correction, not the original gesture: healthy timing and
        # details are not penalized merely for moving fast.
        velocity=(delta[1:]-delta[:-1]).square().mean() if len(q)>1 else delta.sum()*0
        accel=(delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean() if len(q)>2 else delta.sum()*0
        # A generous shared guard addresses branch jumps left by native IK;
        # unlike a low global smoothness cost it is zero on ordinary motion.
        step_excess=torch.relu((variable[1:]-variable[:-1]).abs()-np.deg2rad(720.)/fps).square().mean() if len(q)>1 else delta.sum()*0
        return direction+2*relation+.25*torso+palm+2*rate**2*jitter+.01*delta.square().mean()+.08*rate**2*velocity+.8*rate**4*accel+20*step_excess
    before=[float(t.detach()) for t in terms(q)]
    if prepare_only:
        if optimize_trunk:raise ValueError('Partitioned joint solve must not optimize trunk from arm loss')
        from types import SimpleNamespace
        return SimpleNamespace(variable=variable,mask=mask,objective=objective,terms=terms,
            lo=lo[mask],hi=hi[mask],limited=limited[mask],learning_rate=learning_rate,
            before_terms=before,events=events,palm_frames=palm_frames,upper=upper,
            reference=seed,follow_contact_reference=False)
    for step in range(iterations):
        optim.zero_grad();full=q.clone();full[:,mask]=variable
        loss=objective(full)
        loss.backward();optim.step()
        with torch.no_grad():variable.copy_(torch.where(limited[mask],variable.maximum(lo[mask]).minimum(hi[mask]),variable))
        if step==0 or (step+1)%25==0 or step==iterations-1:
            history.append({'iteration':step+1,'loss':float(loss.detach())})
    result=q.clone();result[:,mask]=variable.detach()
    return result,upper,dict(iterations=iterations,solve_seconds=time.perf_counter()-start,
        before_terms=before,after_terms=[float(t.detach()) for t in terms(result)],history=history,
        term_names=['arm_direction','near_body_relation','torso_frame','palm_axes','semantic_error_temporal'],
        native_palm_frames=palm_frames,relation_tolerance_arm_lengths=relation_tolerance,
        anatomical_audit_before=terms(q,True),anatomical_audit_after=terms(result,True),
        target='source body-relative anatomical directions and loose near-body wrist relations; no absolute EE targets',
        optimize_trunk=optimize_trunk,
        optimized_parameters=('arms and upper trunk subtree jointly; saved locomotion protected in this experiment' if optimize_trunk else 'arms only; baseline trunk/head/root/lower-body trajectories retained frame by frame'),
        source_mapping='SOMA anatomical names; other adapters require explicit equivalent mapping',
        seed_repair_events=events,branch_speed_guard_deg_s=720.,
        head_anchor_joint=spec.joint_names[head],head_anchor_semantics=head_anchor_semantics,
        visual_review='pending',physical_validation='not performed')
