"""Fast compact-artifact sole penetration/slip screen using authoritative FK.

No controller, dynamics certificate or manipulation harness. Support is inferred
from source phase and geometry, never from measured low foot velocity.
"""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np
from .compact import load_compact,rot6d_matrix,native_visual_frame
from ._bvh import load_bvh
from .physics_audit import intervals

def source_support_phase(landmark, fps, height_cm=8., speed_cm_s=25.):
    """Independent source proxy; never gate support using target foot speed."""
    source_speed=np.zeros(len(landmark))
    source_speed[1:]=np.linalg.norm(np.diff(landmark[:,:,[0,2]],axis=0),axis=-1).min(1)*fps
    low=landmark[:,:,1].min(1)<height_cm
    return low & (source_speed<speed_cm_s), low, source_speed

def audit(manifest,robot,descriptor,output,slip_cm_s=5.,penetration_cm=.5,bad_duration_s=.1,
          *,progress_every=1,support_config=None,kimodo_metrics=False,result_sink=None,fixed_series_sink=None):
    import torch,mujoco as mj
    from scipy.ndimage import minimum_filter1d
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine import constraints as c
    from alphamotion.engine.spatial import key_joints
    from alphamotion.embodiment.mjcf_build import AX,build_merge
    from greenwich_umi_proof.sole_geometry import build_foot_sole_model
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
    desc=json.loads(gzip.decompress(Path(descriptor).read_bytes()))
    if not np.allclose(np.array(desc['zup_to_yup']).reshape(3,3),AX):raise ValueError('Viewer coordinate basis disagrees with official descriptor AX')
    spec,dof,rest,qnames,*_=build_from_mjcf(robot['xml'],robot['body'])
    sole=build_foot_sole_model(robot['xml'],spec,rest,['left_foot','right_foot'],key_joints(spec)[0][4:6])
    native=mj.MjModel.from_xml_path(robot['xml']);data=mj.MjData(native);keep,frames=build_merge(native)
    results=[]
    records=manifest['clips'] if isinstance(manifest,dict) else json.loads(Path(manifest).read_text(encoding='utf-8'))['clips']
    for row in records:
        motion=row.get('_compact_motion')
        if motion is None:motion=load_compact(row['archive'],member=row['member'],robot=robot['name'])
        fps=motion['fps'];T=len(motion['q'])
        q=torch.as_tensor(motion['q'],dtype=torch.float64);root_R=rot6d_matrix(motion['root_rot6d'])
        R,p=c.fk_from_angles(q,spec,torch.as_tensor(dof),rest=torch.as_tensor(rest),root_R=torch.as_tensor(root_R))
        rotation=R.numpy();world=p.numpy()+motion['root_t_cm'][:,None]
        if row.get('source_canonical'):
            from .motion import MotionClip
            clip=MotionClip.load(row['source_canonical'])
            delta=np.diff(clip.timestamps_s)
            if len(clip.timestamps_s)!=T or not np.allclose(delta,delta.mean(),atol=1e-6,rtol=1e-4) or abs(1/delta.mean()-fps)>.02:
                raise ValueError(f'Source/target timeline mismatch: {row["clip_id"]}')
            source=clip.world_position_cm;names=clip.names
        else:
            bvh=row.get('_source_bvh')
            if bvh is None:bvh=load_bvh(row['source_bvh'])
            stride=round(1/bvh['dt']/fps);take=np.arange(T)*stride
            if stride<1 or abs(1/bvh['dt']/stride-fps)>.02 or take[-1]>=len(bvh['values']):
                raise ValueError(f'Source/target timeline mismatch: {row["clip_id"]}')
            source=bvh['positions'][take].copy();source[:,:,[0,2]]-=bvh['positions'][0,1,[0,2]]
            names=bvh['names']
        feet={};series=[];reconstructed_heights=[];robot_toes=[];source_toes=[]
        for side,(role,foot) in enumerate(sole.feet.items()):
            ids=(row['source_foot_indices'][side*2:side*2+2] if row.get('source_foot_indices') is not None else
                 [names.index(x) for x in (("Left" if side==0 else "Right")+'Foot',("Left" if side==0 else "Right")+'ToeBase')])
            landmark=source[:,ids]
            # Match the generation proxy, including its independent source-speed
            # test. Height alone incorrectly treats low airborne feet as stance.
            phase,coarse_phase,source_speed=source_support_phase(landmark,fps)
            # Source phase is independent of target speed, so a sliding foot is
            # not silently classified as airborne. Erode to exclude strike/off.
            stable=minimum_filter1d(phase.astype(np.uint8),size=2*round(.1*fps)+1,mode='constant')>0
            vertices=np.einsum('tij,vj->tvi',rotation[:,foot.joint],foot.vertices_local_cm)+world[:,foot.joint,None]
            heights=vertices[:,:,1].min(1)
            reconstructed_heights.append(heights)
            vel=np.linalg.norm(np.diff(vertices[:,:,[0,2]],axis=0),axis=-1)*fps
            near=(vertices[:,:,1]<=1)&(vertices[:,:,1]>=-penetration_cm)
            near&=vertices[:,:,1]<=heights[:,None]+.5
            speed=np.full(T,np.nan)
            for t in range(1,T):
                mask=near[t]&near[t-1]
                if stable[t] and stable[t-1] and mask.any():speed[t]=np.median(vel[t-1,mask])
            # A stricter diagnostic requires quiet source landmarks and a flat
            # native underside. It excludes obvious roll, but cannot establish
            # measured load-bearing contact or certify excluded frames.
            from greenwich_umi_proof.sole_geometry import _neutral_descriptor_fk
            _,neutral_R=_neutral_descriptor_fk(spec,np.asarray(rest))
            v=foot.vertices_local_cm;neutral=v@neutral_R[foot.joint].T
            bottom=np.flatnonzero(neutral[:,1]<neutral[:,1].min()+.5)
            along=neutral[bottom,2]
            patch=np.stack([v[bottom[along<=np.quantile(along,.25)]].mean(0),
                            v[bottom[along>=np.quantile(along,.75)]].mean(0)])
            patch_world=np.einsum('tij,vj->tvi',rotation[:,foot.joint],patch)+world[:,foot.joint,None]
            if kimodo_metrics:
                # Native robots have no anatomical ToeBase joint. The front
                # quarter of the neutral sole is the declared robot toe proxy.
                robot_toes.append(patch_world[:,1]);source_toes.append(landmark[:,1])
            tilt=np.degrees(np.arctan2(np.abs(patch_world[:,1,1]-patch_world[:,0,1]),
                np.linalg.norm(patch_world[:,1,[0,2]]-patch_world[:,0,[0,2]],axis=1)))
            quiet_source=minimum_filter1d((coarse_phase&(source_speed<10.)).astype(np.uint8),
                size=2*round(.1*fps)+1,mode='constant')>0
            flat=quiet_source & (tilt<5.) & (patch_world[:,:,1].max(1)<=1.)
            confirmed=np.isfinite(speed)&flat
            flat_events=intervals(confirmed&(np.nan_to_num(speed)>slip_cm_s),fps,bad_duration_s)
            drift=[]
            reference=np.einsum('tij,j->ti',rotation[:,foot.joint],foot.sole_point_local_cm)+world[:,foot.joint]
            for event in flat_events:
                a,b=event['start_frame'],event['end_frame_exclusive']
                xy=reference[a:b][:,[0,2]]
                event['sole_reference_net_displacement_cm']=float(np.linalg.norm(xy[-1]-xy[0]))
                event['sole_reference_path_length_cm']=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
                event['median_material_speed_cm_s']=float(np.median(speed[a:b]))
            for episode in intervals(stable&(heights<=1)&(heights>=-penetration_cm),fps,.1):
                a,b=episode['start_frame'],episode['end_frame_exclusive']
                drift.append(float(np.linalg.norm(reference[a:b][:,[0,2]]-reference[a,[0,2]],axis=1).max()))
            slip_events=intervals(np.nan_to_num(speed)>slip_cm_s,fps,bad_duration_s)
            pen_events=intervals(heights < -penetration_cm,fps,bad_duration_s)
            lost_events=intervals(stable&(heights>3),fps,bad_duration_s)
            feet[role]={'minimum_sole_height_cm':float(heights.min()),'max_penetration_cm':float(np.maximum(-heights,0).max()),
                'support_slip_p95_cm_s':float(np.nanpercentile(speed,95)) if np.isfinite(speed).any() else None,
                'support_slip_max_cm_s':float(np.nanmax(speed)) if np.isfinite(speed).any() else None,
                'evaluated_support_frames':int(np.isfinite(speed).sum()),
                'source_slow_flat_support_frames':int(confirmed.sum()),
                'source_slow_flat_slip_p95_cm_s':float(np.percentile(speed[confirmed],95)) if confirmed.any() else None,
                'source_slow_flat_slip_events':flat_events,
                'source_height_only_candidate_frames':int(coarse_phase.sum()),
                'source_height_and_speed_candidate_frames':int(phase.sum()),
                'max_stable_episode_reference_drift_cm':max(drift) if drift else None,
                'slip_events':slip_events,'penetration_events':pen_events,'source_support_lost_events':lost_events}
            series.extend((heights,speed,stable.astype(float)))
            if support_config is not None:
                from .fixed_support import measure_fixed_support
                fixed,fixed_speed,fixed_evaluated=measure_fixed_support(landmark,vertices,rotation[:,foot.joint],
                    fps,support_config,slip_cm_s,penetration_cm,bad_duration_s)
                feet[role].update(fixed)
                if fixed_series_sink is not None:fixed_series_sink(row['clip_id'],role,fps,fixed_speed,fixed_evaluated)
                if output is not None:
                    np.savetxt(output/(row['clip_id']+'__'+role+'__fixed.csv'),
                        np.column_stack([np.arange(T)/fps,fixed_speed,fixed_evaluated]),delimiter=',',
                        header='time_s,fixed_patch_speed_cm_s,evaluated',comments='')
        # Native FK cross-check against saved-angle descriptor FK, rather than
        # comparing two renderers that may share the same wrong world basis.
        errors=[]
        for t in np.unique(np.linspace(0,T-1,min(T,5)).astype(int)):
            data.qpos[:]=native.qpos0
            for j,bindings in enumerate(qnames):
                for k,name in enumerate(bindings):
                    jid=mj.mj_name2id(native,mj.mjtObj.mjOBJ_JOINT,name);data.qpos[native.jnt_qposadr[jid]]=motion['q'][t,j,k]
            mj.mj_forward(native,data)
            delta=root_R[t]@AX@data.xmat[frames[keep[0]]].reshape(3,3).T
            realized=motion['root_t_cm'][t]+(data.xpos[keep]-data.xpos[keep[0]])@delta.T*100
            errors.append(float(np.linalg.norm(realized-world[t],axis=1).max()))
        def facing(pos,left,right):
            vector=pos[:,right]-pos[:,left];return np.cross(np.array([0.,1,0]),vector)[:,[0,2]]
        headings=None
        if all(n in names for n in ('LeftShoulder','RightShoulder')) and all(n in spec.joint_names for n in ('left_shoulder_pitch_link','right_shoulder_pitch_link')):
            sf=facing(source,names.index('LeftShoulder'),names.index('RightShoulder'))
            tf=facing(world,spec.joint_names.index('left_shoulder_pitch_link'),spec.joint_names.index('right_shoulder_pitch_link'))
            cosine=(sf*tf).sum(1)/np.maximum(np.linalg.norm(sf,axis=1)*np.linalg.norm(tf,axis=1),1e-8)
            headings=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        failures=[]
        for role,f in feet.items():
            for tag in ('slip_events','penetration_events'):
                if f[tag]:failures.append(role+':'+tag)
        uncertain=any(f['evaluated_support_frames']==0 for f in feet.values())
        result={'clip_id':row['clip_id'],'video':row.get('video'),'variant':row.get('variant'),
            'iterations':row.get('iterations'),'source_clip_id':row.get('source_clip_id',row['clip_id']),
            'frames':T,'fps':fps,'feet':feet,
            'status':'fail' if failures else ('insufficient_support' if uncertain else 'pass_screen'),
            'slip_status':'fail' if any(f['slip_events'] for f in feet.values()) else ('insufficient_support' if uncertain else 'pass_screen'),
            'penetration_status':'fail' if any(f['penetration_events'] for f in feet.values()) else 'pass_screen',
            'failures':failures,'native_joint_fk_max_error_cm':max(errors),
            'initial_torso_heading_difference_deg':float(headings[0]) if headings is not None else None,'torso_heading_difference_p95_deg':float(np.percentile(headings,95)) if headings is not None else None,
            'saved_vs_reconstructed_sole_max_error_cm':float(np.max(np.abs(np.stack(reconstructed_heights,axis=1)-motion['sole_height_cm'])))}
        if support_config is not None:
            result['fixed_support_status']='drift_flag' if any(f['fixed_patch_slip_events'] for f in feet.values()) else (
                'no_drift_flag' if any(f['fixed_patch_evaluated_frames'] for f in feet.values()) else 'no_confirmed_fixed_support')
        if kimodo_metrics:
            from .kimodo_foot_skate import toe_metrics
            result['kimodo']={}
            for actor,toes in (('robot',robot_toes),('source',source_toes)):
                metrics,velocity,contact,skating=toe_metrics(np.stack(toes,axis=1)/100.,fps)
                result['kimodo'][actor]=metrics
                if output is not None:
                    np.savetxt(output/(row['clip_id']+'__kimodo_'+actor+'.csv'),
                        np.column_stack([np.arange(T-1)/fps,velocity,contact,skating]),delimiter=',',
                        header='time_s,left_speed_m_s,right_speed_m_s,left_contact,right_contact,left_skating,right_skating',comments='')
        results.append(result)
        if result_sink is not None:result_sink(result)
        if output is not None:
            np.savetxt(output/(row['clip_id']+'.csv'),np.column_stack([np.arange(T)/fps,*series,headings if headings is not None else np.full(T,np.nan)]),delimiter=',',
                       header='time_s,left_sole_cm,left_slip_cm_s,left_source_stable,right_sole_cm,right_slip_cm_s,right_source_stable,torso_heading_difference_deg',comments='')
        if len(results)%progress_every==0:
            if kimodo_metrics:
                print(json.dumps({'evaluated':len(results),'kimodo_robot_clips_with_skating_frames':sum(r['kimodo']['robot']['skating_toe_frame_pairs']>0 for r in results)}),flush=True)
            elif support_config is not None:
                from collections import Counter
                print(json.dumps({'evaluated':len(results),'fixed_support_counts':dict(Counter(r['fixed_support_status'] for r in results))}),flush=True)
            else: print(json.dumps({'evaluated':len(results),'slip_fail':sum(r['slip_status']=='fail' for r in results),
                'slip_pass':sum(r['slip_status']=='pass_screen' for r in results),
                'insufficient_support':sum(r['slip_status']=='insufficient_support' for r in results),
                'last_clip':row['clip_id']}),flush=True)
    report={'schema':'greenwich.locomotion.sole-screen.v2','thresholds':{'slip_cm_s':slip_cm_s,'penetration_cm':penetration_cm,'bad_duration_s':bad_duration_s,
            'source_support_height_cm':8.,'source_support_speed_cm_s':25.,'diagnostic_source_speed_cm_s':10.,'diagnostic_flat_tilt_deg':5.},
        'interpretation':'Necessary geometry/stance-slip checks only; no dynamic feasibility certificate. Reference-point drift includes foot roll; patch speed is the slip gate.',
        'coordinate_basis':AX.tolist(),'results':results}
    if support_config is not None:
        from dataclasses import asdict
        report['fixed_support_config']=asdict(support_config)
        report['fixed_support_interpretation']='Conservative source-stationary persistent patch diagnostics; partial sole support allowed. Engineering tolerances are not yet human-calibrated; excluded frames are not passes.'
    if kimodo_metrics:
        from .kimodo_foot_skate import REFERENCE
        report['kimodo_protocol']={'reference':REFERENCE,'height_m':.05,'speed_m_s':.2,
            'velocity':'3D norm, adjacent frames, no smoothing','contact':'toe height < 0.05 m at both frames',
            'minimum_duration_s':None,'clip_acceptance_threshold':None,
            'robot_landmark_adapter':'centroid of front-quarter neutral sole underside mesh, transformed by native FK',
            'source_landmarks':['LeftToeBase','RightToeBase'],
            'note':'Official toe-metric formulas with declared robot mesh landmark adapter. Not a native anatomical-toe benchmark. No source-static gate, roll exclusion or local persistence filter.'}
    if output is not None:(output/'locomotion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def annotate_review(directory,report):
    """Expose verdicts and seekable failure intervals beside each HTML video."""
    import html,re
    from .render_review import package_review
    directory=Path(directory)
    if not isinstance(report,dict): report=json.loads(Path(report).read_text(encoding='utf-8'))
    rows=report['results'];cfg=report['thresholds']
    page=(directory/'index.html').read_text(encoding='utf-8')
    page=re.sub(r'<!--AUDIT:.*?<!--END_AUDIT-->', '',page,flags=re.S)
    passed=sum(r['status']=='pass_screen' for r in rows)
    summary=f'<!--AUDIT:summary--><section style="padding:12px;border:1px solid #888"><b>数值筛查：{passed}/{len(rows)} 通过脚滑与穿地检查</b><p>穿地阈值 {cfg["penetration_cm"]} cm；接触脚面滑动速度 {cfg["slip_cm_s"]} cm/s；超限持续 {cfg["bad_duration_s"]} s。通过仅表示这两项未发现持续问题，不是动力学可执行认证。</p><p>人体蓝色，机器人灰色；官方坐标转换已修正。网格 50 cm，每个画面共用一个地面。点下面的超限区间可跳到视频对应时间。</p><a style="color:#8cf" href="locomotion_report.json" download>下载完整数值报告 JSON</a></section><!--END_AUDIT-->'
    page=page.replace('</h2>','</h2>'+summary,1)
    for r in rows:
        heading=html.escape(r['video']);fail=r['status']!='pass_screen'
        block=f'<!--AUDIT:{r["clip_id"]}--><div style="padding:10px;border-left:4px solid {"#eb9959" if fail else "#8cc"}"><b>{"脚滑筛查未通过" if fail else "脚滑与穿地筛查通过"}</b> · 初始躯干朝向差 {r["initial_torso_heading_difference_deg"]:.1f}°<table cellpadding="5"><tr><th>脚</th><th>接触滑动 P95 / 最大 cm/s</th><th>最大穿地 cm</th></tr>'
        for side,f in r['feet'].items():
            p95=f['support_slip_p95_cm_s'];maximum=f['support_slip_max_cm_s']
            speed='无有效支撑采样' if p95 is None else f'{p95:.2f} / {maximum:.2f}'
            block+=f'<tr><td>{"左脚" if side=="left_foot" else "右脚"}</td><td>{speed}</td><td>{f["max_penetration_cm"]:.3f}</td></tr>'
        block+='</table><div>脚滑超限区间：'
        for side,f in r['feet'].items():
            for event in f['slip_events']:
                start=event['start_s'];end=event['end_s']
                block+=f'<button style="margin:3px" data-video="{html.escape(r["video"],quote=True)}" data-start="{start}" onclick="const v=Array.from(document.querySelectorAll(\'video\')).find(v=>v.getAttribute(\'src\')===this.dataset.video);v.currentTime=Number(this.dataset.start);v.play();v.scrollIntoView({{block:\'center\'}})">{"左" if side=="left_foot" else "右"} {start:.2f}–{end:.2f}s</button>'
        if not any(f['slip_events'] for f in r['feet'].values()):block+='无持续超限'
        block+='</div></div><!--END_AUDIT-->'
        page=page.replace(f'<h3>{heading}</h3>',f'<h3>{heading}</h3>'+block,1)
    (directory/'locomotion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (directory/'index.html').write_text(page,encoding='utf-8')
    package_review(directory)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','robots','descriptor','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--slip-cm-s',type=float,default=5.);p.add_argument('--penetration-cm',type=float,default=.5);p.add_argument('--bad-duration-s',type=float,default=.1)
    a=p.parse_args();robot=json.loads(a.robots.read_text(encoding='utf-8-sig'))[0]
    audit(a.manifest,robot,a.descriptor,a.output,a.slip_cm_s,a.penetration_cm,a.bad_duration_s)
if __name__=='__main__':main()
