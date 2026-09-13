"""Enrich a compact release index with source semantics and portable provenance.

All dataset metadata stays in memory unless an explicit output directory is given.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

SOURCE_FIELDS={
    'move_name':'move_name','package':'package','category':'category',
    'content_name':'action_name','content_short_description_2':'semantic_label',
    'content_natural_desc_4':'natural_description','content_technical_description':'technical_description',
    'content_type_of_movement':'movement_type','content_body_position':'body_position_annotation',
    'content_uniform_style':'style','content_props':'props','is_mirror':'is_mirror',
    'content_horizontal_move':'horizontal_move_annotation','content_vertical_move':'vertical_move_annotation',
    'content_complex_action':'complex_action_annotation','content_repeated_action':'repeated_action_annotation',
    'move_duration_frames':'frame_count_metadata','take_org_name':'take_group',
    'take_date':'take_date_metadata','actor_uid':'actor_group',
}
ANTHROPOMETRY=('actor_height_cm','actor_foot_cm','actor_collarbone_height_cm','actor_collarbone_span_cm',
              'actor_elbow_span_cm','actor_wrist_span_cm','actor_shoulder_span_cm','actor_hips_height_cm',
              'actor_hips_bones_span_cm','actor_knee_height_cm','actor_ankle_height_cm','actor_weight_kg')


def safe_value(value):
    if value is None:return None
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def encoded(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')


def robot_metadata(xml,body,descriptor):
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.embodiment.mjcf_build import AX
    import mujoco as mj
    spec,dof,rest,qnames,_=build_from_mjcf(str(xml),body)
    import gzip
    raw=Path(descriptor).read_bytes()
    visual=json.loads(gzip.decompress(raw) if str(descriptor).endswith('.gz') else raw)
    if visual['joint_names']!=spec.joint_names or not np.allclose(np.reshape(visual['zup_to_yup'],(3,3)),AX):
        raise ValueError('Portable/native descriptor names or coordinate basis disagree')
    model=mj.MjModel.from_xml_path(str(xml));slots=[]
    for node,names in enumerate(qnames):
        for slot,name in enumerate(names):
            jid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name)
            if jid<0:raise ValueError(f'Unknown native joint {name}')
            limits=model.jnt_range[jid].tolist() if model.jnt_limited[jid] else None
            slots.append({'node_index':node,'slot_index':slot,'native_joint_name':name,
                'native_joint_id':jid,'qpos_index':int(model.jnt_qposadr[jid]),
                'qvel_index':int(model.jnt_dofadr[jid]),'axis_native_local':model.jnt_axis[jid].tolist(),
                'pivot_native_local_m':model.jnt_pos[jid].tolist(),'limit_rad':limits,
                'default_angle_rad':float(model.qpos0[model.jnt_qposadr[jid]])})
    return {'schema':'greenwich.dataset.robot-metadata.v1','robot':'h2','native_body':body,
        'mjcf_filename':Path(xml).name,'mjcf_sha256':hashlib.sha256(Path(xml).read_bytes()).hexdigest(),
        'mjcf_hash_scope':'root XML file only; external meshes/includes are not certified by this hash',
        'canonical_world':{'handedness':'right','up':'Y','floor_y_cm':0,'root_translation_unit':'cm',
            'native_xyz_to_canonical_yup':AX.tolist(),'rot6d':'matrix column 0 concatenated with column 1'},
        'merged_skeleton':{'joint_names':spec.joint_names,'parents':spec.parents.tolist(),
            'rest_offsets_cm':spec.rest_offsets.tolist(),'rest_rotations':rest.tolist(),
            'dof_features':dof.tolist(),'dof_feature_schema':'3 slots × [axis3, limited, half_range/pi, center/pi], followed by DOF_count/3',
            'root_origin':'root_t_cm is the world position of the canonical root; do not add a standing root-height offset'},
        'native_slots':slots,'mapped_hinge_count':len(slots),
        'mapped_hinge_count_meaning':'kinematic parameter slots, including auxiliary hinges; not a claimed motor count',
        'native_kinematic_tree':{'root_body':visual['root_body'],'root_frame':visual['root_frame'],
            'bodies':visual['bodies'],'position_unit':'m','rotation_storage':'row-major 3x3'},
        'native_constraints':'Passive/closed-chain constraints and full dynamics still require the matching vendor MJCF.',
        'meshes_included':False}


def build_release_metadata(index,source_metadata,archives,xml,body,descriptor,checkpoint_config=None):
    import pandas as pd
    rows=[json.loads(line) for line in Path(index).read_text(encoding='utf-8').splitlines() if line.strip()]
    source=pd.read_parquet(source_metadata)
    lookup={}
    for item in source.to_dict('records'):
        stem=Path(str(item['move_soma_uniform_path'])).stem
        if stem in lookup:lookup[stem]=None
        else:lookup[stem]=item
    manifests=[];generation={}
    for archive in sorted({Path(r['archive']).name for r in rows}):
        with zipfile.ZipFile(Path(archives)/archive) as z:
            names=[n for n in z.namelist() if n.endswith('/manifest.json')]
            if len(names)!=1:raise ValueError(f'Unexpected worker manifest count: {archive}')
            manifests.append(json.loads(z.read(names[0])))
            ledger_name=next(n for n in z.namelist() if n.endswith('/ledger.jsonl'))
            for line in z.read(ledger_name).decode().splitlines():
                entry=json.loads(line)
                if entry['status']=='complete':
                    generation[(archive,entry['stem'])]={'status':'complete','scope':'numerical export completion, not visual acceptance',
                        'completed_at_utc':datetime.fromtimestamp(entry['time_unix_s'],timezone.utc).isoformat(),
                        'source_selection_index':entry['index']}
    base=manifests[0]
    for manifest in manifests[1:]:
        for key in ('source_rows','excluded_ground_system_rows','global_selected_rows','fps','contact_iterations','contact_config','ground_filter_regex'):
            if manifest[key]!=base[key]:raise ValueError(f'Inconsistent worker recipe: {key}')
    categories=Counter();packages=Counter();actions=Counter();missing_generation=0;parts=[]
    for row in rows:
        item=lookup.get(row['clip_id'])
        if item is None:raise ValueError(f'Missing or ambiguous authoritative source metadata: {row["clip_id"]}')
        details={target:safe_value(item.get(field)) for field,target in SOURCE_FIELDS.items()}
        details['archive_member']=item['move_soma_uniform_path']
        details['representation']='soma_uniform'
        details['anthropometry']={field.removeprefix('actor_'):safe_value(item.get(field)) for field in ANTHROPOMETRY}
        if not details['natural_description']:details['natural_description']=safe_value(item.get('content_natural_desc_1'))
        if not details['semantic_label']:details['semantic_label']=details['action_name']
        gen=generation.get((Path(row['archive']).name,row['clip_id']))
        if gen is None:missing_generation+=1
        enriched={**row,'source':details,'generation':gen,'split':'unspecified',
                  'last_timestamp_s':(row['frames']-1)/row['fps']}
        parts.append(json.dumps(enriched,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        categories[details['category']]+=1;packages[details['package']]+=1;actions[details['semantic_label']]+=1
    if missing_generation:raise ValueError(f'{missing_generation} indexed motions lack completed ledger provenance')
    seconds=sum(r['duration_s'] for r in rows)
    durations=np.array([r['duration_s'] for r in rows])
    robot=robot_metadata(xml,body,descriptor)
    summary={'schema':'greenwich.dataset.release-metadata.v1','motion_count':len(rows),'total_duration_s':seconds,
        'hours':seconds/3600,'archive_count':len(manifests),'fps':base['fps'],
        'duration_s':{'min':float(durations.min()),'median':float(np.median(durations)),
                      'p95':float(np.percentile(durations,95)),'max':float(durations.max())},
        'categories':dict(categories),'packages':dict(packages),'semantic_label_count':len(actions),
        'semantic_label_counts':dict(actions),'source_metadata_file':Path(source_metadata).name,
        'source_metadata_sha256':hashlib.sha256(Path(source_metadata).read_bytes()).hexdigest(),
        'source_matching':'exact stem of move_soma_uniform_path; ambiguity is rejected',
        'source_rows':base['source_rows'],'excluded_source_rows':base['excluded_ground_system_rows'],
        'eligible_source_rows':base['global_selected_rows'],'subset':'completed budget-limited subset',
        'source_filter':{'regex':base['ground_filter_regex'],'ignore_case':True,
            'fields':['move_name','category','content_name','content_short_description_2','content_type_of_movement','content_body_position']},
        'robot':'h2','native_metadata':'metadata/robot_h2.json',
        'generator':{'model':'AlphaMotion v0.2 Prime','parameter_count_approx':57700000,
                     'immutable_model_weight_revision':None,'weight_revision_status':'not recorded in cloud export'},
        'refinement':{'iterations':base['contact_iterations'],'configuration':base['contact_config'],
            'root_xyz_optimizable':True,'root_rotation':'model output reference','source_horizontal_guidance':'XZ',
            'trajectory_scale':1.0},
        'contact':{'kind':'predicted boolean per foot','threshold':0.8,'order':['left','right'],
            'area_fraction':None,'force':None,'sole_height':'lowest final mesh Y above canonical floor in cm'},
        'timing':{'duration_definition':'T/fps','last_timestamp_definition':'(T-1)/fps',
            'timestamps':'uniform, t[k]=k/fps, starting at zero','source_sampling':'integer stride; no 120 Hz interpolation',
            'source_fps_assumption_in_bulk_selection':120.0048,'source_fps_assumption_scope':'source selection duration estimate, not per-file measured FPS'},
        'acceptance':{'human_review':'mild visually tolerable foot drift accepted','foot_slip_metrics':'advisory',
                      'full_human_review_complete':False,'dynamics_validated':False},
        'splits':{'provided':False,'advice':'Group original/mirrored takes and related variants to avoid leakage.'}}
    schema={'schema':'greenwich.dataset.index-metadata.v1',
        'existing_index_fields':'Existing clip_id/archive/member/frames/fps/sha256 fields retain their original meanings.',
        'source_fields':SOURCE_FIELDS,'anthropometry_fields':list(ANTHROPOMETRY),
        'annotation_scope':'Source labels describe the human take, not commanded robot posture. Uniform skeleton geometry need not equal the original actor anthropometry.',
        'anthropometry_units':'cm except weight_kg; measurements are static metadata, not pelvis XYZ trajectories',
        'source_frame_count':'Original metadata frame count; actual source-file header was not individually revalidated for every clip.',
        'nulls':'missing source annotations are null; no guessed posture or measured contact values are filled',
        'generation_scope':'Completion timestamp/index comes from worker ledger; not motion acceptance or model inference timing',
        'split':'unspecified; no authoritative train/test split exists'}
    files={'index.jsonl':''.join(parts).encode('utf-8'),'metadata/dataset.json':encoded(summary),
           'metadata/robot_h2.json':encoded(robot),'metadata/index_schema.json':encoded(schema)}
    if checkpoint_config is not None:
        files['metadata/generator_config.json']=encoded(json.loads(Path(checkpoint_config).read_text()))
        summary['generator']['configuration']='metadata/generator_config.json'
        summary['generator']['configuration_scope']='current local Prime reference configuration; cloud did not record immutable weight/config hashes'
        files['metadata/dataset.json']=encoded(summary)
    return files,summary


def metadata_card_section(card):
    heading='## Release metadata\n'
    section='''## Release metadata

The enriched [index.jsonl](index.jsonl) preserves shard paths, hashes and timing
and adds authoritative source semantics: action/category, natural/technical
description, movement type, human posture annotation, mirror flag, take/actor
grouping and static anthropometry. These labels describe the **source human**;
they do not prescribe a standing/crouching robot pose. `soma_uniform` uses a
uniform skeleton, so actor measurements are not a trajectory scaling factor.

- [metadata/dataset.json](metadata/dataset.json): counts, duration distribution,
  semantic/category coverage, source filter, units, contacts and acceptance policy.
- [metadata/robot_h2.json](metadata/robot_h2.json): mesh-free native/merged
  skeleton, rest transforms, axes, limits and NPZ-slot ↔ native-joint mapping.
- [metadata/index_schema.json](metadata/index_schema.json): field provenance,
  annotation scope, null handling and grouping/split guidance.
- [metadata/generator_config.json](metadata/generator_config.json): current
  local Prime reference configuration; immutable cloud weight revision was
  not recorded and is not implied by this file.

No official train/test split is assigned. Static human hip-height metadata is
not a time-varying pelvis trajectory. Completion timestamps are worker-ledger
events, not measured per-stage inference latency or energy consumption.

'''
    if heading not in card:card=card.replace('## Files and access\n',section+'## Files and access\n',1)
    return card


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','source-metadata','archives','xml','descriptor'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--body',default='unitree_h2');p.add_argument('--checkpoint-config',type=Path)
    p.add_argument('--output',type=Path,help='Optional dataset metadata destination; omit to validate in memory')
    p.add_argument('--publish',action='store_true');p.add_argument('--repo',default='sajio/alphamotion-soma-h2')
    a=p.parse_args();files,summary=build_release_metadata(a.index,a.source_metadata,a.archives,a.xml,a.body,a.descriptor,a.checkpoint_config)
    if a.output:
        for name,payload in files.items():
            path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    if a.publish:
        import io
        from huggingface_hub import HfApi,CommitOperationAdd
        result=HfApi().create_commit(repo_id=a.repo,repo_type='dataset',
            operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=io.BytesIO(value)) for name,value in files.items()],
            commit_message='Enrich motion metadata with source semantics and native H2 mapping')
        summary['published_revision']=result.oid
    print(json.dumps({key:summary[key] for key in ('motion_count','hours','archive_count','semantic_label_count','categories','duration_s')},ensure_ascii=False))
    print(json.dumps({'files':{name:len(value) for name,value in files.items()},'all_source_rows_matched':True,'all_completed_ledgers_matched':True}))


if __name__=='__main__':main()
