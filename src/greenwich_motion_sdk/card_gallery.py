"""Build a reusable four-motion GIF and publish dataset/model documentation.

Motion images and GIF bytes stay in memory unless --asset-directory is supplied.
Upload only small card/preview assets; existing dataset shards are untouched.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np

DEFAULT_CLIPS=('pels_air_punch_001__A495','jog_avoid_bump_270_R_003__A168',
               'walk_sideway_045_start_004__A023','jump_left_003__A033')
LABELS=('Air punch','Jog around obstacle','Sideways walk','Side jump')


def gallery_review_sheet(gif, samples=4):
    """Decode evenly spaced GIF times into a reproducible visual QA sheet."""
    from PIL import Image, ImageDraw
    source=Image.open(io.BytesIO(gif))
    tiles=[]
    for index in np.linspace(0,source.n_frames-1,samples).round().astype(int):
        source.seek(int(index));frame=source.convert('RGB')
        tile=Image.new('RGB',(frame.width,frame.height+24),(32,32,32))
        tile.paste(frame,(0,24));ImageDraw.Draw(tile).text((8,5),f'GIF frame {index}',fill='white')
        tiles.append(tile)
    sheet=Image.new('RGB',(tiles[0].width*2,tiles[0].height*math.ceil(len(tiles)/2)),(32,32,32))
    for index,tile in enumerate(tiles):sheet.paste(tile,((index%2)*tile.width,(index//2)*tile.height))
    output=io.BytesIO();sheet.save(output,format='PNG');return output.getvalue()


def build_gallery(manifest, clips=DEFAULT_CLIPS, width=960, fps=10, max_seconds=5., colors=128,
                  *, columns=2, contact_labels=False, descriptor=None, additional_manifests=(), hide_ui=False, contact_glow=False):
    import cv2
    from PIL import Image
    if not clips or columns<1 or len(clips)%columns or width<640 or width%columns or fps<1 or max_seconds<=0 or not 2<=colors<=256:
        raise ValueError('Gallery requires full rows, divisible width >=640 and positive FPS/duration')
    manifest=Path(manifest);data={'clips':[]}
    for path in (manifest,*map(Path,additional_manifests)):
        for original in json.loads(path.read_text(encoding='utf-8'))['clips']:
            row=dict(original);row.setdefault('source_clip_id',row['clip_id']);row.setdefault('variant','cloud100')
            row['video']=str((path.parent/row['video']).resolve())
            row['archive']=str((path.parent/row['archive']).resolve())
            if not any(r['source_clip_id']==row['source_clip_id'] and r['variant']==row['variant'] for r in data['clips']):data['clips'].append(row)
    rows=[]
    for clip in clips:
        matches=[r for r in data['clips'] if r.get('source_clip_id',r['clip_id'])==clip and r.get('variant')=='cloud100']
        if len(matches)!=1:raise ValueError(f'Expected exactly one published cloud100 preview: {clip}')
        rows.append(matches[0])
    seconds=min(max_seconds,max(r['frames']/r['fps'] for r in rows))
    count=max(1,math.ceil(seconds*fps));cell_w=width//columns
    header_h=0 if hide_ui else (54 if contact_labels else 36);cell_h=round(cell_w*9/16)+header_h
    contacts=[];floor_contacts=[];surface_contacts=[];surface_names=[]
    if contact_labels:
        from .compact import load_compact
        for row in rows:
            archive=Path(row['archive'])
            if not archive.is_absolute():archive=manifest.parent/archive
            motion=load_compact(archive,member=row.get('member'),robot=row['robot_id'])
            if len(motion['q'])!=row['frames'] or motion['fps']!=row['fps']:
                raise ValueError('Contact/video timeline mismatch')
            contacts.append(motion['model_contact'])
            floor_contacts.append(motion['sole_height_cm']<=1.)
            if descriptor is not None:
                import gzip
                from .compact import native_visual_motion
                from .surface_audit import mesh_extrema
                desc=json.loads(gzip.decompress(Path(descriptor).read_bytes()))
                p,r=native_visual_motion(desc,motion['q'],motion['root_t_cm'],motion['root_rot6d'])
                labels=[];heights=[]
                for k,g in enumerate(desc['geoms']):
                    hull=mesh_extrema(desc['meshes'][g['mesh']]['vertices'])
                    heights.append((r[:,k,1]@hull.T).min(1)*100+p[:,k,1]*100)
                    body=desc['bodies'][g['body']];slot=next((j['slot'] for j in reversed(body['joints']) if j['slot'] is not None),None)
                    labels.append(desc['joint_names'][slot[0]] if slot is not None else f'body_{g["body"]}')
                heights=np.stack(heights,1);surface_contacts.append((heights<=1)&(heights>=-.5));surface_names.append(labels)
    captures=[cv2.VideoCapture(str(manifest.parent/r['video'])) for r in rows]
    frames=[]
    try:
        for i in range(count):
            cells=[]
            for n,(row,cap) in enumerate(zip(rows,captures)):
                # Every panel shares elapsed time. Short clips hold their final
                # frame; no task motion is stretched or sped up.
                index=min(int(i/fps*row['fps']),row['frames']-1)
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok:raise ValueError(f'Video decode failed: {row["video"]}, frame {index}')
                frame=frame[56:]  # Remove only the baked-in filename/title band.
                if contact_glow:
                    # Existing geometric paint is alpha blended in the renderer.
                    # Saturate that green mask for small GIF panels; contact
                    # decisions and trajectory coordinates stay unchanged.
                    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                    green=(hsv[:,:,0]>=45)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=35)&(hsv[:,:,2]>=40)
                    frame[green]=(95,240,35)
                panel=np.full((cell_h,cell_w,3),32,dtype=np.uint8)
                target_h=cell_h-header_h
                scale=min(cell_w/frame.shape[1],target_h/frame.shape[0])
                resized=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)),interpolation=cv2.INTER_AREA)
                y=header_h+(target_h-resized.shape[0])//2;x=(cell_w-resized.shape[1])//2
                panel[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
                if hide_ui:
                    # Keep the baked geometric ground-contact paint in the
                    # scene, with no title, timer, legend or floating lamps.
                    cells.append(panel);continue
                label=LABELS[n] if tuple(clips)==DEFAULT_CLIPS else row.get('source',{}).get('semantic_label',row['source_clip_id'].split('__')[0].replace('_',' '))[:43]
                cv2.putText(panel,f'{label} | {index/row["fps"]:.1f}s',(8,15),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(panel,'SOMA: blue | H2: gray | green: near floor',(8,30),cv2.FONT_HERSHEY_SIMPLEX,.36,(225,225,225),1,cv2.LINE_AA)
                if contact_labels:
                    active=contacts[n][index]
                    floor=floor_contacts[n][index]
                    cv2.putText(panel,'Floor:',(8,47),cv2.FONT_HERSHEY_SIMPLEX,.36,(225,225,225),1,cv2.LINE_AA)
                    for side,name in enumerate(('L','R')):
                        xdot=56+side*62;color=(115,205,35) if floor[side] else (110,110,110)
                        cv2.circle(panel,(xdot,43),5,color,-1,cv2.LINE_AA)
                        cv2.putText(panel,f'{name}:{int(floor[side])}',(xdot+10,47),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,cv2.LINE_AA)
                    if surface_contacts:
                        count_surfaces=int(surface_contacts[n][index].sum())
                        cv2.putText(panel,f'All surfaces: {count_surfaces} near floor',(190,47),cv2.FONT_HERSHEY_SIMPLEX,.36,(35,205,115),1,cv2.LINE_AA)
                    else:
                        color=(50,195,250) if active.any() else (165,165,165)
                        cv2.putText(panel,f'Model: L:{int(active[0])} R:{int(active[1])}',(190,47),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,cv2.LINE_AA)
                cells.append(panel)
            image=np.vstack([np.hstack(cells[j:j+columns]) for j in range(0,len(cells),columns)])
            frames.append(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
    finally:
        for cap in captures:cap.release()
    # One global palette prevents framewise colour flicker. GIF timing uses
    # centiseconds; choose 10 FPS by default for exact 100 ms frames.
    sample=Image.fromarray(np.vstack([np.asarray(f.resize((240,160))) for f in frames[::max(1,count//12)]]))
    # Contact patches occupy few pixels. Reserve their exact colors so the
    # GIF quantizer cannot merge contact green/amber into skeleton blue.
    reserved=min(4,colors//2)
    palette=sample.quantize(colors=colors-reserved)
    values=palette.getpalette()
    label_colors=([35,240,95] if contact_glow else [35,205,115])+[250,195,50, 235,55,45, 15,102,219]
    values[(colors-reserved)*3:colors*3]=label_colors[:reserved*3]
    palette.putpalette(values)
    indexed=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
    output=io.BytesIO();indexed[0].save(output,format='GIF',save_all=True,append_images=indexed[1:],
        duration=round(1000/fps/10)*10,loop=0,optimize=False,disposal=2)
    gif=output.getvalue();decoded=Image.open(io.BytesIO(gif))
    total_ms=0
    for i in range(decoded.n_frames):decoded.seek(i);total_ms+=decoded.info.get('duration',0)
    if abs(total_ms-count*round(1000/fps/10)*10)>1:raise ValueError('GIF timeline verification failed')
    png=io.BytesIO();frames[min(len(frames)-1,len(frames)//2)].save(png,format='PNG')
    info={'schema':'greenwich.card-gallery.v1','width':width,'height':cell_h*(len(clips)//columns),'frames':decoded.n_frames,
          'duration_s':total_ms/1000,'fps':fps,'bytes':len(gif),'sha256':hashlib.sha256(gif).hexdigest(),
          'motion_regenerated':False,'variant':'cloud100','clips':[
              {'clip_id':r['source_clip_id'],'video':r['video'],'duration_s':r['frames']/r['fps'],
               'source_time_max_difference_s':r.get('source_time_max_difference_s')} for r in rows],
          'timing':'shared elapsed time; shorter clips hold their last frame',
          'columns':columns,'rows':len(clips)//columns,
          'ui_visible':not hide_ui,
          'contact_glow':contact_glow,
          'contact_labels':('Scene only: green ground-contact regions; no floating lamps or UI.' if hide_ui else 'Floor L/R proximity; all-surface proximity when descriptor supplied, otherwise separate model-foot diagnostic.') if contact_labels else None,
          'contact_counts':[{ 'clip_id':r['source_clip_id'],'model_true_frames':c.sum(0).tolist(),
            'floor_proximity_frames':g.sum(0).tolist()} for r,c,g in zip(rows,contacts,floor_contacts)] if contact_labels else None,
          'whole_surface_contact_counts':[{ 'clip_id':r['source_clip_id'],'surface_names':names,'contact_frames':c.sum(0).tolist()} for r,names,c in zip(rows,surface_names,surface_contacts)],
          'surface_colors':'All robot visual surfaces, including hands/knees/pelvis: green=geometric ground proximity; red=penetration; not measured forces'}
    return gif,png.getvalue(),info


def build_cards(base_card, repo):
    card=Path(base_card).read_text(encoding='utf-8')
    card=card.replace('sajio/alphamotion-prime-soma-h2-26k',repo)
    title='# SOMA to H2: generated kinematic motion references\n'
    preview='''
## Four-motion comparison preview

![SOMA human reference and generated H2 motion in four panels](assets/soma-h2-four-grid.gif)

Clockwise from top left: air punch, jog around an obstacle, side jump,
sideways walk. **Blue: original SOMA human skeleton. Gray: generated H2.**
Every panel shows the published **Prime + 100-step refinement** version,
not the older 500-step comparison. Green shows geometry close to the floor;
it is not a force measurement. Panels share elapsed time; shorter clips hold
their last frame. The GIF is a reduced-frame-rate preview, not the 30 Hz data.

## Generating model

**AlphaMotion v0.2 Prime**, approximately **57.7M parameters**, generates the
cross-embodiment pose. A shared native-joint/contact refiner then produces the
published H2 reference. See [MODEL_CARD.md](MODEL_CARD.md) for architecture,
inputs, output heads, refinement settings, intended use and provenance limits.
The repository contains motion data, not model weights.

'''
    if '## Four-motion comparison preview' not in card:card=card.replace(title,title+preview,1)
    old='''Only 1/10
passed the combined sole/slip screen: nine had inferred stable-contact patch
speed above 5 cm/s lasting at least 0.1 s.'''
    new='''The historical v1 audit reported 1/10
passing its combined sole/slip screen. **That is a preliminary proxy-screen
result, not an accepted-motion success rate:** its support inference omitted
the source foot-speed criterion and could include low airborne motion.
The corrected v2 audit additionally checks source speed and reports a separate
slow-source/flat-foot diagnostic. A five-capture old/new comparison still found
residual near-ground drift in both versions, rather than a universal regression
in the 100-step version. Mild visible foot drift is acceptable for the current
human review; slip metrics are advisory and do not alone reject a motion.'''
    card=card.replace(old,new)
    model='''# Generating model card: AlphaMotion Prime → Unitree H2

This card documents the model and postprocessing used to generate this dataset.
It is not a model-weight release or a dynamics certification.

## Model identity and architecture

| Property | Configured value |
| --- | --- |
| Family / variant | AlphaMotion v0.2 / Prime (L) |
| Approximate model size | 57.7 million parameters, per release contract |
| Hidden width / attention heads | 384 / 12 |
| Encoder / decoder depth | 6 / 6 configured layers |
| Position / rotation latent slots | 4 / 128 per frame |
| Quantization | Single-stage FSQ; 13 dimensions, 32 levels each |
| Quantized code shape | [T, 132, 13] |
| Joint-name semantics | SigLIP, 768-dimensional embeddings |
| Geometry conditioning | Skeleton geometry, joint axes and DoF/limit descriptors |

The position and rotation streams share a cross-embodiment spatial codec.
Position and rotation decoder branches condition on the target body's geometry.
The v2 code space is shared by Bumblebee, Megatron and Prime; it is not compatible
with the older v0.1 Greenwich/Equator code space without conversion or retraining.

## Inputs and output heads

Input pose features include per-joint global rotation-6D and root-relative
positions, together with skeleton geometry and optional visibility/contact
conditioning. This dataset encodes **all 77 anatomical SOMA joints directly**;
only the identity scene Root is removed. It is not an EE-only experiment and
does not route SOMA through a SMPL-22 intermediary.

- **Rotation head:** free global rotation-6D for each target joint. This is not
  already a hardware-ordered native motor-angle vector.
- **Position head:** three components per joint. The root slot carries the
  model's height above ground in metres, according to the v2 codec convention.
  The present pipeline uses external source XZ as a locomotion guide; it does
  not claim independently generated horizontal root displacement.
- **Contact head:** a per-joint contact logit. Published compact data retains
  the two foot predictions thresholded at sigmoid >0.8.
- **Judge head:** 16 training-embodiment critic outputs in the checkpoint.
  These are not calibrated physical-success probabilities, and are not used
  as acceptance labels for the published motions.
- **Separate root head:** disabled in the configured checkpoint; root-height
  information comes from the position branch.

The release notes describe training on LAFAN v3 and timebase-fixed AMASS.
The local checkpoint configuration identifies source_run `L_L20k_0903`.
This card does not independently audit the training corpus or claim additional
SOMA/H2-specific fine-tuning beyond the recorded generator configuration.

## Shared refinement recipe

1. Decode target global rotations and initial root height.
2. Project global rotations into H2's actual native joint parameters and limits.
3. Optimize native joint parameters and root XYZ for **100 Adam iterations**
   around the decoded reference. Root orientation remains the model reference.
4. Apply mesh-based penetration/contact terms, a sole-reference-point slip
   penalty, stable front/rear sole-height penalties, source swing/flight
   clearance terms and joint/root temporal regularization.

Saved configuration: stable margin 0.1 s, swing threshold 3 cm, stable-patch
weight 20, swing-clearance weight 12, flight-root weight 5,
analytic_vertical_init 1. Native-angle learning rate is 0.008; root-offset
learning rate is 0.25 in canonical centimetres. No procedural gait or constant
pelvis-height preset is injected. Trajectories are not resized for H2.

Ground remains SOMA Y=0. Root XYZ is adjustable; source horizontal displacement
provides guidance. Native mesh reconstruction uses the official AX mapping
`native (x,y,z) → canonical (y,z,x)`. A display-only rotation is not a substitute.

## Published outputs and intended use

26,111 clips, approximately 50.723 hours at 30 Hz, in 16 ZIP shards. Compact NPZ
stores native joint slots, root rotation/translation, joint names, final lowest
sole heights and predicted foot-contact booleans. See [schema.json](schema.json)
and [load_motion.py](load_motion.py). Root translation is in centimetres, native
joint angles in radians, and rot6D concatenates rotation-matrix columns 0 and 1.

Use as kinematic motion references, retargeting examples and candidate reference
trajectories for controller training. Contact flags do not encode fractional
sole contact area. Torques, contact forces, object loads and control inputs are
not included. Simulation/hardware tracking must be validated separately.

## Evaluation and provenance limits

ZIP CRC, archive SHA-256, motion counts and per-artifact checks passed; integrity
is separate from motion quality. Human review accepts mild visually tolerable
foot drift. Foot-slip metrics remain diagnostic, not automatic acceptance.
The historical ten-clip v1 report is preliminary and not a population success rate.
The four GIF panels are examples chosen for demonstration, not a random benchmark.

Current local release code/configuration documents the model architecture and
the cloud export manifest records the 100-step refinement settings. The export
did not record an immutable model-weight revision/SHA, so exact checkpoint
identity cannot be independently proven from the exported motions alone.

Source motion attribution: **Motion Data by Bones Studio**,
[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed).
The underlying [BONES-SEED license](https://bones.studio/info/seed-license)
continues to apply; this card does not grant a different source-data license.
'''
    return card,model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--base-card',type=Path)
    p.add_argument('--repo',default='sajio/alphamotion-soma-h2')
    p.add_argument('--clips',nargs='+',default=DEFAULT_CLIPS);p.add_argument('--width',type=int,default=960)
    p.add_argument('--columns',type=int,default=2);p.add_argument('--contact-labels',action='store_true')
    p.add_argument('--hide-ui',action='store_true',help='Scene-only panels; retain ground-contact paint and remove titles/timers/legends')
    p.add_argument('--contact-glow',action='store_true',help='Illuminate existing geometric contact green; no change to contact decisions')
    p.add_argument('--preview-only',action='store_true',help='Only local GIF/PNG/manifest; no cards or publication')
    p.add_argument('--asset-name',default='soma-h2-four-grid')
    p.add_argument('--qa-sheet',action='store_true',help='Also decode four GIF times into a visual review PNG')
    p.add_argument('--separate',action='store_true',help='Export one independent source/robot GIF per requested clip')
    p.add_argument('--fps',type=int,default=10);p.add_argument('--duration',type=float,default=5.)
    p.add_argument('--draft-directory',type=Path,help='Documentation only; no motion images written here')
    p.add_argument('--asset-directory',type=Path,help='Optional disk destination for GIF/PNG; omit for memory-only')
    p.add_argument('--preview-base64',action='store_true');p.add_argument('--publish',action='store_true')
    p.add_argument('--source-metadata',type=Path);p.add_argument('--release-index',type=Path)
    p.add_argument('--archive-root',type=Path);p.add_argument('--robot-xml',type=Path)
    p.add_argument('--viewer-descriptor',type=Path);p.add_argument('--checkpoint-config',type=Path)
    p.add_argument('--additional-manifests',nargs='*',type=Path,default=[])
    a=p.parse_args()
    if a.preview_only:
        if a.publish or not a.asset_directory:p.error('Preview-only requires asset-directory and forbids publish')
        if Path(a.asset_name).name!=a.asset_name:p.error('asset-name must be a basename')
        if a.separate:
            a.asset_directory.mkdir(parents=True,exist_ok=True)
            outputs=[]
            for number,clip in enumerate(a.clips,1):
                if Path(clip).name!=clip or '/' in clip or '\\' in clip:p.error('Clip identifier must be a basename')
                gif,png,info=build_gallery(a.manifest,(clip,),a.width,a.fps,a.duration,
                    columns=1,contact_labels=a.contact_labels,descriptor=a.viewer_descriptor,
                    additional_manifests=a.additional_manifests,hide_ui=a.hide_ui,contact_glow=a.contact_glow)
                name=f'{number:02d}_{clip}'
                for suffix,payload in (('.gif',gif),('.png',png),('.json',json.dumps(info,indent=2).encode())):
                    (a.asset_directory/(name+suffix)).write_bytes(payload)
                if a.qa_sheet:(a.asset_directory/(name+'-qa.png')).write_bytes(gallery_review_sheet(gif,samples=8))
                outputs.append({'clip_id':clip,'gif':name+'.gif','bytes':len(gif),'sha256':info['sha256'],'duration_s':info['duration_s']})
                print(json.dumps(outputs[-1]),flush=True)
            (a.asset_directory/'individual-gifs.json').write_text(json.dumps({'clips':outputs},indent=2),encoding='utf-8')
            return
        gif,png,info=build_gallery(a.manifest,tuple(a.clips),a.width,a.fps,a.duration,
                                 columns=a.columns,contact_labels=a.contact_labels,descriptor=a.viewer_descriptor,additional_manifests=a.additional_manifests,hide_ui=a.hide_ui,contact_glow=a.contact_glow)
        a.asset_directory.mkdir(parents=True,exist_ok=True)
        for suffix,payload in (('.gif',gif),('.png',png),('.json',json.dumps(info,indent=2).encode())):
            (a.asset_directory/(a.asset_name+suffix)).write_bytes(payload)
        if a.qa_sheet:(a.asset_directory/(a.asset_name+'-qa.png')).write_bytes(gallery_review_sheet(gif,samples=8))
        print(json.dumps(info));return
    if not a.base_card:p.error('Card generation requires base-card')
    if len(a.clips)!=4 or a.columns!=2:p.error('Published card layout remains four panels; use preview-only for other grids')
    card,model=build_cards(a.base_card,a.repo)
    metadata_files={};metadata_summary=None
    if a.source_metadata:
        if not all((a.release_index,a.archive_root,a.robot_xml,a.viewer_descriptor)):
            p.error('Source metadata requires release index, archive root, robot XML and viewer descriptor')
        from .dataset_metadata import build_release_metadata,metadata_card_section
        metadata_files,metadata_summary=build_release_metadata(a.release_index,a.source_metadata,a.archive_root,
            a.robot_xml,'unitree_h2',a.viewer_descriptor,a.checkpoint_config)
        card=metadata_card_section(card)
    gif,png,info=build_gallery(a.manifest,tuple(a.clips),a.width,a.fps,a.duration,
                             columns=a.columns,contact_labels=a.contact_labels)
    if metadata_summary:
        info['metadata_validation']={key:metadata_summary[key] for key in ('motion_count','semantic_label_count','categories')}
        info['metadata_files']={name:len(value) for name,value in metadata_files.items()}
    if a.draft_directory:
        a.draft_directory.mkdir(parents=True,exist_ok=True)
        for name,value in (('README.md',card),('MODEL_CARD.md',model),('gallery_manifest.json',json.dumps(info,indent=2))):
            (a.draft_directory/name).write_text(value,encoding='utf-8')
    if a.asset_directory:
        a.asset_directory.mkdir(parents=True,exist_ok=True)
        (a.asset_directory/'soma-h2-four-grid.gif').write_bytes(gif)
        (a.asset_directory/'soma-h2-four-grid.png').write_bytes(png)
    if a.publish:
        from huggingface_hub import HfApi,CommitOperationAdd
        additions={'README.md':card.encode(),'MODEL_CARD.md':model.encode(),
                   'assets/soma-h2-four-grid.gif':gif,'assets/gallery_manifest.json':json.dumps(info,indent=2).encode()}
        additions.update(metadata_files)
        api=HfApi();result=api.create_commit(repo_id=a.repo,repo_type='dataset',
            operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=io.BytesIO(value)) for name,value in additions.items()],
            commit_message='Document Prime generator and add four-motion SOMA/H2 GIF preview')
        remote=api.repo_info(a.repo,repo_type='dataset',files_metadata=True,revision=result.oid)
        asset=next(s for s in remote.siblings if s.rfilename=='assets/soma-h2-four-grid.gif')
        if asset.size!=len(gif):raise ValueError('Remote GIF size mismatch')
        info['published_revision']=result.oid
    if a.preview_base64:
        from PIL import Image
        preview=Image.open(io.BytesIO(png)).convert('RGB');preview.thumbnail((960,640))
        encoded=io.BytesIO();preview.save(encoded,format='JPEG',quality=75)
        info['preview_jpeg_base64']=base64.b64encode(encoded.getvalue()).decode()
    print(json.dumps(info))


if __name__=='__main__': main()
