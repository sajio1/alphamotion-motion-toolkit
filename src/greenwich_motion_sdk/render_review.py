"""Render diverse saved compact clips without inference, fitting or grounding."""
import argparse, gzip, html, json, subprocess, time, zipfile
from pathlib import Path
import numpy as np
from .compact import load_compact, native_visual_frame
from .surface_contacts import ground_surface_patches

def choose(clips, count):
    groups = {}
    for row in clips: groups.setdefault(row.get('review_category', 'other'), []).append(row)
    selected = []
    while len(selected) < count:
        progressed = False
        for rows in groups.values():
            if rows:
                selected.append(rows.pop(0)); progressed = True
                if len(selected) == count: break
        if not progressed: break
    if len(selected) != count: raise ValueError('Not enough clips')
    return selected

def package_review(directory):
    """Package the page and its exact videos for portable offline review."""
    directory=Path(directory)
    rows=json.loads((directory/'videos.json').read_text(encoding='utf-8'))['clips']
    page=(directory/'index.html').read_text(encoding='utf-8')
    robot=rows[0].get('robot_id','robot')
    height=rows[0].get('video_height',720)
    filename=f'{robot}-{len(rows)}-videos-{height}p.zip'
    # An offline page has no dependency on a running server or the ZIP itself.
    link=f'<p><a style="color:#8cf" download href="{filename}">Download all videos + HTML (ZIP)</a></p>'
    page=page.replace(link,'')
    posters=[]
    for row in rows:
        poster=row['video'].split('_',1)[0]+'_frame_0000.jpg'
        if (directory/poster).is_file():
            posters.append(poster)
            if f'poster="{poster}"' not in page:
                page=page.replace(f'src="{html.escape(row["video"])}"',
                                  f'poster="{poster}" src="{html.escape(row["video"])}"')
    with zipfile.ZipFile(directory/filename,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('index.html',page)
        for row in rows:z.write(directory/row['video'],row['video'])
        for poster in posters:z.write(directory/poster,poster)
        if (directory/'locomotion_report.json').is_file():
            z.write(directory/'locomotion_report.json','locomotion_report.json')
        z.writestr('timing.json',json.dumps([{k:v for k,v in row.items() if k in
            ('video','frames','fps','duration_s','source_stride','source_time_max_difference_s','comparison')}
            for row in rows],indent=2))
        if z.testzip() is not None:raise ValueError('ZIP integrity failure')
    (directory/'index.html').write_text(page.replace('</h2>','</h2>'+link,1),encoding='utf-8')
    return directory/filename

def render(manifest, descriptor_path, xml, output, ffmpeg, count=10, height=720, crf=26, compare_source=False):
    import mujoco as mj, cv2
    from scipy.spatial import ConvexHull
    manifest, output = Path(manifest), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = Path(descriptor_path).read_bytes()
    descriptor = json.loads(gzip.decompress(raw) if str(descriptor_path).endswith('.gz') else raw)
    rows = choose(json.loads(manifest.read_text(encoding='utf-8'))['clips'], count)
    width = round(height*16/9/2)*2
    native = mj.MjModel.from_xml_path(str(xml))
    ids = [i for i in range(native.ngeom) if native.geom_type[i] == mj.mjtGeom.mjGEOM_MESH
           and native.geom_contype[i] == 0 and native.geom_conaffinity[i] == 0]
    if len(ids) != len(descriptor['geoms']): raise ValueError('Native visual roster mismatch')
    scene = mj.MjSpec.from_file(str(xml))
    scene.worldbody.add_geom(name='review_floor', type=mj.mjtGeom.mjGEOM_PLANE,
                             size=[100,100,.1], rgba=[.87,.84,.76,1])
    scene.worldbody.add_camera(name='review_camera', fovy=42)
    model = scene.compile()
    # Adding a world-floor geom shifts native indices. Resolve the compiled
    # visual roster again instead of reusing pre-compilation geom IDs.
    ids = [i for i in range(model.ngeom) if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH
           and model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0]
    if len(ids) != len(descriptor['geoms']): raise ValueError('Compiled visual roster mismatch')
    visible = set(ids)
    for i in range(model.ngeom):
        if i not in visible and mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,i) != 'review_floor':
            model.geom_rgba[i,3] = 0
            model.geom_group[i] = 5
    model.vis.global_.offwidth = width; model.vis.global_.offheight = height
    model.vis.quality.offsamples = 4
    model.vis.headlight.ambient[:] = .6; model.vis.headlight.diffuse[:] = .35
    data = mj.MjData(model); mj.mj_forward(model,data)
    cid = mj.mj_name2id(model,mj.mjtObj.mjOBJ_CAMERA,'review_camera')
    camera = mj.MjvCamera(); camera.type = mj.mjtCamera.mjCAMERA_FIXED; camera.fixedcamid = cid
    renderer = mj.Renderer(model,height=height,width=width)
    options = mj.MjvOption(); options.sitegroup[:] = 0
    options.geomgroup[5] = 0
    options.flags[mj.mjtVisFlag.mjVIS_TENDON] = False
    # Canonical Y-up -> native display Z-up; display conversion only.
    basis = np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    geometry = [(np.asarray(descriptor['meshes'][g['mesh']]['vertices']).reshape(-1,3),
                 np.asarray(descriptor['meshes'][g['mesh']]['faces']).reshape(-1,3)) for g in descriptor['geoms']]
    result = []
    try:
        for number,row in enumerate(rows,1):
            started = time.perf_counter()
            archive = Path(row['archive'])
            if not archive.is_absolute(): archive = manifest.parent/archive
            motion = load_compact(archive,member=row['member'],robot=descriptor['robot'])
            if motion['joint_names'].tolist() != descriptor['joint_names']: raise ValueError('Joint mismatch')
            frames = len(motion['q']); fps = motion['fps']
            direction = np.array([3.,-4.,1.0]); direction /= np.linalg.norm(direction)
            right = np.cross(-direction,[0.,0.,1.]); right /= np.linalg.norm(right)
            up = np.cross(right,-direction)
            source = None; robot_lane=np.zeros(3)
            if compare_source:
                from ._bvh import load_bvh
                if not row.get('source_bvh'): raise ValueError('Source comparison requires source_bvh in the manifest')
                bvh=load_bvh(row['source_bvh'])
                stride=round(1/bvh['dt']/fps)
                if stride<1 or abs(1/bvh['dt']/stride-fps)>.02: raise ValueError('Source FPS does not match saved generation stride')
                take=np.arange(frames)*stride
                if take[-1]>=len(bvh['values']): raise ValueError('Source shorter than robot timeline')
                source=bvh['positions'][take].copy()
                source[:,:,[0,2]]-=bvh['positions'][0,1,[0,2]]
                source=source/100@basis.T-right*.85
                robot_lane=right*.85
            poses = [native_visual_frame(descriptor,q,p,r) for q,p,r in zip(motion['q'],motion['root_t_cm'],motion['root_rot6d'])]
            # Fit the entire motion relative to its moving horizontal root.
            roots = motion['root_t_cm']/100 @ basis.T; roots[:,2] = 0
            if source is not None:
                source_roots=source[:,1]+right*.85; source_roots[:,2]=0
                roots=(roots+source_roots)/2
            clouds = []
            for f,(positions,rotations) in enumerate(poses):
                for j,(vertices,_) in enumerate(geometry):
                    world = (vertices@rotations[j].T+positions[j])@basis.T+robot_lane-roots[f]
                    # Retain bounding corners, not a duplicate of every vertex
                    # at every timestamp. Camera fitting needs only extrema.
                    lo,hi = world.min(0),world.max(0)
                    clouds.append(np.array([[x,y,z] for x in (lo[0],hi[0])
                                           for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]))
                if source is not None: clouds.append(source[f,1:]-roots[f])
            cloud = np.concatenate(clouds); low,high = cloud.min(0),cloud.max(0)
            target = (low+high)/2
            rel = cloud-target; tan = np.tan(np.deg2rad(21))
            distance = np.max(rel@direction + np.maximum((np.abs(rel@right)+.16)/(tan*width/height*.94),
                                                        (np.abs(rel@up)+.18)/(tan*.86)))
            del clouds,cloud,rel
            path = output/f'{number:02d}_{motion["stem"]}.mp4'
            writer = subprocess.Popen([str(ffmpeg),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
                                       '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264',
                                       '-preset','fast','-crf',str(crf),'-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            try:
                for f,(positions,rotations) in enumerate(poses):
                    data.geom_xpos[ids] = positions@basis.T+robot_lane
                    data.geom_xmat[ids] = (basis@rotations).reshape(-1,9)
                    look = target+roots[f]; eye = look+direction*distance
                    camrot = np.column_stack((right,up,direction))
                    data.cam_xpos[cid] = eye; data.cam_xmat[cid] = camrot.reshape(-1)
                    renderer.update_scene(data,camera,scene_option=options)
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
                    def line(start,end,radius,color):
                        if renderer.scene.ngeom>=renderer.scene.maxgeom: raise ValueError('Scene geometry capacity exceeded')
                        geom=renderer.scene.geoms[renderer.scene.ngeom]
                        mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array(color))
                        mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,radius,start,end)
                        renderer.scene.ngeom+=1
                    # One 50 cm world-grid for source and robot, at ground Z=0.
                    for axis in (0,1):
                        other=1-axis
                        for tick in np.arange(np.floor(look[axis]-8)*2,np.ceil(look[axis]+8)*2+1):
                            start=look.copy();end=look.copy();start[2]=end[2]=.002
                            start[axis]=end[axis]=tick*.5
                            start[other]-=8;end[other]+=8
                            line(start,end,.002,[.43,.42,.39,1])
                    if source is not None:
                        for j,parent in enumerate(bvh['parents']):
                            if parent>=1: line(source[f,parent],source[f,j],.016,[.06,.40,.86,1])
                    image = renderer.render().copy(); paint = image.copy()
                    def project_display(points):
                        local = (points-eye)@camrot
                        good = local[:,2] < -1e-4; local = local[good]
                        focal = height/(2*tan)
                        return np.column_stack((width/2+focal*local[:,0]/-local[:,2],
                                                height/2-focal*local[:,1]/-local[:,2]))
                    for j,(vertices,faces) in enumerate(geometry):
                        world = vertices@rotations[j].T+positions[j]
                        if world[:,1].min() > .01: continue
                        for (points,_),color in zip(ground_surface_patches(world,faces),((35,205,115),(235,55,45))):
                            if len(points)<3: continue
                            xy = project_display(points@basis.T+robot_lane)
                            if len(xy)<3: continue
                            try: hull = xy[ConvexHull(xy).vertices].astype(np.int32)
                            except Exception: continue
                            cv2.fillConvexPoly(paint,hull,color)
                    if source is not None:
                        # The source has bones rather than a closed mesh. Ground
                        # colors on it are anatomical landmark footprint proxies.
                        names=bvh['names']
                        for side in ('Left','Right'):
                            heel=source[f,names.index(side+'Foot')].copy()
                            toe=source[f,names.index(side+'ToeBase')].copy()
                            if min(heel[2],toe[2])>.08: continue
                            heading=toe-heel;heading[2]=0;heading/=max(np.linalg.norm(heading),1e-6)
                            across=np.cross(heading,[0,0,1])*.045
                            patch=np.array([heel-heading*.04+across,toe+heading*.04+across,
                                            toe+heading*.04-across,heel-heading*.04-across]);patch[:,2]=.003
                            xy=project_display(patch)
                            if len(xy)==4: cv2.fillConvexPoly(paint,xy.astype(np.int32),(35,205,115))
                        for j,p in enumerate(source[f]):
                            if j<1 or p[2]>.01 or names[j] in ('LeftFoot','RightFoot','LeftToeBase','RightToeBase'): continue
                            patch=np.array([p+[-.035,-.035,0],p+[.035,-.035,0],p+[.035,.035,0],p+[-.035,.035,0]])
                            patch[:,2]=.003;xy=project_display(patch)
                            if len(xy)==4:cv2.fillConvexPoly(paint,xy.astype(np.int32),(235,55,45) if p[2]<-.005 else (35,205,115))
                    image = cv2.addWeighted(paint,.45,image,.55,0)
                    cv2.rectangle(image,(0,0),(width,56),(40,40,38),-1)
                    variant = f' [{row["variant"]}: {row["iterations"]} iterations]' if 'variant' in row else ''
                    title = f'{number:02d} {"SOMA (blue) + H2" if source is not None else "H2"}{variant} | {motion["stem"]}'
                    cv2.putText(image,title[:110],(15,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
                    cv2.putText(image,f'{f/fps:.2f} / {frames/fps:.2f}s | Green: ground proximity | Red: penetration',
                                (15,45),cv2.FONT_HERSHEY_SIMPLEX,.42,(225,225,225),1,cv2.LINE_AA)
                    if f in {0,frames//2,frames-1}: cv2.imwrite(str(output/f'{number:02d}_frame_{f:04d}.jpg'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    writer.stdin.write(image.tobytes())
            finally:
                writer.stdin.close()
                if writer.wait() != 0: raise RuntimeError('Video encoding failed')
            cap = cv2.VideoCapture(str(path)); actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            if actual != frames: raise ValueError(f'Encoded frame mismatch: {actual} != {frames}')
            item = {**row,'video':path.name,'frames':frames,'duration_s':frames/fps,'bytes':path.stat().st_size,
                    'render_elapsed_s':time.perf_counter()-started,'verified_frames':actual,
                    'video_height':height,'video_width':width}
            if source is not None:item.update(source_stride=stride,source_frames=len(bvh['values']),
                source_time_max_difference_s=float(np.max(np.abs(take*bvh['dt']-np.arange(frames)/fps))),
                comparison='exact source frames used for generation; display lanes only; no motion scaling')
            result.append(item)
            (output/'videos.json').write_text(json.dumps({'clips':result,'motion_modified':False},indent=2),encoding='utf-8')
            print(f'[{number}/{count}] {path.name} {item["render_elapsed_s"]:.1f}s {item["bytes"]/1e6:.2f}MB',flush=True)
    finally: renderer.close()
    page = f'<meta name="viewport" content="width=device-width"><body style="background:#222;color:white;font:16px sans-serif"><h2>{"SOMA + " if compare_source else ""}H2: {count} sampled motions</h2>'
    for row in result:
        page += f'<h3>{html.escape(row["video"])}</h3><video controls preload="none" style="width:100%;max-width:960px" src="{html.escape(row["video"])}"></video><p><a style="color:#8cf" download href="{html.escape(row["video"])}">Download MP4</a></p>'
    (output/'index.html').write_text(page,encoding='utf-8')
    package_review(output)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','descriptor','xml','output','ffmpeg'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=10); p.add_argument('--height',type=int,default=720); p.add_argument('--crf',type=int,default=26)
    p.add_argument('--compare-source',action='store_true')
    a=p.parse_args()
    if a.count<1 or a.height<360 or a.height%2: p.error('Invalid count or height')
    render(a.manifest,a.descriptor,a.xml,a.output,a.ffmpeg,a.count,a.height,a.crf,a.compare_source)
if __name__=='__main__': main()
