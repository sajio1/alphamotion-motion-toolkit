"""Render saved SOMA and any number of robot results on one synchronized stage."""
import argparse,json,sys,os,subprocess,html
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','pipeline','root','robots','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--layout',choices=['auto','row','grid'],default='auto')
    p.add_argument('--indices',help='Optional comma-separated subset of saved selection')
    p.add_argument('--height',type=int,default=1080,help='Primary MP4 height; width uses 16:9')
    p.add_argument('--formation-scale',type=float,default=1.,help='Scale display-only actor spacing')
    p.add_argument('--camera-elevation',type=float,default=.65,help='Camera direction vertical component')
    p.add_argument('--camera-zoom',type=float,default=1.,help='Multiplier on fitted camera distance; below 1 moves closer')
    p.add_argument('--camera-azimuth',type=float,default=-45.,help='Camera azimuth in degrees; +X is robot front and +Y is robot left')
    p.add_argument('--lookat-height-bias',type=float,default=0.,help='Metres added to the fitted look-at height')
    p.add_argument('--hide-robot-labels',action='store_true',help='Hide per-robot names and the named contact status footer')
    p.add_argument('--lead-source',action='store_true',help='Put SOMA front-centre with robots in rows behind it')
    p.add_argument('--contact-overlay-alpha',type=float,default=.85,help='Opacity of green/red ground-contact surface overlays')
    p.add_argument('--ffmpeg',required=True);a=p.parse_args()
    if a.layout=='grid':
        from render_soma_robot_grid import main as render_grid
        return render_grid()
    sys.dont_write_bytecode=True;sys.path[:0]=[str(a.repo/'src'),str(a.pipeline/'src'),str(a.pipeline/'scripts')]
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HOME']=str(a.repo/'.local/huggingface');os.environ['ALPHAMOTION_CACHE']=str(a.root/'cache')
    import mujoco as mj,cv2
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
    from greenwich_motion_sdk.contact_labels import export_contacts
    from greenwich_motion_sdk.surface_contacts import export_surface_contacts
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine import constraints as constraints
    from alphamotion.engine.spatial import key_joints
    from greenwich_umi_proof.sole_geometry import build_foot_sole_model
    from scipy.spatial import ConvexHull
    import torch
    from scipy.spatial.transform import Rotation
    from greenwich_umi_proof.interactive_viewer import _sample_greenwich_head,YUP_TO_ZUP
    from render_hiw500_transfer_showcase import _configure_model,_set_fixed_camera,_writer,_close_writer
    robots=json.loads(a.robots.read_text(encoding='utf-8-sig'));selection=json.loads((a.root/'selection.json').read_text())
    if a.indices:selection=[r for r in selection if r['index'] in set(map(int,a.indices.split(',')))]
    robot_names='_'.join(r['name'].upper() for r in robots);videos=[]
    if a.height < 360 or a.height % 2:p.error('--height must be an even integer >= 360')
    if not 0.35 <= a.formation_scale <= 2.:p.error('--formation-scale must be in [0.35, 2.0]')
    if not 0.05 <= a.camera_elevation <= 1.5:p.error('--camera-elevation must be in [0.05, 1.5]')
    if not 0.75 <= a.camera_zoom <= 1.5:p.error('--camera-zoom must be in [0.75, 1.5]')
    if not 0. <= a.contact_overlay_alpha <= 1.:p.error('--contact-overlay-alpha must be in [0, 1]')
    HEIGHT=a.height;WIDTH=round(a.height*16/9/2)*2
    a.output.mkdir(parents=True,exist_ok=True)
    azimuth=np.deg2rad(a.camera_azimuth)
    direction=np.array([np.cos(azimuth),np.sin(azimuth),a.camera_elevation if len(robots)>4 else min(a.camera_elevation,.35)]);direction/=np.linalg.norm(direction)
    right=np.cross(-direction,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,-direction)
    for row in selection:
        motions=[np.load(a.root/row['stem']/r['name']/'motion.npz') for r in robots]
        contacts=[export_contacts(m,a.root/row['stem']/r['name']) for m,r in zip(motions,robots)]
        T=len(motions[0]['root_t']);fps=float(motions[0]['fps'])
        assert all(len(m['root_t'])==T and float(m['fps'])==fps for m in motions)
        source=motions[0]['source_positions_cm']@YUP_TO_ZUP.T/100.
        if len(robots)>4:
            depth=direction.copy();depth[2]=0;depth/=np.linalg.norm(depth)
            if a.lead_source:
                columns=6
                lane=[depth*2.6*a.formation_scale]
                lane += [right*((i%columns)-(columns-1)/2)*2.3*a.formation_scale-depth*(i//columns)*2.5*a.formation_scale for i in range(len(robots))]
            else:
                columns=5;rows=int(np.ceil((len(robots)+1)/columns))
                lane=[right*((i%columns)-(columns-1)/2)*2.6*a.formation_scale-depth*((i//columns)-(rows-1)/2)*3.0*a.formation_scale for i in range(len(robots)+1)]
        else:lane=[right*(i-len(robots)/2)*2.0*a.formation_scale for i in range(len(robots)+1)]
        source+=lane[0];parents=motions[0]['source_parents'];tracks=[];points=[source[:,1:]]
        patches=[]
        scene=mj.MjSpec();scene.add_material(name='floor_mat',rgba=[.84,.80,.69,1])
        scene.worldbody.add_geom(name='ground',type=mj.mjtGeom.mjGEOM_PLANE,size=[100,100,.1],material='floor_mat')
        scene.worldbody.add_camera(name='camera',fovy=42)
        for index,(r,m) in enumerate(zip(robots,motions)):
            positions,quats,*_=_sample_greenwich_head(a.root/row['stem']/r['name']/'motion.npz',r['xml'],r['body'],rotation_key='rot6d',world_position_key='world_position_cm',floor_y_cm=0.)
            # The mesh helper recentres initial XZ; put it in the same saved frame.
            origin=m['root_t'][0].copy();origin[1]=0
            positions+=origin@YUP_TO_ZUP.T/100.+lane[index+1]
            matrices=Rotation.from_quat(quats.reshape(-1,4),scalar_first=True).as_matrix().reshape(*quats.shape[:-1],3,3)
            tracks.append((positions,matrices));points.append(m['world_position_cm']@YUP_TO_ZUP.T/100.+lane[index+1])
            spec,dof,rest,*_=build_from_mjcf(r['xml'],r['body'])
            # Upstream caches depth by id(spec); short-lived render descriptors can
            # reuse an old id. Invalidate only this descriptor's stale entries.
            from alphamotion.engine import spatial
            for cache_key in [k for k in spatial._DEPTH if k[0]==id(spec)]:del spatial._DEPTH[cache_key]
            feet=key_joints(spec)[0][4:6]
            sole=build_foot_sole_model(r['xml'],spec,rest,['left_foot','right_foot'],feet)
            rotations=constraints.rot6d_to_matrix(torch.as_tensor(m['rot6d'])).numpy()
            feet_world=[]
            for foot in sole.feet.values():
                vertices=np.einsum('tij,vj->tvi',rotations[:,foot.joint],foot.vertices_local_cm)+m['world_position_cm'][:,foot.joint,None,:]
                feet_world.append(vertices)
            patches.append(feet_world)
            scene.attach(mj.MjSpec.from_file(r['xml']),prefix=f'actor{index}_',frame=scene.worldbody.add_frame())
        model=scene.compile();_configure_model(model);model.vis.headlight.ambient[:]=.55;model.vis.headlight.diffuse[:]=.25
        data=mj.MjData(model);mj.mj_forward(model,data);idsets=[]
        for index,(r,(positions,_)) in enumerate(zip(robots,tracks)):
            ids=[i for i in range(model.ngeom) if model.geom_type[i]==mj.mjtGeom.mjGEOM_MESH and (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or '').startswith(f'actor{index}_')]
            visual=[i for i in ids if model.geom_contype[i]==0 and model.geom_conaffinity[i]==0];ids=visual or ids
            assert len(ids)==positions.shape[1],(r['name'],len(ids),positions.shape[1]);idsets.append(ids)
        visible=set(sum(idsets,[]))
        for i in range(model.ngeom):model.geom_rgba[i,3]=int(i in visible or mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,i)=='ground')
        # Build labels from every rendered mesh surface.  Feet, knees, hands,
        # forearms and the torso therefore use one geometry/ground rule.
        surface_sets=[]
        for index,(r,ids,(positions,matrices)) in enumerate(zip(robots,idsets,tracks)):
            names=[];locals_=[];heights=[]
            for local_index,gid in enumerate(ids):
                mesh_id=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mesh_id]);count=int(model.mesh_vertnum[mesh_id])
                vertices=np.asarray(model.mesh_vert[start:start+count],dtype=float)
                world=np.einsum('tij,vj->tvi',matrices[:,local_index],vertices)+positions[:,local_index,None,:]
                body=mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[gid])) or f'geom_{gid}'
                names.append(body.removeprefix(f'actor{index}_'));locals_.append(vertices);heights.append(world[:,:,2].min(1)*100.)
            labelled=export_surface_contacts(np.stack(heights,1),names,fps,a.root/row['stem']/r['name'])
            surface_sets.append([{'gid':gid,'local':local,'contact':labelled['contact'][:,j],
                                  'penetration':labelled['penetration'][:,j]}
                                 for j,(gid,local) in enumerate(zip(ids,locals_))])
        centre=source[:,1]-lane[0];centre=centre.copy();centre[:,2]=.85
        cloud=np.concatenate(points,1)-centre[:,None];low=cloud.min((0,1))-.15;high=cloud.max((0,1))+.15;target=(low+high)/2
        # Fit the actual moving formation, not its inflated world-axis box.
        relative=(cloud-target).reshape(-1,3)
        ty=np.tan(np.deg2rad(21))*.86;tx=np.tan(np.deg2rad(21))*WIDTH/HEIGHT*.94
        distance=np.max(relative@direction+np.maximum((np.abs(relative@right)+.25)/tx,(np.abs(relative@up)+.25)/ty))*1.03*a.camera_zoom
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,WIDTH);model.vis.global_.offheight=max(model.vis.global_.offheight,HEIGHT)
        lookat_bias=np.array([0.,0.,a.lookat_height_bias])
        camera=_set_fixed_camera(model,data,mj,'camera',target+centre[0]+direction*distance,target+centre[0]+lookat_bias);renderer=mj.Renderer(model,height=HEIGHT,width=WIDTH)
        video=a.output/(row['stem']+'_SOMA_'+('18robots' if len(robots)>4 else robot_names)+'.mp4')
        writer=subprocess.Popen([a.ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE);tiles=[]
        def line(start,end,radius,color):
            geom=renderer.scene.geoms[renderer.scene.ngeom]
            mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array(color))
            mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,radius,start,end);renderer.scene.ngeom+=1
        try:
            for f in range(T):
                for ids,(pos,rot) in zip(idsets,tracks):data.geom_xpos[ids]=pos[f];data.geom_xmat[ids]=rot[f].reshape(-1,9)
                _set_fixed_camera(model,data,mj,'camera',target+centre[f]+direction*distance,target+centre[f]+lookat_bias);renderer.update_scene(data,camera)
                for axis in (0,1):
                    for tick in range(-30,31):
                        start=np.array([-20.,-20.,.002]);end=np.array([20.,20.,.002]);start[axis]=end[axis]=tick*.5
                        line(start,end,.002,[.4,.38,.33,1])
                for j,parent in enumerate(parents):
                    if parent>=1:line(source[f,parent],source[f,j],.018,[.08,.4,.9,1])
                im=renderer.render().copy()
                def project(vertices):
                    cid=camera.fixedcamid
                    local=(vertices-data.cam_xpos[cid])@data.cam_xmat[cid].reshape(3,3)
                    focal=HEIGHT/(2*np.tan(np.deg2rad(float(model.cam_fovy[cid]))/2));z=np.maximum(-local[:,2],.001)
                    return np.column_stack([WIDTH/2+focal*local[:,0]/z,HEIGHT/2-focal*local[:,1]/z]).astype(np.int32)
                # Fill the projected native sole contact region, not a ring or full-foot recolor.
                paint=im.copy()
                snames=motions[0]['source_joint_names'].tolist()
                # SOMA is a skeleton rather than a closed human mesh, so non-foot
                # contacts are explicitly small joint-centred surface proxies.
                for joint,point in enumerate(source[f]):
                    if point[2]>.01 or snames[joint] in ('LeftFoot','RightFoot','LeftToeBase','RightToeBase'):continue
                    radius=.035;region=np.array([[point[0]-radius,point[1]-radius,.003],
                        [point[0]+radius,point[1]-radius,.003],[point[0]+radius,point[1]+radius,.003],
                        [point[0]-radius,point[1]+radius,.003]])
                    cv2.fillConvexPoly(paint,project(region),(240,35,35) if point[2]<-.005 else (20,235,55))
                for k,side in enumerate(['Left','Right']):
                    if not contacts[0]['source'][f,k]:continue
                    heel=source[f,snames.index(side+'Foot')].copy();toe=source[f,snames.index(side+'ToeBase')].copy()
                    heading=toe-heel;heading[2]=0;heading/=max(np.linalg.norm(heading),1e-6)
                    across=np.cross(heading,[0,0,1])*.045
                    region=np.array([heel-heading*.04+across,toe+heading*.04+across,toe+heading*.04-across,heel-heading*.04-across]);region[:,2]=.003
                    cv2.fillConvexPoly(paint,project(region),(20,235,55))
                for surfaces in surface_sets:
                    for surface in surfaces:
                        if not (surface['contact'][f] or surface['penetration'][f]):continue
                        gid=surface['gid'];rotation=data.geom_xmat[gid].reshape(3,3)
                        vertices=surface['local']@rotation.T+data.geom_xpos[gid]
                        near=vertices[vertices[:,2]<=.01]
                        if len(near)<3:continue
                        xy=near[:,:2]
                        try:hull=ConvexHull(xy).vertices
                        except Exception:continue
                        region=near[hull].copy();region[:,2]=.003
                        cv2.fillConvexPoly(paint,project(region),(240,35,35) if surface['penetration'][f] else (20,235,55))
                im=cv2.addWeighted(paint,a.contact_overlay_alpha,im,1.-a.contact_overlay_alpha,0)
                if not a.hide_robot_labels:
                    for index,m in enumerate(motions):
                        anchor=m['root_t'][f]@YUP_TO_ZUP.T/100.+lane[index+1];anchor[2]=0
                        pixel=project(anchor[None])[0];cv2.putText(im,robots[index]['name'].upper(),tuple(pixel+[0,35]),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,30),2)
                    cv2.putText(im,'SOMA',tuple(project(source[f,1:2])[0]+[-35,-140]),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,70,160),2)
                cv2.rectangle(im,(0,0),(WIDTH,104),(226,216,192),-1)
                cv2.putText(im,row['semantic_type'],(32,43),cv2.FONT_HERSHEY_SIMPLEX,1,(30,30,30),2)
                label=('SOMA (blue) + '+str(len(robots))+' robots' if a.hide_robot_labels else 'SOMA (blue) + '+' + '.join(r['name'].upper() for r in robots))+' | All-body ground contact: green; penetration: red'
                cv2.putText(im,f'{label}     Same scale, floor, time: {f/fps:.2f} / {T/fps:.2f}s',(32,83),cv2.FONT_HERSHEY_SIMPLEX,.72,(30,30,30),2)
                if not a.hide_robot_labels:
                    cv2.rectangle(im,(0,HEIGHT-94),(WIDTH,HEIGHT),(226,216,192),-1)
                    contact_columns=[('SOMA proxy',contacts[0]['source'][f],None)]+[(r['name'].upper()+' geometry',c['geometry'][f],c['penetration'][f]) for r,c in zip(robots,contacts)]
                    for i,(name,bits,pens) in enumerate(contact_columns):
                        x=10+int(i*WIDTH/len(contact_columns))
                        cv2.putText(im,name,(x,HEIGHT-62),cv2.FONT_HERSHEY_SIMPLEX,.48 if len(robots)>4 else .65,(30,30,30),1)
                        state='  '.join(side+': '+('PEN' if pens is not None and pens[k] else 'ON' if bits[k] else 'OFF') for k,side in enumerate(['L','R']))
                        cv2.putText(im,state,(x,HEIGHT-23),cv2.FONT_HERSHEY_SIMPLEX,.48 if len(robots)>4 else .72,(30,30,30),1)
                writer.stdin.write(im.tobytes())
                if f%max(1,round(fps/10))==0:tiles.append(cv2.cvtColor(cv2.resize(im,(640,360)),cv2.COLOR_RGB2BGR))
        finally:
            writer.stdin.close();code=writer.wait();renderer.close()
            if code:raise RuntimeError('FFmpeg failed')
        for page,k in enumerate(range(0,len(tiles),16)):
            batch=tiles[k:k+16]
            while len(batch)%4:batch.append(np.zeros_like(batch[0]))
            cv2.imwrite(str(a.output/(row['stem']+f'_review_{page}.jpg')),np.concatenate([np.concatenate(batch[j:j+4],1) for j in range(0,len(batch),4)],0))
        print(str(video),flush=True)
        videos.append(video)
    # Compact remote previews and a combined reel are reproducible render outputs.
    playlist=a.output/'concat.txt'
    playlist.write_text(''.join("file '"+v.name+"'\n" for v in videos),encoding='utf-8')
    combined=a.output/('SOMA_'+('18robots' if len(robots)>4 else robot_names)+'_all.mp4')
    subprocess.run([a.ffmpeg,'-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy','-movflags','+faststart','-y',str(combined)],check=True)
    for video in videos+[combined]:
        small=video.with_name(video.stem+'_small.mp4')
        preview_height=min(540,HEIGHT);preview_width=round(preview_height*16/9/2)*2
        subprocess.run([a.ffmpeg,'-hide_banner','-loglevel','error','-i',str(video),'-vf',f'scale={preview_width}:{preview_height}','-c:v','libx264','-preset','fast','-b:v','450k','-maxrate','550k','-bufsize','900k','-pix_fmt','yuv420p','-an','-movflags','+faststart','-y',str(small)],check=True)
    print('Preview reel: '+str(combined.with_name(combined.stem+'_small.mp4')),flush=True)
    # A portable local review page; no server or additional listening port.
    cards=[]
    for row,video in zip(selection,videos):
        title=html.escape(row['semantic_type'])
        small=html.escape(video.stem+'_small.mp4',quote=True)
        full=html.escape(video.name,quote=True)
        cards.append(f'<section><h2>{title}</h2><video controls preload="none" src="{small}"></video><p><a href="{full}">Full resolution</a></p></section>')
    reel=html.escape(combined.stem+'_small.mp4',quote=True)
    page='<!doctype html><meta charset="utf-8"><title>SOMA comparison</title><style>body{background:#efe6d4;color:#25231f;font:17px system-ui;max-width:1400px;margin:32px auto;padding:16px}video{width:100%;background:#181818}section{margin:32px 0}a{color:#164c88}</style>'
    page+=f'<h1>SOMA / {html.escape(robot_names.replace("_"," / "))}</h1><p>Same source, scale, floor and timestamps. Shared refinement settings.</p><h2>All actions</h2><video controls preload="metadata" src="{reel}"></video>'+''.join(cards)
    (a.output/'index.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
