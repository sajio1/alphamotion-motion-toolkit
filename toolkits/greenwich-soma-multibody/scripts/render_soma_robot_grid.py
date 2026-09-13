"""Synchronized multi-embodiment grid with foot contact rings, at shared physical scale."""
import argparse,json,sys,os,subprocess,math
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','pipeline','root','robots','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--layout',default='grid')
    p.add_argument('--indices')
    p.add_argument('--ffmpeg',required=True);a=p.parse_args()
    sys.path[:0]=[str(a.repo/'src'),str(a.pipeline/'src'),str(a.pipeline/'scripts'),str(Path(__file__).resolve().parents[3]/'src')]
    os.environ['ALPHAMOTION_CACHE']=str(a.root/'cache');os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HOME']=str(a.repo/'.local/huggingface')
    import mujoco as mj,cv2
    from scipy.spatial.transform import Rotation
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine.spatial import key_joints
    from greenwich_umi_proof.interactive_viewer import _sample_greenwich_head,YUP_TO_ZUP
    from render_hiw500_transfer_showcase import _configure_model,_set_fixed_camera
    from greenwich_motion_sdk.contact_labels import export_contacts
    from greenwich_motion_sdk.video import export
    robots=json.loads(a.robots.read_text(encoding='utf-8-sig'));selection=json.loads((a.root/'selection.json').read_text())
    if a.indices:selection=[r for r in selection if r['index'] in set(map(int,a.indices.split(',')))]
    a.output.mkdir(parents=True,exist_ok=True);videos=[];width,height=640,480
    direction=np.array([1.,-1.,.3]);direction/=np.linalg.norm(direction)
    for row in selection:
        tile_dir=a.output/(row['stem']+'_tiles');tile_dir.mkdir(exist_ok=True)
        first=np.load(a.root/row['stem']/robots[0]['name']/'motion.npz')
        source=first['source_positions_cm']@YUP_TO_ZUP.T/100.;parents=first['source_parents'];names=first['source_joint_names'].tolist()
        centres=source[:,1].copy();centres[:,2]=.95
        fps=float(first['fps']);T=len(centres);tile_videos=[]
        # Same scale and camera distance for all panels; no per-robot resizing.
        distance=max(4.2,float(np.max(source[:,:,2]))/0.45)
        for robot in robots:
            with np.load(a.root/row['stem']/robot['name']/'motion.npz') as m:
                distance=max(distance,float(m['world_position_cm'][:,:,1].max()/100)/.45)
        for index,robot in enumerate([None]+robots):
            name='SOMA' if robot is None else robot['name'];tile=tile_dir/(name+'.mp4');tile_videos.append(tile)
            scene=mj.MjSpec();scene.add_material(name='floor_mat',rgba=[.84,.80,.69,1])
            scene.worldbody.add_geom(name='ground',type=mj.mjtGeom.mjGEOM_PLANE,size=[100,100,.1],material='floor_mat')
            scene.worldbody.add_camera(name='camera',fovy=42)
            scene.worldbody.add_light(pos=[3,-4,7],dir=[-.2,.3,-1],diffuse=[.5]*3)
            if robot:
                path=a.root/row['stem']/name/'motion.npz';m=np.load(path)
                if len(m['root_t'])!=T or float(m['fps'])!=fps:raise ValueError('Unsynchronized grid input')
                labels=export_contacts(m,path.parent)
                pos,quat,*_=_sample_greenwich_head(path,robot['xml'],robot['body'],rotation_key='rot6d',world_position_key='world_position_cm',floor_y_cm=0.)
                origin=m['root_t'][0].copy();origin[1]=0;pos+=origin@YUP_TO_ZUP.T/100.
                rot=Rotation.from_quat(quat.reshape(-1,4),scalar_first=True).as_matrix().reshape(*quat.shape[:-1],3,3)
                spec,*_=build_from_mjcf(robot['xml'],robot['body']);feet=key_joints(spec)[0][4:6]
                foot=m['world_position_cm'][:,feet]@YUP_TO_ZUP.T/100.
                foot[:,:,2]=np.maximum(m['sole_surface_height_cm']/100.,0)+.012
                scene.attach(mj.MjSpec.from_file(robot['xml']),prefix='actor_',frame=scene.worldbody.add_frame())
            else:
                labels=dict(geometry=first['source_contact_proxy'],penetration=np.zeros((T,2),bool))
                foot=source[:,[names.index('LeftFoot'),names.index('RightFoot')]].copy();foot[:,:,2]=.012
            model=scene.compile();_configure_model(model);model.vis.headlight.ambient[:]=.5
            data=mj.MjData(model);mj.mj_forward(model,data)
            ids=[i for i in range(model.ngeom) if model.geom_type[i]==mj.mjtGeom.mjGEOM_MESH]
            visual=[i for i in ids if model.geom_contype[i]==0 and model.geom_conaffinity[i]==0];ids=visual or ids
            if robot and len(ids)!=pos.shape[1]:raise ValueError('Native mesh ownership mismatch: '+name)
            for i in range(model.ngeom):model.geom_rgba[i,3]=int(i in ids or mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,i)=='ground')
            camera=_set_fixed_camera(model,data,mj,'camera',centres[0]+direction*distance,centres[0])
            renderer=mj.Renderer(model,height=height,width=width)
            cmd=[a.ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(tile)]
            writer=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            def line(start,end,radius,color):
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array(color))
                mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,radius,start,end);renderer.scene.ngeom+=1
            try:
                for f in range(T):
                    if robot:data.geom_xpos[ids]=pos[f];data.geom_xmat[ids]=rot[f].reshape(-1,9)
                    _set_fixed_camera(model,data,mj,'camera',centres[f]+direction*distance,centres[f]);renderer.update_scene(data,camera)
                    for axis in (0,1):
                        for tick in range(-12,13):
                            start=np.array([-10.,-10.,.002]);end=np.array([10.,10.,.002]);start[axis]=end[axis]=tick*.5;line(start,end,.002,[.35,.33,.30,1])
                    if robot is None:
                        for j,parent in enumerate(parents):
                            if parent>=1:line(source[f,parent],source[f,j],.018,[.08,.4,.9,1])
                    for k in range(2):
                        color=[1.,.05,.05,1] if labels['penetration'][f,k] else [0.,.9,.15,1] if labels['geometry'][f,k] else [1.,.55,0.,1]
                        ring=foot[f,k]+np.column_stack([.13*np.cos(np.linspace(0,2*np.pi,21)),.13*np.sin(np.linspace(0,2*np.pi,21)),np.zeros(21)])
                        for j in range(20):line(ring[j],ring[j+1],.013,color)
                    im=renderer.render().copy();cv2.rectangle(im,(0,0),(width,40),(226,216,192),-1)
                    cv2.putText(im,name.upper()+f'  {f/fps:.2f}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.72,(30,30,30),2)
                    cv2.rectangle(im,(0,height-32),(width,height),(226,216,192),-1)
                    state='  '.join(side+': '+('PEN' if labels['penetration'][f,k] else 'ON' if labels['geometry'][f,k] else 'OFF') for k,side in enumerate(['L','R']))
                    cv2.putText(im,state+('  source proxy' if robot is None else '  geometry'),(12,height-10),cv2.FONT_HERSHEY_SIMPLEX,.58,(30,30,30),1)
                    writer.stdin.write(im.tobytes())
            finally:
                writer.stdin.close();code=writer.wait();renderer.close()
                if robot:m.close()
            if code:raise RuntimeError('FFmpeg failed: '+name)
        columns=min(5,len(tile_videos));layout='|'.join(f'{i%columns*width}_{i//columns*height}' for i in range(len(tile_videos)))
        video=a.output/(row['stem']+'_contact_grid.mp4');cmd=[a.ffmpeg,'-y','-v','error','-filter_complex_threads','1']
        for tile in tile_videos:cmd+=['-i',str(tile)]
        cmd+=['-filter_complex',f'xstack=inputs={len(tile_videos)}:layout={layout}:fill=0xe2d8c0','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        subprocess.run(cmd,check=True);videos.append(video);print(str(video),flush=True)
        first.close()
    playlist=a.output/'grid_concat.txt';playlist.write_text(''.join("file '"+v.name+"'\n" for v in videos))
    reel=a.output/'SOMA_all_robots_contact_grid.mp4'
    subprocess.run([a.ffmpeg,'-y','-v','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy','-movflags','+faststart',str(reel)],check=True)
    export(reel,reel.with_stem(reel.stem+'_1080p'),a.ffmpeg,height=1080,crf=26)
    (a.output/'index.html').write_text('<meta charset="utf-8"><h1>SOMA + all robots</h1><p>Green: near floor; orange: off floor; red: penetration. Geometry/proxy labels, not measured contact forces.</p><video controls style="width:100%" src="SOMA_all_robots_contact_grid_1080p.mp4"></video>')

if __name__=='__main__':main()
