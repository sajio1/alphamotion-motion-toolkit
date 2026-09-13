"""Re-read saved SOMA runs, check video lengths, export review index and sampled previews."""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    rows=json.loads((a.root/'results.json').read_text());selected=json.loads((a.root/'selection.json').read_text())
    review=a.root/'review';review.mkdir(exist_ok=True);summary=[]
    robots=list(dict.fromkeys(r['robot'] for r in rows))
    md=[f'# Prime × BoneSeed：{len(selected)} 条全身动作','',
        rows[0]['input']+'。'+rows[0]['projection']+'。'+rows[0]['ground'],'',
        '这是诊断结果，不是已通过的动作库。支撑状态有模型预测和源脚踝/脚趾近地代理两种记录；源代理不是实际接触力。','',
        '|动作|'+'|'.join(x.upper() for x in robots)+'|','|'+'---|'*(len(robots)+1)]
    for item in selected:
        source=item['stem'];tiles=[]
        for row in [r for r in rows if r['source']==source]:
            folder=a.root/source/row['robot'];video=folder/'preview.mp4';cap=cv2.VideoCapture(str(video))
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
            if n!=row['frames'] or abs(fps-row['fps'])>.01:raise ValueError(f'Video timing mismatch {video}')
            for frame in np.linspace(0,n-1,12,dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,im=cap.read()
                if not ok:raise ValueError('Unreadable frame')
                im=cv2.resize(im,(320,180));cv2.rectangle(im,(0,0),(200,21),(0,0,0),-1)
                cv2.putText(im,f'{row["robot"]} {frame/fps:.2f}s',(4,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);tiles.append(im)
            cap.release()
            summary.append({'source':source,'robot':row['robot'],'frames':n,'duration_s':n/fps,'max_penetration_cm':row['max_penetration_cm'],'source_support_clearance_p95_cm':row['source_support_clearance_p95_cm'],'root_min_cm':row['root_height_cm'][0],'root_max_cm':row['root_height_cm'][1],'encode_s':row['timing_s']['encode_shared'],'decode_s':row['timing_s']['decode'],'projection_s':row['timing_s']['joint_projection'],'render_s':row['timing_s']['render']})
        if tiles:
            while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(review/(source+'.jpg')),np.concatenate([np.concatenate(tiles[k:k+4],axis=1) for k in range(0,len(tiles),4)],axis=0))
        md.append('|'+item['semantic_type']+'|'+'|'.join(f'[视频]({source}/{r}/preview.mp4)' for r in robots)+'|')
    with (a.root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    audit={'clips':len(selected),'robot_results':len(rows),'video_timing_checked':True,'preview_sampling':'12 frames per video, recorded in review/*.jpg; not exhaustive playback','results_with_penetration_over_1cm':sum(x['max_penetration_cm']>1 for x in summary),'all_numeric_motion_results_accepted':False,'total_video_seconds':sum(x['duration_s'] for x in summary)}
    (a.root/'summary.json').write_text(json.dumps(audit,indent=2));(a.root/'INDEX.md').write_text('\n'.join(md),encoding='utf-8');print(json.dumps(audit))


if __name__=='__main__':main()
