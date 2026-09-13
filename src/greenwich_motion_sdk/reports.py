"""Rebuild comparison reports from saved audits; never rerun inference for thresholds."""
from pathlib import Path
import csv,html,json,os
import numpy as np

def compare_runs(runs:dict[str,Path],output:Path,review_notes=()):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];links=[]
    for label,root in runs.items():
        root=Path(root)
        audit=json.loads((root/'accuracy.json').read_text())
        results=audit['results']
        for robot in dict.fromkeys(r['robot'] for r in results):
            rr=[r for r in results if r['robot']==robot]
            rows.append({'input':label,'robot':robot,'clips':len(rr),
                'left_hand_mean_clip_p95_cm':float(np.mean([r['root_relative_difference_cm']['l_wrist']['p95'] for r in rr])),
                'right_hand_mean_clip_p95_cm':float(np.mean([r['root_relative_difference_cm']['r_wrist']['p95'] for r in rr])),
                'max_penetration_cm':max(r['max_penetration_cm'] for r in rr),
                'support_slip_mean_clip_p95_cm_s':float(np.mean([r['support_slip_p95_cm_s'] for r in rr if r['support_slip_p95_cm_s'] is not None])),
                'motion_compute_mean_s':float(np.mean([r['motion_compute_s'] for r in rr]))})
        videos=sorted((root/'vid').glob('*_all.mp4'))
        if videos:links.append((label,Path(os.path.relpath(videos[0],output)).as_posix()))
    text=['# Prime 全身转换小样本对照','',
        f'共 {len(runs)} 组、{sum(r["clips"] for r in rows)} 个转换结果；只统计调用方指定的目录。',
        'SOMA77 直接编码；SMPL22 为同源派生对照。BVH 是这些 SOMA 动作的容器格式，不是第三套独立数据。',
        '没有配对机器人真值。下表手部数值是相对骨盆的源/机器人位置差，保留原尺寸，包含身体比例差异；不是精修 EE 优化残差，也不是硬件精度。',
        'P95 为每条动作分别计算，再对组内片段取均值；穿地为组内最大值。模型计时包含共享编码的单条成本，不包含冷启动和渲染。','',
        '|输入|机器人|左手位置差 cm|右手位置差 cm|最大穿地 cm|支撑滑移 cm/s|生成含精修 s|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"|{r['input']}|{r['robot']}|{r['left_hand_mean_clip_p95_cm']:.2f}|{r['right_hand_mean_clip_p95_cm']:.2f}|{r['max_penetration_cm']:.3f}|{r['support_slip_mean_clip_p95_cm_s']:.2f}|{r['motion_compute_mean_s']:.2f}|")
    text+=['','## 解读','',
        '接地误差小不代表动作已准确。',
        '全程使用相同接地配置；source XZ 是输入引导，不作为模型自主水平位移能力的证据。',
        *list(review_notes),'', '## 视频','']
    text += [f'- [{label}：源骨架 + A3/H2/G1]({url})' for label,url in links]
    (output/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    (output/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    page=['<!doctype html><meta charset="utf-8"><title>Prime 输入表示对照</title>',
          '<style>body{background:#eee5d3;color:#282520;font:16px system-ui;max-width:1280px;margin:30px auto}video{width:100%}pre{white-space:pre-wrap}</style>',
          '<h1>Prime 输入表示对照 · A3 / H2 / G1</h1>',
          '<p>同源派生 SMPL22 对照，不是独立 SMPL 数据集。数值完成不等于审核通过。</p>']
    for label,url in links:page.append(f'<h2>{html.escape(label)}</h2><video controls preload="metadata" src="{html.escape(url)}"></video>')
    page.append('<pre>'+html.escape('\n'.join(text))+'</pre>')
    (output/'index.html').write_text('\n'.join(page),encoding='utf-8')
    return rows
