"""Aggregate locomotion ground penetration and support-height error from NPZ."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np


def stats(values):
    x=np.asarray(values,float).reshape(-1);x=x[np.isfinite(x)]
    return {"mean":float(x.mean()),"p95":float(np.percentile(x,95)),"max":float(x.max())} if len(x) else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[];all_pen=[];all_error=[]
    for path in sorted(a.root.glob('*/*/motion.npz')):
        with np.load(path,allow_pickle=False) as m:
            h=np.asarray(m['sole_surface_height_cm'],float);support=np.asarray(m['source_contact_proxy'],bool)
            penetration=np.maximum(-h,0);error=np.abs(h[support])
            row={"clip":path.parents[1].name,"robot":path.parent.name,"frames":len(h),
                 "penetration_cm":stats(penetration),"support_ground_error_cm":stats(error),
                 "support_frames":int(support.sum())}
            rows.append(row);all_pen.append(penetration.reshape(-1));
            if len(error):all_error.append(error)
    report={"schema":"greenwich-locomotion-ground-error-v1","motions":len(rows),
            "definitions":{"penetration_cm":"max(0, -lowest native sole mesh Y), ground Y=0",
                           "support_ground_error_cm":"abs(lowest native sole mesh Y) only where SOMA source support proxy is true"},
            "aggregate":{"penetration_cm":stats(np.concatenate(all_pen)),
                         "support_ground_error_cm":stats(np.concatenate(all_error)) if all_error else None},
            "results":rows}
    (a.root/'ground_error_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (a.root/'ground_error_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['clip','robot','frames','support_frames','penetration_max_cm','support_error_p95_cm','support_error_max_cm']);w.writeheader()
        for r in rows:w.writerow({'clip':r['clip'],'robot':r['robot'],'frames':r['frames'],'support_frames':r['support_frames'],
            'penetration_max_cm':r['penetration_cm']['max'],'support_error_p95_cm':r['support_ground_error_cm']['p95'] if r['support_ground_error_cm'] else None,
            'support_error_max_cm':r['support_ground_error_cm']['max'] if r['support_ground_error_cm'] else None})
    ordered=sorted(rows,key=lambda r:(r['support_ground_error_cm'] or {'max':-1})['max'],reverse=True)
    body=['<!doctype html><meta charset="utf-8"><title>Ground error</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}</style>',
          '<h1>Locomotion ground error</h1>',
          f"<p>Penetration P95/max: {report['aggregate']['penetration_cm']['p95']:.3f}/{report['aggregate']['penetration_cm']['max']:.3f} cm. Support-ground error P95/max: {report['aggregate']['support_ground_error_cm']['p95']:.3f}/{report['aggregate']['support_ground_error_cm']['max']:.3f} cm.</p>",
          '<table><tr><th>Clip</th><th>Penetration max cm</th><th>Support error P95 cm</th><th>Support error max cm</th></tr>']
    for r in ordered:
        e=r['support_ground_error_cm'];body.append(f"<tr><td>{html.escape(r['clip'])}</td><td>{r['penetration_cm']['max']:.3f}</td><td>{e['p95']:.3f}</td><td>{e['max']:.3f}</td></tr>")
    body.append('</table>');(a.root/'ground_error_report.html').write_text(''.join(body),encoding='utf-8')
    index=a.root/'vid'/'index.html'
    if index.exists():
        page=index.read_text(encoding='utf-8');marker='<!-- ground-error-link -->'
        if marker not in page:
            page=page.replace('</h1>',f'</h1>{marker}<p><a href="../ground_error_report.html">Ground penetration and support-height error report</a></p>',1)
            index.write_text(page,encoding='utf-8')
    print(json.dumps(report['aggregate'],indent=2))


if __name__=='__main__':main()
