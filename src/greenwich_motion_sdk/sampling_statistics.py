"""Finite-population proportion estimates for SRS without replacement."""
import math
from scipy.stats import hypergeom,norm


def proportion_interval(successes,sample_size,population,alpha=.05):
    """Equal-tail exact hypergeometric inversion, conservative discrete coverage."""
    k,n,N=int(successes),int(sample_size),int(population)
    if not 0<=k<=n<=N or n<1 or not 0<alpha<1:raise ValueError('Invalid sampling counts or alpha')
    lo,hi=0,N
    while lo<hi:
        mid=(lo+hi)//2
        if hypergeom.sf(k-1,N,mid,n)>=alpha/2:hi=mid
        else:lo=mid+1
    lower=lo
    lo,hi=0,N
    while lo<hi:
        mid=(lo+hi+1)//2
        if hypergeom.cdf(k,N,mid,n)>=alpha/2:lo=mid
        else:hi=mid-1
    upper=lo;p=k/n
    worst_margin=float(norm.ppf(1-alpha/2)*math.sqrt(.25/n*(N-n)/(N-1))) if N>1 else 0.
    return {'successes':k,'sample_size':n,'population':N,'alpha':alpha,
        'confidence_level':1-alpha,'estimated_proportion':p,
        'confidence_interval':[lower/N,upper/N],
        'population_count_interval':[lower,upper],
        'estimated_population_count':p*N,
        'worst_case_planning_margin':worst_margin,
        'method':'Exact equal-tail hypergeometric inversion for a fixed SRS without replacement; planning margin uses normal approximation with finite population correction.'}


def report_markdown(stats):
    """Human-readable estimate with policy and finite-population scope explicit."""
    good=stats['good'];flagged=stats['flagged'];unknown=stats['accepted_unconfirmed_support']
    lines=['# H2 支撑脚滑动随机抽样报告','',
        f'总体：{good["population"]:,} 条唯一动作；样本：{good["sample_size"]:,} 条；不放回简单随机抽样，随机种子 {stats["seed"]}。',
        '判断：明确固定支撑的脚面速度超过 15 cm/s，连续至少 0.1 s 才提示。无法确认固定支撑的按用户要求计为好，原始测量标记保留。','',
        '| 结果 | 样本条数 | 比例估计 | 总体比例的95%置信区间 |',
        '|---|---:|---:|---:|']
    for name,value in (('好',good),('滑动提示',flagged),('无法确认支撑、计为好（包含在“好”中）',unknown)):
        low,high=value['confidence_interval']
        lines.append(f'| {name} | {value["successes"]} | {value["estimated_proportion"]:.2%} | {low:.2%}–{high:.2%} |')
    lines+=['',
        f'估计总体约 {good["estimated_population_count"]:,.0f} 条好；对应95%区间为 {good["population_count_interval"][0]:,}–{good["population_count_interval"][1]:,} 条。',
        f'估计总体约 {flagged["estimated_population_count"]:,.0f} 条触发提示；对应95%区间为 {flagged["population_count_interval"][0]:,}–{flagged["population_count_interval"][1]:,} 条。',
        f'抽样规划的最坏情况误差约 ±{good["worst_case_planning_margin"]*100:.2f} 个百分点（正态近似并作有限总体修正）；实际区间使用超几何分布精确反演，显著性水平 α=0.05。',
        '置信区间只描述这批固定数据按当前规则分类的比例，不包含机筛误判的不确定性，也不等于动力学可执行率。','',
        '## 分类别结果','',
        '| 类别 | 总体条数 | 样本条数 | 好的估计比例 | 95%区间 |',
        '|---|---:|---:|---:|---:|']
    for name,row in sorted(stats['categories'].items()):
        value=row['good']
        p=f'{value["estimated_proportion"]:.2%}' if value else '未抽中'
        ci=f'{value["confidence_interval"][0]:.2%}–{value["confidence_interval"][1]:.2%}' if value else '—'
        lines.append(f'| {name} | {row["population"]} | {row["sample_count"]} | {p} | {ci} |')
    lines+=['','类别区间没有作多重比较修正，样本较少的类别精度有限。','',
        'AQL（可接受质量水平）属于整批验收抽样，需要先约定 AQL/RQL 以及双方风险。本次没有擅自设定这些业务参数，使用总体比例估计。','',
        '方法参考：[NIST 样本量规划](https://www.itl.nist.gov/div898/handbook/ppc/section3/ppc333.htm)、[NIST 验收抽样](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc23.htm)。','']
    return '\n'.join(lines)


def main():
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description='Render a cached statistical report without rerunning the audit')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();stats=json.loads(a.input.read_text(encoding='utf-8'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(report_markdown(stats),encoding='utf-8')


if __name__=='__main__':main()
