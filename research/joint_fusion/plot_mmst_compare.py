"""Committed training probes and paired three-seed final prediction comparisons."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from cropnet_leo.remote import sha
from .mmst_io import write_json

METHODS=('centralized','fedavg','hierarchical_async')
LABELS=('Centralized','FedAvg','Hierarchical async (balanced)')
COLORS=('#2665a0','#299c7c','#de793d')
KEYS=('rmse','r2_author_corr_squared','pearson_corr')
TITLES=('RMSE (BU/ACRE)','Author R² = Corr²','Pearson Corr')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def committed_points(out,method,seed):
    arm=out/'training'/f'{method}_seed{seed}'
    pointer=arm/'recovery.json'
    limit=read(pointer)['event_count'] if pointer.exists() else 0
    points=[]
    for path in sorted((out/'curves'/f'{method}_seed{seed}').glob('point_*.json')):
        if path.name.endswith('.sha256.json'):continue
        sidecar=path.with_suffix('.sha256.json')
        if not sidecar.exists():continue  # The writer may still be committing this point.
        point=read(path)
        if point['event_count']>limit:continue
        if sha(path)!=read(sidecar)['sha256']:raise ValueError('Training probe checksum mismatch')
        if point['identity']['scope']!='fixed_training_probe_not_heldout':raise ValueError('Unexpected probe scope')
        if (point['identity']['method'],point['identity']['seed'])!=(method,seed):raise ValueError('Probe arm mismatch')
        points.append(point)
    return points


def save_formats(fig,folder,name):
    for ext in ('png','svg','pdf'):fig.savefig(folder/f'{name}.{ext}',dpi=220,bbox_inches='tight')


def plot_curves(out):
    out=Path(out);config=read(out/'config.json');plt=plotting()
    points={(m,s):committed_points(out,m,s) for m in METHODS for s in config['seeds']}
    if not any(points.values()):return 'No committed training probes yet.'
    folder=out/'curve_reports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    keys=(*KEYS,'mean_error');titles=(*TITLES,'Mean prediction error (BU/ACRE)')
    counts={}
    for method,label,color in zip(METHODS,LABELS,COLORS):
        available=[s for s in config['seeds'] if points[method,s]]
        counts[method]={str(s):len(points[method,s]) for s in config['seeds']}
        for si,seed in enumerate(available):
            rows=points[method,seed];x=[p['accepted_epoch_equivalent'] for p in rows]
            for ax,key in zip(axes.flat,keys):
                y=[np.nan if p['metrics'][key] is None else p['metrics'][key] for p in rows]
                ax.plot(x,y,color=color,lw=.85,alpha=.55,ls=('-', '--', ':')[config['seeds'].index(seed)%3],
                        label=label+f' (n={len(available)})' if si==0 else None)
        # Mean/SD only at shared events after every planned seed is available.
        if len(available)==len(config['seeds']) and len(available)>1:
            byseed=[{p['event_count']:p for p in points[method,s]} for s in available]
            common=sorted(set.intersection(*(set(d) for d in byseed)))
            x=[byseed[0][i]['accepted_epoch_equivalent'] for i in common]
            for d in byseed[1:]:
                if [d[i]['accepted_epoch_equivalent'] for i in common]!=x:raise ValueError('Seed budgets differ')
            for ax,key in zip(axes.flat,keys):
                values=np.array([[np.nan if d[i]['metrics'][key] is None else d[i]['metrics'][key] for i in common] for d in byseed])
                mean,sd=values.mean(0),values.std(0,ddof=1)
                ax.plot(x,mean,color=color,lw=1.8);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.12)
    for ax,title in zip(axes.flat,titles):
        ax.set(xlabel='Accepted epoch equivalent (accepted counties / training counties)',ylabel=title)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[1,1].axhline(0,color='#888',lw=.7);axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Fixed training probe — not validation or test\nThin lines: individual seeds; thick / band: all-seed mean / sample SD',fontsize=12)
    save_formats(fig,folder,'training_metric_curves');plt.close(fig)
    write_json(folder/'observations.json',{'scope':'fixed_training_probe_not_heldout','committed_points':counts,
               'observations':[{ 'method':m,'seed':s,'points':points[m,s]} for m in METHODS for s in config['seeds']]})
    write_json(folder/'artifacts_sha256.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    write_json(out/'latest_curves.json',{'path':str(folder),'scope':'training_probe_not_heldout'})
    return str(folder/'training_metric_curves.png')


def statistics(values):
    if any(v is None for v in values):return {'mean':None,'sample_sd':None,'n':len(values)}
    return {'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'n':len(values)}


def format_stat(item):
    if item['mean'] is None:return 'undefined'
    return f"{item['mean']:.4f}"+(f" ± {item['sample_sd']:.4f}" if item['sample_sd'] is not None else '')


def plot_results(out):
    out=Path(out);config=read(out/'config.json');seeds=config['seeds'];results=read(out/'metrics.json')
    expected={(m,s) for m in METHODS for s in seeds}
    if read(out/'status.json')!={'status':'completed','stage':'historical_test','runs':len(expected)}:
        raise ValueError('Final evaluation is incomplete')
    if len(results)!=len(expected) or {(r['method'],r['seed']) for r in results}!=expected:raise ValueError('Evaluation matrix incomplete')
    grouped={m:[next(r for r in results if (r['method'],r['seed'])==(m,s)) for s in seeds] for m in METHODS}
    summary={m:{k:statistics([r['metrics'][k] for r in grouped[m]]) for k in (*KEYS,'r2_coefficient_of_determination')} for m in METHODS}
    for m in METHODS:
        summary[m]['paired_rmse_gap_percent']=statistics([r['rmse_gap_percent'] for r in grouped[m]])
    write_json(out/'seed_summary.json',summary)
    n={r['metrics']['n'] for r in results}
    if len(n)!=1:raise ValueError('Evaluation sample counts differ')
    n=n.pop();plt=plotting()
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    for ax,key,title in zip(axes,KEYS,TITLES):
        means=[summary[m][key]['mean'] for m in METHODS];sds=[summary[m][key]['sample_sd'] for m in METHODS]
        for i,(mean,sd,color) in enumerate(zip(means,sds,COLORS)):
            if mean is None:ax.text(i,0,'undefined',ha='center');continue
            ax.bar(i,mean,yerr=sd,capsize=4,color=color,alpha=.75)
            ax.scatter(i+np.linspace(-.1,.1,len(seeds)),[r['metrics'][key] for r in grouped[METHODS[i]]],color='#333',s=14,zorder=5)
        ax.set_xticks(range(3),['Centralized','FedAvg','Balanced\nasync']);ax.set_ylabel(title)
        ax.spines[['top','right']].set_visible(False);ax.margins(y=.15)
    fig.suptitle(f'Soybeans / {n} counties / historical {config["evaluation_year"]} evaluation\n{len(seeds)} paired seeds; fixed final {config["epochs"]} epochs; mean ± sample SD')
    save_formats(fig,out,'centralized_vs_federated');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    for ax,method,label,color in zip(axes,METHODS,LABELS,COLORS):
        bounds=[]
        for seed,marker in zip(seeds,('o','x','+')):
            rows=read(out/f'{method}_seed{seed}_predictions.json')
            y=np.array([r['truth'] for r in rows]);p=np.array([r['prediction'] for r in rows])
            ax.scatter(y,p,s=12,alpha=.35,c=color,marker=marker,label=f'seed {seed}')
            bounds.extend([y.min(),y.max(),p.min(),p.max()])
        lo,hi=min(bounds),max(bounds);ax.plot([lo,hi],[lo,hi],'--',c='#777',lw=.8)
        ax.set(title=label,xlabel='Observed yield (BU/ACRE)',ylabel='Predicted yield (BU/ACRE)');ax.legend(frameon=False,fontsize=8)
    save_formats(fig,out,'prediction_scatter');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3.7));ax.axis('off')
    body=[[label,*[format_stat(summary[m][k]) for k in KEYS]] for m,label in zip(METHODS,LABELS)]
    table=ax.table(cellText=body,colLabels=['Training method','RMSE ↓','R² = Corr² ↑','Corr ↑'],colWidths=[.36,.22,.21,.21],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2)
    for (row,col),cell in table.get_celld().items():
        cell.visible_edges='TB' if row==0 else ('B' if row==3 else '');cell.set_linewidth(.7)
    ax.set_title(f'Soybeans / same MMST-ViT backbone / {len(seeds)} paired seeds\nHistorical {config["evaluation_year"]} evaluation: {n} counties',fontsize=13)
    fig.text(.5,.035,f'Fixed final {config["epochs"]} epochs per run. Mean ± sample SD across seeds; not a confidence interval.',ha='center',fontsize=10)
    save_formats(fig,out,'paper_table');plt.close(fig)
    lines=['# 三种子三算法大豆对比','',f'种子 {seeds}；每组固定最终{config["epochs"]}轮；{config["evaluation_year"]}年{n}县历史基准复测。表内为逐种子指标的均值±样本标准差（ddof=1）。',
           '','|训练方式|RMSE↓|作者R²=Corr²↑|Corr↑|标准决定系数|同种子RMSE相对集中式差距（%）|','|---|---:|---:|---:|---:|---:|']
    for m,label in zip(METHODS,LABELS):
        lines.append('|'+label+'|'+'|'.join(format_stat(summary[m][k]) for k in (*KEYS,'r2_coefficient_of_determination','paired_rmse_gap_percent'))+'|')
    lines+=['','![三算法对比表](paper_table.png)','','![均值与种子波动](centralized_vs_federated.png)','','![逐县预测](prediction_scatter.png)','',
            'R²按作者公开指标实现取Corr²；另列标准决定系数。全部分数来自相同评价县及固定最终模型，没有按测试结果挑选轮次。',
            '误差线是三个种子的样本标准差，不是置信区间。相同训练样本处理量不等于相同优化步数、时间或通信开销。',
            '训练过程曲线是固定训练探针，不能当验证或测试曲线。本轮为Tiny CropNet大豆比较，不是原论文四作物主表复现，也不预设三种方法精度相等。']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'artifacts_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifacts_sha256.json'})
