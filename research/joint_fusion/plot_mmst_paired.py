"""Plot measured paired scores only; never substitute literature or ridge values."""
import json
from pathlib import Path
import numpy as np


def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    status = json.loads((folder/'status.json').read_text())
    if status.get('status') != 'completed' or status.get('stage') != 'historical_test':
        raise ValueError('Only a completed historical-test evaluation may produce comparison plots')
    results = json.loads((folder/'metrics.json').read_text())
    methods = ['centralized', 'fedavg', 'hierarchical_async']
    labels = ['Centralized', 'FedAvg', 'Hierarchical async']
    colors = ['#2665a0', '#8b99a6', '#de793d']
    seeds = sorted({r['seed'] for r in results})
    if len(results) != len(methods)*len(seeds) or any(sum(r['seed']==s and r['method']==m for r in results)!=1 for s in seeds for m in methods):
        raise ValueError('Incomplete or duplicated paired training matrix')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for i, m in enumerate(methods):
        values = np.array([r['metrics']['rmse'] for r in results if r['method'] == m])
        axes[0].bar(i, values.mean(), yerr=values.std(ddof=1) if len(values)>1 else 0, color=colors[i], capsize=4)
        axes[0].scatter(np.linspace(i-.12, i+.12, len(values)), values, color='#172534', s=16, zorder=3)
    axes[0].set(xticks=range(3), xticklabels=labels, ylabel='RMSE (BU/ACRE)', title='Same architecture, inputs and example budget')
    # The first predeclared seed, never the best-performing run.
    bounds = []
    for m, color, label in zip([methods[0], methods[2]], [colors[0], colors[2]], [labels[0], labels[2]]):
        predictions = json.loads((folder/f'{m}_seed{seeds[0]}_predictions.json').read_text())
        y = np.array([p['truth'] for p in predictions])
        p = np.array([p['prediction'] for p in predictions])
        axes[1].scatter(y, p, alpha=.6, s=17, color=color, label=label)
        bounds.extend([y.min(), y.max(), p.min(), p.max()])
    lo, hi = min(bounds), max(bounds)
    axes[1].plot([lo, hi], [lo, hi], '--', color='#555555', linewidth=1)
    axes[1].set(xlabel='Observed yield (BU/ACRE)', ylabel='Predicted yield (BU/ACRE)', title=f'Paired county predictions: seed {seeds[0]}')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    for suffix in ('png', 'svg'):
        fig.savefig(folder/f'centralized_vs_federated.{suffix}', dpi=300)
    plt.close(fig)
    lines = ['# MMST-ViT 集中式与联邦训练配对结果', '',
             '作者公开 Tiny CropNet 县表上的扩展协议；不是 KDD 主表复现。2022 是历史基准复测。', '',
             '|训练方式|RMSE 均值±种子标准差|作者口径 R²|Corr|配对 RMSE 差值|',
             '|---|---:|---:|---:|---:|']
    for method, label in zip(methods, labels):
        group = [r for r in results if r['method'] == method]
        rmse = np.array([r['metrics']['rmse'] for r in group])
        def avg(key):
            vals = [r['metrics'][key] for r in group]
            return f'{np.mean(vals):.4f}' if all(v is not None for v in vals) else 'undefined'
        gap = np.mean([r['rmse_gap'] for r in group])
        sd = rmse.std(ddof=1) if len(rmse)>1 else 0.
        lines.append(f'|{label}|{rmse.mean():.4f} ± {sd:.4f}|{avg("r2_author_corr_squared")}|{avg("pearson_corr")}|{gap:+.4f}|')
    lines += ['', '差值 = 联邦 RMSE − 同种子集中式 RMSE；正数表示精度下降。种子标准差不是独立县域泛化置信区间。',
              '所有指标来自同一个固定最终检查点。未预设非劣效界值，不仅凭差值小就宣称统计等价。']
    (folder/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
