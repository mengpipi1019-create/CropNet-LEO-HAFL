"""User-owned foreground first comparison, with durable pause and auto-resume."""
import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import types

import numpy as np
import torch

from cropnet_leo.remote import sha
from . import run_mmst_full_repaired as full
from . import download_mmst_data as downloader
from .download_mmst_data import fetch,download_lock
from .mmst_bridge import ROOT,make_model
from .mmst_data import records
from .mmst_full_data import CountyData,fit_statistics,prepare,read
from .mmst_full_engine import train_arm
from .mmst_io import write_json
from .mmst_training import assign
from .run_mmst_paired import METHODS,digest
from .author_evaluation import evaluate


DEFAULT_OUT=ROOT/'results/joint_fusion/mmst_local_first_20260929_01'
BASE_RUN=ROOT/'results/joint_fusion/mmst_full_repaired_20260929_01'
DEFAULT_CONFIG=ROOT/'research/joint_fusion/configs/mmst_local_first_v1.json'
ALLOWED_CHANGES={'experiment','seeds','checkpoint_every_epochs','scope'}


class LocalPause(BaseException):
    """Cooperative stop; unlike an error, never retried by the downloader."""


def display_status(out):
    if not (out/'status.json').exists():return {'status':'not_started','message':'Local run has not started.'}
    status=read(out/'status.json')
    if status.get('status')=='running' and os.name=='nt' and status.get('pid'):
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x00100000,False,int(status['pid']))
        if handle:
            active=kernel.WaitForSingleObject(handle,0)==258
            kernel.CloseHandle(handle)
            if not active:status={**status,'status':'interrupted','message':'Process exited. Run the same start command to resume.'}
        else:
            status={**status,'status':'process_unverified','message':'Cannot confirm the saved process is alive. Check the training window before starting another copy.'}
    return status


def require_same_inputs(config,cache_config):
    changed={k for k in set(config)|set(cache_config) if config.get(k)!=cache_config.get(k)}
    if changed-ALLOWED_CHANGES:raise ValueError('Input/model/optimizer protocol changed: '+str(changed-ALLOWED_CHANGES))


class PauseControl:
    def __init__(self,out):self.out=Path(out);self.phase='starting'
    def request(self):
        (self.out/'PAUSE').write_text('User requested pause\n',encoding='utf-8')
        print('\nPause requested. During training, wait for the next saved epoch boundary.',flush=True)
    def check(self):
        if (self.out/'PAUSE').exists():raise LocalPause()
    def signal(self,*_):
        self.request()
        if self.phase!='train':raise LocalPause()
    def completed_checkpoint(self):self.check()


@contextmanager
def keep_awake():
    active=False
    if os.name=='nt':
        active=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        if not active:print('Automatic sleep prevention unavailable; keep the computer awake.',flush=True)
    try:yield
    finally:
        if active:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def run_training(config,cache_config,prepared,out,update,control):
    """Bind the audited training loop to its unchanged cache and safe save hook.

    The original module and its globals stay untouched. Only data identity and
    the after-save pause hook differ; parameter updates use the frozen engine.
    """
    require_same_inputs(config,cache_config)
    def data_for_cache(path,unused_config,evaluation):
        return CountyData(path,cache_config,evaluation)
    def pausable_arm(model,initial,data,clients,cfg,seed,method,emit,save,resume=None,optimizer_store=None):
        control.check()
        def durable_save(state):
            save(state)
            control.completed_checkpoint()
        return train_arm(model,initial,data,clients,cfg,seed,method,emit,durable_save,resume,optimizer_store)
    bound=types.FunctionType(full.train.__code__,{**full.train.__globals__,
                             'CountyData':data_for_cache,'train_arm':pausable_arm})
    bound(config,prepared,out,update)


def fetch_with_pause(manifest,raw,max_gib,progress,control):
    def request(url,headers=None):
        control.check()
        return downloader.request(url,headers)
    bound=types.FunctionType(downloader._fetch.__code__,{**downloader._fetch.__globals__,'request':request})
    with download_lock(raw):bound(manifest,raw,max_gib,progress)


def evaluate_models(config,training,prepared,out,control,update):
    summary=read(training/'training_summary.json')
    if read(training/'status.json').get('status')!='completed':raise ValueError('Train all three arms first')
    if {(r['seed'],r['method']) for r in summary}!={(s,m) for s in config['seeds'] for m in METHODS}:
        raise ValueError('Incomplete training matrix')
    stats=read(training/'normalization.json');data=CountyData(prepared,config,True,stats)
    out.mkdir(parents=True,exist_ok=True);rows=records(config,True)
    identity={'config_sha256':digest(config),'summary_sha256':sha(training/'training_summary.json'),
              'test_manifest_sha256':sha(prepared/'manifest.json'),'normalization_sha256':sha(training/'normalization.json')}
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Evaluation identity changed')
    write_json(out/'identity.json',identity)
    model=make_model(config).cuda();results=[]
    write_json(out/'status.json',{'status':'running','stage':'historical_test'})
    for item in summary:
        control.check();prefix=f"{item['method']}_seed{item['seed']}"
        checkpoint=training/item['checkpoint'];prediction_path=out/(prefix+'_predictions.json');result_path=out/(prefix+'_result.json')
        if sha(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Final model changed')
        if result_path.exists():
            result=read(result_path)
            if result['evaluation_identity']!=identity or result['checkpoint_sha256']!=item['checkpoint_sha256']:
                raise ValueError('Completed evaluation identity mismatch')
            if sha(prediction_path)!=result['prediction_sha256']:raise ValueError('Saved predictions changed')
        else:
            assign(model,torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            mu,sd=stats['target']['mean'][1],stats['target']['std'][1];truth=[];pred=[]
            with torch.no_grad():
                for i in range(len(data)):
                    control.check()
                    x,s,l,y=[a.cuda() for a in data(i)];p=model(x,ys=s,yl=l)
                    truth.append(float(torch.exp(y[0,1].double()*sd+mu)))
                    pred.append(float(torch.exp(p[0,1].double()*sd+mu)))
                    if (i+1)%20==0 or i+1==len(data):update('evaluate',method=item['method'],completed=i+1,total=len(data))
            write_json(prediction_path,[{'fips':r['fips'],'year':2022,'truth':a,'prediction':b} for r,a,b in zip(rows,truth,pred)])
            result={**item,'metrics':evaluate(truth,pred),'year':2022,'scope':'single-seed full Tiny CropNet soybean historical retest',
                    'evaluation_identity':identity,'prediction_sha256':sha(prediction_path)}
            write_json(result_path,result)
        # Independently recompute stored scores on every resumed evaluation.
        prediction_rows=read(prediction_path)
        assert [r['fips'] for r in prediction_rows]==[r['fips'] for r in rows]
        checked=evaluate([r['truth'] for r in prediction_rows],[r['prediction'] for r in prediction_rows])
        if checked!=result['metrics']:raise ValueError('Metrics do not match saved predictions')
        results.append(result)
    del model;torch.cuda.empty_cache()
    for result in results:
        ref=next(r for r in results if r['method']=='centralized' and r['seed']==result['seed'])
        result['rmse_gap']=result['metrics']['rmse']-ref['metrics']['rmse']
        result['rmse_gap_percent']=100*result['rmse_gap']/ref['metrics']['rmse'] if ref['metrics']['rmse'] else None
    write_json(out/'metrics.json',results);write_json(out/'config.json',config)
    write_json(out/'status.json',{'status':'completed','stage':'historical_test','runs':len(results)})
    plot_results(out)


def plot_results(out):
    """One predeclared seed: no across-seed standard deviation is fabricated."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if read(out/'status.json')!={'status':'completed','stage':'historical_test','runs':3}:
        raise ValueError('Only a completed first-seed comparison may be plotted')
    results=read(out/'metrics.json')
    if len(results)!=3 or {(r['seed'],r['method']) for r in results}!={(10,m) for m in METHODS}:
        raise ValueError('Expected the three seed10 arms')
    labels=['Centralized','FedAvg','Hierarchical async'];colors=['#2665a0','#299c7c','#de793d']
    ordered=[next(r for r in results if r['method']==m) for m in METHODS]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    bars=axes[0].bar(labels,[r['metrics']['rmse'] for r in ordered],color=colors)
    axes[0].bar_label(bars,fmt='%.3f',padding=3);axes[0].margins(y=.15)
    axes[0].set(ylabel='RMSE (BU/ACRE)',title='254 counties / historical 2022 evaluation')
    bounds=[]
    for r,label,color in zip(ordered,labels,colors):
        rows=read(out/f"{r['method']}_seed10_predictions.json")
        y=np.array([a['truth'] for a in rows]);p=np.array([a['prediction'] for a in rows])
        axes[1].scatter(y,p,s=14,alpha=.45,color=color,label=label);bounds.extend([y.min(),y.max(),p.min(),p.max()])
    lo,hi=min(bounds),max(bounds);axes[1].plot([lo,hi],[lo,hi],'--',color='#555')
    axes[1].set(xlabel='Observed yield (BU/ACRE)',ylabel='Predicted yield (BU/ACRE)',title='Fixed seed 10 / same final checkpoints');axes[1].legend(frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'centralized_vs_federated.{ext}',dpi=240)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4));ax.axis('off');body=[]
    for r,label in zip(ordered,labels):
        body.append([label,*[f"{r['metrics'][key]:.4f}" if r['metrics'][key] is not None else 'undefined'
                            for key in ['rmse','r2_author_corr_squared','pearson_corr']]])
    table=ax.table(cellText=body,colLabels=['Training method','RMSE ↓','R² = Corr² ↑','Corr ↑'],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,1.9)
    for (r,c),cell in table.get_celld().items():
        cell.visible_edges='TB' if r==0 else ('B' if r==3 else '');cell.set_linewidth(.7)
    ax.set_title('Soybeans / same MMST-ViT backbone\n222 training counties / 254 evaluation counties / seed 10',fontsize=13)
    fig.text(.5,.06,'120 epochs and 26,640 processed training examples per arm. Single-seed historical retest.',ha='center',fontsize=10)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'paper_table.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    lines=['# 首版三算法大豆对比','','2021全222县训练；2022全254县历史基准复测；固定seed10，各120轮/26640样本处理。单种子无种子标准差。','',
           '|方式|RMSE↓|作者口径R²↑|Corr↑|标准决定系数|RMSE相对集中式|','|---|---:|---:|---:|---:|---:|']
    def fmt(value):return 'undefined' if value is None else f'{value:.4f}'
    for r,label in zip(ordered,labels):
        m=r['metrics'];gap='undefined' if r['rmse_gap_percent'] is None else f"{r['rmse_gap_percent']:+.2f}%"
        lines.append('|'+label+'|'+'|'.join(fmt(m[k]) for k in ['rmse','r2_author_corr_squared','pearson_corr','r2_coefficient_of_determination'])+'|'+gap+'|')
    lines+=['','![对比表](paper_table.png)','','![预测结果](centralized_vs_federated.png)','',
            '全部分数来自固定最终模型与相同评价县，无测试挑选轮次。R²按公开MMST-ViT指标代码取Corr²，标准决定系数另外列出。',
            '这是首版探索性对比，不是四作物主表复现；单种子不能证明稳定性或精度等价。相同样本处理量不等于相同优化步数或资源成本。']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'artifacts_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifacts_sha256.json'})


def snapshot(out,config):
    full.snapshot(out,config)
    hashes=read(out/'source_sha256.json')
    for relative in ['research/joint_fusion/run_mmst_local.py','research/joint_fusion/MMST_LOCAL_FIRST_PROTOCOL.md',
                     'run_mmst_local.ps1','start_mmst_local.cmd','pause_mmst_local.cmd']:
        dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
        hashes[relative]=sha(dest)
    write_json(out/'source_sha256.json',hashes)


def preflight(config,base):
    cache_config=read(base/'config.json');require_same_inputs(config,cache_config)
    full.check_sources(base)
    data=CountyData(base/'prepared_train',cache_config,False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in the selected Python environment')
    return cache_config,data


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--action',choices=['run','pause','status','check'],default='run')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--base',type=Path,default=BASE_RUN)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    args=parser.parse_args();args.out=args.out.resolve()
    if args.action=='pause':
        if not (args.out/'config.json').exists():print('No local run has started; nothing to pause.');return
        PauseControl(args.out).request();return
    if args.action=='status':
        print(json.dumps(display_status(args.out),indent=2,ensure_ascii=False))
        return
    config=read(args.config);cache_config,data=preflight(config,args.base)
    print(f'Inputs verified: {len(data)} counties. CUDA: {torch.cuda.get_device_name(0)}. Three arms x 120 epochs, seed 10.',flush=True)
    if config['seeds']!=[10] or config['epochs']!=120 or config['checkpoint_every_epochs']!=1:
        raise ValueError('Use a new protocol for a different first-comparison budget')
    if args.action=='check':print('CHECK PASSED. No training or download started.',flush=True);return
    # A common lock prevents two local-run directories from training on this GPU at once.
    with download_lock(ROOT/'results/joint_fusion/mmst_local_gpu_lock'),keep_awake():
        manifest=read(args.base/'download_manifest.json')
        if (manifest['repository'],manifest['revision'],manifest['include_evaluation'])!=(config['dataset_repository'],config['dataset_revision'],True):
            raise ValueError('Pinned complete download manifest required')
        binding={'base':str(args.base.resolve()),'prepared_train':str((args.base/'prepared_train').resolve()),
                 'manifest_sha256':sha(args.base/'prepared_train/manifest.json'),'cache_config_sha256':digest(cache_config)}
        if args.out.exists():
            if read(args.out/'config.json')!=config or read(args.out/'cache_binding.json')!=binding:
                raise ValueError('Existing local run identity changed')
            full.check_sources(args.out)
            if (args.out/'status.json').exists() and read(args.out/'status.json').get('status')=='completed':
                print('Already completed: '+str(args.out/'evaluation/comparison.md'));return
            if (args.out/'PAUSE').exists():
                (args.out/'PAUSE').replace(args.out/f'PAUSE_acknowledged_{time.time_ns()}')
        else:
            args.out.mkdir(parents=True);snapshot(args.out,config)
            write_json(args.out/'cache_binding.json',binding);write_json(args.out/'download_manifest.json',manifest)
        control=PauseControl(args.out);state={'stage':'starting'}
        def update(stage,**extra):
            state.clear();state.update(stage=stage,**extra);control.phase=stage
            write_json(args.out/'status.json',{'status':'running',**state,'pid':os.getpid(),'updated_at':full.now(),
                                             'performance_results_available':False})
            if stage=='train' and extra.get('version') is not None:
                accepted=extra['accepted_examples_total'];percent=100*accepted/26640
                print(f"[{extra['method']}] accepted {accepted}/26640 ({percent:.1f}%), completed {extra['completed_runs']}/3 arms",flush=True)
            elif stage in ('prepare_test','evaluate') and ('prepared' in extra or 'completed' in extra):
                print(stage,extra,flush=True)
        old_handler=signal.signal(signal.SIGINT,control.signal)
        try:
            torch.set_num_threads(4);update('train')
            print('Training starts/resumes. To pause safely: use pause_mmst_local.cmd, then wait for PAUSED.',flush=True)
            run_training(config,cache_config,args.base/'prepared_train',args.out/'training',update,control)
            control.check();full.check_sources(args.out);update('download_test')
            print('All three final models frozen. Resuming required evaluation downloads...',flush=True)
            fetch_with_pause(manifest,ROOT/'data/author_tiny_cropnet/raw',config['max_download_gib'],args.out/'download_progress.json',control)
            control.check();update('prepare_test');cache=args.out/'prepared_test'
            if not (cache/'status.json').exists() or read(cache/'status.json')['status']!='completed':
                def prepared(row):update('prepare_test',**row);control.check()
                prepare(config,ROOT/'data/author_tiny_cropnet/raw',cache,True,prepared)
            control.check();full.check_sources(args.out);update('evaluate')
            evaluate_models(config,args.out/'training',cache,args.out/'evaluation',control,update)
            write_json(args.out/'status.json',{'status':'completed','stage':'complete','updated_at':full.now(),'performance_results_available':True})
            print('COMPLETED: '+str(args.out/'evaluation/comparison.md'),flush=True)
        except LocalPause:
            for arm in (args.out/'training').glob('*_seed*') if (args.out/'training').exists() else []:
                if (arm/'status.json').exists() and read(arm/'status.json').get('status')=='running':
                    previous=read(arm/'status.json')
                    write_json(arm/'status.json',{**previous,'status':'paused','updated_at':full.now()})
            write_json(args.out/'status.json',{'status':'paused',**state,'updated_at':full.now(),'performance_results_available':False})
            print('PAUSED. Durable progress preserved. Run the same start command to continue.',flush=True)
        except BaseException as exc:
            write_json(args.out/'status.json',{'status':'failed',**state,'error':repr(exc),'updated_at':full.now(),'performance_results_available':False})
            traceback.print_exc();raise
        finally:signal.signal(signal.SIGINT,old_handler)


if __name__=='__main__':main()
