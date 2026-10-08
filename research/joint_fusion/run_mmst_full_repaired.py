"""Frozen full-soybean repair transfer: prepare, equal-budget train, test, plot."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from cropnet_leo.remote import sha
from .author_evaluation import evaluate
from .download_mmst_data import fetch, download_lock
from .mmst_bridge import ROOT,make_model
from .mmst_data import records,required_paths,client_partition
from .mmst_full_data import CountyData,fit_statistics,prepare,read
from .mmst_full_engine import train_arm
from .mmst_optimizer_store import OptimizerStore
from .mmst_io import write_json
from .mmst_training import seed_all,vector,assign
from .run_mmst_paired import source_hashes,digest,METHODS


def now():return datetime.now(timezone.utc).isoformat()


def snapshot(out,config):
    files=source_hashes()
    for name in ['mmst_full_data.py','mmst_full_engine.py','mmst_optimizer_store.py','run_mmst_full_repaired.py',
                 'mmst_async_annealed.py','mmst_repair_federated.py','mmst_repair.py',
                 'audit_mmst_quick.py','MMST_FULL_REPAIRED_PROTOCOL.md']:
        path=ROOT/'research/joint_fusion'/name
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    files['cropnet_leo/remote.py']=sha(ROOT/'cropnet_leo/remote.py')
    for relative in files:
        dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,dest)
    write_json(out/'source_sha256.json',files);write_json(out/'config.json',config)


def check_sources(out):
    for relative,expected in read(out/'source_sha256.json').items():
        if sha(ROOT/relative)!=expected or sha(out/'source'/relative)!=expected:
            raise ValueError('Frozen source changed: '+relative)


def predictions(model,data,stats):
    model.eval();truth=[];pred=[];device=next(model.parameters()).device
    mu,sd=stats['target']['mean'][1],stats['target']['std'][1]
    with torch.no_grad():
        for i in range(len(data)):
            x,s,l,y=[a.to(device) for a in data(i)]
            p=model(x,ys=s,yl=l)
            truth.append(float(torch.exp(y[0,1].double()*sd+mu)))
            pred.append(float(torch.exp(p[0,1].double()*sd+mu)))
    return truth,pred


def train(config,prepared,out,update):
    out.mkdir(parents=True,exist_ok=True)
    data=CountyData(prepared,config,False)
    stats=fit_statistics(data);data.stats=stats
    if (out/'normalization.json').exists() and read(out/'normalization.json')!=stats:
        raise ValueError('Training normalization changed')
    write_json(out/'normalization.json',stats)
    write_json(out/'data_manifest.json',data.manifest)
    clients=client_partition(records(config),config['clients_per_state'])
    write_json(out/'clients.json',clients)
    manifest_hash=sha(prepared/'manifest.json');config_hash=digest(config)
    completed=[]
    for seed in config['seeds']:
        seed_all(seed);model=make_model(config).cuda()
        initial_path=out/f'initial_seed{seed}.pt'
        generated=vector(model)
        if initial_path.exists():
            initial=torch.load(initial_path,map_location='cpu',weights_only=True)
            if not torch.equal(initial,generated):raise ValueError('Initialization changed')
        else: initial=generated;torch.save(initial,initial_path)
        initial_hash=hashlib.sha256(initial.numpy().tobytes()).hexdigest()
        for method in METHODS:
            arm=out/f'{method}_seed{seed}';arm.mkdir(exist_ok=True)
            identity={'method':method,'seed':seed,'config_sha256':config_hash,
                      'data_manifest_sha256':manifest_hash,'initial_sha256':initial_hash}
            if (arm/'identity.json').exists():
                if read(arm/'identity.json')!=identity:raise ValueError('Arm identity changed')
            else:write_json(arm/'identity.json',identity)
            if (arm/'result.json').exists():
                result=read(arm/'result.json')
                if sha(out/result['checkpoint'])!=result['checkpoint_sha256']:raise ValueError('Completed model changed')
                completed.append(result);continue
            resume=None;log=arm/'training.jsonl'
            if (arm/'recovery.json').exists():
                info=read(arm/'recovery.json');path=arm/info['path']
                if sha(path)!=info['sha256']:raise ValueError('Recovery checksum mismatch')
                resume=torch.load(path,map_location='cpu',weights_only=False)
                if resume['identity']!=identity:raise ValueError('Recovery identity mismatch')
            # Preserve uncommitted events rather than count them twice.
            if log.exists():
                lines=log.read_text(encoding='utf-8').splitlines(keepends=True)
                keep=resume['event_count'] if resume else 0
                if len(lines)<keep:raise ValueError('Recovery is ahead of durable event log')
                if len(lines)>keep:
                    backup=arm/f'training_before_recovery_{time.time_ns()}.jsonl'
                    shutil.copy2(log,backup);log.write_text(''.join(lines[:keep]),encoding='utf-8')
            started=time.perf_counter()
            write_json(arm/'status.json',{'status':'running','method':method,'seed':seed,'updated_at':now()})
            with log.open('a',encoding='utf-8') as handle:
                def emit(row):
                    row={**row,'segment_seconds':time.perf_counter()-started,'updated_at':now()}
                    handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
                    write_json(arm/'status.json',{'status':'running',**row})
                    update('train',method=method,seed=seed,completed_runs=len(completed),total_runs=len(METHODS)*len(config['seeds']),**{k:row[k] for k in ['processed_examples','accepted_examples_total','version','local_loss','lr']})
                    print(json.dumps(row),flush=True)
                def save(state):
                    old=read(arm/'recovery.json') if (arm/'recovery.json').exists() else {'slot':1}
                    slot=1-old['slot'];name=f'recovery_{slot}.pt';temp=arm/(name+'.partial')
                    opts=state['opts'].snapshot(slot)
                    torch.save({**state,'opts':opts,'identity':identity},temp);temp.replace(arm/name)
                    write_json(arm/'recovery.json',{'slot':slot,'path':name,'sha256':sha(arm/name),
                                                  'event_count':state['event_count'],'updated_at':now()})
                final,count=train_arm(model,initial,data,clients,config,seed,method,emit,save,resume,OptimizerStore(arm/'optimizer_state'))
            del resume
            path=arm/'final.pt';torch.save(final,path)
            y,p=predictions(model,data,stats)
            result={**identity,'checkpoint':str(path.relative_to(out)), 'checkpoint_sha256':sha(path),
                    'processed_examples':count,'training_metrics':evaluate(y,p),
                    'checkpoint_rule':config['checkpoint_rule']}
            write_json(arm/'training_predictions.json',{'truth':y,'prediction':p})
            write_json(arm/'result.json',result);write_json(arm/'status.json',{'status':'completed',**result,'updated_at':now()})
            completed.append(result);write_json(out/'training_summary.json',completed)
        del model
        torch.cuda.empty_cache()
    assert len(completed)==len(METHODS)*len(config['seeds'])
    assert {r['processed_examples'] for r in completed}=={len(data)*config['epochs']}
    write_json(out/'training_summary.json',completed)
    write_json(out/'status.json',{'status':'completed','stage':'training','runs':len(completed),'test_evaluated':False})


def test(config,training,prepared,out):
    if read(training/'status.json').get('status')!='completed':raise ValueError('Training matrix incomplete')
    summary=read(training/'training_summary.json')
    if {(r['seed'],r['method']) for r in summary}!={(s,m) for s in config['seeds'] for m in METHODS}:
        raise ValueError('Paired matrix incomplete')
    if out.exists():raise ValueError('Evaluation output already exists; preserve any partial scores')
    out.mkdir(parents=True)
    stats=read(training/'normalization.json');data=CountyData(prepared,config,True,stats)
    model=make_model(config).cuda();results=[];rows=records(config,True)
    write_json(out/'status.json',{'status':'running','stage':'historical_test'})
    for item in summary:
        checkpoint=training/item['checkpoint']
        if sha(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Final model checksum mismatch')
        assign(model,torch.load(checkpoint,map_location='cpu',weights_only=True))
        y,p=predictions(model,data,stats)
        result={**item,'metrics':evaluate(y,p),'year':2022,'scope':'full Tiny CropNet soybean historical benchmark retest'}
        results.append(result)
        write_json(out/f"{item['method']}_seed{item['seed']}_predictions.json",
                   [{'fips':r['fips'],'year':2022,'truth':a,'prediction':b} for r,a,b in zip(rows,y,p)])
    for r in results:
        ref=next(a for a in results if a['seed']==r['seed'] and a['method']=='centralized')
        r['rmse_gap']=r['metrics']['rmse']-ref['metrics']['rmse']
        r['rmse_gap_percent']=100*r['rmse_gap']/ref['metrics']['rmse'] if ref['metrics']['rmse'] else None
    write_json(out/'metrics.json',results);write_json(out/'config.json',config)
    write_json(out/'provenance.json',{'training':str(training.resolve()),'test_manifest_sha256':sha(prepared/'manifest.json'),
                                     'normalization_sha256':sha(training/'normalization.json')})
    write_json(out/'status.json',{'status':'completed','stage':'historical_test','runs':len(results)})
    from .plot_mmst_paired import plot
    plot(out)
    # A publication-style metric table with across-seed variation, not best-seed selection.
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,3.7));ax.axis('off')
    labels=['Centralized','FedAvg','Hierarchical async'];body=[]
    for method,label in zip(METHODS,labels):
        group=[r for r in results if r['method']==method];values=[]
        for key in ['rmse','r2_author_corr_squared','pearson_corr']:
            a=[r['metrics'][key] for r in group]
            values.append(f'{np.mean(a):.4f} ± {np.std(a,ddof=1):.4f}' if all(v is not None for v in a) else 'undefined')
        body.append([label,*values])
    table=ax.table(cellText=body,colLabels=['Training method','RMSE ↓','R² = Corr² ↑','Corr ↑'],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.8)
    for (r,c),cell in table.get_celld().items():
        cell.visible_edges='TB' if r==0 else ('B' if r==len(body) else '')
        cell.set_linewidth(.7)
    ax.set_title('Soybeans · full public Tiny CropNet · 2022 historical retest\n222 training / 254 test counties · 3 paired seeds',fontsize=12)
    fig.text(.5,.08,f"Mean ± sample SD across seeds. {len(records(config))*config['epochs']} processed training examples per run.",ha='center',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'paper_table.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    write_json(out/'artifacts_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'research/joint_fusion/configs/mmst_full_repaired_v1.json')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--download-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/author_tiny_cropnet/raw')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=read(args.config);manifest=read(args.manifest)
    if (manifest['repository'],manifest['revision'],manifest['include_evaluation'])!=(config['dataset_repository'],config['dataset_revision'],True):
        raise ValueError('Full pinned train/test manifest required')
    if args.resume:
        if read(args.out/'config.json')!=config:raise ValueError('Cannot resume with changed config')
        check_sources(args.out)
        if read(args.out/'status.json')['status']=='completed':raise ValueError('Run already completed')
    else:
        args.out.mkdir(parents=True,exist_ok=False);snapshot(args.out,config)
        shutil.copy2(args.manifest,args.out/'download_manifest.json')
    stage='wait_training_download'
    def update(new_stage,**extra):
        nonlocal stage
        stage=new_stage
        write_json(args.out/'status.json',{'status':'running','stage':stage,'updated_at':now(),
             'performance_training_started':stage in ('train','download_test','prepare_test','evaluate','complete'),
             'performance_results_available':stage=='complete',**extra})
    # Reuse the OS lock mechanism on a separate run directory to prohibit duplicate trainers.
    with download_lock(args.out/'runner_lock'):
        try:
            torch.set_num_threads(4);update(stage)
            while True:
                downloaded=read(args.download_run/'status.json')
                if downloaded['status']=='completed':break
                if downloaded['status']=='failed':raise RuntimeError('Training downloader failed: '+downloaded.get('error',''))
                update('wait_training_download',download_run=str(args.download_run.resolve()))
                time.sleep(30)
            check_sources(args.out)
            for evaluation in [False]:
                update('prepare_train')
                cache=args.out/'prepared_train'
                if not (cache/'status.json').exists() or read(cache/'status.json')['status']!='completed':
                    prepare(config,args.raw,cache,False,lambda row:update('prepare_train',**row))
            update('train')
            train(config,cache,args.out/'training',update)
            check_sources(args.out);update('download_test')
            fetch(manifest,args.raw,config['max_download_gib'],args.out/'download_progress.json')
            update('prepare_test');cache=args.out/'prepared_test'
            if not (cache/'status.json').exists() or read(cache/'status.json')['status']!='completed':
                prepare(config,args.raw,cache,True,lambda row:update('prepare_test',**row))
            check_sources(args.out);update('evaluate')
            test(config,args.out/'training',cache,args.out/'evaluation')
            write_json(args.out/'status.json',{'status':'completed','stage':'complete','updated_at':now(),
                                             'performance_training_started':True,'performance_results_available':True})
        except BaseException as exc:
            write_json(args.out/'status.json',{'status':'failed','stage':stage,'error':repr(exc),'updated_at':now(),
                                             'performance_results_available':False})
            traceback.print_exc();raise


if __name__=='__main__':main()
