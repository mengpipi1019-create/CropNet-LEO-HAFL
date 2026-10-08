"""User-run three-method comparison with the calibrated balanced asynchronous arm."""
import argparse
import hashlib
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
from . import run_mmst_local as local
from . import run_mmst_calibration as cal
from . import mmst_full_engine as engine
from .mmst_io import write_json
from .mmst_training import vector,assign
from .plot_mmst_compare import plot_curves,plot_results
from .mmst_compare_runtime import fetch_resilient,capacity_check,scan_cache,retryable,checked_sleep
from .mmst_compare_recovery import archive_uncommitted

ROOT=local.ROOT
BASE=ROOT/'results/joint_fusion/portable_inputs'
CONFIG=ROOT/'research/joint_fusion/configs/mmst_compare_balanced_v1.json'
OUT=ROOT/'results/joint_fusion/mmst_compare_balanced_20260930_01'
SHORT=ROOT/'evidence/async_short'
GAIN=.9066011235955057
ALLOWED=local.ALLOWED_CHANGES|{'server_gain','staleness_power'}


def require_shared_inputs(config,cache_config):
    changed={k for k in set(config)|set(cache_config) if config.get(k)!=cache_config.get(k)}
    if changed-ALLOWED:raise ValueError('Input/model/optimizer field changed: '+str(changed-ALLOWED))
    if config['server_gain']!=GAIN or config['staleness_power']!=0:
        raise ValueError('This entry point requires the frozen balanced policy')


def raw_inventory(manifest,raw):
    missing=[];ready=0;missing_bytes=0;orphan=[]
    for item in manifest['files']:
        path=raw/item['path'];meta=path.with_suffix(path.suffix+'.source.json')
        if not path.exists():missing.append(item['path']);missing_bytes+=item['size'];continue
        if path.stat().st_size!=item['size']:raise ValueError('Unverified raw file size: '+str(path))
        if not meta.exists():orphan.append(item['path']);continue
        info=cal.read(meta)
        if info.get('revision')!=manifest['revision'] or not info.get('sha256'):raise ValueError('Raw revision/checksum metadata missing')
        ready+=1
    return {'manifest_files':len(manifest['files']),'size_and_metadata_ready':ready,
            'missing_files':len(missing),'missing_bytes':missing_bytes,'missing_paths':missing,
            'raw_content_hashes_rechecked_here':False,'orphan_files_needing_upstream_verification':orphan}


def preflight(config,base,out=OUT):
    if not (base/'config.json').exists():
        raise RuntimeError('Prepare this checkout first: python -m research.joint_fusion.prepare_mmst_compare')
    cache=cal.read(base/'config.json');require_shared_inputs(config,cache);local.full.check_sources(base)
    if config['seeds']!=[10,20,30] or config['epochs']!=120 or config['checkpoint_every_epochs']!=1:
        raise ValueError('Frozen seeds10/20/30 and 120-epoch comparison required')
    if sha(SHORT/'summary.json')!=cal.read(SHORT/'report_sha256.json')['summary.json']:
        raise ValueError('Short diagnostic evidence changed')
    if not cal.read(SHORT/'summary.json')['decision']['short_screen_passed']:raise ValueError('Short screen did not pass')
    # Separate cache identity from the new aggregation protocol.
    data=local.CountyData(base/'prepared_train',cache,False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    inventory=raw_inventory(cal.read(base/'download_manifest.json'),ROOT/'data/author_tiny_cropnet/raw')
    capacity=capacity_check(config,inventory,out)
    print(json.dumps({'training_counties':len(data),'seeds':config['seeds'],'methods':3,'epochs_per_method':120,
          'training_examples_total':len(data)*120*3*len(config['seeds']),'balanced_gain':config['server_gain'],
          'balanced_power':config['staleness_power'],'cuda':torch.cuda.get_device_name(0),
          'evaluation_missing_files':inventory['missing_files'],'evaluation_missing_gib':inventory['missing_bytes']/1024**3,
          'capacity':capacity,'probe_curve_scope':'20 fixed training counties; not heldout; no test curves'},ensure_ascii=False),flush=True)
    return cache,data,inventory


def probe_measurement(model,weights,probe,stats):
    previous,mode=vector(model),model.training
    devices=[next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            assign(model,weights);y,p=local.full.predictions(model,probe,stats)
            return {'metrics':{**cal.score(y,p),'mean_error':float(np.mean(np.array(p)-np.array(y)))},
                    'predictions':[{'fips':r['record']['fips'],'year':r['record']['year'],'truth':a,'prediction':b}
                                   for r,a,b in zip(probe.manifest['records'],y,p)]}
    finally:assign(model,previous);model.train(mode)


def make_observer(model,initial,data,clients,cfg,seed,method,curve_root,update):
    ids=sorted(min(c['indices']) for c in clients)
    if len(set(ids))!=len(clients):raise ValueError('Overlapping probe counties')
    probe=cal.View(data,ids)
    if {r['record']['year'] for r in probe.manifest['records']}!={cfg['train_year']}:
        raise ValueError('Probe must contain training-year counties only')
    folder=curve_root/f'{method}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    identity={'training_config_sha256':cal.digest(cfg),'data_manifest_sha256':cal.digest(data.manifest),
              'normalization_sha256':cal.digest(data.stats),'method':method,'seed':seed,
              'probe_fips':[r['record']['fips'] for r in probe.manifest['records']],
              'scope':'fixed_training_probe_not_heldout'}
    def observe(row,weights):
        index=row['event_count'];path=folder/f'point_{index:05d}.json'
        digest=hashlib.sha256(weights.detach().cpu().contiguous().numpy()).hexdigest()
        if path.exists() and path.with_suffix('.sha256.json').exists():
            saved=cal.read(path)
            if sha(path)!=cal.read(path.with_suffix('.sha256.json'))['sha256'] or saved['identity']!=identity or saved['model_weights_sha256']!=digest:
                raise ValueError('Saved training curve identity or weights changed')
            return
        # Keep cooperative pauses at the next durable training checkpoint.
        if index:update('train',method=method,seed=seed,completed_runs=cfg['seeds'].index(seed)*3+local.METHODS.index(method),total_runs=3*len(cfg['seeds']),
                        activity='training_probe',version=row['version'],accepted_examples_total=row['accepted_examples_total'],
                        processed_examples=row['processed_examples'])
        measured=probe_measurement(model,weights,probe,data.stats)
        value={'identity':identity,'model_weights_sha256':digest,'event_count':index,
               'accepted_examples':row['accepted_examples_total'],'processed_examples':row['processed_examples'],
               'accepted_epoch_equivalent':row['accepted_examples_total']/len(data),
               'ring':row.get('ring'),'staleness':row.get('staleness'),**measured}
        write_json(path,value);write_json(path.with_suffix('.sha256.json'),{'sha256':sha(path)})
        print('TRAINING_PROBE '+json.dumps({'method':method,'accepted_epoch':value['accepted_epoch_equivalent'],**value['metrics']}),flush=True)
    observe({'event_count':0,'version':0,'accepted_examples_total':0,'processed_examples':0},initial)
    return observe


def run_training(config,cache_config,prepared,out,update,control):
    require_shared_inputs(config,cache_config)
    def data_for_cache(path,unused_config,evaluation):return local.CountyData(path,cache_config,evaluation)
    def observed_arm(model,initial,data,clients,cfg,seed,method,emit,save,resume=None,optimizer_store=None):
        control.check()
        archive_uncommitted(out.parent/'curves',method,seed,resume['event_count'] if resume else 0)
        observe=make_observer(model,initial,data,clients,cfg,seed,method,out.parent/'curves',update)
        # A checkpoint is committed before a synchronous probe. Recover a probe
        # interrupted in that interval directly from its saved global weights.
        if resume is not None:
            observe({'event_count':resume['event_count'],'version':resume['version'],
                     'accepted_examples_total':resume['accepted'],'processed_examples':resume['processed']},resume['current'])
        latest={}
        def fusion(*args,**kwargs):
            value=engine.fuse(*args,**kwargs);latest['weights']=value;return value
        def record(row):
            latest['row']=row;emit(row)
            if method=='hierarchical_async':observe(row,latest['weights'])
        def durable_save(state):
            save(state)
            if method!='hierarchical_async':observe(latest['row'],state['current'])
            control.check()
        bound=types.FunctionType(engine.train_arm.__code__,{**engine.train_arm.__globals__,'fuse':fusion})
        return bound(model,initial,data,clients,cfg,seed,method,record,durable_save,resume,optimizer_store)
    bound=types.FunctionType(local.full.train.__code__,{**local.full.train.__globals__,
                             'CountyData':data_for_cache,'train_arm':observed_arm})
    bound(config,prepared,out,update)


def evaluate_models(config,training,prepared,out,control,update):
    """Evaluate every frozen final model once, preserving per-seed paired scores."""
    summary=cal.read(training/'training_summary.json')
    if cal.read(training/'status.json').get('status')!='completed':raise ValueError('Train all nine arms first')
    expected={(s,m) for s in config['seeds'] for m in local.METHODS}
    if len(summary)!=len(expected) or {(r['seed'],r['method']) for r in summary}!=expected:
        raise ValueError('Incomplete training matrix')
    stats=cal.read(training/'normalization.json');data=local.CountyData(prepared,config,True,stats)
    out.mkdir(parents=True,exist_ok=True);rows=local.records(config,True)
    identity={'config_sha256':cal.digest(config),'summary_sha256':sha(training/'training_summary.json'),
              'test_manifest_sha256':sha(prepared/'manifest.json'),'normalization_sha256':sha(training/'normalization.json')}
    if (out/'identity.json').exists() and cal.read(out/'identity.json')!=identity:raise ValueError('Evaluation identity changed')
    write_json(out/'identity.json',identity);model=local.make_model(config).cuda();results=[]
    write_json(out/'status.json',{'status':'running','stage':'historical_test'})
    try:
        for item in summary:
            control.check();prefix=f"{item['method']}_seed{item['seed']}"
            checkpoint=training/item['checkpoint'];prediction_path=out/(prefix+'_predictions.json');result_path=out/(prefix+'_result.json')
            if sha(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Final model changed')
            if result_path.exists():
                result=cal.read(result_path)
                if result['evaluation_identity']!=identity or result['checkpoint_sha256']!=item['checkpoint_sha256']:
                    raise ValueError('Completed evaluation identity mismatch')
                if sha(prediction_path)!=result['prediction_sha256']:raise ValueError('Saved predictions changed')
            else:
                assign(model,torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
                mu,sd=stats['target']['mean'][1],stats['target']['std'][1];truth=[];pred=[]
                with torch.no_grad():
                    for i in range(len(data)):
                        control.check();x,s,l,y=[a.cuda() for a in data(i)];p=model(x,ys=s,yl=l)
                        truth.append(float(torch.exp(y[0,1].double()*sd+mu)))
                        pred.append(float(torch.exp(p[0,1].double()*sd+mu)))
                        if (i+1)%20==0 or i+1==len(data):
                            update('evaluate',method=item['method'],seed=item['seed'],completed=i+1,total=len(data),completed_runs=len(results),total_runs=len(expected))
                write_json(prediction_path,[{'fips':r['fips'],'year':config['evaluation_year'],'truth':a,'prediction':b} for r,a,b in zip(rows,truth,pred)])
                result={**item,'metrics':local.evaluate(truth,pred),'year':config['evaluation_year'],
                        'scope':'three-seed full Tiny CropNet soybean historical retest',
                        'evaluation_identity':identity,'prediction_sha256':sha(prediction_path)}
                write_json(result_path,result)
            predictions=cal.read(prediction_path)
            if [r['fips'] for r in predictions]!=[r['fips'] for r in rows]:raise ValueError('Evaluation counties changed')
            checked=local.evaluate([r['truth'] for r in predictions],[r['prediction'] for r in predictions])
            if checked!=result['metrics']:raise ValueError('Metrics do not match saved predictions')
            results.append(result)
    finally:del model;torch.cuda.empty_cache()
    for result in results:
        ref=next(r for r in results if r['method']=='centralized' and r['seed']==result['seed'])
        result['rmse_gap']=result['metrics']['rmse']-ref['metrics']['rmse']
        result['rmse_gap_percent']=100*result['rmse_gap']/ref['metrics']['rmse'] if ref['metrics']['rmse'] else None
    write_json(out/'metrics.json',results);write_json(out/'config.json',config)
    write_json(out/'status.json',{'status':'completed','stage':'historical_test','runs':len(results)})
    plot_results(out)


def snapshot(out,config):
    local.full.snapshot(out,config);hashes=cal.read(out/'source_sha256.json')
    for relative in ['research/joint_fusion/prepare_mmst_compare.py',
                     'research/joint_fusion/run_mmst_compare.py','research/joint_fusion/plot_mmst_compare.py',
                     'research/joint_fusion/run_mmst_local.py','research/joint_fusion/run_mmst_calibration.py',
                     'research/joint_fusion/MMST_COMPARE_BALANCED_PROTOCOL.md','research/joint_fusion/configs/mmst_compare_balanced_v1.json',
                     'run_mmst_compare.ps1','start_mmst_compare.cmd','pause_mmst_compare.cmd','plot_mmst_compare.cmd',
                     'download_mmst_compare_eval.cmd','本地新三算法对比运行说明.md','tests/test_mmst_compare.py',
                     'research/joint_fusion/mmst_compare_runtime.py','research/joint_fusion/supervise_mmst_compare.py',
                     'research/joint_fusion/mmst_compare_recovery.py',
                     'research/joint_fusion/MMST_UNATTENDED_READINESS.md',
                     'research/joint_fusion/audit_mmst_compare_unattended.py','tests/test_mmst_compare_unattended.py']:
        dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,dest);hashes[relative]=sha(dest)
    write_json(out/'source_sha256.json',hashes)


def input_cache_root(config):
    return ROOT/'results/joint_fusion/mmst_compare_input_readiness'/cal.digest(config)


def ensure_inputs(config,cache_config,manifest,control,update):
    """Share one immutable cache and one preparation owner across entry points."""
    cache_root=input_cache_root(config);prepared=cache_root/'prepared_test'
    lock=local.download_lock(cache_root)
    while True:
        control.check()
        try:lock.__enter__();break
        except RuntimeError as exc:
            if 'Another downloader already holds' not in str(exc):raise
            update('waiting_for_input_preparation',cache=str(cache_root));checked_sleep(15,control)
            lock=local.download_lock(cache_root)
    try:
        write_json(cache_root/'config.json',config)
        if not (prepared/'status.json').exists() or cal.read(prepared/'status.json')['status']!='completed':
            update('download_inputs',cache=str(cache_root))
            fetch_resilient(manifest,ROOT/'data/author_tiny_cropnet/raw',config['max_download_gib'],cache_root/'download_progress.json',control,update)
            control.check();update('prepare_test_inputs',cache=str(cache_root))
            def progress(row):update('prepare_test_inputs',**row);control.check()
            local.prepare(config,ROOT/'data/author_tiny_cropnet/raw',prepared,True,progress)
        scans={}
        for label,path,cfg,is_test in [('train',BASE/'prepared_train',cache_config,False),('evaluation',prepared,config,True)]:
            update('scan_'+label+'_inputs')
            scans[label]=scan_cache(local.CountyData(path,cfg,is_test),config,control)
        readiness={'status':'completed','test_scores_computed':False,'scans':scans,
                   'evaluation_cache':str(prepared),'evaluation_manifest_sha256':sha(prepared/'manifest.json')}
        write_json(cache_root/'input_readiness.json',readiness)
        return prepared,readiness
    finally:lock.__exit__(None,None,None)


def run_pipeline(config,cache_config,manifest,out,control,update):
    """Input readiness precedes expensive training; no test predictions until all models freeze."""
    prepared,readiness=ensure_inputs(config,cache_config,manifest,control,update)
    write_json(out/'input_readiness.json',readiness)
    control.check();local.full.check_sources(out);update('train')
    run_training(config,cache_config,BASE/'prepared_train',out/'training',update,control)
    control.check();local.full.check_sources(out);update('plot_training_curves');plot_curves(out)
    control.check();update('evaluate')
    evaluate_models(config,out/'training',prepared,out/'evaluation',control,update)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--action',choices=['run','pause','status','check','plot','download-eval','prepare-inputs'],default='run')
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--supervised',action='store_true');parser.add_argument('--acknowledge-pause',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if args.action=='status':print(json.dumps(local.display_status(out),indent=2,ensure_ascii=False));return
    if args.action=='pause':
        if not (out/'config.json').exists():print('No comparison has started.');return
        local.PauseControl(out).request();return
    if args.action=='plot':
        if not (out/'config.json').exists():print('No training observations yet.');return
        local.full.check_sources(out);print(plot_curves(out));return
    config=cal.read(args.config)
    if args.action=='download-eval':
        require_shared_inputs(config,cal.read(BASE/'config.json'))
        manifest=cal.read(BASE/'download_manifest.json')
        if (manifest['repository'],manifest['revision'],manifest['include_evaluation'])!=(config['dataset_repository'],config['dataset_revision'],True):
            raise ValueError('Pinned complete manifest mismatch')
        print('Downloading pinned raw files only; no training or evaluation scores.',flush=True)
        download_root=ROOT/'results/joint_fusion/mmst_compare_input_download';download_root.mkdir(exist_ok=True)
        if (download_root/'PAUSE').exists():(download_root/'PAUSE').replace(download_root/f'PAUSE_acknowledged_{time.time_ns()}')
        control=local.PauseControl(download_root)
        def download_update(stage,**row):
            control.phase=stage;write_json(download_root/'status.json',{'status':'running','stage':stage,**row})
        previous=signal.signal(signal.SIGINT,control.signal)
        try:
            with local.keep_awake():fetch_resilient(manifest,ROOT/'data/author_tiny_cropnet/raw',config['max_download_gib'],download_root/'progress.json',control,download_update)
        except local.LocalPause:
            write_json(download_root/'status.json',{'status':'paused'});return
        finally:signal.signal(signal.SIGINT,previous)
        write_json(download_root/'status.json',{'status':'completed','stage':'raw_verified'})
        print('RAW DOWNLOAD COMPLETE',flush=True);return
    cache_config,data,inventory=preflight(config,BASE,out)
    if args.action=='check':
        ready=input_cache_root(config)/'input_readiness.json'
        print('TRAINING PREREQUISITES PASSED. Full input readiness: '+('completed (rechecked at Run)' if ready.exists() else 'pending; Run prepares inputs before training')+'. No training/download/evaluation launched.',flush=True);return
    if args.action=='prepare-inputs':
        root=input_cache_root(config);root.mkdir(parents=True,exist_ok=True);control=local.PauseControl(root)
        if (root/'PAUSE').exists():(root/'PAUSE').replace(root/f'PAUSE_acknowledged_{time.time_ns()}')
        def update_inputs(stage,**row):
            control.phase=stage;write_json(root/'status.json',{'status':'running','stage':stage,**row,'pid':os.getpid()})
        previous=signal.signal(signal.SIGINT,control.signal)
        try:
            with local.keep_awake():ensure_inputs(config,cache_config,cal.read(BASE/'download_manifest.json'),control,update_inputs)
            write_json(root/'status.json',{'status':'completed','stage':'all_inputs_verified','test_scores_computed':False})
            print('ALL INPUTS VERIFIED. No training or test predictions performed.',flush=True)
        except local.LocalPause:write_json(root/'status.json',{'status':'paused'})
        except BaseException as exc:
            write_json(root/'status.json',{'status':'failed','error':repr(exc)});raise
        finally:signal.signal(signal.SIGINT,previous)
        return
    with local.download_lock(ROOT/'results/joint_fusion/mmst_local_gpu_lock'),local.keep_awake():
        manifest=cal.read(BASE/'download_manifest.json')
        if (manifest['repository'],manifest['revision'],manifest['include_evaluation'])!=(config['dataset_repository'],config['dataset_revision'],True):
            raise ValueError('Pinned complete manifest mismatch')
        binding={'base':str(BASE),'prepared_train':str(BASE/'prepared_train'),
                 'manifest_sha256':sha(BASE/'prepared_train/manifest.json'),'cache_config_sha256':cal.digest(cache_config)}
        if out.exists():
            if cal.read(out/'config.json')!=config or cal.read(out/'cache_binding.json')!=binding:raise ValueError('Run identity changed')
            local.full.check_sources(out)
            if (out/'status.json').exists() and cal.read(out/'status.json')['status']=='completed':
                print('Already completed: '+str(out/'evaluation/comparison.md'));return
            if (out/'PAUSE').exists():
                if args.supervised and not args.acknowledge_pause:
                    write_json(out/'status.json',{'status':'paused','stage':'before_resume'});return
                (out/'PAUSE').replace(out/f'PAUSE_acknowledged_{time.time_ns()}')
        else:
            out.mkdir(parents=True);snapshot(out,config);write_json(out/'cache_binding.json',binding)
            write_json(out/'download_manifest.json',manifest);write_json(out/'raw_inventory_at_start.json',inventory)
            write_json(out/'candidate_evidence.json',{'run':str(SHORT),'summary_sha256':sha(SHORT/'summary.json'),
                       'gain':config['server_gain'],'power':config['staleness_power']})
        control=local.PauseControl(out);state={}
        def update(stage,**extra):
            state.clear();state.update(stage=stage,**extra);control.phase=stage
            write_json(out/'status.json',{'status':'running',**state,'pid':os.getpid(),'updated_at':local.full.now()})
        previous=signal.signal(signal.SIGINT,control.signal)
        try:
            torch.set_num_threads(4);update('train')
            print('NEW balanced comparison: same command resumes. Safe pause: pause_mmst_compare.cmd',flush=True)
            run_pipeline(config,cache_config,manifest,out,control,update)
            write_json(out/'status.json',{'status':'completed','stage':'complete','updated_at':local.full.now(),'performance_results_available':True})
            print('COMPLETED: '+str(out/'evaluation/comparison.md'),flush=True)
        except local.LocalPause:
            write_json(out/'status.json',{'status':'paused',**state,'updated_at':local.full.now()})
            print('PAUSED. Run start_mmst_compare.cmd again to resume.',flush=True)
        except BaseException as exc:
            transient=retryable(exc) or (isinstance(exc,RuntimeError) and 'CUDA' in str(exc) and 'out of memory' not in str(exc).lower())
            write_json(out/'status.json',{'status':'interrupted',**state,'error':repr(exc),'retryable':transient,'updated_at':local.full.now()})
            traceback.print_exc();raise
        finally:signal.signal(signal.SIGINT,previous)


if __name__=='__main__':main()
