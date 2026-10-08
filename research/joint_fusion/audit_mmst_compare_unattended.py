"""Bounded operational rehearsal. Artifacts are QA, never paper results."""
import argparse
import contextlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from cropnet_leo.remote import sha
from . import run_mmst_compare as compare
from . import run_mmst_local as local
from . import run_mmst_full_repaired as full
from . import run_mmst_calibration as cal
from . import mmst_full_engine as engine
from .mmst_compare_runtime import scan_cache,capacity_check
from .mmst_io import write_json


def rehearsal(out,mode):
    out.mkdir(parents=True,exist_ok=False);started=time.time();cfg=cal.read(compare.CONFIG)
    production=cal.read(compare.BASE/'config.json');torch.set_num_threads(4)
    if mode=='gpu':
        data=local.CountyData(compare.BASE/'prepared_train',production,False)
        stats=full.fit_statistics(data);data.stats=stats
        selected=[]
        for state in sorted({r['record']['state'] for r in data.manifest['records']}):
            selected.append(max((i for i,r in enumerate(data.manifest['records']) if r['record']['state']==state),key=lambda i:data.manifest['records'][i]['grids']))
        selected.sort();view=cal.View(data,selected);view.stats=stats
        rows=[r['record'] for r in view.manifest['records']]
        class Data:
            def __new__(cls,*args):return view
        cfg={**cfg,'epochs':2,'checkpoint_rule':'QA ONLY two epochs; not performance evidence','clients_per_state':1}
        model_factory=full.make_model
        detail={'selected_training_counties':[{'fips':r['record']['fips'],'grids':r['grids']} for r in view.manifest['records']],
                'real_MMST_ViT':True,'evaluation_2022_used':False}
    else:
        from tests.test_mmst_compare import Data,Toy,CompareTests
        cfg={**CompareTests().config(),'epochs':120,'warmup_epochs':3,'decay_after_epoch':40}
        stats={'target':{'mean':[0.,0.],'std':[1.,1.]}}
        rows=[{'state':str(i//2),'fips':str(i)} for i in range(8)]
        model_factory=lambda _:Toy();detail={'real_MMST_ViT':False,'synthetic_fixture_only':True,'evaluation_2022_used':False}
    cfg['scope']='QA ONLY: operational rehearsal, not scientific comparison'
    schedule=engine.scheduled_lr
    # The QA execution budget is two epochs; its first two learning rates
    # must still come from the untouched production 120-epoch schedule.
    qa_schedule=(lambda c,e:schedule({**c,'epochs':120},e)) if mode=='gpu' else schedule
    write_json(out/'config.json',cfg);prepared=out/'prepared_fixture';prepared.mkdir()
    write_json(prepared/'manifest.json',{'scope':'QA ONLY','records':rows})
    write_json(out/'source_sha256.json',{p:sha(compare.ROOT/p) for p in [
        'research/joint_fusion/run_mmst_compare.py','research/joint_fusion/mmst_full_engine.py',
        'research/joint_fusion/run_mmst_full_repaired.py','research/joint_fusion/mmst_compare_runtime.py',
        'research/joint_fusion/audit_mmst_compare_unattended.py']})
    control=local.PauseControl(out);pause_sent=False
    def update(stage,**row):
        nonlocal pause_sent
        write_json(out/'status.json',{'status':'running','scope':'QA ONLY','stage':stage,**row})
        if not pause_sent and row.get('method')=='centralized' and row.get('version')==1:
            pause_sent=True;control.request()
    torch.cuda.reset_peak_memory_stats()
    with patch.object(local,'CountyData',Data),patch.object(full,'fit_statistics',lambda _:stats), \
         patch.object(full,'make_model',model_factory),patch.object(full,'records',lambda _:rows),patch.object(engine,'scheduled_lr',qa_schedule):
        with (out/'training_console.log').open('w',encoding='utf-8') as console,contextlib.redirect_stdout(console):
            try:compare.run_training(cfg,cfg,prepared,out/'training',update,control)
            except local.LocalPause:pass
            else:raise AssertionError('Injected pause was not reached')
            (out/'PAUSE').replace(out/'PAUSE_audit_acknowledged')
            compare.run_training(cfg,cfg,prepared,out/'training',update,control)
    summary=cal.read(out/'training/training_summary.json')
    assert len(summary)==9
    expected_count=len(rows)*cfg['epochs']
    assert {r['processed_examples'] for r in summary}=={expected_count}
    checkpoints=[]
    for r in summary:
        path=out/'training'/r['checkpoint'];assert sha(path)==r['checkpoint_sha256']
        weights=torch.load(path,map_location='cpu',weights_only=True);assert torch.isfinite(weights).all()
        checkpoints.append({'method':r['method'],'seed':r['seed'],'parameters':len(weights),'finite':True})
    image=compare.plot_curves(out)
    reporting_checked=False
    if mode=='long-logic':
        evaluation_fixture=out/'synthetic_evaluation_fixture';evaluation_fixture.mkdir()
        write_json(evaluation_fixture/'manifest.json',{'scope':'SYNTHETIC QA ONLY; no CropNet evaluation data'})
        with patch.object(local,'CountyData',Data),patch.object(local,'make_model',model_factory),patch.object(local,'records',lambda *a:rows):
            compare.evaluate_models(cfg,out/'training',evaluation_fixture,out/'synthetic_reporting',control,lambda *a,**k:None)
        report=out/'synthetic_reporting/comparison.md'
        report.write_text('> SYNTHETIC QA ONLY. Toy network and artificial samples. Never use these numbers as CropNet results.\n\n'+report.read_text(encoding='utf-8'),encoding='utf-8')
        reporting_checked=True
    write_json(out/'audit.json',{'status':'passed','scope':'QA ONLY, not prediction comparison evidence','mode':mode,
               'runs':9,'epochs':cfg['epochs'],'training_examples_per_run':expected_count,'checkpoints':checkpoints,
               'pause_resume_completed':True,'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
               'seconds':time.time()-started,'curve_image':image,'synthetic_final_reporting_checked':reporting_checked,**detail})
    write_json(out/'status.json',{'status':'completed','scope':'QA ONLY','passed':True})
    print(json.dumps(cal.read(out/'audit.json'),ensure_ascii=False),flush=True)


def inputs(out):
    out.mkdir(parents=True,exist_ok=True);cfg=cal.read(compare.CONFIG);cache,data,inventory=compare.preflight(cfg,compare.BASE,out)
    control=local.PauseControl(out)
    def update(stage,**row):write_json(out/'status.json',{'status':'running','stage':stage,**row})
    with local.keep_awake():
        prepared,readiness=compare.ensure_inputs(cfg,cache,cal.read(compare.BASE/'download_manifest.json'),control,update)
    write_json(out/'audit.json',{'status':'passed','scope':'input validation only',**readiness,'test_predictions_computed':False})
    write_json(out/'status.json',{'status':'completed','passed':True})
    print(json.dumps(cal.read(out/'audit.json')),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--mode',choices=['gpu','long-logic','inputs'],required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:
        if args.mode=='inputs':inputs(args.out)
        else:
            with local.download_lock(compare.ROOT/'results/joint_fusion/mmst_local_gpu_lock'),local.keep_awake():rehearsal(args.out,args.mode)
    except BaseException as exc:
        if args.out.exists():write_json(args.out/'status.json',{'status':'failed','error':repr(exc),'scope':'QA ONLY'})
        raise


if __name__=='__main__':main()
