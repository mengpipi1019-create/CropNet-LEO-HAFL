"""Independent calibrated local-Adam federation; training fit only."""
import argparse
from datetime import datetime, timezone
import heapq
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .aggregation import fuse
from .mmst_bridge import make_model, ROOT
from .mmst_training import assign, vector, seed_all, optimizer
from .mmst_repair import read, load_train, transform, metrics
from .mmst_io import write_json
from .audit_mmst_quick import sha


def local_step(model,data,ids,opt,config,lr):
    device=next(model.parameters()).device
    model.train(); opt.zero_grad(set_to_none=True)
    for group in opt.param_groups: group['lr']=lr
    loss_sum=0.
    for i in ids:
        x,s,l,y=[t.to(device) for t in data[i]]
        loss=torch.nn.functional.mse_loss(model(x,ys=s,yl=l),y)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite local loss')
        (loss/len(ids)).backward(); loss_sum+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite local gradient')
    opt.step()
    return loss_sum/len(ids)


def run(model,initial,data,clients,config,emit):
    current=initial.clone(); opts={}; processed=accepted=0
    epochs=config['epochs']; n=len(data)
    def aggregate(base,members,epoch):
        nonlocal processed
        total=sum(len(c['indices']) for c in members)
        average=torch.zeros_like(base); loss=0.
        lr=config['learning_rate']*min(1.,(epoch+1)/config['warmup_epochs'])
        for client in members:
            assign(model,base)
            if config['local_state']=='reset' or client['id'] not in opts:
                opts[client['id']]=optimizer(model,config)
            local_loss=local_step(model,data,client['indices'],opts[client['id']],config,lr)
            weight=len(client['indices'])/total
            average.add_(vector(model),alpha=weight)
            loss+=local_loss*weight; processed+=len(client['indices'])
        return average,total,loss,lr
    if config['method']=='fedavg':
        for epoch in range(epochs):
            current,count,loss,lr=aggregate(current,clients,epoch)
            accepted+=count; assign(model,current)
            emit({'method':'fedavg','epoch':epoch+1,'version':epoch+1,'processed_examples':processed,
                  'accepted_examples_total':accepted,'local_loss':loss,'lr':lr},model)
    elif config['method']=='hierarchical_async':
        groups={r:[c for c in clients if c['ring']==r] for r in sorted({c['ring'] for c in clients})}
        done={r:0 for r in groups}; jobs=[]; version=0; clock=0.; effective=0.
        def dispatch(ring,rank):
            base=current.clone()
            updated,count,loss,lr=aggregate(base,groups[ring],done[ring])
            heapq.heappush(jobs,(clock+config['logical_ring_delays'][rank],rank,ring,version,base,updated,count,loss,lr))
        for rank,ring in enumerate(groups): dispatch(ring,rank)
        while jobs:
            clock,rank,ring,base_version,base,updated,count,loss,lr=heapq.heappop(jobs)
            age=version-base_version; mass=sum(len(c['indices']) for c in groups[ring])
            alpha=config['server_gain']*mass/n
            weight=alpha/(1+age)**config['staleness_power']
            current=fuse(current,[updated],[base],[mass],[age],'delta',alpha,config['staleness_power'])
            version+=1; done[ring]+=1; accepted+=count; effective+=weight
            assign(model,current)
            emit({'method':'hierarchical_async','version':version,'ring':ring,'ring_epoch':done[ring],
                  'base_version':base_version,'staleness':age,'effective_coefficient':weight,
                  'cumulative_effective_coefficient':effective,'processed_examples':processed,
                  'accepted_examples_total':accepted,'local_loss':loss,'lr':lr,'logical_time':clock},model)
            if done[ring]<epochs: dispatch(ring,rank)
    else: raise ValueError(config['method'])
    assert processed==accepted==epochs*n
    assign(model,current)
    return current,processed


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--central',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not read(args.central/'final_metrics.json')['fit_pass']:
        raise ValueError('Centralized training fit gate must pass first')
    config=read(args.config); central_config=read(args.central/'config.json')
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'config.json',config)
    write_json(args.out/'central_protocol.json',central_config)
    source_files=['mmst_repair_federated.py','mmst_repair.py','mmst_bridge.py','mmst_training.py','aggregation.py','MMST_REPAIR_PROTOCOL.md']
    for file in source_files:
        shutil.copy2(ROOT/'research/joint_fusion'/file,args.out/file)
    write_json(args.out/'source_sha256.json',{file:sha(args.out/file) for file in source_files})
    start=time.perf_counter()
    def status(value):
        write_json(args.out/'status.json',{**value,'scope':'2021 training-only calibration; unequal fitting budgets disclosed',
            'updated_at':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start})
    try:
        torch.set_num_threads(4); seed_all(10)
        raw,manifest=load_train(args.run)
        stats=read(args.central/'normalization.json')
        data=transform(raw,stats,central_config['input_mode']); del raw
        write_json(args.out/'normalization.json',stats)
        write_json(args.out/'training_manifest.json',manifest)
        cfg={**read(args.run/'config.json'),'image_chunk_size':central_config['image_chunk_size']}
        model=make_model(cfg).cuda()
        initial=torch.load(args.run/'initial_seed10.pt',map_location='cpu',weights_only=True)
        write_json(args.out/'initial.json',{'sha256':hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            'source':str((args.run/'initial_seed10.pt').resolve()),'no_centralized_warm_start':True})
        clients=read(args.run/'clients.json'); write_json(args.out/'clients.json',clients)
        with (args.out/'training.jsonl').open('x',encoding='utf-8') as logfile:
            def emit(row,model):
                if row['version']%config['evaluate_every']==0:
                    m=metrics(model,data,stats); row.update(m); write_json(args.out/'latest_metrics.json',m)
                row['seconds']=time.perf_counter()-start
                logfile.write(json.dumps(row)+'\n'); logfile.flush()
                status({'status':'running','detail':row})
                print(json.dumps({k:v for k,v in row.items() if k not in ['prediction_yield','truth_yield']}),flush=True)
            final,count=run(model,initial,data,clients,config,emit)
        m=metrics(model,data,stats); write_json(args.out/'final_metrics.json',m)
        torch.save(final,args.out/'final.pt')
        write_json(args.out/'checkpoint.json',{'sha256':sha(args.out/'final.pt'),'processed_examples':count})
        status({'status':'completed','fit_pass':m['fit_pass'],'detail':m,'processed_examples':count})
        write_json(args.out/'artifacts_sha256.json',{p.name:sha(p) for p in args.out.iterdir() if p.is_file()})
    except BaseException as exc:
        status({'status':'failed','error':repr(exc)}); raise


if __name__=='__main__': main()
