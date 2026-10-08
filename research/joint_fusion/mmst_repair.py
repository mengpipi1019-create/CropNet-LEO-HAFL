"""Training-only MMST repair: explicit input/target transforms and fit gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .mmst_bridge import make_model, ROOT
from .mmst_training import assign, vector, seed_all, optimizer
from .mmst_io import write_json
from .audit_mmst_quick import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_train(run):
    manifest = read(run/'prepared_train/manifest.json')
    if manifest['evaluation'] or any(e['record']['year']!=2021 for e in manifest['entries']):
        raise ValueError('Repair accepts 2021 training inputs only')
    data=[]
    for entry in manifest['entries']:
        p=run/'prepared_train'/entry['path']
        if sha(p)!=entry['sha256']:
            raise ValueError('Input hash changed')
        with np.load(p,allow_pickle=False) as a:
            data.append(tuple(torch.from_numpy(a[k].copy()) for k in ['x','short','long','target']))
    return data, manifest


def statistics(data):
    target=torch.cat([d[3] for d in data]).double()
    short=torch.cat([d[1].reshape(-1,9) for d in data]).double()
    long=torch.cat([d[2].reshape(-1,9) for d in data]).double()
    return {k: {'mean':a.mean(0).float().tolist(),'std':a.std(0,unbiased=False).clamp_min(1e-6).float().tolist()}
            for k,a in [('target',target),('short',short),('long',long)]}


def transform(data, stats, input_mode):
    answer=[]
    for x,s,l,y in data:
        if input_mode=='unit_rgb_weather_z':
            # Invert the archived transform, then convert raw uint8 scale to 0..1.
            mean=x.new_tensor([.466,.471,.380]).view(1,1,1,3,1,1)
            std=x.new_tensor([.195,.194,.192]).view(1,1,1,3,1,1)
            x=((x*std+mean)/255-mean)/std
            s=(s-s.new_tensor(stats['short']['mean']))/s.new_tensor(stats['short']['std'])
            l=(l-l.new_tensor(stats['long']['mean']))/l.new_tensor(stats['long']['std'])
        elif input_mode!='legacy':
            raise ValueError(input_mode)
        y=(y-y.new_tensor(stats['target']['mean']))/y.new_tensor(stats['target']['std'])
        answer.append((x,s,l,y))
    return answer


def metrics(model,data,stats):
    model.eval()
    targets,pred=[],[]
    with torch.no_grad():
        for x,s,l,y in data:
            pred.append(model(x.cuda(),ys=s.cuda(),yl=l.cuda())[0].double().cpu().numpy())
            targets.append(y[0].double().numpy())
    target,pred=np.array(targets),np.array(pred)
    mu,sd=np.array(stats['target']['mean']),np.array(stats['target']['std'])
    truth=np.exp(target[:,1]*sd[1]+mu[1]); yhat=np.exp(pred[:,1]*sd[1]+mu[1])
    corr=float(np.corrcoef(truth,yhat)[0,1]) if np.ptp(yhat)>0 else 0.
    ratio=float(np.std(yhat)/np.std(truth))
    rmse=float(np.sqrt(np.mean((truth-yhat)**2)))
    return {'train_rmse':rmse,'train_corr':corr,'prediction_std_ratio':ratio,
            'prediction_range':[float(yhat.min()),float(yhat.max())],
            'standardized_mse':float(np.mean((pred-target)**2)),
            'fit_pass':rmse<=1.0 and corr>=.9 and .5<=ratio<=1.5,
            'prediction_yield':yhat.tolist(),'truth_yield':truth.tolist()}


def backward_batch(model,data,ids,opt,config):
    model.train(); opt.zero_grad(set_to_none=True)
    loss_sum=0.
    for i in ids:
        x,s,l,y=[t.cuda() for t in data[int(i)]]
        loss=torch.nn.functional.mse_loss(model(x,ys=s,yl=l),y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        (loss/len(ids)).backward(); loss_sum+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradients')
    opt.step()
    return loss_sum/len(ids),float(norm)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    config=read(args.config)
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'config.json',config)
    source=Path(__file__)
    shutil.copy2(source,args.out/source.name)
    write_json(args.out/'source_sha256.json',{str(source.relative_to(ROOT)):sha(source),
        'research/joint_fusion/mmst_bridge.py':sha(ROOT/'research/joint_fusion/mmst_bridge.py'),
        'research/joint_fusion/mmst_training.py':sha(ROOT/'research/joint_fusion/mmst_training.py')})
    start=time.perf_counter()
    def status(value):
        write_json(args.out/'status.json',{**value,'scope':'2021 training-only repair; no 2022 evaluation',
            'updated_at':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start})
    try:
        torch.set_num_threads(4); seed_all(10)
        raw,manifest=load_train(args.run)
        stats=statistics(raw)
        write_json(args.out/'normalization.json',stats)
        write_json(args.out/'training_manifest.json',manifest)
        data=transform(raw,stats,config['input_mode']); del raw
        model_config={**read(args.run/'config.json'),'image_chunk_size':config['image_chunk_size']}
        model=make_model(model_config).cuda()
        initial=torch.load(args.run/'initial_seed10.pt',map_location='cpu',weights_only=True)
        assign(model,initial)
        write_json(args.out/'initial.json',{'sha256':hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            'source':str((args.run/'initial_seed10.pt').resolve()),'parameters':initial.numel()})
        opt=optimizer(model,config)
        initial_metrics=metrics(model,data,stats)
        write_json(args.out/'initial_metrics.json',initial_metrics)
        rng=np.random.default_rng(10)
        success_count=0
        with (args.out/'training.jsonl').open('x',encoding='utf-8') as logfile:
            for epoch in range(config['epochs']):
                lr=config['learning_rate']*min(1.,(epoch+1)/config['warmup_epochs'])
                for group in opt.param_groups: group['lr']=lr
                order=rng.permutation(len(data))
                losses=[]; norms=[]
                for begin in range(0,len(order),config['county_batch_size']):
                    loss,norm=backward_batch(model,data,order[begin:begin+config['county_batch_size']],opt,config)
                    losses.append(loss); norms.append(norm)
                row={'epoch':epoch+1,'lr':lr,'loss':float(np.mean(losses)),'max_gradient_norm':max(norms),
                    'processed_examples':len(data)*(epoch+1),'seconds':time.perf_counter()-start}
                if (epoch+1)%config['evaluate_every']==0:
                    m=metrics(model,data,stats); row.update(m)
                    success_count=success_count+1 if m['fit_pass'] else 0
                    write_json(args.out/'latest_metrics.json',m)
                logfile.write(json.dumps(row)+'\n'); logfile.flush()
                status({'status':'running','detail':row}); print(json.dumps(row),flush=True)
                if success_count>=config['consecutive_passes']:
                    break
        final=metrics(model,data,stats)
        checkpoint=args.out/'final.pt'; torch.save(vector(model),checkpoint)
        write_json(args.out/'final_metrics.json',final)
        write_json(args.out/'checkpoint.json',{'sha256':sha(checkpoint),'epoch':epoch+1,'processed_examples':len(data)*(epoch+1)})
        status({'status':'completed','fit_pass':final['fit_pass'],'detail':final,'epochs':epoch+1})
    except BaseException as exc:
        status({'status':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    main()
