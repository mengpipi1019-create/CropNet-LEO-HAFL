"""Equal-budget repaired training with CPU optimizer state and exact recovery."""
import heapq
import math

import numpy as np
import torch

from .aggregation import fuse
from .mmst_async_annealed import scheduled_lr
from .mmst_training import assign, vector, optimizer, seed_all


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def batch_step(model,get_sample,ids,opt,lr,clip):
    device=next(model.parameters()).device
    model.train(); opt.zero_grad(set_to_none=True)
    for group in opt.param_groups: group['lr']=lr
    total=0.
    for i in ids:
        x,s,l,y=[t.to(device) for t in get_sample(int(i))]
        loss=torch.nn.functional.mse_loss(model(x,ys=s,yl=l),y)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
        (loss/len(ids)).backward(); total+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite training gradient')
    opt.step(); opt.zero_grad(set_to_none=True)
    return total


def train_arm(model,initial,get_sample,clients,config,seed,method,emit,save,resume=None,optimizer_store=None):
    if method not in ('centralized','fedavg','hierarchical_async'): raise ValueError(method)
    if list(model.buffers()): raise ValueError('Unaccounted model buffers')
    ids=[i for c in clients for i in c['indices']]; n=len(ids)
    if not n or sorted(ids)!=list(range(n)): raise ValueError('Invalid client partition')
    if len({c['id'] for c in clients})!=len(clients): raise ValueError('Repeated client id')
    epochs=config['epochs']; interval=config['checkpoint_every_epochs']
    seed_all(seed); rng=np.random.default_rng(seed)
    state={'current':initial.clone(),'opts':{},'processed':0,'accepted':0,'epoch':0,'version':0,'clock':0.,
           'jobs':[],'done':{},'started':{},'event_count':0,'method':method,'seed':seed}
    if resume is not None:
        state=resume
        if (state['method'],state['seed'])!=(method,seed): raise ValueError('Recovery arm mismatch')
        rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    if optimizer_store is not None:
        if resume is not None:optimizer_store.restore(state['opts'])
        state['opts']=optimizer_store
    def checkpoint():
        state['numpy_rng']=rng.bit_generator.state
        state['torch_rng']=torch.get_rng_state()
        state['cuda_rng']=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        save(state)
    def record(row):
        state['event_count']+=1
        emit({'method':method,'seed':seed,'event_count':state['event_count'],
              'processed_examples':state['processed'],'accepted_examples_total':state['accepted'],**row})
    def average(base,members,epoch):
        mass=sum(len(c['indices']) for c in members)
        avg=torch.zeros_like(base); total_loss=0.
        lr=scheduled_lr(config,epoch)
        for c in members:
            assign(model,base); opt=optimizer(model,config)
            if c['id'] in state['opts']: opt.load_state_dict(state['opts'].pop(c['id']))
            loss=batch_step(model,get_sample,c['indices'],opt,lr,config['gradient_clip'])
            state['opts'][c['id']]=cpu_tree(opt.state_dict())
            del opt
            avg.add_(vector(model),alpha=len(c['indices'])/mass)
            total_loss+=loss; state['processed']+=len(c['indices'])
        return avg,mass,total_loss/mass,lr
    if method=='centralized':
        assign(model,state['current']); opt=optimizer(model,config)
        if 'central' in state['opts']: opt.load_state_dict(state['opts'].pop('central'))
        for epoch in range(state['epoch'],epochs):
            order=rng.permutation(n); loss=0.; lr=scheduled_lr(config,epoch)
            for begin in range(0,n,config['county_batch_size']):
                batch=order[begin:begin+config['county_batch_size']]
                loss+=batch_step(model,get_sample,batch,opt,lr,config['gradient_clip'])
                state['processed']+=len(batch);state['accepted']+=len(batch)
            state['current']=vector(model);state['epoch']=epoch+1;state['version']=epoch+1
            record({'epoch':epoch+1,'version':epoch+1,'local_loss':loss/n,'lr':lr})
            if (epoch+1)%interval==0 or epoch+1==epochs:
                state['opts']['central']=cpu_tree(opt.state_dict());checkpoint();del state['opts']['central']
    elif method=='fedavg':
        for epoch in range(state['epoch'],epochs):
            state['current'],count,loss,lr=average(state['current'],clients,epoch)
            state['accepted']+=count;state['epoch']=epoch+1;state['version']=epoch+1
            record({'epoch':epoch+1,'version':epoch+1,'local_loss':loss,'lr':lr})
            if (epoch+1)%interval==0 or epoch+1==epochs: checkpoint()
    else:
        groups={r:[c for c in clients if c['ring']==r] for r in sorted({c['ring'] for c in clients})}
        delays=config['logical_ring_delays'];lead=config['max_ring_lead']
        if len(groups)!=len(delays) or any(d<=0 or not math.isfinite(d) for d in delays) or lead<1:
            raise ValueError('Invalid asynchronous schedule')
        if not state['done']:
            state['done']={r:0 for r in groups};state['started']={r:0 for r in groups}
        def dispatch():
            busy={j[2] for j in state['jobs']}
            for rank,ring in enumerate(groups):
                if ring in busy or state['started'][ring]>=epochs or state['started'][ring]+1>min(state['done'].values())+lead:
                    continue
                base=state['current'].clone();epoch=state['started'][ring]
                update,count,loss,lr=average(base,groups[ring],epoch)
                state['started'][ring]+=1
                heapq.heappush(state['jobs'],(state['clock']+delays[rank],rank,ring,state['version'],base,update,count,loss,lr))
        dispatch()
        while state['jobs']:
            clock,rank,ring,base_version,base,update,count,loss,lr=heapq.heappop(state['jobs'])
            state['clock']=clock;age=state['version']-base_version
            alpha=config['server_gain']*count/n
            state['current']=fuse(state['current'],[update],[base],[count],[age],'delta',alpha,config['staleness_power'])
            state['version']+=1;state['accepted']+=count;state['done'][ring]+=1
            assert max(state['done'].values())-min(state['done'].values())<=lead
            record({'version':state['version'],'ring':ring,'ring_epoch':state['done'][ring],
                    'ring_epochs':dict(state['done']),'base_version':base_version,'staleness':age,
                    'effective_coefficient':alpha/(1+age)**config['staleness_power'],
                    'logical_time':clock,'local_loss':loss,'lr':lr})
            # Recovery snapshots include all dispatches following this event.
            dispatch()
            if state['version']%(interval*len(groups))==0 or not state['jobs']: checkpoint()
        assert all(v==epochs for v in state['done'].values())
    assert state['processed']==state['accepted']==n*epochs
    assign(model,state['current'])
    return state['current'],state['processed']
