"""Bound ring progress lead while preserving stale asynchronous model deltas."""
import heapq
import math
import torch

from .aggregation import fuse
from .mmst_training import assign, vector, optimizer
from .mmst_repair_federated import local_step


def scheduled_lr(config, epoch):
    """Learning rate at a zero-based local epoch; fixed before training."""
    base=config['learning_rate']
    warmup=config['warmup_epochs']
    start=config['decay_after_epoch']; end=config['epochs']
    floor=config['minimum_lr_ratio']
    if not (warmup <= start < end and 0 < floor <= 1):
        raise ValueError('Invalid warmup/decay schedule')
    step=epoch+1
    if not 1 <= step <= end: raise ValueError('Local epoch outside frozen budget')
    if step <= warmup: return base*step/warmup
    progress=max(0.,(step-start)/(end-start))
    return base*(floor+(1-floor)*0.5*(1+math.cos(math.pi*progress)))


def run_balanced(model,initial,data,clients,config,emit):
    current=initial.clone(); opts={}; processed=accepted=version=0
    n=len(data); limit=config['epochs']; lead=config['max_ring_lead']
    assert lead>=1
    groups={r:[c for c in clients if c['ring']==r] for r in sorted({c['ring'] for c in clients})}
    assert len(groups)==len(config['logical_ring_delays'])
    done={r:0 for r in groups}; started={r:0 for r in groups}
    busy=set(); jobs=[]; clock=0.; effective=0.
    def dispatch_ready():
        nonlocal processed
        for rank,ring in enumerate(groups):
            if ring in busy or started[ring]>=limit or started[ring]+1>min(done.values())+lead:
                continue
            epoch=started[ring]; base=current.clone(); updated=torch.zeros_like(base)
            members=groups[ring]; mass=sum(len(c['indices']) for c in members)
            lr=scheduled_lr(config,epoch)
            loss=0.
            for c in members:
                assign(model,base)
                if config['local_state']=='reset' or c['id'] not in opts:
                    opts[c['id']]=optimizer(model,config)
                loss+=len(c['indices'])/mass*local_step(model,data,c['indices'],opts[c['id']],config,lr)
                updated.add_(vector(model),alpha=len(c['indices'])/mass)
                processed+=len(c['indices'])
            started[ring]+=1; busy.add(ring)
            heapq.heappush(jobs,(clock+config['logical_ring_delays'][rank],rank,ring,version,base,updated,mass,loss,lr))
    dispatch_ready()
    while jobs:
        clock,rank,ring,base_version,base,updated,count,loss,lr=heapq.heappop(jobs)
        age=version-base_version; alpha=config['server_gain']*count/n
        coefficient=alpha/(1+age)**config['staleness_power']
        current=fuse(current,[updated],[base],[count],[age],'delta',alpha,config['staleness_power'])
        version+=1; done[ring]+=1; accepted+=count; effective+=coefficient; busy.remove(ring)
        assert max(done.values())-min(done.values())<=lead
        assign(model,current)
        emit({'method':'hierarchical_async','version':version,'ring':ring,'ring_epoch':done[ring],
              'ring_epochs':dict(done),'base_version':base_version,'staleness':age,
              'effective_coefficient':coefficient,'cumulative_effective_coefficient':effective,
              'processed_examples':processed,'accepted_examples_total':accepted,
              'local_loss':loss,'lr':lr,'logical_time':clock,'max_ring_lead':lead},model)
        dispatch_ready()
    assert processed==accepted==limit*n and all(e==limit for e in done.values())
    assign(model,current)
    return current,processed
