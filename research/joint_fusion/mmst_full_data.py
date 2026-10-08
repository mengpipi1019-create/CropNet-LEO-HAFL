"""Resumable uint8 county cache and streaming train-only normalization."""
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

from cropnet_leo.remote import sha
from .mmst_data import records, required_paths, sample, read_rows, value
from .mmst_io import write_json
from .run_mmst_paired import digest


KEYS=('image_uint8','short','long','target')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def compact(arrays):
    x,s,l,y=arrays
    mean=x.new_tensor([.466,.471,.380]).view(1,1,1,3,1,1)
    sd=x.new_tensor([.195,.194,.192]).view(1,1,1,3,1,1)
    raw=x*sd+mean; rounded=raw.round()
    if not torch.allclose(raw,rounded,atol=3e-5,rtol=0) or rounded.min()<0 or rounded.max()>255:
        raise ValueError('Archived image cannot be losslessly represented as uint8')
    return rounded.to(torch.uint8),s,l,y


def prepare(config,raw,out,evaluation,emit=lambda r:None):
    raw,out=Path(raw),Path(out);rows=records(config,evaluation)
    out.mkdir(parents=True,exist_ok=True)
    identity={'config_sha256':digest(config),'evaluation':evaluation,'format':'uint8_full_county_v1'}
    if (out/'identity.json').exists():
        if read(out/'identity.json')!=identity: raise ValueError('Cache identity mismatch')
    else: write_json(out/'identity.json',identity)
    def status(state,**other):write_json(out/'status.json',{'status':state,'stage':'prepare',**other})
    status('running',completed=0,total=len(rows))
    try:
        raw_hashes={}
        for p in required_paths(rows):
            meta=read((raw/p).with_suffix(Path(p).suffix+'.source.json'))
            h=sha(raw/p)
            if meta['revision']!=config['dataset_revision'] or meta['sha256']!=h:
                raise ValueError('Raw data provenance mismatch: '+p)
            raw_hashes[p]=h
        weather_cache={}; state_key=None
        def cached_rows(path):
            key=str(path)
            if key not in weather_cache:
                parsed=read_rows(path)
                if parsed and 'Daily/Monthly' in parsed[0]:
                    # Monthly rows are used by the long window. Daily rows only
                    # matter in this split's six short-weather months/days.
                    short_year=config['evaluation_year'] if evaluation else config['train_year']
                    parsed=[r for r in parsed if r['Daily/Monthly']=='Monthly' or
                            (int(value(r['Year']))==short_year and int(value(r['Month'])) in config['months']
                             and r['Daily/Monthly']=='Daily' and 1<=int(value(r['Day']))<=28)]
                weather_cache[key]=parsed
            return weather_cache[key]
        # Reuse the frozen author-compatible sampler with a read-through cache;
        # this does not mutate mmst_data globals or alter any selected value.
        cached_sample=types.FunctionType(sample.__code__,{**sample.__globals__,'read_rows':cached_rows})
        entries=[]
        for i,r in enumerate(rows):
            if state_key!=r['state']:weather_cache.clear();state_key=r['state']
            name=f"{r['fips']}_{r['year']}.npz";path=out/name;meta=out/(name+'.json')
            dependencies={p:raw_hashes[p] for p in required_paths([r])}
            if path.exists() and meta.exists():
                entry=read(meta)
                if entry['record']!=r or entry['raw_sha256']!=dependencies or sha(path)!=entry['sha256']:
                    raise ValueError('Existing prepared county mismatch: '+name)
            else:
                arrays=compact(cached_sample(raw,r,config))
                temp=out/(name+'.partial')
                with temp.open('wb') as f:np.savez_compressed(f,**{k:a.numpy() for k,a in zip(KEYS,arrays)})
                temp.replace(path)
                entry={'record':r,'path':name,'sha256':sha(path),'raw_sha256':dependencies,
                       'grids':arrays[0].shape[2]}
                write_json(meta,entry)
            entries.append(entry);status('running',completed=i+1,total=len(rows))
            emit({'prepared':i+1,'total':len(rows),'fips':r['fips'],'evaluation':evaluation})
        write_json(out/'manifest.json',{**identity,'records':entries,'raw_sha256':raw_hashes})
        status('completed',completed=len(rows),total=len(rows))
    except BaseException as exc:
        status('failed',error=str(exc));raise


class CountyData:
    def __init__(self,root,config,evaluation,stats=None):
        self.root=Path(root);self.stats=stats
        if read(self.root/'status.json')['status']!='completed':raise ValueError('Incomplete cache')
        self.manifest=read(self.root/'manifest.json')
        if self.manifest['config_sha256']!=digest(config) or self.manifest['evaluation']!=evaluation:
            raise ValueError('Cache protocol/split mismatch')
        if [r['record'] for r in self.manifest['records']]!=records(config,evaluation):
            raise ValueError('County list mismatch')
        for entry in self.manifest['records']:
            if sha(self.root/entry['path'])!=entry['sha256']:raise ValueError('County cache changed')
    def __len__(self):return len(self.manifest['records'])
    def __getitem__(self,i):
        if i<0 or i>=len(self):raise IndexError(i)
        with np.load(self.root/self.manifest['records'][i]['path'],allow_pickle=False) as d:
            x,s,l,y=[torch.from_numpy(d[k]) for k in KEYS]
        if self.stats is None:return x,s,l,y
        x=x.float()/255
        mean=x.new_tensor([.466,.471,.380]).view(1,1,1,3,1,1)
        sd=x.new_tensor([.195,.194,.192]).view(1,1,1,3,1,1)
        x=(x-mean)/sd
        out=[x]
        for key,tensor in [('short',s),('long',l),('target',y)]:
            out.append((tensor-tensor.new_tensor(self.stats[key]['mean']))/tensor.new_tensor(self.stats[key]['std']))
        return tuple(out)
    __call__=__getitem__


def fit_statistics(data):
    if data.manifest['evaluation'] or {r['record']['year'] for r in data.manifest['records']}!={2021}:
        raise ValueError('Statistics accept 2021 training only')
    accum={}
    for entry in data.manifest['records']:
        # Do not decompress images to fit weather and target statistics.
        with np.load(data.root/entry['path'],allow_pickle=False) as a:
            for key,width in [('short',9),('long',9),('target',2)]:
                values=torch.from_numpy(a[key]).reshape(-1,width).double()
                count=len(values);mean=values.mean(0);m2=((values-mean)**2).sum(0)
                if key not in accum:accum[key]=(count,mean,m2)
                else:
                    n,old,ss=accum[key];delta=mean-old;total=n+count
                    accum[key]=(total,old+delta*count/total,ss+m2+delta**2*n*count/total)
    return {k:{'mean':mean.float().tolist(),'std':(ss/n).sqrt().clamp_min(1e-6).float().tolist()}
            for k,(n,mean,ss) in accum.items()}
