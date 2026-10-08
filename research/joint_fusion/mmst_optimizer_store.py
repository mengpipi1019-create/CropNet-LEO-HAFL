"""Disk-backed client moments; hard-linked recovery generations on one volume."""
import os
from pathlib import Path

import torch

from cropnet_leo.remote import sha


class OptimizerStore:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);self.refs={}
    def __contains__(self,key):return key in self.refs
    def __delitem__(self,key):del self.refs[key]
    def _path(self,relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Optimizer path outside run')
        return path
    def pop(self,key):
        entry=self.refs[key];path=self._path(entry['path'])
        if sha(path)!=entry['sha256']:raise ValueError('Client optimizer checksum mismatch')
        value=torch.load(path,map_location='cpu',weights_only=True)
        del self.refs[key];return value
    def __setitem__(self,key,value):
        if not key.replace('_','').isalnum():raise ValueError('Invalid optimizer client key')
        path=self.root/(key+'.pt');temp=self.root/(key+'.partial')
        torch.save(value,temp);temp.replace(path)
        self.refs[key]={'path':path.name,'sha256':sha(path)}
    def snapshot(self,slot):
        refs={};folder=self.root/f'recovery_{slot}';folder.mkdir(exist_ok=True)
        for key,entry in self.refs.items():
            source=self._path(entry['path']);dest=folder/(key+'.pt');temp=folder/(key+'.link')
            if temp.exists():temp.unlink()
            os.link(source,temp);temp.replace(dest)
            refs[key]={'path':str(dest.relative_to(self.root)),'sha256':entry['sha256']}
        return {'disk_optimizer_refs':refs}
    def restore(self,value):
        refs=value['disk_optimizer_refs']
        for entry in refs.values():
            if sha(self._path(entry['path']))!=entry['sha256']:raise ValueError('Recovery optimizer changed')
        self.refs={k:dict(v) for k,v in refs.items()}
