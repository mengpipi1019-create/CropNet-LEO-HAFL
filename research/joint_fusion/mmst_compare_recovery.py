"""Keep curve events beyond the durable model checkpoint as recovery evidence."""
from pathlib import Path
import re
import time

from cropnet_leo.remote import sha
from .mmst_io import write_json


def archive_uncommitted(curve_root,method,seed,committed):
    root=Path(curve_root).resolve();folder=(root/f'{method}_seed{seed}').resolve()
    if not folder.is_relative_to(root):raise ValueError('Curve path outside run')
    files=[]
    for path in folder.glob('point_*.json'):
        match=re.fullmatch(r'point_(\d+)\.(?:sha256\.)?json',path.name)
        if match and int(match[1])>committed:files.append(path)
    if not files:return None
    destination=folder/f'uncommitted_before_recovery_{time.time_ns()}'
    destination.mkdir();records=[]
    for path in files:
        source=path.resolve();target=(destination/path.name).resolve()
        if not source.is_relative_to(folder) or not target.is_relative_to(folder):raise ValueError('Recovery move outside arm')
        records.append({'name':path.name,'sha256':sha(source)})
        source.replace(target)
    write_json(destination/'archive.json',{'committed_event_count':committed,'files':records,
               'reason':'Observations after the durable checkpoint will be recomputed, not reused.'})
    return destination
