"""Operational checks and resumable downloads; no changes to learning rules."""
import errno
import hashlib
import http.client
import json
import shutil
import ssl
import time
import types
import urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
import threading
from pathlib import Path

import numpy as np
from cropnet_leo.remote import sha
from . import download_mmst_data as downloader
from .mmst_io import write_json


def retryable(exc):
    if isinstance(exc,ssl.SSLCertVerificationError):return False
    if isinstance(exc,urllib.error.HTTPError):return exc.code in (408,425,429,500,502,503,504)
    if isinstance(exc,urllib.error.URLError):return retryable(exc.reason) if isinstance(exc.reason,BaseException) else True
    if isinstance(exc,(TimeoutError,ConnectionError,http.client.IncompleteRead,http.client.RemoteDisconnected)):return True
    if isinstance(exc,OSError):
        return exc.errno in (errno.EAGAIN,errno.EINTR,errno.ECONNRESET,errno.ECONNREFUSED,errno.ETIMEDOUT) or getattr(exc,'winerror',None) in (32,33,10051,10052,10053,10054,10060,10061,11001,11002)
    if isinstance(exc,ValueError):return str(exc)=='Incomplete range block'
    return False


def checked_sleep(seconds,control):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        control.check();time.sleep(min(1.,max(0.,deadline-time.monotonic())))


def recover_raw_metadata(entry,manifest,raw):
    """Recover only a rename-before-sidecar crash after checking upstream content."""
    dest=raw/entry['path'];meta=dest.with_suffix(dest.suffix+'.source.json')
    if not dest.exists() or meta.exists():return False
    if dest.stat().st_size!=entry['size']:raise ValueError('Orphan raw size mismatch: '+str(dest))
    checksum=sha(dest);expected=entry.get('lfs',{}).get('oid')
    if expected:
        if checksum!=expected:raise ValueError('Orphan raw SHA256 mismatch: '+str(dest))
    else:
        h=hashlib.sha1(f"blob {entry['size']}\0".encode())
        with dest.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=entry['oid']:raise ValueError('Orphan raw Git blob mismatch: '+str(dest))
    write_json(meta,{'repository':manifest['repository'],'revision':manifest['revision'],'sha256':checksum,
                    'size':entry['size'],'upstream_oid':expected or entry['oid']})
    return True


def fetch_resilient(manifest,raw,max_gib,progress,control,update):
    """Keep completed files, retry temporary failures, never bypass a checksum."""
    raw=Path(raw);raw.mkdir(parents=True,exist_ok=True)
    lock=downloader.download_lock(raw)
    while True:
        control.check()
        try:lock.__enter__();break
        except RuntimeError as exc:
            if 'Another downloader already holds' not in str(exc):raise
            update('waiting_for_download_lock',message=str(exc));checked_sleep(15,control)
            lock=downloader.download_lock(raw)
    try:
        # Account for the whole outstanding download before starting any file.
        pending=sum(e['size'] for e in manifest['files'] if not (raw/e['path']).exists())
        if pending>max_gib*1024**3:raise ValueError('Pinned download exceeds configured size limit')
        if shutil.disk_usage(raw).free<pending+5*1024**3:raise OSError(errno.ENOSPC,'Insufficient raw download space')
        outstanding=[];verified_bytes=0;verified_files=0;states={};guard=threading.Lock();cancel=threading.Event()
        class CancelledDownload(BaseException):pass
        def check():
            control.check()
            if cancel.is_set():raise CancelledDownload()
        def publish():
            # Caller owns guard; writes are serialized across download threads.
            value={'status':'downloading','verified_files':verified_files,'total_files':len(manifest['files']),
                   'verified_bytes':verified_bytes,'total_bytes':sum(e['size'] for e in manifest['files']),
                   'active_files':dict(states),'updated_at_unix':time.time()}
            downloader.write_progress(progress,value)
        for index,entry in enumerate(manifest['files']):
            check()
            dest=(raw/entry['path']).resolve()
            if not dest.is_relative_to(raw.resolve()):raise ValueError('Dataset path escaped root')
            recover_raw_metadata(entry,manifest,raw)
            if dest.exists():
                meta=json.loads(dest.with_suffix(dest.suffix+'.source.json').read_text(encoding='utf-8'))
                if meta.get('revision')!=manifest['revision'] or meta.get('sha256')!=sha(dest) or dest.stat().st_size!=entry['size']:
                    raise ValueError('Unverified existing file: '+str(dest))
                verified_files+=1;verified_bytes+=entry['size']
            else:outstanding.append((index,entry))
        publish()
        def one_file(index,entry):
            nonlocal verified_files,verified_bytes
            attempt=0
            def request(url,headers=None):check();return downloader.request(url,headers)
            def progress_row(path,row):
                with guard:states[entry['path']]=row;publish()
            fetch=types.FunctionType(downloader._fetch.__code__,{**downloader._fetch.__globals__,'request':request,'write_progress':progress_row})
            while True:
                check()
                try:
                    with guard:update('download_inputs',file_number=index+1,total_files=len(manifest['files']),current_file=entry['path'],parallel_files=min(8,len(outstanding)))
                    fetch({**manifest,'files':[entry]},raw,max_gib,progress);break
                except Exception as exc:
                    if not retryable(exc):raise
                    attempt+=1;delay=min(300,5*2**min(attempt-1,6))
                    with guard:update('waiting_for_network',file_number=index+1,current_file=entry['path'],attempt=attempt,
                                      retry_in_seconds=delay,error=repr(exc))
                    print(f'Temporary download failure; retry {attempt} in {delay}s: {exc}',flush=True)
                    class DownloadControl:
                        def check(self):check()
                    checked_sleep(delay,DownloadControl())
            with guard:
                verified_files+=1;verified_bytes+=entry['size'];states.pop(entry['path'],None);publish()
        with ThreadPoolExecutor(max_workers=8,thread_name_prefix='cropnet-input') as pool:
            futures=[pool.submit(one_file,index,entry) for index,entry in outstanding]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                cancel.set()
                for future in futures:future.cancel()
                raise
        write_json(progress,{'status':'completed','files':len(manifest['files']),'verified_bytes':sum(e['size'] for e in manifest['files'])})
    finally:lock.__exit__(None,None,None)


def capacity_check(config,inventory,out,parameter_bytes=256*1024**2):
    """Conservative space bound including three distinct Adam generations."""
    target=Path(out).resolve()
    while not target.exists():target=target.parent
    seeds=len(config['seeds']);clients=config['clients_per_state']*len(config['logical_ring_delays'])
    optimizer=(2*clients+1)*seeds*2*3*parameter_bytes
    checkpoints=seeds*24*parameter_bytes
    evaluation_cache=254*6*config['grid_token_capacity']*3*224*224
    # Full-run bound is intentionally retained on recovery; current outputs count as already allocated.
    estimate=optimizer+checkpoints+evaluation_cache+inventory['missing_bytes']+20*1024**3
    allocated=0;seen=set()
    for path in Path(out).rglob('*') if Path(out).exists() else []:
        if path.is_file():
            st=path.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:allocated+=st.st_size;seen.add(key)
    required=max(20*1024**3,estimate-allocated);free=shutil.disk_usage(target).free
    if free<required:raise OSError(errno.ENOSPC,f'Need conservative {required/1024**3:.1f} GiB free; available {free/1024**3:.1f}')
    return {'free_gib':free/1024**3,'conservative_remaining_gib':required/1024**3,'parameter_byte_bound':parameter_bytes}


def scan_cache(data,config,control=None):
    """Decompress every input now, so late corrupt/oversized counties fail early."""
    max_grids=0
    for entry in data.manifest['records']:
        if control:control.check()
        with np.load(data.root/entry['path'],allow_pickle=False) as arrays:
            x,s,l,y=[arrays[k] for k in ('image_uint8','short','long','target')]
            if x.dtype!=np.uint8 or x.ndim!=6 or x.shape[0:2]!=(1,6) or x.shape[3:]!=(3,224,224):
                raise ValueError('Invalid imagery tensor: '+entry['path'])
            # The author's positional table includes the CLS token.
            if not 1<=x.shape[2]<config['grid_token_capacity']:raise ValueError('Grid capacity including CLS exceeded: '+entry['path'])
            if s.shape!=(1,6,x.shape[2],28,9) or l.shape!=(1,len(config['long_term_years']),12,9) or y.shape!=(1,2):
                raise ValueError('Invalid weather/target shape')
            if not all(np.isfinite(a).all() for a in (s,l,y)):raise ValueError('Nonfinite raw weather/target')
            max_grids=max(max_grids,x.shape[2])
    return {'counties':len(data),'max_grids':max_grids,'maximum_input_grids':config['grid_token_capacity']-1,
            'all_arrays_decompressed_and_finite':True}
