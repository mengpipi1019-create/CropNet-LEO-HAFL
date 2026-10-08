import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import threading
import time
import numpy as np
import threading
import time

from cropnet_leo.remote import sha
from research.joint_fusion import mmst_compare_runtime as runtime
from research.joint_fusion import supervise_mmst_compare as supervisor
from research.joint_fusion import run_mmst_compare as compare
from research.joint_fusion import run_mmst_local as local
from research.joint_fusion.mmst_io import write_json


class RuntimeTests(unittest.TestCase):
    def test_input_capacity_reserves_cls_and_weather_dimensions_are_exact(self):
        cfg=local.read(compare.CONFIG)
        class Data:
            root=Path('.')
            manifest={'records':[{'path':'fixture.npz'}]}
            def __len__(self):return 1
        def tensors(grids):
            return {'image_uint8':np.broadcast_to(np.zeros((),dtype=np.uint8),(1,6,grids,3,224,224)),
                    'short':np.zeros((1,6,grids,28,9)),'long':np.zeros((1,5,12,9)),'target':np.zeros((1,2))}
        values=tensors(63)
        with patch.object(runtime.np,'load',return_value=contextlib.nullcontext(values)):
            self.assertEqual(runtime.scan_cache(Data(),cfg)['max_grids'],63)
        with patch.object(runtime.np,'load',return_value=contextlib.nullcontext(tensors(64))):
            with self.assertRaisesRegex(ValueError,'including CLS'):runtime.scan_cache(Data(),cfg)
        values['short']=np.zeros((1,6,62,28,9))
        with patch.object(runtime.np,'load',return_value=contextlib.nullcontext(values)):
            with self.assertRaisesRegex(ValueError,'shape'):runtime.scan_cache(Data(),cfg)
    def test_parallel_downloads_commit_separate_files_and_preserve_total_inventory(self):
        payloads={f'{i}.bin':bytes([i])*100 for i in range(4)}
        manifest={'repository':'fixture/repo','revision':'fixed','files':[
            {'path':name,'size':len(value),'lfs':{'oid':hashlib.sha256(value).hexdigest()}} for name,value in payloads.items()]}
        active=0;peak=0;guard=threading.Lock()
        class Response(io.BytesIO):
            status=206;headers={'Content-Range':'bytes 0-99/100'}
        def request(url,headers):
            nonlocal active,peak
            with guard:active+=1;peak=max(peak,active)
            time.sleep(.05)
            name=url.split('?')[0].rsplit('/',1)[-1]
            with guard:active-=1
            return Response(payloads[name])
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            with patch.object(runtime.downloader,'request',request):
                runtime.fetch_resilient(manifest,root/'raw',1,root/'progress.json',local.PauseControl(root),lambda *a,**k:None)
            self.assertGreater(peak,1)
            self.assertEqual(local.read(root/'progress.json')['verified_bytes'],400)
            self.assertEqual(local.read(root/'progress.json')['files'],4)
            for name,payload in payloads.items():self.assertEqual((root/'raw'/name).read_bytes(),payload)
    def test_parallel_downloads_commit_separate_files_and_preserve_total_inventory(self):
        payloads={f'{i}.bin':bytes([i])*100 for i in range(4)}
        manifest={'repository':'fixture/repo','revision':'fixed','files':[
            {'path':name,'size':len(value),'lfs':{'oid':hashlib.sha256(value).hexdigest()}} for name,value in payloads.items()]}
        active=0;peak=0;guard=threading.Lock()
        class Response(io.BytesIO):
            status=206;headers={'Content-Range':'bytes 0-99/100'}
        def request(url,headers):
            nonlocal active,peak
            with guard:active+=1;peak=max(peak,active)
            time.sleep(.05)
            name=url.split('?')[0].rsplit('/',1)[-1]
            with guard:active-=1
            return Response(payloads[name])
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            with patch.object(runtime.downloader,'request',request):
                runtime.fetch_resilient(manifest,root/'raw',1,root/'progress.json',local.PauseControl(root),lambda *a,**k:None)
            self.assertGreater(peak,1)
            self.assertEqual(local.read(root/'progress.json')['verified_bytes'],400)
            self.assertEqual(local.read(root/'progress.json')['files'],4)
            for name,payload in payloads.items():self.assertEqual((root/'raw'/name).read_bytes(),payload)
    def test_retry_classification_stops_bad_data_auth_and_storage_errors(self):
        self.assertTrue(runtime.retryable(TimeoutError()))
        self.assertTrue(runtime.retryable(urllib.error.URLError('DNS temporarily unavailable')))
        for status,expected in [(403,False),(404,False),(429,True),(503,True)]:
            self.assertEqual(runtime.retryable(urllib.error.HTTPError('url',status,'error',{},None)),expected)
        self.assertFalse(runtime.retryable(ValueError('LFS SHA256 mismatch')))
        self.assertFalse(runtime.retryable(OSError(28,'No space left')))

    def test_download_retries_preserve_integrity_and_orphan_metadata_recovers(self):
        payload=b'actual bounded fixture bytes';entry={'path':'fixture.bin','size':len(payload),'lfs':{'oid':hashlib.sha256(payload).hexdigest()}}
        manifest={'repository':'fixture/repo','revision':'pinned','files':[entry]}
        class Response(io.BytesIO):
            status=206;headers={'Content-Range':f'bytes 0-{len(payload)-1}/{len(payload)}'}
        calls=0
        def request(*a,**k):
            nonlocal calls
            calls+=1
            if calls<=3:raise TimeoutError('injected outage')
            return Response(payload)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);control=local.PauseControl(root);events=[]
            with patch.object(runtime.downloader,'request',request),patch.object(runtime.downloader.time,'sleep'),patch.object(runtime,'checked_sleep'):
                runtime.fetch_resilient(manifest,root/'raw',1,root/'progress.json',control,lambda stage,**kw:events.append(stage))
            self.assertEqual(calls,4);self.assertIn('waiting_for_network',events)
            path=root/'raw/fixture.bin';self.assertEqual(path.read_bytes(),payload)
            self.assertEqual(local.read(path.with_suffix('.bin.source.json'))['sha256'],sha(path))
            path.with_suffix('.bin.source.json').unlink()
            self.assertTrue(runtime.recover_raw_metadata(entry,manifest,root/'raw'))
            path.with_suffix('.bin.source.json').unlink();path.write_bytes(b'x'*len(payload))
            with self.assertRaises(ValueError):runtime.recover_raw_metadata(entry,manifest,root/'raw')

    def test_wait_respects_pause_and_supervisor_never_restarts_permanent_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'PAUSE').touch()
            with self.assertRaises(local.LocalPause):runtime.checked_sleep(60,local.PauseControl(root))
        self.assertEqual(supervisor.decision(1,{'status':'interrupted','retryable':False},False),'failed')
        self.assertEqual(supervisor.decision(1,{'status':'running'},True),'paused')
        self.assertEqual(supervisor.decision(17,{'status':'running'},False),'retry')

    def test_real_subprocess_abrupt_exit_is_automatically_restarted(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);out=root/'run';out.mkdir()
            script=root/'worker.py'
            script.write_text("import json,os,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);marker=p/'once'\nif not marker.exists():\n marker.touch();(p/'status.json').write_text(json.dumps({'status':'running'}));os._exit(17)\n(p/'status.json').write_text(json.dumps({'status':'completed'}))\n",encoding='utf-8')
            code=supervisor.supervise([sys.executable,str(script),str(out)],out,delay=0)
            self.assertEqual(code,0)
            self.assertEqual(local.read(root/'run_supervisor.json')['attempts'],2)

    def test_supervisor_is_exclusive_and_restart_budget_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            out=Path(folder)/'run';out.mkdir();calls=[]
            with supervisor.supervisor_lock(out):
                with self.assertRaises(OSError):
                    with supervisor.supervisor_lock(out):pass
            def crash(command):
                calls.append(command);write_json(out/'status.json',{'status':'running'});return 17
            self.assertEqual(supervisor.supervise(['fixture'],out,max_restarts=2,delay=0,launch=crash),17)
            self.assertEqual(len(calls),3)
            self.assertIn('--acknowledge-pause',calls[0]);self.assertNotIn('--acknowledge-pause',calls[1])

    def test_second_block_network_failure_resumes_without_duplicate_bytes(self):
        block=b'x'*(8*1024*1024);tail=b'end';payload=block+tail
        entry={'path':'large.bin','size':len(payload),'lfs':{'oid':hashlib.sha256(payload).hexdigest()}}
        manifest={'repository':'fixture/repo','revision':'fixed','files':[entry]};attempts=[]
        class Response(io.BytesIO):
            status=206
            def __init__(self,content,start,end):
                super().__init__(content);self.headers={'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
        def request(url,headers):
            value=headers['Range'];attempts.append(value)
            if value.startswith('bytes=0-'):return Response(block,0,len(block)-1)
            if len(attempts)<5:raise TimeoutError('injected failure after the first durable block')
            return Response(tail,len(block),len(payload)-1)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            with patch.object(runtime.downloader,'request',request),patch.object(runtime.downloader.time,'sleep'),patch.object(runtime,'checked_sleep'):
                runtime.fetch_resilient(manifest,root/'raw',1,root/'progress.json',local.PauseControl(root),lambda *a,**k:None)
            self.assertEqual((root/'raw/large.bin').read_bytes(),payload)
            self.assertEqual(sum(v.startswith('bytes=0-') for v in attempts),1)

    def test_pipeline_prepares_and_scans_all_inputs_before_training_and_evaluates_last(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'run';out.mkdir();events=[];cfg=local.read(compare.CONFIG)
            def prepare(*args):
                events.append('prepare');write_json(args[2]/'status.json',{'status':'completed'});write_json(args[2]/'manifest.json',{'fixture':True})
            def train(*args):events.append('train')
            def evaluate(*args):events.append('evaluate')
            with patch.object(compare,'input_cache_root',lambda cfg:root/'shared_inputs'),patch.object(compare,'fetch_resilient',lambda *a:events.append('download')), \
                 patch.object(local,'prepare',prepare),patch.object(local,'CountyData',lambda *a:None), \
                 patch.object(compare,'scan_cache',lambda *a:events.append('scan') or {}), \
                 patch.object(local.full,'check_sources'),patch.object(compare,'run_training',train), \
                 patch.object(compare,'plot_curves',lambda *a:events.append('plot')),patch.object(compare,'evaluate_models',evaluate):
                compare.run_pipeline(cfg,cfg,{},out,local.PauseControl(out),lambda *a,**k:None)
            self.assertEqual(events,['download','prepare','scan','scan','train','plot','evaluate'])


if __name__=='__main__':unittest.main()
