"""Small foreground supervisor: bounded process recovery, no parameter changes."""
import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .mmst_io import write_json

ROOT=Path(__file__).resolve().parents[2]


@contextmanager
def awake():
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:yield
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


@contextmanager
def supervisor_lock(out):
    """One owner per output, including intervals when its worker is restarting."""
    path=out.parent/(out.name+'_supervisor.lock');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)


def decision(exit_code,status,pause_exists):
    if pause_exists or status.get('status')=='paused':return 'paused'
    if exit_code==0 and status.get('status')=='completed':return 'completed'
    if status.get('status')=='interrupted':return 'retry' if status.get('retryable',False) else 'failed'
    # A worker that vanished without recording an exception gets a bounded retry.
    if status.get('status')=='running' or exit_code in (3221225477,-1073741819):return 'retry'
    return 'failed'


def supervise(command,out,max_restarts=3,delay=30,launch=subprocess.call):
    out=Path(out);record=out.parent/(out.name+'_supervisor.json');stopping=False
    def pause(*_):
        nonlocal stopping
        stopping=True
        if out.exists():(out/'PAUSE').write_text('User requested pause\n',encoding='utf-8')
        print('Pause requested; waiting for worker checkpoint.',flush=True)
    previous=signal.signal(signal.SIGINT,pause)
    try:
        with supervisor_lock(out),awake():
            for attempt in range(max_restarts+1):
                if stopping:return 0
                write_json(record,{'status':'running','attempt':attempt+1,'pid':os.getpid()})
                code=launch(command+(['--acknowledge-pause'] if attempt==0 else []))
                status=json.loads((out/'status.json').read_text(encoding='utf-8')) if (out/'status.json').exists() else {}
                action=decision(code,status,(out/'PAUSE').exists())
                if action in ('completed','paused'):
                    write_json(record,{'status':action,'attempts':attempt+1,'exit_code':code});return 0
                if action!='retry' or attempt==max_restarts:
                    write_json(record,{'status':'failed','attempts':attempt+1,'exit_code':code,'worker':status})
                    print('Stopped on a permanent error or exhausted bounded recovery. See status/log; protocol was not changed.',flush=True)
                    return code or 1
                write_json(record,{'status':'retry_wait','attempts':attempt+1,'delay_seconds':delay,'worker':status})
                print(f'Recoverable worker exit; automatically resuming in {delay}s ({attempt+1}/{max_restarts}).',flush=True)
                for _ in range(delay):
                    if stopping or (out/'PAUSE').exists():return 0
                    time.sleep(1)
    finally:signal.signal(signal.SIGINT,previous)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    command=[sys.executable,'-X','utf8','-u','-m','research.joint_fusion.run_mmst_compare','--action','run','--out',str(args.out.resolve()),'--supervised']
    raise SystemExit(supervise(command,args.out.resolve()))


if __name__=='__main__':main()
