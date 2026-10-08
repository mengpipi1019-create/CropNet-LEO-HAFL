"""Pinned Tiny CropNet download manifest and resumable, hash-checked downloads."""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
import uuid

from cropnet_leo.remote import sha
from .mmst_io import write_json, write_progress
from .mmst_bridge import ROOT
from .mmst_data import records, required_paths


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'CropNet-paired-research/1.0', **(headers or {})}), timeout=45)


def inventory(config, evaluation):
    rows = records(config) + (records(config, True) if evaluation else [])
    required = required_paths(rows)
    repo, revision = config['dataset_repository'], config['dataset_revision']
    base = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/'
    def listing(directory):
        with request(base + urllib.parse.quote(directory, safe='/')) as response:
            entries = json.load(response)
        if len(entries) >= 1000:
            raise ValueError('Pagination required; refusing incomplete directory')
        return entries
    parents = sorted({str(Path(p).parent).replace('\\', '/') for p in required})
    with ThreadPoolExecutor(max_workers=4) as executor:
        entries = {e['path']: e for group in executor.map(listing, parents) for e in group}
    missing = [p for p in required if p not in entries]
    if missing:
        raise ValueError(f'Files absent from pinned public repository: {missing}')
    return {'repository': repo, 'revision': revision, 'include_evaluation': evaluation,
            'total_bytes': sum(entries[p]['size'] for p in required), 'files': [entries[p] for p in required],
            'contents_read': False}


@contextmanager
def download_lock(root):
    """OS releases the lock after a crash; concurrent resumptions cannot append."""
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.download.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another downloader already holds this raw-data directory') from exc
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def fetch(manifest, root, max_gib, progress_path=None):
    with download_lock(root):
        _fetch(manifest, root, max_gib, progress_path)


def _fetch(manifest, root, max_gib, progress_path):
    root.mkdir(parents=True, exist_ok=True)
    pending = []
    for entry in manifest['files']:
        dest = (root/entry['path']).resolve()
        if not dest.is_relative_to(root.resolve()):
            raise ValueError('Dataset path escaped selected root')
        meta = dest.with_suffix(dest.suffix+'.source.json')
        if dest.exists():
            record = json.loads(meta.read_text()) if meta.exists() else {}
            if record.get('revision') == manifest['revision'] and record.get('sha256') == sha(dest) and dest.stat().st_size == entry['size']:
                continue
            raise ValueError(f'Unverified existing file: {dest}')
        pending.append(entry)
    size = sum(e['size'] for e in pending)
    if size > max_gib * 1024**3:
        raise ValueError(f'Download needs {size/1024**3:.2f} GiB; exceeds frozen {max_gib} GiB budget')
    if shutil.disk_usage(root).free < size + 5*1024**3:
        raise ValueError('Insufficient space for raw downloads plus 5 GiB reserve; prepared tensors need additional space')
    for i, entry in enumerate(pending):
        dest = root/entry['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix+'.partial')
        start = part.stat().st_size if part.exists() else 0
        if start > entry['size']:
            raise ValueError('Partial file larger than pinned source')
        while start < entry['size']:
            end = min(start + 8*1024*1024, entry['size']) - 1
            base_url = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{manifest['revision']}/" + urllib.parse.quote(entry['path'], safe='/')
            # Bounded ranges and fresh redirect URLs avoid a stalled multi-GB
            # response. Only fully received blocks are appended to the partial.
            for attempt in range(3):
                try:
                    url = base_url + '?download=true&request_id=' + uuid.uuid4().hex
                    with request(url, {'Range': f'bytes={start}-{end}', 'Cache-Control': 'no-cache'}) as response:
                        whole_small = start == 0 and end == entry['size']-1 and response.status == 200
                        if not whole_small and (response.status != 206 or response.headers.get('Content-Range') != f"bytes {start}-{end}/{entry['size']}"):
                            raise ValueError('Server did not honor bounded range; refusing corrupt append')
                        block = response.read(end-start+2)
                    if len(block) != end-start+1:
                        raise ValueError('Incomplete range block')
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(.5*(attempt+1))
            with part.open('ab' if start else 'wb') as handle:
                handle.write(block)
            start = end+1
            if progress_path:
                write_progress(progress_path, {'current_file': entry['path'], 'current_file_bytes': start,
                                           'current_file_total_bytes': entry['size'], 'pending_file_number': i+1,
                                           'pending_files': len(pending), 'status': 'downloading',
                                           'updated_at_unix': time.time()})
        if part.stat().st_size != entry['size']:
            raise ValueError('Incomplete download; partial retained for explicit rerun')
        checksum = sha(part)
        expected = entry.get('lfs', {}).get('oid')
        if expected and expected != checksum:
            raise ValueError('LFS SHA256 mismatch; partial retained for inspection')
        if not expected:
            blob = hashlib.sha1(f"blob {entry['size']}\0".encode() + part.read_bytes()).hexdigest()
            if blob != entry['oid']:
                raise ValueError('Git blob checksum mismatch')
        part.replace(dest)
        write_json(dest.with_suffix(dest.suffix+'.source.json'), {
            'repository': manifest['repository'], 'revision': manifest['revision'], 'sha256': checksum,
            'size': entry['size'], 'upstream_oid': expected or entry['oid']})
        print(f"downloaded {i+1}/{len(pending)} {entry['path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'research/joint_fusion/configs/mmst_paired_v1.json')
    parser.add_argument('--raw', type=Path, default=ROOT/'data/author_tiny_cropnet/raw')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--include-evaluation', action='store_true')
    parser.add_argument('--fetch', action='store_true', help='Without this flag only list remote file metadata')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        if manifest['repository'] != config['dataset_repository'] or manifest['revision'] != config['dataset_revision'] or manifest['include_evaluation'] != args.include_evaluation:
            raise ValueError('Existing download manifest differs from requested protocol')
    else:
        manifest = inventory(config, args.include_evaluation)
        write_json(args.manifest, manifest)
    print(json.dumps({'files': len(manifest['files']), 'total_gib': manifest['total_bytes']/1024**3, 'fetch': args.fetch}), flush=True)
    if args.fetch:
        status_path = args.manifest.parent/'status.json'
        write_json(status_path, {'status': 'running', 'stage': 'data_download', 'files': len(manifest['files']),
                                 'total_bytes': manifest['total_bytes'], 'performance_results_available': False})
        try:
            fetch(manifest, args.raw, config['max_download_gib'], args.manifest.parent/'download_progress.json')
            write_json(status_path, {'status': 'completed', 'stage': 'data_download', 'files': len(manifest['files']),
                                     'performance_results_available': False})
        except BaseException as exc:
            write_json(status_path, {'status': 'failed', 'stage': 'data_download', 'error': str(exc),
                                     'performance_results_available': False})
            raise


if __name__ == '__main__':
    main()
