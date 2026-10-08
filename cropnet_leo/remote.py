"""Pinned public downloads and strict range reads; never silently fetch a whole HDF5."""
from pathlib import Path
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO = 'CropNet/CropNet'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def request(url, headers=None, limit=16*1024*1024):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CropNet-LEO-research-bridge/1.0', **(headers or {})})
            with urllib.request.urlopen(req, timeout=40) as response:
                if headers and 'Range' in headers and response.status != 206:
                    raise RuntimeError('Server ignored HTTP Range; full-file download refused.')
                data = response.read(limit+1)
                if len(data) > limit:
                    raise RuntimeError('Response exceeds the download limit.')
                return data, dict(response.headers), response.status
        except Exception:
            if attempt == 2:
                raise
            time.sleep(.5 * (attempt+1))


def api(url):
    return json.loads(request(url)[0])


class Store:
    def __init__(self, root=None):
        self.root = Path(root or ROOT / 'data' / 'cropnet')
        self.root.mkdir(parents=True, exist_ok=True)
        self.pin_path = self.root / 'upstream_pin.json'
        if self.pin_path.exists():
            self.pin = json.loads(self.pin_path.read_text(encoding='utf-8'))
        else:
            info = api('https://huggingface.co/api/datasets/' + REPO)
            self.pin = {'repository': REPO, 'revision': info['sha'], 'source': 'official Hugging Face dataset'}
            write_json(self.pin_path, self.pin)
        self.revision = self.pin['revision']

    def listing(self, prefix):
        key = hashlib.sha256(prefix.encode()).hexdigest()[:24]
        cached = self.root / 'metadata' / f'{key}.json'
        if cached.exists():
            return json.loads(cached.read_text(encoding='utf-8'))
        url = f'https://huggingface.co/api/datasets/{REPO}/tree/{self.revision}/' + urllib.parse.quote(prefix, safe='/')
        data = api(url)
        if len(data) >= 1000:
            raise RuntimeError('Directory requires pagination; use smaller state/year prefixes.')
        write_json(cached, data)
        return data

    def url(self, relative):
        return f'https://huggingface.co/datasets/{REPO}/resolve/{self.revision}/' + urllib.parse.quote(relative, safe='/')

    def download(self, entry):
        rel = entry['path']
        path = self.root / 'raw' / rel
        meta = path.with_suffix(path.suffix + '.source.json')
        if path.exists() and meta.exists():
            info = json.loads(meta.read_text(encoding='utf-8'))
            if path.stat().st_size == entry['size'] and sha(path) == info['sha256']:
                return path
            raise RuntimeError(f'Cached file failed integrity check: {path}')
        if entry['size'] > 30*1024*1024:
            raise RuntimeError('Whole-file downloads limited to 30 MB; HDF5 uses range reads.')
        data, _, _ = request(self.url(rel), limit=entry['size'])
        if len(data) != entry['size']:
            raise RuntimeError(f'Truncated download: {rel}')
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + '.part')
        part.write_bytes(data)
        part.replace(path)
        write_json(meta, {'repository': REPO, 'revision': self.revision, 'path': rel,
                          'size': len(data), 'sha256': sha(path), 'url': self.url(rel)})
        return path


class RangeFile(io.RawIOBase):
    """Seekable read-only HDF5 range cache, bound to a pinned dataset revision."""
    def __init__(self, store, entry, max_bytes=96*1024*1024, block_size=512*1024):
        super().__init__()
        self.store, self.entry = store, entry
        self.size, self.pos = int(entry['size']), 0
        self.block_size, self.max_bytes, self.new_bytes = block_size, max_bytes, 0
        identity = store.revision + '/' + entry['path']
        self.cache = store.root / 'ranges' / hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.cache.mkdir(parents=True, exist_ok=True)
        write_json(self.cache / 'source.json', {'revision': store.revision, **entry, 'block_size': block_size})

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if pos < 0: raise ValueError('Negative seek')
        self.pos = pos
        return pos

    def _block(self, i):
        path = self.cache / f'{i:08d}.bin'
        start, end = i*self.block_size, min((i+1)*self.block_size, self.size)-1
        expected = end-start+1
        if path.exists():
            value = path.read_bytes()
            digest = path.with_suffix('.sha256')
            if len(value) != expected or not digest.exists() or hashlib.sha256(value).hexdigest() != digest.read_text():
                raise RuntimeError('Corrupted range cache')
            return value
        if self.new_bytes + expected > self.max_bytes:
            raise RuntimeError('Per-file range download budget reached; increase it explicitly.')
        # A fresh query also prevents reuse of a stale signed CDN redirect.
        for attempt in range(3):
            url = self.store.url(self.entry['path']) + f'?download=true&range_block={i}&request_id={uuid.uuid4().hex}'
            value, headers, _ = request(url, {'Range': f'bytes={start}-{end}', 'Cache-Control':'no-cache'}, limit=expected)
            cr = next((v for k,v in headers.items() if k.lower() == 'content-range'), '')
            if cr == f'bytes {start}-{end}/{self.size}' and len(value) == expected:
                break
            if attempt == 2:
                raise RuntimeError(f'Invalid Content-Range: received {cr}, expected bytes {start}-{end}/{self.size}')
        path.write_bytes(value)
        path.with_suffix('.sha256').write_text(hashlib.sha256(value).hexdigest())
        self.new_bytes += len(value)
        return value

    def read(self, size=-1):
        if size < 0: size = self.size-self.pos
        size = max(0, min(size, self.size-self.pos))
        chunks = []
        while size:
            i, offset = divmod(self.pos, self.block_size)
            value = self._block(i)
            count = min(size, len(value)-offset)
            chunks.append(value[offset:offset+count])
            self.pos += count
            size -= count
        return b''.join(chunks)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)
