"""Windows-tolerant atomic status writes for long MMST jobs."""
import json
import os
from pathlib import Path
import time
import uuid


def write_json(path, value, attempts=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    try:
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                # Readers and antivirus can briefly deny FILE_SHARE_DELETE.
                time.sleep(min(.05 * 2**attempt, .8))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass


def write_progress(path, value):
    """Telemetry must not turn a verified downloaded block into a failed job."""
    try:
        write_json(path, value, attempts=3)
        return True
    except OSError as exc:
        print(f'Progress record delayed ({type(exc).__name__}); download continues.', flush=True)
        return False
