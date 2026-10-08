"""Bootstrap a fresh checkout without any historical local experiment folders."""
import argparse
import json
import signal
import time

from cropnet_leo.remote import sha
from . import run_mmst_compare as compare
from .mmst_data import records, required_paths
from .mmst_io import write_json


def validate_manifest(config, manifest):
    expected = set(required_paths(records(config) + records(config, True)))
    paths = [item['path'] for item in manifest['files']]
    if (manifest['repository'], manifest['revision'], manifest['include_evaluation']) != (
            config['dataset_repository'], config['dataset_revision'], True):
        raise ValueError('Pinned dataset identity mismatch')
    if len(paths) != len(set(paths)) or set(paths) != expected:
        raise ValueError('Download manifest does not match the complete county protocol')
    if sum(item['size'] for item in manifest['files']) != manifest['total_bytes']:
        raise ValueError('Download manifest size mismatch')


def prepare_inputs(config, manifest, base, control, update):
    """Download/verify raw data, prepare both splits, and scan before training."""
    compare.require_shared_inputs(config, config)
    validate_manifest(config, manifest)
    with compare.local.download_lock(base), compare.local.keep_awake():
        if (base/'config.json').exists():
            if compare.cal.read(base/'config.json') != config:
                raise ValueError('Preparation configuration changed; use a separate checkout')
            compare.local.full.check_sources(base)
        else:
            compare.snapshot(base, config)
            write_json(base/'download_manifest.json', manifest)
        if compare.cal.read(base/'download_manifest.json') != manifest:
            raise ValueError('Preparation manifest changed')
        raw = compare.ROOT/'data/author_tiny_cropnet/raw'
        compare.fetch_resilient(manifest, raw, config['max_download_gib'], base/'download_progress.json', control, update)
        control.check()
        train = base/'prepared_train'
        if not (train/'status.json').exists() or compare.cal.read(train/'status.json')['status'] != 'completed':
            def progress(row):
                update('prepare_training_inputs', **row)
                control.check()
            compare.local.prepare(config, raw, train, False, progress)
        # BASE is this module's explicit input cache, not the old calibration run.
        prepared, readiness = compare.ensure_inputs(config, config, manifest, control, update)
        write_json(base/'input_readiness.json', readiness)
        return prepared


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', action='store_true', help='Validate pinned county/manifest metadata only; no download or training')
    args = parser.parse_args()
    config = compare.cal.read(compare.CONFIG)
    manifest = compare.cal.read(compare.ROOT/'evidence/download_manifest.json')
    validate_manifest(config, manifest)
    if args.plan:
        print(json.dumps({'training_counties':len(records(config)), 'evaluation_counties':len(records(config, True)),
                          'raw_files':len(manifest['files']), 'raw_gib':manifest['total_bytes']/1024**3,
                          'seeds':config['seeds'], 'epochs':config['epochs'], 'methods':3,
                          'dataset_revision':manifest['revision'], 'download_started':False}, indent=2))
        return
    base = compare.BASE
    base.mkdir(parents=True, exist_ok=True)
    if (base/'PAUSE').exists():
        (base/'PAUSE').replace(base/f'PAUSE_acknowledged_{time.time_ns()}')
    control = compare.local.PauseControl(base)
    def update(stage, **row):
        control.phase = stage
        write_json(base/'status.json', {'status':'running', 'stage':stage, **row})
        print(stage, json.dumps(row), flush=True)
    previous = signal.signal(signal.SIGINT, control.signal)
    try:
        prepare_inputs(config, manifest, base, control, update)
        write_json(base/'status.json', {'status':'completed', 'stage':'all_inputs_verified'})
        print('ALL INPUTS VERIFIED. Run start_mmst_compare.cmd to train. No test scores computed.')
    except compare.local.LocalPause:
        write_json(base/'status.json', {'status':'paused'})
        print('PAUSED. Repeat the preparation command to resume.')
    except BaseException as exc:
        write_json(base/'status.json', {'status':'failed','error':repr(exc)})
        raise
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == '__main__':
    main()
