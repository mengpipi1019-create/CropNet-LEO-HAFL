"""One authorized local job: download -> prepare -> train -> evaluate and plot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

from .download_mmst_data import fetch
from .mmst_io import write_json
from .mmst_bridge import ROOT
from .mmst_data import records, required_paths
from .run_mmst_paired import read_json, prepare, train, test_run, snapshot


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'research/joint_fusion/configs/mmst_paired_v1.json')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/author_tiny_cropnet/raw')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config, manifest = read_json(args.config), read_json(args.manifest)
    if (manifest['repository'], manifest['revision'], manifest['include_evaluation']) != (config['dataset_repository'], config['dataset_revision'], True):
        raise ValueError('Pipeline requires the frozen complete train/test download manifest')
    snapshot(args.out, config)
    shutil.copy2(args.manifest, args.out/'download_manifest.json')
    stage = 'download_train'
    def status(state, **extra):
        write_json(args.out/'status.json', {'status': state, 'stage': stage,
                   'updated_at': datetime.now(timezone.utc).isoformat(),
                   'performance_training_started': stage in ('train', 'download_test', 'prepare_test', 'evaluate', 'complete'),
                   'performance_results_available': stage == 'complete', **extra})
    try:
        train_paths = set(required_paths(records(config)))
        train_files = [e for e in manifest['files'] if e['path'] in train_paths]
        if {e['path'] for e in train_files} != train_paths:
            raise ValueError('Manifest omits required training inputs')
        train_manifest = {**manifest, 'files': train_files, 'total_bytes': sum(e['size'] for e in train_files)}
        print('STAGE download_train: resume pinned training inputs before test inputs', flush=True)
        status('running')
        fetch(train_manifest, args.raw, config['max_download_gib'], args.out/'download_progress.json')
        stage = 'prepare_train'
        status('running')
        print('STAGE prepare_train: all 222 county inputs must validate', flush=True)
        prepare(config, args.raw, args.out/'prepared_train', False)
        stage = 'train'
        status('running')
        print('STAGE train: three methods x three seeds', flush=True)
        train(config, args.out/'prepared_train', args.out/'training', 'cuda')
        stage = 'download_test'
        status('running')
        fetch(manifest, args.raw, config['max_download_gib'], args.out/'download_progress.json')
        stage = 'prepare_test'
        status('running')
        prepare(config, args.raw, args.out/'prepared_test', True)
        stage = 'evaluate'
        status('running')
        test_run(args.out/'training', args.out/'prepared_test', args.out/'evaluation', 'cuda')
        stage = 'complete'
        status('completed')
        print(f'COMPLETED: {args.out / "evaluation/comparison.md"}', flush=True)
    except BaseException as exc:
        traceback.print_exc()
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
