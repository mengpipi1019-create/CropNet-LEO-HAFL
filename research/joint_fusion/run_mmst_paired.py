"""Author-network paired experiment, with explicit data/preparation/train/eval stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import torch

from cropnet_leo.remote import sha
from .mmst_io import write_json
from .author_evaluation import evaluate
from .mmst_bridge import ROOT, SOURCE, REVISION, make_model
from .mmst_data import records, required_paths, client_partition, sample
from .mmst_training import seed_all, vector, assign, train_method

METHODS = ('centralized', 'fedavg', 'hierarchical_async')
DEFAULT_CONFIG = ROOT / 'research/joint_fusion/configs/mmst_paired_v1.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_hashes():
    files = list(SOURCE.rglob('*.py')) + list(SOURCE.glob('LICENSE')) + list(SOURCE.glob('input/county_info_*.csv'))
    files += [ROOT / 'research/joint_fusion' / name for name in (
        'run_mmst_paired.py', 'mmst_bridge.py', 'mmst_data.py', 'mmst_training.py',
        'aggregation.py', 'author_evaluation.py', 'plot_mmst_paired.py',
        'download_mmst_data.py', 'verify_mmst_model.py', 'mmst_io.py',
        'run_mmst_pipeline.py', 'MMST_PAIRED_PROTOCOL.md')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def plan(config, raw):
    train, test = records(config), records(config, evaluation=True)
    needed_train, needed_test = required_paths(train), required_paths(test)
    missing = [p for p in needed_train if not (raw/p).is_file()]
    return {'config': config, 'config_sha256': digest(config), 'upstream_revision': REVISION,
            'train_records': train, 'test_records': test, 'clients': client_partition(train, config['clients_per_state']),
            'counts': {'train': len(train), 'historical_test': len(test), 'training_files': len(needed_train),
                       'all_files': len(set(needed_train+needed_test)), 'missing_training_files': len(missing)},
            'missing_training_files': missing, 'raw_root': str(raw.resolve()), 'source_sha256': source_hashes(),
            'test_labels_read': False, 'performance_results_available': False}


def prepare(config, raw, out, evaluation):
    rows = records(config, evaluation)
    paths = required_paths(rows)
    missing = [p for p in paths if not (raw/p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} required input files missing; first: {missing[0]}. Run the data download stage first.')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'status.json', {'status': 'running', 'stage': 'prepare'})
    try:
        raw_hashes = {p: sha(raw/p) for p in paths}
        entries = []
        for i, r in enumerate(rows):
            arrays = sample(raw, r, config)
            path = out/f"{r['fips']}_{r['year']}.npz"
            np.savez_compressed(path, **{k: a.numpy() for k, a in zip(('x', 'short', 'long', 'target'), arrays)})
            entries.append({'record': r, 'path': path.name, 'sha256': sha(path)})
            print(f"prepared {i+1}/{len(rows)} {r['fips']}/{r['year']}", flush=True)
        write_json(out/'manifest.json', {'config_sha256': digest(config), 'records': entries,
                   'raw_sha256': raw_hashes, 'source_sha256': source_hashes(), 'evaluation': evaluation})
        write_json(out/'status.json', {'status': 'completed', 'stage': 'prepare', 'samples': len(rows)})
    except BaseException as exc:
        write_json(out/'status.json', {'status': 'failed', 'stage': 'prepare', 'error': str(exc)})
        raise


class Prepared:
    def __init__(self, root, config, evaluation):
        self.root = Path(root)
        status = read_json(self.root/'status.json')
        if status.get('status') != 'completed':
            raise ValueError('Prepared data did not complete')
        self.manifest = read_json(self.root/'manifest.json')
        if self.manifest['config_sha256'] != digest(config) or self.manifest['evaluation'] != evaluation:
            raise ValueError('Prepared data protocol or split mismatch')
        expected = records(config, evaluation)
        if [r['record'] for r in self.manifest['records']] != expected:
            raise ValueError('Prepared data county/year list differs from frozen protocol')
        for r in self.manifest['records']:
            if sha(self.root/r['path']) != r['sha256']:
                raise ValueError('Prepared input checksum mismatch')

    def __call__(self, i):
        with np.load(self.root/self.manifest['records'][i]['path'], allow_pickle=False) as data:
            return tuple(torch.from_numpy(data[k]) for k in ('x', 'short', 'long', 'target'))


def snapshot(out, config):
    write_json(out/'config.json', config)
    hashes = source_hashes()
    for relative in hashes:
        dest = out/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, dest)
    write_json(out/'source_sha256.json', hashes)


def train(config, prepared, out, device):
    # Verify all inputs before reserving a run directory or allocating a GPU.
    data = Prepared(prepared, config, False)
    clients = client_partition(records(config), config['clients_per_state'])
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out, config)
    write_json(out/'data_manifest.json', data.manifest)
    write_json(out/'clients.json', clients)
    write_json(out/'status.json', {'status': 'running', 'test_evaluated': False})
    try:
        results = []
        for seed in config['seeds']:
            seed_all(seed)
            model = make_model(config).to(device)
            initial = vector(model)
            initial_hash = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
            torch.save(initial, out/f'initial_seed{seed}.pt')
            for method in METHODS:
                log = out/f'{method}_seed{seed}.jsonl'
                with log.open('x', encoding='utf-8') as handle:
                    def emit(row):
                        handle.write(json.dumps(row, allow_nan=False)+'\n')
                        handle.flush()
                        print(json.dumps({'seed': seed, **row}), flush=True)
                    final, processed = train_method(model, initial, data, clients, config, seed, method, emit)
                path = out/f'{method}_seed{seed}.pt'
                torch.save(final, path)
                results.append({'method': method, 'seed': seed, 'initial_sha256': initial_hash,
                                'checkpoint': path.name, 'checkpoint_sha256': sha(path),
                                'processed_examples': processed, 'checkpoint_rule': config['checkpoint_rule']})
                write_json(out/'training_summary.json', results)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_json(out/'status.json', {'status': 'completed', 'stage': 'training', 'test_evaluated': False,
                                       'runs': len(results), 'performance_results_available': False})
    except BaseException as exc:
        write_json(out/'status.json', {'status': 'failed', 'stage': 'training', 'error': str(exc)})
        raise


def test_run(run, prepared, out, device):
    status = read_json(run/'status.json')
    if status.get('status') != 'completed' or status.get('stage') != 'training':
        raise ValueError('Training matrix must complete before historical-test evaluation')
    for relative, expected in read_json(run/'source_sha256.json').items():
        if sha(ROOT/relative) != expected:
            raise ValueError('Code changed since training; evaluate with the archived source')
    config = read_json(run/'config.json')
    data = Prepared(prepared, config, True)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'status.json', {'status': 'running', 'stage': 'historical_test'})
    write_json(out/'data_manifest.json', data.manifest)
    write_json(out/'config.json', config)
    write_json(out/'training_summary.json', read_json(run/'training_summary.json'))
    try:
        model = make_model(config).to(device)
        model.eval()
        rows = records(config, True)
        results = []
        for trained in read_json(run/'training_summary.json'):
            path = run/trained['checkpoint']
            if sha(path) != trained['checkpoint_sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            assign(model, torch.load(path, map_location='cpu', weights_only=True))
            truth, predictions = [], []
            with torch.no_grad():
                for i in range(len(rows)):
                    x, short, long, target = [a.to(device) for a in data(i)]
                    y = model(x, ys=short, yl=long)
                    truth.append(float(target[0, -1].double().exp()))
                    predictions.append(float(y[0, -1].double().exp()))
            # Values are taken from the SAME final checkpoint; no independent
            # maxima/minima across epochs as in the upstream printed best list.
            result = {**trained, 'metrics': evaluate(truth, predictions), 'year': config['evaluation_year'],
                      'scope': 'historical benchmark retest; no claim of untouched confirmatory testing'}
            results.append(result)
            write_json(out/f"{trained['method']}_seed{trained['seed']}_predictions.json",
                       [{'fips': r['fips'], 'year': r['year'], 'truth': y, 'prediction': p}
                        for r, y, p in zip(rows, truth, predictions)])
            print(json.dumps(result), flush=True)
        for r in results:
            ref = next(v for v in results if v['seed'] == r['seed'] and v['method'] == 'centralized')
            difference = r['metrics']['rmse'] - ref['metrics']['rmse']
            r['rmse_gap'] = difference
            r['rmse_gap_percent'] = 100*difference/ref['metrics']['rmse'] if ref['metrics']['rmse'] else None
        write_json(out/'metrics.json', results)
        write_json(out/'provenance.json', {'training_run': str(run.resolve()), 'source_sha256': source_hashes(),
                                          'test_manifest_sha256': sha(prepared/'manifest.json')})
        write_json(out/'status.json', {'status': 'completed', 'stage': 'historical_test', 'runs': len(results)})
        from .plot_mmst_paired import plot
        plot(out)
    except BaseException as exc:
        write_json(out/'status.json', {'status': 'failed', 'stage': 'historical_test', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['plan', 'prepare', 'train', 'evaluate', 'plot'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/author_tiny_cropnet/raw')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-evaluation', action='store_true', help='prepare 2022 historical test inputs; never train on them')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = read_json(args.config)
    if config['upstream_revision'] != REVISION:
        raise ValueError('This adapter is pinned to one audited upstream revision')
    if args.stage == 'plan':
        args.out.mkdir(parents=True, exist_ok=False)
        report = plan(config, args.raw)
        write_json(args.out/'readiness.json', report)
        write_json(args.out/'status.json', {'status': 'completed', 'stage': 'readiness_audit',
                                           'training_ready': not report['missing_training_files']})
        print(json.dumps(report['counts']))
    elif args.stage == 'prepare':
        prepare(config, args.raw, args.out, args.include_evaluation)
    elif args.stage == 'train':
        if not args.prepared:
            parser.error('--prepared is required')
        train(config, args.prepared, args.out, args.device)
    elif args.stage == 'evaluate':
        if not args.prepared or not args.run:
            parser.error('--prepared and --run are required')
        test_run(args.run, args.prepared, args.out, args.device)
    else:
        from .plot_mmst_paired import plot
        plot(args.out)


if __name__ == '__main__':
    main()
