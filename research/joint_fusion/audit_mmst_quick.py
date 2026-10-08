"""Read-only audit of a completed small MMST paired run; save outside the run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .mmst_bridge import ROOT, make_model
from .mmst_training import assign


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(run):
    status, config = read(run/'status.json'), read(run/'config.json')
    assert status['status'] == 'completed' and status['performance_results_available']
    counts = {}
    for manifest, base in [('artifacts_sha256.json', run), ('source_sha256.json', ROOT),
                           ('quick_source_sha256.json', ROOT/'research/joint_fusion')]:
        entries = read(run/manifest)
        for name, expected in entries.items():
            assert sha(base/name) == expected, f'Hash mismatch: {manifest}/{name}'
        counts[manifest] = len(entries)
    assert config['seeds'] == [10] and config['epochs'] == 20
    rows = read(run/'metrics.json')
    methods = ['centralized', 'fedavg', 'hierarchical_async']
    assert [r['method'] for r in rows] == methods
    initial = torch.load(run/'initial_seed10.pt', map_location='cpu', weights_only=True)
    initial_sha = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
    test = read(run/'prepared_test/manifest.json')
    assert test['evaluation'] and len(test['entries']) == 10
    assert [e['record']['fips'] for e in test['entries']] == config['quick_counties']
    assert all(e['record']['year'] == 2022 for e in test['entries'])
    torch.set_num_threads(4)
    model = make_model(config).cuda().eval()
    report = []
    for row in rows:
        method = row['method']
        assert row['initial_sha256'] == initial_sha and row['processed_examples'] == 200
        checkpoint = run/row['checkpoint']
        assert sha(checkpoint) == row['checkpoint_sha256']
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert weights.shape == initial.shape and torch.isfinite(weights).all()
        delta = float((weights-initial).abs().max())
        assert delta > 0
        events = [json.loads(s) for s in (run/f'{method}_training.jsonl').read_text().splitlines()]
        assert all(np.isfinite(e['log_target_mse']) for e in events)
        assert events[-1]['processed_examples'] == 200
        max_age = None
        if method == 'hierarchical_async':
            assert len(events) == 40 and sum(e['accepted_examples'] for e in events) == 200
            assert all([e['ring_epoch'] for e in events if e['ring'] == ring] == list(range(1,21)) for ring in ['0','1'])
            max_age = max(e['staleness'] for e in events)
            assert max_age > 0
        else:
            assert [e['epoch'] for e in events] == list(range(1,21))
        predictions = read(run/f'{method}_predictions.json')
        assert [p['fips'] for p in predictions] == config['quick_counties']
        y = np.array([p['truth'] for p in predictions])
        pred = np.array([p['prediction'] for p in predictions])
        yc, pc = y-y.mean(), pred-pred.mean()
        corr = float(yc @ pc / np.sqrt((yc @ yc)*(pc @ pc)))
        mse = float(np.mean((pred-y)**2))
        independent = {'rmse': float(np.sqrt(mse)), 'pearson_corr': corr,
                       'r2_author_corr_squared': corr*corr,
                       'r2_coefficient_of_determination': 1-float(np.sum((pred-y)**2)/(yc @ yc))}
        for key, actual in independent.items():
            assert np.isclose(actual, row['metrics'][key], rtol=1e-12, atol=1e-12), key
        baseline = rows[0]['metrics']['rmse']
        assert np.isclose(row['rmse_gap'], independent['rmse']-baseline, atol=1e-12)
        assert np.isclose(row['rmse_gap_percent'], 100*(independent['rmse']-baseline)/baseline, atol=1e-12)
        assign(model, weights)
        replay = []
        with torch.no_grad():
            for entry, saved in zip(test['entries'], predictions):
                with np.load(run/'prepared_test'/entry['path'], allow_pickle=False) as data:
                    x, short, long, target = [torch.from_numpy(data[k]).cuda() for k in ['x','short','long','target']]
                actual = float(model(x, ys=short, yl=long)[0,-1].double().exp())
                assert np.isclose(float(target[0,-1].double().exp()), saved['truth'], rtol=0, atol=1e-8)
                assert np.isclose(actual, saved['prediction'], rtol=1e-6, atol=1e-5)
                replay.append(abs(actual-saved['prediction']))
        report.append({'method': method, 'independent_metrics': independent,
                       'max_prediction_reload_error': max(replay), 'max_parameter_change': delta,
                       'training_events': len(events), 'max_staleness': max_age})
        print(json.dumps(report[-1]), flush=True)
    return {'status': 'passed', 'run': str(run.resolve()), 'verified_files': counts,
            'model_parameters': int(initial.numel()), 'methods': report,
            'training_only_reference_caution': 'This audit checks reproduction, not predictive adequacy or statistical equivalence.',
            'verified_at': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), 'Audit output must be new'
    assert not args.out.resolve().is_relative_to(args.run.resolve()), 'Do not alter the frozen run'
    report = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
