"""Recompute archived numerical evidence and plot without data/model downloads."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cropnet_leo.remote import sha
from research.joint_fusion.author_evaluation import evaluate
from research.joint_fusion.plot_mmst_compare import plot_curves, plot_results, committed_points, METHODS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_metrics(rows, expected):
    actual = evaluate([r['truth'] for r in rows], [r['prediction'] for r in rows])
    for key, value in actual.items():
        saved = expected[key]
        if value is None or saved is None:
            if value != saved: raise ValueError('Undefined metric mismatch: '+key)
        elif abs(value-saved) > 1e-9:
            raise ValueError('Recomputed metric mismatch: '+key)
    return actual


def verify(source):
    results = read(source/'evaluation/metrics.json')
    seen = set()
    prediction_count = 0
    for item in results:
        key = (item['method'], item['seed'])
        if key in seen: raise ValueError('Repeated evaluation arm')
        seen.add(key)
        path = source/'evaluation'/f'{key[0]}_seed{key[1]}_predictions.json'
        if sha(path) != item['prediction_sha256']: raise ValueError('Final prediction checksum mismatch')
        rows = read(path)
        if len(rows) != 254 or {r['year'] for r in rows} != {2022}:
            raise ValueError('Unexpected historical evaluation split')
        verify_metrics(rows, item['metrics'])
        prediction_count += len(rows)
    if seen != {(m,s) for m in METHODS for s in (10,20,30)}:
        raise ValueError('Incomplete 3-by-3 comparison')
    probe_count = 0
    for method,seed in sorted(seen):
        points = committed_points(source, method, seed)
        if not points or abs(points[-1]['accepted_epoch_equivalent']-120)>1e-9:
            raise ValueError('Missing final training probe')
        for point in points:
            verify_metrics(point['predictions'], point['metrics'])
            probe_count += 1
    return {'evaluation_predictions_verified':prediction_count,'training_probe_points_verified':probe_count,
            'scope':'archived predictions; no model inference; probe curves are training-only'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/figures')
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    source = ROOT/'examples/completed'
    report = verify(source)
    print(json.dumps(report,indent=2))
    if args.verify_only: return
    out = args.out.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Figure output must be separate from bundled evidence')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output already contains files; select a fresh --out directory')
    shutil.copytree(source, out, dirs_exist_ok=True)
    plot_results(out/'evaluation')
    curve = plot_curves(out)
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Final comparison:',out/'evaluation/centralized_vs_federated.png')
    print('All training rounds:',curve)


if __name__ == '__main__': main()
