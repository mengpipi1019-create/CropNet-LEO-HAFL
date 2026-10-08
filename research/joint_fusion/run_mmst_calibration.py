"""Staged, resumable 2021-only development calibration of the frozen full engine."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from cropnet_leo.remote import sha
from . import run_mmst_full_repaired as full
from .author_evaluation import evaluate
from .download_mmst_data import download_lock
from .mmst_bridge import ROOT, make_model
from .mmst_data import client_partition
from .mmst_full_data import CountyData, fit_statistics, read
from .mmst_full_engine import train_arm
from .mmst_io import write_json
from .mmst_optimizer_store import OptimizerStore
from .mmst_training import assign, seed_all, vector
from .run_mmst_local import BASE_RUN, LocalPause, PauseControl, display_status, keep_awake
from .run_mmst_paired import METHODS, digest

DEFAULT_CONFIG = ROOT / 'research/joint_fusion/configs/mmst_calibration_v1.json'
DEFAULT_OUT = ROOT / 'results/joint_fusion/mmst_calibration_20260929_01'


class StageComplete(BaseException):
    pass


def split_counties(rows, seed, fraction):
    """Use state and county identity only, never outcomes, to select holdout counties."""
    if not 0 < fraction < .5 or {r['year'] for r in rows} != {2021}:
        raise ValueError('2021-only development split required')
    if len({r['fips'] for r in rows}) != len(rows):
        raise ValueError('Repeated county')
    train, validation = [], []
    rng = np.random.default_rng(seed)
    for state in sorted({r['state'] for r in rows}):
        ids = sorted([i for i, r in enumerate(rows) if r['state'] == state], key=lambda i: rows[i]['fips'])
        count = max(1, round(len(ids) * fraction))
        chosen = set(int(i) for i in rng.permutation(ids)[:count])
        validation.extend(chosen)
        train.extend(i for i in ids if i not in chosen)
    return {'train': sorted(train), 'validation': sorted(validation)}


class View:
    def __init__(self, base, indices):
        self.base, self.indices, self.root = base, list(indices), base.root
        self.manifest = {**base.manifest, 'records': [base.manifest['records'][i] for i in indices]}

    def __len__(self):
        return len(self.indices)

    def __call__(self, i):
        return self.base(self.indices[i])


def truth(data, stats):
    values = []
    # Read only targets, avoiding decompression of county imagery.
    for entry in data.manifest['records']:
        with np.load(data.root / entry['path'], allow_pickle=False) as a:
            values.append(float(np.exp(np.float64(a['target'][0, 1]))))
    return values


def score(y, p):
    result = evaluate(y, p)
    result['prediction_std_ratio'] = float(np.std(p) / np.std(y)) if np.std(y) else None
    result['prediction_range'] = [float(min(p)), float(max(p))]
    return result


def measurement(model, weights, train, validation, stats):
    # Validation must not change dropout RNG, the active model, or optimizer parameters.
    previous, mode = vector(model), model.training
    devices = [next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            assign(model, weights)
            result = {}
            for label, data in [('train', train), ('validation', validation)]:
                y, p = full.predictions(model, data, stats)
                result[label] = {'metrics': score(y, p), 'predictions': [
                    {'fips': entry['record']['fips'], 'year': 2021, 'truth': a, 'prediction': b}
                    for entry, a, b in zip(data.manifest['records'], y, p)]}
            return result
    finally:
        assign(model, previous)
        model.train(mode)


def advance(model, initial, train, validation, stats, clients, cfg, method, arm,
            stage, identity, control, update):
    """Stop after a durable engine checkpoint; retain all pending asynchronous jobs."""
    arm.mkdir(parents=True, exist_ok=True)
    result_path = arm / f'validation_{stage:03d}.json'
    if result_path.exists():
        result = read(result_path)
        if result['identity'] != identity or sha(arm / result['prediction_file']) != result['prediction_sha256']:
            raise ValueError('Saved validation identity changed')
        return result
    resume = None
    pointer = arm / 'recovery.json'
    if pointer.exists():
        info = read(pointer)
        if sha(arm / info['path']) != info['sha256']:
            raise ValueError('Recovery checksum mismatch')
        resume = torch.load(arm / info['path'], map_location='cpu', weights_only=False)
        if resume['identity'] != identity:
            raise ValueError('Recovery identity mismatch')
    log = arm / 'training.jsonl'
    keep = resume['event_count'] if resume else 0
    if log.exists():
        lines = log.read_text(encoding='utf-8').splitlines(keepends=True)
        if len(lines) < keep:
            raise ValueError('Event log behind recovery')
        if len(lines) > keep:
            shutil.copy2(log, arm / f'events_before_recovery_{time.time_ns()}.jsonl')
            log.write_text(''.join(lines[:keep]), encoding='utf-8')
    store = OptimizerStore(arm / 'optimizer_state')
    target = stage * len(train)

    def assess(state):
        update('validate', method=method, target_epoch=stage, accepted=state['accepted'])
        measured = measurement(model, state['current'], train, validation, stats)
        pred_path = arm / f'predictions_{stage:03d}.json'
        write_json(pred_path, {key: value['predictions'] for key, value in measured.items()})
        result = {'identity': identity, 'method': method, 'stage_epoch': stage,
                  'accepted_examples': state['accepted'], 'processed_examples': state['processed'],
                  'accepted_epoch_equivalent': state['accepted'] / len(train),
                  'ring_epochs': state['done'], 'pending_jobs': len(state['jobs']),
                  'checkpoint_sha256': read(pointer)['sha256'], 'prediction_file': pred_path.name,
                  'prediction_sha256': sha(pred_path), 'updated_at': full.now(),
                  **{key: value['metrics'] for key, value in measured.items()}}
        write_json(result_path, result)
        print('VALIDATION ' + json.dumps(result, ensure_ascii=False), flush=True)
        return result

    # If the process stopped during validation, reuse its already-saved weights first.
    if resume is not None and resume['accepted'] >= target:
        result = assess(resume)
        control.check()
        return result
    with log.open('a', encoding='utf-8') as handle:
        def emit(row):
            row = {**row, 'updated_at': full.now()}
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            update('train', method=method, target_epoch=stage,
                   accepted=row['accepted_examples_total'], processed=row['processed_examples'],
                   event_count=row['event_count'], local_loss=row['local_loss'], lr=row['lr'])

        def save(state):
            old = read(pointer) if pointer.exists() else {'slot': 1}
            slot = 1 - old['slot']
            path = arm / f'recovery_{slot}.pt'
            temporary = path.with_suffix('.partial')
            opts = state['opts'].snapshot(slot)
            torch.save({**state, 'opts': opts, 'identity': identity}, temporary)
            temporary.replace(path)
            write_json(pointer, {'slot': slot, 'path': path.name, 'sha256': sha(path),
                                 'event_count': state['event_count'], 'updated_at': full.now()})
            control.check()
            if state['accepted'] >= target:
                assess(state)
                raise StageComplete()
        try:
            train_arm(model, initial, train, clients, cfg, cfg['seeds'][0], method,
                      emit, save, resume, store)
        except StageComplete:
            pass
    if not result_path.exists():
        raise RuntimeError('Training ended before requested calibration checkpoint')
    return read(result_path)


def decide(histories, baseline, rules):
    """Predeclared diagnostic gates; never force federated/centralized parity."""
    stage = min(history[-1]['stage_epoch'] for history in histories.values())
    details = {}
    for method, history in histories.items():
        latest = history[-1]
        m = latest['validation']
        signal = (latest['train']['rmse'] <= baseline['train']['rmse'] * rules['train_mean_ratio']
                  and m['rmse'] <= baseline['validation']['rmse'] * rules['validation_mean_ratio']
                  and (m['pearson_corr'] or 0.) > 0
                  and (m['prediction_std_ratio'] or 0.) >= rules['minimum_std_ratio'])
        recent = [r['validation']['rmse'] for r in history[-rules['plateau_points']:]]
        plateau = (len(recent) == rules['plateau_points']
                   and (max(recent) - min(recent)) / min(recent) <= rules['plateau_relative_range'])
        earlier = [r['validation']['rmse'] for r in history[:-2]]
        overfit_risk = (bool(earlier) and all(r['validation']['rmse'] > min(earlier) * rules['deterioration_ratio']
                                            for r in history[-2:]))
        details[method] = {'signal_pass': signal, 'recent_plateau': plateau,
                           'validation_deterioration': overfit_risk,
                           'validation_rmse': m['rmse'], 'validation_corr': m['pearson_corr'],
                           'prediction_std_ratio': m['prediction_std_ratio']}
    decision = 'continue_calibration'
    if stage >= rules['screen_epoch'] and any(not r['signal_pass'] for r in details.values()):
        decision = 'needs_revision_no_validation_signal'
    elif stage >= rules['screen_epoch'] and any(r['validation_deterioration'] for r in details.values()):
        decision = 'needs_revision_validation_deterioration'
    elif stage >= rules['earliest_plateau_epoch'] and all(r['recent_plateau'] for r in details.values()):
        decision = 'ready_for_exploratory_comparison'
    elif stage >= rules['maximum_epoch']:
        decision = 'budget_not_settled_at_120'
    return {'decision': decision, 'stage_epoch': stage, 'methods': details,
            'test_2022_used': False, 'formal_training_launched': False,
            'scope': 'Single-seed 2021 development validation; no temporal-generalization or parity guarantee'}


def report(out, histories, baseline, decision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for method, history in histories.items():
        x = [r['accepted_epoch_equivalent'] for r in history]
        for ax, key in zip(axes, ['rmse', 'pearson_corr', 'prediction_std_ratio']):
            ax.plot(x, [r['validation'][key] for r in history], marker='o', label=method)
    axes[0].axhline(baseline['validation']['rmse'], color='gray', linestyle='--', label='Train-mean predictor (diagnostic)')
    for ax, title in zip(axes, ['Validation RMSE (BU/ACRE)', 'Validation signed correlation', 'Prediction / truth standard deviation']):
        ax.set_title(title); ax.set_xlabel('Accepted training-example budget / training counties'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('2021 county holdout calibration — development only')
    fig.tight_layout()
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out / ('validation_curves.' + extension), dpi=180)
    plt.close(fig)
    lines = ['# 2021 内部校准证据', '', '状态：' + decision['decision'], '',
             '这些是2021内部开发验证结果，不是2022测试成绩；不证明跨年泛化或联邦精度等价。', '',
             '| 方法 | 已接纳等效轮次 | 验证RMSE | Corr | 预测/真值标准差比 |', '|---|---:|---:|---:|---:|']
    for method, history in histories.items():
        row = history[-1]; m = row['validation']
        def fmt(v): return 'undefined' if v is None else f'{v:.4f}'
        lines.append(f"| {method} | {row['accepted_epoch_equivalent']:.2f} | {fmt(m['rmse'])} | {fmt(m['pearson_corr'])} | {fmt(m['prediction_std_ratio'])} |")
    lines.extend(['', f"只预测训练集平均单产的验证RMSE为 {baseline['validation']['rmse']:.4f}，仅用于检查模型是否学到信号，不作为论文算法。",
                  '', '异步检查点保留在途任务，阶段横轴使用实际接纳样本量；最终120轮才严格排空。',
                  '同样本预算不等于同优化步数。阈值只是预登记的工程筛查条件，不是统计显著性或已证明收敛。',
                  '若未通过，不自动开启原正式三算法实验，也不据此换种子或读取2022分数。', '', '![验证曲线](validation_curves.png)'])
    (out / 'evidence.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'decision.json', decision)
    write_json(out / 'report_sha256.json', {name: sha(out / name) for name in
               ['evidence.md', 'decision.json', 'validation_curves.png', 'validation_curves.svg', 'validation_curves.pdf']})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--action', choices=['check', 'run', 'status', 'pause'], default='run')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(); out = args.out.resolve()
    if args.action == 'status':
        print(json.dumps(display_status(out), ensure_ascii=False, indent=2)); return
    if args.action == 'pause':
        if out.exists(): PauseControl(out).request()
        else: print('Calibration has not started.')
        return
    config = read(args.config); cache_config = read(BASE_RUN / 'config.json')
    cfg = read(ROOT / 'research/joint_fusion/configs/mmst_local_first_v1.json')
    full.check_sources(BASE_RUN)
    if digest(cfg) != config['training_config_sha256']:
        raise ValueError('Proposed training configuration changed')
    base = CountyData(BASE_RUN / 'prepared_train', cache_config, False)
    rows = [e['record'] for e in base.manifest['records']]
    split = split_counties(rows, config['split_seed'], config['validation_fraction'])
    train, validation = View(base, split['train']), View(base, split['validation'])
    stats = fit_statistics(train)
    clients = client_partition([rows[i] for i in split['train']], cfg['clients_per_state'])
    binding = {'manifest_sha256': sha(BASE_RUN / 'prepared_train/manifest.json'),
               'train_indices': split['train'], 'validation_indices': split['validation'],
               'train_fips': [rows[i]['fips'] for i in split['train']],
               'validation_fips': [rows[i]['fips'] for i in split['validation']],
               'normalization_sha256': digest(stats), 'evaluation_years': [2021]}
    train_truth, val_truth = truth(train, stats), truth(validation, stats)
    mean = float(np.mean(train_truth))
    baseline = {'training_mean_yield': mean,
                'train': evaluate(train_truth, [mean] * len(train_truth)),
                'validation': evaluate(val_truth, [mean] * len(val_truth))}
    base.stats = stats
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    print(f'CHECK PASSED: {len(train)} train / {len(validation)} validation, 2021 only, {len(clients)} clients. No 2022 data.', flush=True)
    if args.action == 'check': return
    with download_lock(ROOT / 'results/joint_fusion/mmst_local_gpu_lock'), keep_awake():
        if out.exists():
            if read(out / 'config.json') != config or read(out / 'binding.json') != binding:
                raise ValueError('Calibration identity changed')
            full.check_sources(out)
            if read(out / 'normalization.json') != stats: raise ValueError('Normalization changed')
        else:
            out.mkdir(parents=True)
            full.snapshot(out, config)
            hashes = read(out / 'source_sha256.json')
            for relative in ['research/joint_fusion/run_mmst_calibration.py', 'research/joint_fusion/run_mmst_local.py',
                             'research/joint_fusion/MMST_CALIBRATION_PROTOCOL.md',
                             'research/joint_fusion/configs/mmst_local_first_v1.json',
                             'research/joint_fusion/configs/mmst_calibration_v1.json',
                             'run_mmst_calibration.ps1', 'start_mmst_calibration.cmd', 'pause_mmst_calibration.cmd']:
                destination = out / 'source' / relative; destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination); hashes[relative] = sha(destination)
            write_json(out / 'source_sha256.json', hashes)
            write_json(out / 'binding.json', binding); write_json(out / 'normalization.json', stats)
            write_json(out / 'clients.json', clients); write_json(out / 'baseline.json', baseline)
            write_json(out / 'training_config.json', cfg)
        if (out / 'decision.json').exists() and read(out / 'decision.json')['decision'] != 'continue_calibration':
            print('Calibration decision already saved: ' + str(out / 'evidence.md')); return
        if (out / 'PAUSE').exists(): (out / 'PAUSE').replace(out / f'PAUSE_acknowledged_{time.time_ns()}')
        control = PauseControl(out); state = {}
        def update(phase, **extra):
            state.clear(); state.update(stage=phase, **extra)
            # During both training and validation, Ctrl+C requests a safe boundary.
            control.phase = 'train'
            value = {'status': 'running', **state, 'pid': os.getpid(), 'updated_at': full.now(), 'test_2022_used': False}
            write_json(out / 'status.json', value)
            print(json.dumps(value), flush=True)
        old_handler = signal.signal(signal.SIGINT, control.signal)
        try:
            torch.set_num_threads(4); update('initialization')
            seed_all(cfg['seeds'][0]); model = make_model(cfg).cuda(); initial = vector(model)
            initial_path = out / 'initial.pt'
            if initial_path.exists():
                if not torch.equal(torch.load(initial_path, weights_only=True), initial): raise ValueError('Initialization changed')
            else: torch.save(initial, initial_path)
            identity = {'config_sha256': digest(config), 'binding_sha256': digest(binding),
                        'initial_sha256': sha(initial_path), 'source_sha256': sha(out / 'source_sha256.json')}
            if not (out / 'initial_predictions.json').exists():
                update('initial_validation')
                write_json(out / 'initial_predictions.json', measurement(model, initial, train, validation, stats))
            histories = {method: [] for method in METHODS}
            for stage in config['stages']:
                for method in METHODS:
                    control.check()
                    row = advance(model, initial, train, validation, stats, clients, cfg, method,
                                  out / method, stage, {**identity, 'method': method}, control, update)
                    histories[method].append(row)
                full.check_sources(out)
                decision = decide(histories, baseline, config['rules'])
                report(out, histories, baseline, decision)
                if decision['decision'] != 'continue_calibration':
                    write_json(out / 'status.json', {'status': 'completed', **decision, 'updated_at': full.now()})
                    print('CALIBRATION COMPLETE: ' + json.dumps(decision), flush=True)
                    return
        except LocalPause:
            write_json(out / 'status.json', {'status': 'paused', **state, 'updated_at': full.now()})
            print('PAUSED. Use the same start command to resume.', flush=True)
        except BaseException as exc:
            write_json(out / 'status.json', {'status': 'failed', **state, 'error': repr(exc), 'updated_at': full.now()})
            raise
        finally:
            signal.signal(signal.SIGINT, old_handler)


if __name__ == '__main__':
    main()
