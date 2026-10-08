"""Equal example budgets for central, FedAvg and hierarchical stale deltas.

This is a neural-network extension of aggregation.fuse, not the historical
ridge Hessian/gradient-memory solver and not an orbital resource experiment.
"""
import heapq
import math
import random

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .aggregation import fuse


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vector(model):
    return parameters_to_vector(model.parameters()).detach().cpu().clone()


def assign(model, v):
    device = next(model.parameters()).device
    # Clone prevents a parameter update from modifying the retained stale base.
    vector_to_parameters(v.to(device).clone(), model.parameters())


def learning_rate(config, epoch):
    warmup, total = config['warmup_epochs'], config['epochs']
    if warmup and epoch < warmup:
        return config['learning_rate'] * (epoch + 1) / warmup
    position = (epoch - warmup) / max(1, total - warmup)
    return config['learning_rate'] * .5 * (1. + math.cos(math.pi * position))


def optimizer(model, config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 or name.endswith('.bias') else decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': config['weight_decay']},
                              {'params': no_decay, 'weight_decay': 0.}],
                             lr=config['learning_rate'], betas=(.9, .95))


def train_samples(model, get_sample, ids, opt, lr, rng):
    model.train()
    for group in opt.param_groups:
        group['lr'] = lr
    device = next(model.parameters()).device
    order = rng.permutation(ids)
    total_loss = 0.
    for i in order:
        x, short, long, target = [a.to(device) for a in get_sample(int(i))]
        opt.zero_grad(set_to_none=True)
        prediction = model(x, ys=short, yl=long)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite log-target loss')
        loss.backward()
        # Same safeguard for all three methods; no gradient/label upload.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm')
        opt.step()
        total_loss += loss.item()
    return total_loss, len(order)


def train_method(model, initial, get_sample, clients, config, seed, method, emit=lambda row: None):
    if method not in ('centralized', 'fedavg', 'hierarchical_async'):
        raise ValueError('Unknown training method')
    if config['local_epochs'] != 1:
        raise ValueError('This frozen equal-budget protocol requires local_epochs=1')
    if list(model.buffers()):
        raise ValueError('This runner requires a buffer-free model; do not silently omit federated buffers')
    all_ids = [i for c in clients for i in c['indices']]
    if len(set(all_ids)) != len(all_ids) or sorted(all_ids) != list(range(len(all_ids))):
        raise ValueError('Client partition must cover each training sample exactly once')
    if config['epochs'] < 1 or any(not c['indices'] for c in clients):
        raise ValueError('Empty training budget or client')
    seed_all(seed)
    assign(model, initial)
    rng = np.random.default_rng(seed)
    processed = 0
    n = len(all_ids)
    epochs = config['epochs']
    if method == 'centralized':
        opt = optimizer(model, config)
        for epoch in range(epochs):
            loss, count = train_samples(model, get_sample, all_ids, opt, learning_rate(config, epoch), rng)
            processed += count
            emit({'method': method, 'epoch': epoch+1, 'processed_examples': processed, 'log_target_mse': loss/count})
        return vector(model), processed

    def local_average(base, members, epoch):
        mass = sum(len(c['indices']) for c in members)
        average = torch.zeros_like(base)
        loss_sum, count_sum = 0., 0
        for c in members:
            assign(model, base)
            opt = optimizer(model, config)  # local Adam state resets at dispatch
            loss, count = train_samples(model, get_sample, c['indices'], opt, learning_rate(config, epoch), rng)
            average.add_(vector(model), alpha=len(c['indices'])/mass)
            loss_sum += loss
            count_sum += count
        return average, count_sum, loss_sum / count_sum

    current = initial.clone()
    if method == 'fedavg':
        for epoch in range(epochs):
            current, count, loss = local_average(current, clients, epoch)
            processed += count
            emit({'method': method, 'epoch': epoch+1, 'processed_examples': processed, 'log_target_mse': loss})
    else:
        groups = {r: [c for c in clients if c['ring'] == r] for r in sorted({c['ring'] for c in clients})}
        delays = config['logical_ring_delays']
        if len(groups) != len(delays) or any(not math.isfinite(x) or x <= 0 for x in delays):
            raise ValueError('One positive logical service delay is required per ring')
        jobs, version, clock = [], 0, 0.
        done = {r: 0 for r in groups}
        def dispatch(ring, rank):
            nonlocal processed
            base = current.clone()
            updated, count, loss = local_average(base, groups[ring], done[ring])
            processed += count
            heapq.heappush(jobs, (clock + delays[rank], rank, ring, version, base, updated, count, loss))
        for rank, ring in enumerate(groups):
            dispatch(ring, rank)
        while jobs:
            clock, rank, ring, base_version, base, updated, count, loss = heapq.heappop(jobs)
            age = version - base_version
            mass = sum(len(c['indices']) for c in groups[ring])
            current = fuse(current, [updated], [base], [mass], [age], 'delta',
                           config['server_learning_rate'] * mass / n, config['staleness_power'])
            version += 1
            done[ring] += 1
            emit({'method': method, 'version': version, 'ring': ring, 'ring_epoch': done[ring],
                  'base_version': base_version, 'staleness': age, 'processed_examples': processed,
                  'accepted_examples': count, 'logical_completion_time': clock, 'log_target_mse': loss})
            if done[ring] < epochs:
                dispatch(ring, rank)
    if processed != n * epochs:
        raise AssertionError('Methods must consume identical total training-example budgets')
    assign(model, current)
    return current, processed
