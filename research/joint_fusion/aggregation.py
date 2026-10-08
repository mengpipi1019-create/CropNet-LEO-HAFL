"""Explicit stale snapshot mixing and damped stale delta aggregation."""
import numpy as np
import torch


def fuse(current, updates, bases, masses, ages, rule, alpha, power=1.):
    n = len(updates)
    if n == 0 or any(len(x) != n for x in (bases, masses, ages)):
        raise ValueError('Unaligned aggregation inputs')
    if not 0 < alpha <= 1 or not np.isfinite(power) or power < 0:
        raise ValueError('Invalid aggregation coefficient')
    if any(not np.isfinite(m) or m <= 0 for m in masses):
        raise ValueError('Nonpositive mass')
    if any(not np.isfinite(a) or a < 0 or int(a) != a for a in ages):
        raise ValueError('Invalid model age')
    vectors = [current, *updates, *bases]
    if any(v.shape != current.shape or not torch.isfinite(v).all() for v in vectors):
        raise ValueError('Invalid model vector')
    mass = current.new_tensor(masses)
    attenuation = (1. + current.new_tensor(ages)).pow(-power)
    if rule == 'snapshot':
        weights = mass * attenuation
        average = (torch.stack(updates) * weights[:, None]).sum(0) / weights.sum()
        result = current.lerp(average, alpha)
    elif rule == 'delta':
        deltas = torch.stack([u - b for u, b in zip(updates, bases)])
        # Divide by raw mass: a lone stale update must still be attenuated.
        result = current + alpha * (deltas * (mass * attenuation)[:, None]).sum(0) / mass.sum()
    else:
        raise ValueError('Unknown aggregation rule')
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite fused model')
    return result
