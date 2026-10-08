"""Metrics aligned to the pinned public MMST-ViT evaluator.

Metric compatibility does not imply data/model/protocol compatibility with a
published CropNet result. Keep both R-squared definitions explicitly named.
"""
import numpy as np

UPSTREAM_REVISION = '615666c8d9fcd704acb662c13065703cbf2eab70'
UPSTREAM_PATH = 'util/metrics.py'


def evaluate(y_true, y_pred):
    y, prediction = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if y.ndim != 1 or y.shape != prediction.shape or len(y) < 2:
        raise ValueError('Expected matching 1-D arrays with at least two samples')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Metrics require finite observations and predictions')
    error = prediction - y
    sse = float(error @ error)
    centered = y - y.mean()
    sst = float(centered @ centered)
    correlation = (float(np.corrcoef(y, prediction)[0, 1])
                   if np.ptp(y) > 0 and np.ptp(prediction) > 0 else None)
    return {'n': len(y), 'rmse': float(np.sqrt(np.mean(error * error))),
            'r2_author_corr_squared': correlation * correlation if correlation is not None else None,
            'pearson_corr': correlation,
            'r2_coefficient_of_determination': 1. - sse / sst if sst > 0 else None}
