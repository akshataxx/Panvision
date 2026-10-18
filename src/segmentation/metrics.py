"""Per-case overlap metrics and the statistics used to compare candidate models."""

import numpy as np
import pandas as pd
from scipy import stats


def overlap(pred, ref):
    """Dice, IoU and voxel recall for binary volumes. Both empty counts as perfect; recall is NaN without a reference."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    fn = np.logical_and(~pred, ref).sum()
    if tp + fp + fn == 0:
        return {'dice': 1.0, 'iou': 1.0, 'recall': float('nan')}
    return {'dice': 2 * tp / (2 * tp + fp + fn), 'iou': tp / (tp + fp + fn),
            'recall': tp / (tp + fn) if tp + fn else float('nan')}


def bootstrap_ci(values, n=2000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values))).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(frame, columns, by=None):
    """Mean, SD, median and 95% bootstrap CI per metric (optionally per group)."""
    groups = frame.groupby(by) if by else [((), frame)]
    rows = []
    for key, group in groups:
        for column in columns:
            values = group[column].dropna()
            low, high = bootstrap_ci(values)
            row = dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,))) if by else {}
            row.update(metric=column, n=len(values), mean=values.mean(), sd=values.std(),
                       median=values.median(), ci_low=low, ci_high=high)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon(a, b, metric, key='study_id'):
    """Paired Wilcoxon signed-rank test of one metric between two per-case result tables."""
    merged = a[[key, metric]].merge(b[[key, metric]], on=key, suffixes=('_a', '_b')).dropna()
    if len(merged) < 2 or np.allclose(merged[f'{metric}_a'], merged[f'{metric}_b']):
        return {'n': len(merged), 'mean_diff': 0.0 if len(merged) else float('nan'), 'p_value': float('nan')}
    result = stats.wilcoxon(merged[f'{metric}_a'], merged[f'{metric}_b'])
    return {'n': len(merged), 'mean_diff': float((merged[f'{metric}_a'] - merged[f'{metric}_b']).mean()),
            'p_value': float(result.pvalue)}
