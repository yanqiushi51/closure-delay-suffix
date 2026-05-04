from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from scipy import stats as scipy_stats


def safe_pearson_correlation_with_pvalue(xs: Iterable[float], ys: Iterable[float]):
    xs = np.asarray(list(xs), dtype=float)
    ys = np.asarray(list(ys), dtype=float)
    if len(xs) < 3 or np.allclose(xs.std(), 0.0) or np.allclose(ys.std(), 0.0):
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = scipy_stats.pearsonr(xs, ys)
    return float(r), float(p)


def safe_spearman_correlation(xs: Iterable[float], ys: Iterable[float]):
    xs = np.asarray(list(xs), dtype=float)
    ys = np.asarray(list(ys), dtype=float)
    if len(xs) < 3 or np.allclose(xs.std(), 0.0) or np.allclose(ys.std(), 0.0):
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = scipy_stats.spearmanr(xs, ys)
    return float(rho), float(p)
