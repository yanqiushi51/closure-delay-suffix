from __future__ import annotations

from typing import Iterable

import numpy as np


def fit_length_calibrator(curve_shift_values: Iterable[float], length_ratio_values: Iterable[float]) -> dict:
    xs = np.asarray(list(curve_shift_values), dtype=float)
    ys = np.asarray(list(length_ratio_values), dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) < 2 or np.allclose(xs.std(), 0.0):
        return {
            "type": "linear",
            "available": False,
            "a": None,
            "b": None,
            "r2": None,
            "n": int(len(xs)),
        }
    a, b = np.polyfit(xs, ys, deg=1)
    preds = a * xs + b
    ss_res = float(np.sum((ys - preds) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "type": "linear",
        "available": True,
        "a": float(a),
        "b": float(b),
        "r2": None if r2 is None else float(r2),
        "n": int(len(xs)),
    }


def predict_length_ratio(curve_shift, calibrator: dict):
    if not calibrator or not calibrator.get("available"):
        return None
    try:
        value = float(curve_shift)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return float(calibrator["a"] * value + calibrator["b"])
