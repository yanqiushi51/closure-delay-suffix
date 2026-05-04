"""Target curve helpers for closure delay experiments."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable


def _as_list(values, name):
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc


def _sorted_pairs(fractions, risks):
    fractions_list = _as_list(fractions, "fractions")
    risks_list = _as_list(risks, "risks")
    if len(fractions_list) != len(risks_list):
        raise ValueError("fractions and risks must have the same length")
    if not fractions_list:
        raise ValueError("fractions and risks must not be empty")
    return sorted(zip(fractions_list, risks_list), key=lambda item: item[0])


def _is_query_iterable(query_r):
    if isinstance(query_r, (str, bytes)):
        return False
    return isinstance(query_r, Iterable)


def fit_isotonic_curve(fractions, risks):
    """Fit a non-decreasing isotonic curve with dependency-free PAVA.

    Inputs may be any iterable. Points are sorted by fraction before fitting.
    The return value is a dict with sorted ``fractions`` and fitted ``means``.
    """

    pairs = _sorted_pairs(fractions, risks)
    blocks = []

    for fraction, risk in pairs:
        block = {
            "start": len(blocks),
            "end": len(blocks),
            "weight": 1.0,
            "sum": float(risk),
            "mean": float(risk),
            "fractions": [fraction],
        }
        blocks.append(block)

        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            total = left["sum"] + right["sum"]
            blocks.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "weight": weight,
                    "sum": total,
                    "mean": total / weight,
                    "fractions": left["fractions"] + right["fractions"],
                }
            )

    fitted_fractions = []
    fitted_means = []
    for block in blocks:
        fitted_fractions.extend(block["fractions"])
        fitted_means.extend([block["mean"]] * len(block["fractions"]))

    return {"fractions": fitted_fractions, "means": fitted_means}


def _interpolate_one(sorted_fractions, sorted_risks, query):
    if query <= sorted_fractions[0]:
        return sorted_risks[0]
    if query >= sorted_fractions[-1]:
        return sorted_risks[-1]

    index = bisect_left(sorted_fractions, query)
    left_fraction = sorted_fractions[index - 1]
    right_fraction = sorted_fractions[index]
    left_risk = sorted_risks[index - 1]
    right_risk = sorted_risks[index]

    if right_fraction == left_fraction:
        return right_risk

    weight = (query - left_fraction) / (right_fraction - left_fraction)
    return left_risk + weight * (right_risk - left_risk)


def interpolate_curve(fractions, risks, query_r):
    """Linearly interpolate risks at query fraction(s), clamping to boundaries."""

    pairs = _sorted_pairs(fractions, risks)
    sorted_fractions = [fraction for fraction, _risk in pairs]
    sorted_risks = [float(risk) for _fraction, risk in pairs]

    if _is_query_iterable(query_r):
        return [
            _interpolate_one(sorted_fractions, sorted_risks, query)
            for query in query_r
        ]
    return _interpolate_one(sorted_fractions, sorted_risks, query_r)


def _clean_curve_parts(fractions, clean_curve):
    if isinstance(clean_curve, dict):
        clean_fractions = clean_curve.get("fractions")
        clean_risks = clean_curve.get("means")
        if clean_risks is None:
            clean_risks = clean_curve.get("risks")
        if clean_fractions is None or clean_risks is None:
            raise ValueError("clean_curve dict must contain fractions and means")
        return clean_fractions, clean_risks
    return fractions, clean_curve


def target_curve(fractions, clean_curve, tau):
    """Compute h_tau(r) = h_clean(r / tau) at each requested fraction."""

    if tau == 0:
        raise ValueError("tau must be non-zero")

    raw_fractions = _as_list(fractions, "fractions")
    clean_fractions, clean_risks = _clean_curve_parts(raw_fractions, clean_curve)
    query_fractions = sorted(raw_fractions)
    scaled_queries = [fraction / tau for fraction in query_fractions]
    target_means = interpolate_curve(clean_fractions, clean_risks, scaled_queries)
    return {"fractions": query_fractions, "means": target_means}


def curve_tracking_error(observed_curve, target):
    """Compare observed and target closure curves on shared target fractions."""

    observed_fractions, observed_risks = _clean_curve_parts([], observed_curve)
    target_fractions, target_risks = _clean_curve_parts([], target)
    if not observed_fractions or not target_fractions:
        return {"mse": None, "mae": None, "count": 0}

    observed_at_target = interpolate_curve(observed_fractions, observed_risks, target_fractions)
    errors = [
        float(observed - expected)
        for observed, expected in zip(observed_at_target, target_risks)
    ]
    if not errors:
        return {"mse": None, "mae": None, "count": 0}
    squared = [error * error for error in errors]
    absolute = [abs(error) for error in errors]
    return {
        "mse": sum(squared) / len(squared),
        "mae": sum(absolute) / len(absolute),
        "count": len(errors),
        "fractions": list(target_fractions),
        "errors": errors,
    }
