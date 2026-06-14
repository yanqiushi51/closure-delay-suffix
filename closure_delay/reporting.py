from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if np.isfinite(number) else None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def numeric_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = parse_number(value)
        if number is not None:
            out.append(number)
    return out


def mean_value(values: Sequence[Any]) -> float | None:
    clean = numeric_values(values)
    return float(np.mean(clean)) if clean else None


def bootstrap_mean_ci(
    values: Sequence[Any],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> tuple[float | None, float | None, float | None]:
    clean = np.asarray(numeric_values(values), dtype=float)
    if clean.size == 0:
        return None, None, None
    mean = float(np.mean(clean))
    if clean.size == 1 or int(n_bootstrap) <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, clean.size, size=(int(n_bootstrap), clean.size))
    samples = np.mean(clean[indices], axis=1)
    alpha = 1.0 - float(confidence)
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return mean, lo, hi


def format_mean_ci(
    values: Sequence[Any],
    *,
    digits: int = 2,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> str:
    mean, lo, hi = bootstrap_mean_ci(
        values,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if mean is None or lo is None or hi is None:
        return ""
    half_width = max(mean - lo, hi - mean, 0.0)
    return f"{mean:.{digits}f} +/- {half_width:.{digits}f}"


def summarize_field(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    digits: int = 2,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    values = [row.get(field) for row in rows]
    mean, lo, hi = bootstrap_mean_ci(
        values,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        f"{field}_mean": mean,
        f"{field}_ci_low": lo,
        f"{field}_ci_high": hi,
        f"{field}_mean_ci": format_mean_ci(
            values,
            digits=digits,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }


def paired_metric_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_a: str,
    condition_b: str,
    metric: str,
    id_key: str = "id",
    condition_key: str = "condition",
) -> list[tuple[float, float]]:
    by_key = {
        (str(row.get(id_key)), str(row.get(condition_key))): row
        for row in rows
    }
    pairs: list[tuple[float, float]] = []
    ids = sorted({str(row.get(id_key)) for row in rows})
    for item_id in ids:
        row_a = by_key.get((item_id, condition_a))
        row_b = by_key.get((item_id, condition_b))
        if row_a is None or row_b is None:
            continue
        value_a = parse_number(row_a.get(metric))
        value_b = parse_number(row_b.get(metric))
        if value_a is None or value_b is None:
            continue
        pairs.append((value_a, value_b))
    return pairs


def paired_permutation_pvalue(
    pairs: Sequence[tuple[float, float]],
    *,
    n_permutations: int = 10000,
    seed: int = 12345,
) -> float | None:
    diffs = np.asarray([b - a for a, b in pairs], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    observed = abs(float(np.mean(diffs)))
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(int(seed))
    count = 0
    total = int(n_permutations)
    for _ in range(total):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=diffs.size)
        statistic = abs(float(np.mean(diffs * signs)))
        if statistic >= observed - 1e-12:
            count += 1
    return float((count + 1) / (total + 1))


def paired_binary_pvalue(pairs: Sequence[tuple[float, float]]) -> float | None:
    discordant_ab = 0
    discordant_ba = 0
    for value_a, value_b in pairs:
        a = bool(round(float(value_a)))
        b = bool(round(float(value_b)))
        if a and not b:
            discordant_ab += 1
        elif b and not a:
            discordant_ba += 1
    n = discordant_ab + discordant_ba
    if n == 0:
        return 1.0 if pairs else None
    if n > 60:
        statistic = max(abs(discordant_ab - discordant_ba) - 1.0, 0.0) / math.sqrt(float(n))
        return float(math.erfc(statistic / math.sqrt(2.0)))
    smaller = min(discordant_ab, discordant_ba)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def paired_test_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str]],
    metrics: Sequence[str],
    binary_metrics: set[str] | None = None,
    id_key: str = "id",
    condition_key: str = "condition",
    n_permutations: int = 10000,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    binary_metrics = binary_metrics or set()
    out: list[dict[str, Any]] = []
    for condition_a, condition_b in comparisons:
        for metric in metrics:
            pairs = paired_metric_values(
                rows,
                condition_a=condition_a,
                condition_b=condition_b,
                metric=metric,
                id_key=id_key,
                condition_key=condition_key,
            )
            diffs = [value_b - value_a for value_a, value_b in pairs]
            pvalue = (
                paired_binary_pvalue(pairs)
                if metric in binary_metrics
                else paired_permutation_pvalue(pairs, n_permutations=n_permutations, seed=seed)
            )
            out.append(
                {
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "metric": metric,
                    "n_pairs": len(pairs),
                    "mean_a": mean_value([value_a for value_a, _ in pairs]),
                    "mean_b": mean_value([value_b for _, value_b in pairs]),
                    "mean_delta_b_minus_a": mean_value(diffs),
                    "test": "paired_binary_exact" if metric in binary_metrics else "paired_permutation_two_sided",
                    "p_value": pvalue,
                }
            )
    return out
