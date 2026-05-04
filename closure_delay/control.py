from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _as_float(value: Any):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _condition_name(condition: Any) -> str:
    return str(getattr(condition, "name", condition))


def control_error(length_ratio, target_tau):
    length_ratio = _as_float(length_ratio)
    target_tau = _as_float(target_tau)
    if length_ratio is None or target_tau is None:
        return None
    return abs(length_ratio - target_tau)


def hit_rate(length_ratios: Iterable[float], target_tau, epsilon=0.3):
    target_tau = _as_float(target_tau)
    epsilon = _as_float(epsilon)
    if target_tau is None or epsilon is None:
        return None

    total = 0
    hits = 0
    for length_ratio in length_ratios:
        error = control_error(length_ratio, target_tau)
        if error is None:
            continue
        total += 1
        if error <= epsilon:
            hits += 1
    if total == 0:
        return None
    return hits / total


def monotonicity(example_rows, ordered_conditions):
    condition_order = [_condition_name(condition) for condition in ordered_conditions]
    condition_index = {condition: idx for idx, condition in enumerate(condition_order)}
    by_id = defaultdict(dict)

    for row in example_rows:
        condition = _condition_name(row.get("condition"))
        if condition not in condition_index:
            continue
        length_ratio = _as_float(row.get("length_ratio"))
        if length_ratio is None:
            continue
        by_id[str(row.get("id"))][condition] = length_ratio

    comparable_pairs = 0
    monotone_pairs = 0
    pair_details = []
    for example_id in sorted(by_id):
        ratios = by_id[example_id]
        for left, right in zip(condition_order, condition_order[1:]):
            if left not in ratios or right not in ratios:
                continue
            comparable_pairs += 1
            is_monotone = ratios[left] <= ratios[right]
            if is_monotone:
                monotone_pairs += 1
            pair_details.append(
                {
                    "id": example_id,
                    "left_condition": left,
                    "right_condition": right,
                    "left_length_ratio": ratios[left],
                    "right_length_ratio": ratios[right],
                    "monotone": is_monotone,
                }
            )

    mono_rate = None
    if comparable_pairs:
        mono_rate = monotone_pairs / comparable_pairs
    return {
        "comparable_pairs": comparable_pairs,
        "monotone_pairs": monotone_pairs,
        "mono_rate": mono_rate,
        "pairs": pair_details,
    }


def monotonicity_by_family(example_rows, conditions):
    grouped = defaultdict(list)
    for condition in conditions:
        family = getattr(condition, "family", None) or "unknown"
        target_tau = getattr(condition, "target_tau", None)
        if _as_float(target_tau) is None:
            continue
        grouped[str(family)].append(condition)

    payload = {}
    for family, family_conditions in sorted(grouped.items()):
        ordered = sorted(family_conditions, key=lambda item: float(item.target_tau))
        payload[family] = monotonicity(example_rows, ordered)
        payload[family]["ordered_conditions"] = [
            {
                "name": getattr(item, "name", str(item)),
                "target_tau": float(item.target_tau),
            }
            for item in ordered
        ]
    return payload
