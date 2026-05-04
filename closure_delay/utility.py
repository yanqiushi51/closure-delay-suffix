from __future__ import annotations

import math
import re


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(text: str):
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def numeric_correct(prediction_text: str, gold_answer) -> bool:
    pred = last_number(prediction_text)
    if pred is None:
        return False
    try:
        gold = float(gold_answer)
    except (TypeError, ValueError):
        return False
    return math.isclose(pred, gold, rel_tol=0.0, abs_tol=1e-6)
