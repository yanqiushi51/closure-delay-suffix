from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


EXIT_PROBE_PHRASES = [
    " Therefore, the final answer is",
    " So the answer is",
    " The final answer is",
    " Final answer:",
    " Thus, the answer is",
    " Hence, the answer is",
    " Answer:",
    " The result is",
    " We get",
]

REASONING_PROBE_PHRASES = [
    " Next, we need to",
    " Let's verify this step",
    " Now consider another way",
    " We should check whether",
    " Let's continue",
    " Continue reasoning",
    " We need to check",
    " Let's compute",
    " Another way to see this is",
    " Wait,",
]

EXIT_MARKER_PROBE_PHRASES = [
    " Final",
    " Answer",
    " Therefore",
    " Thus",
    " Hence",
    " So",
]

CONTINUE_MARKER_PROBE_PHRASES = [
    " Next",
    " Continue",
    " Let's",
    " We",
    " Now",
    " Wait",
]

ANSWER_ONSET_PROBE_PHRASES = [
    " Final answer",
    " Final answer:",
    " Answer:",
    " Therefore",
    " Thus",
    " Hence",
    " So the answer",
    " The answer is",
    " \\boxed",
    "\n####",
]

VERIFY_BEHAVIOR_PROBE_PHRASES = [
    " Wait",
    " Wait,",
    " Let's verify",
    " Let us verify",
    " To verify",
    " Verify",
    " verify",
    " Check",
    " Let's check",
    " check",
    " Recalculate",
    " Let's recompute",
    " Recompute",
    " recompute",
    " Another way",
    " another way",
    " Alternative",
    " alternative",
    " Consistency check",
    " Double-check",
    " double-check",
    " Cross-check",
    " cross-check",
    " Review",
    " review",
    " Confirm",
    " confirm",
]

VERIFY_BRANCH_PROBE_PHRASES = [
    " Another",
    " Alternative",
    " Method",
    " Approach",
    " Compare",
    " Cross",
    " Double",
    " Recompute",
    " Check",
    " Verify",
    " Wait",
]

DRIFT_BEHAVIOR_PROBE_PHRASES = [
    " If you",
    " Let me know",
    " I'm here",
    " Thank you",
    " Feel free",
    " As an AI",
    " Python code",
    " Qwen",
    " User",
    " Assistant",
]

CLOSURE_MARKER_PATTERNS = [
    "final answer",
    "therefore",
    "conclusion",
    "answer:",
    "so the answer is",
    "the final answer is",
]

DRIFT_MARKER_PATTERNS = [
    "if you",
    "i'm here",
    "let me know",
    "thank you",
    "<tool_call>",
    "\nuser",
    "feel free",
    "assist you",
    "best regards",
    "qwen",
    "python code",
    "```python",
    "thankyou",
    "service",
]

HAZARD_SCORE_COLUMNS = (
    "exit_hazard_cumlogit",
    "exit_hazard",
    "exit_hazard_cumprob",
    "full_probe_margin",
    "full_probe_risk",
    "full_probe_margin_runmax",
)

LOGIT_FEATURE_KEYS = [
    "exit_logit_margin",
    "exit_logit_exit_logmass",
    "exit_logit_reasoning_logmass",
    "exit_logit_margin_runmax",
    "exit_logit_margin_runmin",
    "exit_logit_margin_pos_cumsum",
    "exit_logit_margin_neg_cumsum",
    "exit_marker_logit_margin",
    "exit_marker_logit_margin_runmax",
    "exit_marker_logit_margin_pos_cumsum",
    "exit_marker_logit_margin_neg_cumsum",
    "exit_logit_pmax",
    "exit_logit_eos_prob",
]


def parse_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if np.isfinite(number) else None


def safe_float(value, default: float = 0.0) -> float:
    number = parse_float(value)
    return float(default) if number is None else float(number)


def load_csv_rows(path: str | Path) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_metrics(path: str | Path, condition: str) -> Dict[str, Dict]:
    rows = load_csv_rows(path)
    return {str(row["id"]): row for row in rows if str(row.get("condition")) == condition}


def load_text_rows(path: str | Path, condition: str) -> Dict[str, Dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in rows if str(row.get("condition")) == condition}


def load_candidate_points(paths: Sequence[str | Path]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for path in paths:
        for row in load_csv_rows(path):
            if row.get("id"):
                grouped.setdefault(str(row["id"]), []).append(row)
    for key, rows in grouped.items():
        grouped[key] = sorted(rows, key=lambda item: safe_float(item.get("token_index")))
    return grouped


def load_hazard_points(paths: Sequence[str | Path]) -> Dict[str, List[Dict]]:
    return load_candidate_points(paths)


def build_prompt_text(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def first_token_ids(tokenizer, phrases: Sequence[str]) -> List[int]:
    ids: List[int] = []
    seen = set()
    for text in phrases:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_id = int(token_ids[0])
        if token_id not in seen:
            seen.add(token_id)
            ids.append(token_id)
    return ids


def first_drift_marker_ratio(text: str, closure_ratio: float | None = None) -> float | None:
    lowered = (text or "").lower()
    if not lowered:
        return None
    start_idx = 0
    if closure_ratio is not None and np.isfinite(closure_ratio):
        start_idx = int(max(0.0, min(1.0, float(closure_ratio))) * len(lowered))
    tail = lowered[start_idx:]
    best = None
    for marker in DRIFT_MARKER_PATTERNS:
        pos = tail.find(marker)
        if pos < 0:
            continue
        idx = start_idx + pos
        if best is None or idx < best:
            best = idx
    return None if best is None else float(best / max(len(lowered), 1))


def event_fractions(metrics_row: Dict, response_text: str) -> tuple[float | None, float | None, float | None]:
    closure_fraction = parse_float(metrics_row.get("first_closure_marker_char_ratio"))
    drift_fraction = first_drift_marker_ratio(response_text, closure_fraction)
    events = [value for value in (closure_fraction, drift_fraction) if value is not None]
    exit_fraction = float(min(events)) if events else None
    return closure_fraction, drift_fraction, exit_fraction


def running_max(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    current = float(values[0])
    for value in values:
        current = max(current, float(value))
        out.append(float(current))
    return out


def running_min(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    current = float(values[0])
    for value in values:
        current = min(current, float(value))
        out.append(float(current))
    return out


def robust_minmax(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(arr, 0.1))
    hi = float(np.quantile(arr, 0.9))
    if hi - lo < 1e-9:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if hi - lo < 1e-9:
        return lo, lo + 1.0
    return lo, hi


def normalize(values: Sequence[float], lo: float, hi: float) -> List[float]:
    scale = max(hi - lo, 1e-9)
    return [float(np.clip((float(value) - lo) / scale, 0.0, 1.0)) for value in values]


def normalized_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float(ys[0]) if ys else float("nan")
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(np.asarray(ys, dtype=float), np.asarray(xs, dtype=float)) / (xs[-1] - xs[0]))
