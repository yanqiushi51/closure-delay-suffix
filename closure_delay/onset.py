from __future__ import annotations

import re
from typing import Dict, Optional


MARKERS = [
    "final answer",
    "the final answer",
    "so the answer",
    "therefore, the answer",
    "therefore the answer",
    "the answer is",
    "answer:",
]


def find_closure_onset(text: str, answer: Optional[str] = None) -> Dict:
    normalized = text or ""
    best = None
    best_marker = None
    for marker in MARKERS:
        pattern = re.compile(re.escape(marker), re.IGNORECASE)
        match = pattern.search(normalized)
        if match and (best is None or match.start() < best):
            best = match.start()
            best_marker = marker

    if best is not None:
        return {
            "found": True,
            "marker": best_marker,
            "char_index": int(best),
            "reason": "marker_found",
        }

    return {
        "found": False,
        "marker": None,
        "char_index": None,
        "reason": "no_marker",
    }


def onset_token_index(tokenizer, text: str, onset: Dict) -> Optional[int]:
    if not onset.get("found") or onset.get("char_index") is None:
        return None
    prefix = (text or "")[: int(onset["char_index"])]
    token_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    return int(len(token_ids))


def onset_metrics(tokenizer, baseline_text: str, attacked_text: str, baseline_length: int) -> Dict:
    baseline = find_closure_onset(baseline_text)
    attacked = find_closure_onset(attacked_text)
    baseline_token = onset_token_index(tokenizer, baseline_text, baseline)
    attacked_token = onset_token_index(tokenizer, attacked_text, attacked)

    onset_delay_ratio = None
    if baseline_token is not None and baseline_token > 0 and attacked_token is not None:
        onset_delay_ratio = attacked_token / baseline_token

    preanswer_token_ratio = None
    if baseline_length and attacked_token is not None:
        preanswer_token_ratio = attacked_token / baseline_length

    return {
        "baseline_onset_found": baseline["found"],
        "baseline_onset_marker": baseline["marker"],
        "baseline_onset_token": baseline_token,
        "attacked_onset_found": attacked["found"],
        "attacked_onset_marker": attacked["marker"],
        "attacked_onset_token": attacked_token,
        "onset_delay_ratio": onset_delay_ratio,
        "preanswer_token_ratio": preanswer_token_ratio,
    }
