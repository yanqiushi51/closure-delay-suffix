"""Label construction from free-generation text for hidden-state probes.

Two label types:
  1. ready labels: binary, 1 if final-answer onset is within delta tokens
  2. direction labels: 5-class, based on future token patterns
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------- ready labels ----------

def build_ready_labels(
    onset_token_idx: Optional[int],
    positions: Sequence[int],
    delta: int = 32,
) -> np.ndarray:
    """y_t = 1 if onset - pos <= delta (and onset is known), else 0.

    If onset_token_idx is None (onset not found), all labels are 0.
    """
    labels = np.zeros(len(positions), dtype=np.float32)
    if onset_token_idx is None:
        return labels
    for i, pos in enumerate(positions):
        if pos <= onset_token_idx and onset_token_idx - pos <= delta:
            labels[i] = 1.0
    return labels


# ---------- direction labels ----------

DIRECTION_PATTERNS: Dict[str, List[str]] = {
    "finalize": [
        r"\bfinal answer\b",
        r"\bthe answer is\b",
        r"\bso the answer\b",
        r"\btherefore[, ]+the answer\b",
        r"\banswer:\s*\d",
        r"\bconclusion\b",
    ],
    "verify": [
        r"\b(check|verify|double.?check)\b",
        r"\bmake sure\b",
        r"\blet('s| us) (check|verify|confirm)\b",
        r"\bensure\b",
        r"\bshould be correct\b",
        r"\brecheck\b",
    ],
    "alternative": [
        r"\banother (way|method|approach)\b",
        r"\balternatively\b",
        r"\bdifferent (way|approach|method)\b",
        r"\bwe (can|could) also\b",
        r"\bor we (can|could)\b",
        r"\bswitch(ing)? (to|approach)\b",
    ],
    "compare": [
        r"\bcompare\b",
        r"\breconcil\w*\b",
        r"\b(on one hand|on the other hand)\b",
        r"\bconsistent\b",
        r"\bmatch(es|ing)?\b",
        r"\bboth (results|answers|values)\b",
        r"\bsame (result|answer)\b",
    ],
    "continue": [
        r"\bnext\b",
        r"\b(now|then) (we|the)\b",
        r"\bcontinuing\b",
        r"\bproceed\b",
        r"\bfollowing\b",
        r"\bso\b",
        r"\bthus\b",
        r"\btherefore\b",
    ],
}


def _classify_future_text(text: str) -> str:
    """Classify a short future text segment into a direction class.

    Priority: finalize > verify > alternative > compare > continue
    (finalize is least ambiguous, continue is the catch-all)
    """
    text_lower = text.lower()
    for direction in ["finalize", "verify", "alternative", "compare"]:
        for pattern in DIRECTION_PATTERNS[direction]:
            if re.search(pattern, text_lower):
                return direction
    return "continue"


def _token_to_char_index(token_ids: Sequence[int], token_idx: int) -> int:
    """Estimate character index from token index. Approximate but good enough for
    onset proximity checks."""
    return token_idx * 3  # rough estimate: ~3 chars per token on average


def build_direction_labels(
    response_text: str,
    positions: Sequence[int],
    onset_token_idx: int | None = None,
    future_window: int = 32,
    delta: int = 32,
    tokenizer=None,
) -> np.ndarray:
    """Build binary direction labels: 0=continue, 1=finalize.

    A position is labeled 'finalize' if it is within `delta` tokens of the
    onset, OR if the future text segment contains a final-answer marker.
    Otherwise it is labeled 'continue'.

    This binary formulation directly answers: "is the model approaching
    final-answer mode at this position?"
    """
    labels = np.zeros(len(positions), dtype=np.int64)
    response_len = len(response_text)

    for i, pos in enumerate(positions):
        # Check proximity to onset
        if onset_token_idx is not None and pos <= onset_token_idx and onset_token_idx - pos <= delta:
            labels[i] = 1
            continue

        # Check future text for final-answer markers
        char_start = pos * 3
        char_end = min(char_start + future_window * 4, response_len)
        if char_start < response_len:
            future_text = response_text[char_start:char_end].lower()
            for pattern in DIRECTION_PATTERNS["finalize"]:
                if re.search(pattern, future_text):
                    labels[i] = 1
                    break

    return labels


def direction_label_summary(labels: np.ndarray) -> Dict:
    """Return counts per direction class."""
    idx_to_name = {0: "continue", 1: "verify", 2: "alternative", 3: "compare", 4: "finalize"}
    total = len(labels)
    if total == 0:
        return {"total": 0}
    counts = {}
    for idx, name in idx_to_name.items():
        counts[name] = int((labels == idx).sum())
    counts["total"] = total
    return counts
