from __future__ import annotations

import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"\S+")


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def distinct_n(text: str, n: int) -> float | None:
    tokens = tokenize_text(text)
    if len(tokens) < n or n <= 0:
        return None
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if not grams:
        return None
    return len(set(grams)) / len(grams)


def repeat_ngram_rate(text: str, n: int = 4) -> float | None:
    tokens = tokenize_text(text)
    if len(tokens) < n or n <= 0:
        return None
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if not grams:
        return None
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def max_repeated_line_count(text: str) -> int:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0
    counts = Counter(lines)
    return max(counts.values())


def repetition_summary(text: str) -> dict:
    return {
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "repeat_4gram_rate": repeat_ngram_rate(text, 4),
        "max_repeated_line_count": max_repeated_line_count(text),
    }
