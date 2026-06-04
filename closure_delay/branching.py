from __future__ import annotations

import re


BRANCH_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\banother (?:way|method|approach|path)\b",
        r"\balternative (?:way|method|approach|path|solution)\b",
        r"\b(?:first|second|third) (?:way|method|approach|path)\b",
        r"\bmethod\s*\d+\b",
        r"\bapproach\s*\d+\b",
        r"\bpath\s*\d+\b",
        r"\bsolution path\b",
        r"\bdouble-check\b",
        r"\bverify (?:this|the|each|our)\b",
    ]
]


def branch_marker_count(text: str) -> int:
    text = text or ""
    return sum(len(pattern.findall(text)) for pattern in BRANCH_PATTERNS)


def branching_summary(text: str, token_count: int | None = None) -> dict:
    count = branch_marker_count(text)
    denominator = max(int(token_count or 0), 1)
    return {
        "branch_marker_count": count,
        "branch_marker_rate": count / denominator,
        "has_branch_marker": count > 0,
    }
