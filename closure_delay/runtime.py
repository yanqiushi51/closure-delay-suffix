"""Small runtime and output helpers for closure-delay experiments."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> int:
    """Seed common RNGs when their packages are available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a UTF-8 JSON file, creating parent directories first."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any] | Any]) -> Path:
    """Write rows to CSV, inferring field order from the row keys."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    normalized_rows = [_row_to_mapping(row) for row in rows]
    fieldnames = _fieldnames(normalized_rows)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return output_path

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return output_path


def now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def summarize_invalid_reasons(items: Iterable[Any]) -> dict[str, int]:
    """Count reasons for items whose valid flag is explicitly False."""

    reasons: Counter[str] = Counter()
    for item in items:
        if _get_value(item, "valid", True) is not False:
            continue

        reason = _get_value(item, "reason", None) or "unknown"
        reasons[str(reason)] += 1
    return dict(reasons)


def _row_to_mapping(row: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    return {key: value for key, value in vars(row).items() if not key.startswith("_")}


def _fieldnames(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def _get_value(item: Any, key: str, default: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)
