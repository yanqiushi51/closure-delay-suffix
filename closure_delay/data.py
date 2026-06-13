import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

GSM8K_ANSWER_PATTERN = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_json(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def split_dataset(dataset: List[Dict], train_size: int, val_size: int = 0):
    train_end = min(train_size, len(dataset))
    val_end = min(train_end + val_size, len(dataset))

    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]

    if not test:
        test = dataset
    return train, val, test


def train_eval_split(dataset: List[Dict], train_size: int):
    train, _, test = split_dataset(dataset, train_size=train_size, val_size=0)
    return train, test


def extract_gsm8k_numeric_answer(answer_text: str) -> Optional[str]:
    m = GSM8K_ANSWER_PATTERN.search(answer_text)
    if not m:
        return None
    return m.group(1)


def _load_local_gsm8k_examples(split: str) -> Optional[List[Dict]]:
    candidate_paths = []
    env_path = os.environ.get("GSM8K_LOCAL_JSONL")
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.extend(
        [
            Path("data") / f"gsm8k_{split}.jsonl",
            Path("data") / "gsm8k.jsonl",
        ]
    )
    for path in candidate_paths:
        if path.exists():
            return load_jsonl(str(path))
    return None


def _gsm8k_record(example: Dict, split: str, index: int) -> Optional[Dict]:
    answer_text = str(example.get("answer", ""))
    answer_str = extract_gsm8k_numeric_answer(answer_text)
    if answer_str is None:
        stripped = answer_text.strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            answer_str = stripped
    if answer_str is None:
        return None

    prompt = example.get("prompt")
    if prompt is None:
        question = str(example.get("question", ""))
        prompt = (
            "Solve this math problem. Show your reasoning step by step. "
            "End with 'Final answer: <number>'.\n\n"
            f"Question: {question}"
        )
    return {
        "id": str(example.get("id", f"gsm8k_{split}_{index}")),
        "prompt": str(prompt),
        "answer": answer_str,
    }


def load_gsm8k_dataset(
    split: str = "train",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    examples = _load_local_gsm8k_examples(split)
    if examples is not None:
        ds = list(examples)
        if n_samples is not None:
            random.Random(seed).shuffle(ds)
            ds = ds[: min(n_samples, len(ds))]
    else:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=split)
        if n_samples is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    records = []
    for i, example in enumerate(ds):
        record = _gsm8k_record(dict(example), split, i)
        if record is not None:
            records.append(record)
    return records
