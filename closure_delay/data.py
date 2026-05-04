import json
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


def load_gsm8k_dataset(
    split: str = "train",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    records = []
    for i, example in enumerate(ds):
        answer_str = extract_gsm8k_numeric_answer(example["answer"])
        if answer_str is None:
            continue
        records.append(
            {
                "id": f"gsm8k_{split}_{i}",
                "prompt": (
                    "Solve this math problem. Show your reasoning step by step. "
                    "End with 'Final answer: <number>'.\n\n"
                    f"Question: {example['question']}"
                ),
                "answer": answer_str,
            }
        )
    return records
