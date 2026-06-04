import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.runtime import ensure_dir, now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple decode-gate shard outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shards", nargs="+", required=True, help="Shard output directories")
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    shard_dirs = [Path(s).resolve() for s in args.shards]

    merged_examples: List[Dict] = []
    merged_texts: List[Dict] = []
    shard_stats: List[Dict] = []

    for shard_dir in shard_dirs:
        shard_tag = shard_dir.name
        examples_path = shard_dir / "example_decode_gate_metrics.csv"
        texts_path = shard_dir / "generation_texts.json"
        summary_path = shard_dir / "summary.json"
        if not examples_path.exists() or not texts_path.exists():
            raise FileNotFoundError(f"Missing outputs in shard: {shard_dir}")

        example_rows = load_csv_rows(examples_path)
        text_rows = json.loads(texts_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        for row in example_rows:
            row = dict(row)
            original_id = str(row.get("id", ""))
            row["original_id"] = original_id
            row["shard"] = shard_tag
            row["id"] = f"{shard_tag}::{original_id}"
            merged_examples.append(row)

        for row in text_rows:
            row = dict(row)
            original_id = str(row.get("id", ""))
            row["original_id"] = original_id
            row["shard"] = shard_tag
            row["id"] = f"{shard_tag}::{original_id}"
            merged_texts.append(row)

        shard_stats.append(
            {
                "shard": shard_tag,
                "example_rows": len(example_rows),
                "text_rows": len(text_rows),
                "n_requested": (
                    summary.get("baseline_filter", {}).get("n_requested")
                    if isinstance(summary, dict)
                    else None
                ),
                "n_included": (
                    summary.get("baseline_filter", {}).get("n_included")
                    if isinstance(summary, dict)
                    else None
                ),
                "n_skipped": (
                    summary.get("baseline_filter", {}).get("n_skipped")
                    if isinstance(summary, dict)
                    else None
                ),
            }
        )

    write_csv(output_dir / "example_decode_gate_metrics.csv", merged_examples)
    write_json(output_dir / "generation_texts.json", merged_texts)
    write_csv(output_dir / "merge_shard_stats.csv", shard_stats)
    write_json(
        output_dir / "merge_summary.json",
        {
            "created_at": now_iso(),
            "n_shards": len(shard_dirs),
            "n_merged_examples": len(merged_examples),
            "n_merged_text_rows": len(merged_texts),
            "shard_dirs": [str(p) for p in shard_dirs],
            "stats": shard_stats,
        },
    )
    print(f"merged shards={len(shard_dirs)} examples={len(merged_examples)} text_rows={len(merged_texts)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
