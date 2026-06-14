import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.rag import evidence_closure_metrics
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute RAG evidence-closure metrics from saved generations.")
    parser.add_argument("--generation-rows", required=True)
    parser.add_argument("--model-path", help="Optional tokenizer path for token-level closure positions.")
    parser.add_argument("--require-answer-for-evidence-closure", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_csv(path: str | Path) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _loads_list(value: str | None) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def main() -> None:
    args = parse_args()
    rows = _read_csv(args.generation_rows)
    tokenizer = _load_tokenizer(args.model_path)
    out_rows: List[Dict] = []
    for row in rows:
        response_ids = [int(value) for value in _loads_list(row.get("response_token_ids_json"))]
        supporting = [str(value) for value in _loads_list(row.get("supporting_doc_ids_json"))]
        aliases = [str(value) for value in _loads_list(row.get("answer_aliases_json"))]
        metrics = evidence_closure_metrics(
            str(row.get("response_text", "")),
            response_ids,
            tokenizer,
            answer=str(row.get("answer", "")),
            supporting_doc_ids=supporting,
            answer_aliases=aliases,
            require_answer_for_evidence_closure=bool(args.require_answer_for_evidence_closure),
        )
        out_rows.append(
            {
                "id": row.get("id"),
                "condition": row.get("condition"),
                "family": row.get("family"),
                "generated_tokens": row.get("generated_tokens"),
                **metrics,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "rag_evidence_closure_rows.csv", out_rows)
    write_json(
        out_dir / "rag_evidence_closure_report.json",
        {
            "created_at": now_iso(),
            "generation_rows": str(args.generation_rows),
            "model_path": str(args.model_path) if args.model_path else None,
            "require_answer_for_evidence_closure": bool(args.require_answer_for_evidence_closure),
            "n_rows": len(out_rows),
        },
    )
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
