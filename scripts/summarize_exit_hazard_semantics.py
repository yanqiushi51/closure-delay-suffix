import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manual semantic labels for exit-hazard alignment.")
    parser.add_argument("manual_labels_csv")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.manual_labels_csv).open(encoding="utf-8")))
    if not rows:
        raise RuntimeError("No manual labels found.")
    by_kind = defaultdict(Counter)
    by_label = Counter()
    by_alignment = Counter()
    for row in rows:
        by_kind[row["candidate_kind"]][row["alignment"]] += 1
        by_label[row["manual_label"]] += 1
        by_alignment[row["alignment"]] += 1
    n = len(rows)
    strict = by_alignment["match"]
    partial = by_alignment["partial_match"]
    payload = {
        "created_at": now_iso(),
        "n_manual_labels": n,
        "strict_match_rate": strict / n,
        "strict_or_partial_rate": (strict + partial) / n,
        "alignment_counts": dict(by_alignment),
        "manual_label_counts": dict(by_label),
        "by_candidate_kind": {kind: dict(counts) for kind, counts in by_kind.items()},
    }
    summary_rows = []
    for kind, counts in sorted(by_kind.items()):
        total = sum(counts.values())
        summary_rows.append(
            {
                "candidate_kind": kind,
                "n": total,
                "match": counts["match"],
                "partial_match": counts["partial_match"],
                "mismatch": total - counts["match"] - counts["partial_match"],
                "strict_match_rate": counts["match"] / total if total else None,
                "strict_or_partial_rate": (counts["match"] + counts["partial_match"]) / total if total else None,
            }
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "semantic_alignment_by_kind.csv", summary_rows)
    write_json(out_dir / "semantic_alignment_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
