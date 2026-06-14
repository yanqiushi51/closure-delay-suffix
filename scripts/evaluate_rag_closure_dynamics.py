import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.dynamics import DynamicsConfig, summarize_dynamics
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute RAG closure dynamics from token process rows.")
    parser.add_argument("--token-process-rows", required=True)
    parser.add_argument("--generation-rows")
    parser.add_argument("--closure-threshold", type=float, default=0.70)
    parser.add_argument("--jump-threshold", type=float, default=0.05)
    parser.add_argument("--jump-quantile", type=float, default=0.90)
    parser.add_argument("--plateau-high-threshold", type=float, default=0.60)
    parser.add_argument("--plateau-slope-threshold", type=float, default=0.01)
    parser.add_argument("--min-plateau-tokens", type=int, default=5)
    parser.add_argument("--min-stage-gap", type=int, default=8)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--local-peak-quantile", type=float, default=0.75)
    parser.add_argument("--local-valley-quantile", type=float, default=0.35)
    parser.add_argument("--local-reset-margin", type=float, default=0.50)
    parser.add_argument("--answer-onset-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_csv(path: str | Path) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: Dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: Sequence) -> float | None:
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            clean.append(number)
    return float(np.mean(clean)) if clean else None


def _config(args: argparse.Namespace) -> DynamicsConfig:
    return DynamicsConfig(
        closure_threshold=float(args.closure_threshold),
        jump_threshold=float(args.jump_threshold),
        jump_quantile=float(args.jump_quantile),
        plateau_high_threshold=float(args.plateau_high_threshold),
        plateau_slope_threshold=float(args.plateau_slope_threshold),
        min_plateau_tokens=int(args.min_plateau_tokens),
        min_stage_gap=int(args.min_stage_gap),
        smooth_window=int(args.smooth_window),
        local_peak_quantile=float(args.local_peak_quantile),
        local_valley_quantile=float(args.local_valley_quantile),
        local_reset_margin=float(args.local_reset_margin),
        answer_onset_threshold=float(args.answer_onset_threshold),
    )


def _generation_lookup(path: str | None) -> Dict[tuple[str, str], Dict]:
    if not path:
        return {}
    return {(row.get("id", ""), row.get("condition", "")): row for row in _read_csv(path)}


def _condition_summary(rows: Sequence[Dict]) -> List[Dict]:
    conditions = sorted({str(row.get("condition")) for row in rows})
    out = []
    fields = [
        "generated_tokens_scored",
        "mean_raw_hazard",
        "mean_cumlogit",
        "max_cumprob",
        "first_cross_token",
        "post_cross_tokens",
        "answer_onset_token",
        "time_above_threshold",
        "auc_qc",
        "pcg_sum",
        "vpcg_sum",
        "verify_mean",
        "drift_mean",
        "jump_count",
        "jump_magnitude_mean",
        "jump_magnitude_max",
        "multi_step_count",
        "multi_step_transition_count",
        "plateau_count",
        "plateau_token_total",
        "plateau_longest",
        "local_peak_count",
        "local_valley_count",
        "local_reset_count",
        "rise_reset_cycle_count",
        "second_rise_rate",
    ]
    for condition in conditions:
        use = [row for row in rows if row.get("condition") == condition]
        out.append(
            {
                "condition": condition,
                "family": next((row.get("family") for row in use if row.get("family")), ""),
                "n": len(use),
                **{f"{field}_mean": _mean([row.get(field) for row in use]) for field in fields},
            }
        )
    return out


def main() -> None:
    args = parse_args()
    rows = _read_csv(args.token_process_rows)
    grouped: dict[tuple[str, str], list[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("id", "")), str(row.get("condition", "")))].append(row)
    generation_by_key = _generation_lookup(args.generation_rows)

    out_rows: List[Dict] = []
    config = _config(args)
    for (item_id, condition), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(float(row.get("token_index") or 0)))
        summary = summarize_dynamics(
            raw_hazard=[_float(row, "exit_hazard") for row in group],
            cumprob=[_float(row, "exit_hazard_cumprob") for row in group],
            cumlogit=[_float(row, "exit_hazard_cumlogit") for row in group],
            q_closure=[_float(row, "q_closure") for row in group],
            answer_survival=[_float(row, "answer_survival") for row in group],
            verify_prob=[_float(row, "verify_prob") for row in group],
            drift_prob=[_float(row, "drift_prob") for row in group],
            pcg=[_float(row, "pcg") for row in group],
            vpcg=[_float(row, "vpcg") for row in group],
            lambda_answer=[_float(row, "lambda_answer") for row in group],
            config=config,
        )
        generation_row = generation_by_key.get((item_id, condition), {})
        out_rows.append(
            {
                "id": item_id,
                "condition": condition,
                "family": group[0].get("family", ""),
                "generated_tokens": generation_row.get("generated_tokens", len(group)),
                "answer_supported": generation_row.get("answer_supported"),
                "answer_contains": generation_row.get("answer_contains"),
                **summary,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "rag_closure_dynamics_rows.csv", out_rows)
    write_csv(out_dir / "rag_closure_dynamics_summary.csv", _condition_summary(out_rows))
    write_json(
        out_dir / "rag_closure_dynamics_report.json",
        {
            "created_at": now_iso(),
            "token_process_rows": str(args.token_process_rows),
            "generation_rows": str(args.generation_rows) if args.generation_rows else None,
            "config": vars(args),
            "n_rows": len(out_rows),
        },
    )
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
