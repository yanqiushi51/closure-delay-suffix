import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard_torch import DifferentiableExitHazardHead
from closure_delay.model import LocalCausalLM
from closure_delay.process import ProcessScoreConfig, score_response_process
from closure_delay.reporting import summarize_field
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate closure dynamics for suffix overthinking generations.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--examples-csv", required=True)
    parser.add_argument("--max-examples-per-condition", type=int, default=0)
    parser.add_argument("--hazard-threshold", type=float, default=0.30)
    parser.add_argument("--closure-threshold", type=float, default=0.70)
    parser.add_argument("--closure-eps", type=float, default=0.08)
    parser.add_argument("--answer-logprob-threshold", type=float, default=-3.50)
    parser.add_argument("--answer-eps", type=float, default=0.60)
    parser.add_argument("--answer-survival-mode", choices=["local", "cumulative"], default="local")
    parser.add_argument("--verify-logprob-threshold", type=float, default=-4.50)
    parser.add_argument("--verify-eps", type=float, default=0.80)
    parser.add_argument("--verify-mode", choices=["absolute", "hybrid"], default="hybrid")
    parser.add_argument("--verify-relative-weight", type=float, default=0.50)
    parser.add_argument("--verify-relative-eps", type=float, default=0.75)
    parser.add_argument("--reasoning-verify-offset", type=float, default=0.75)
    parser.add_argument("--drift-logprob-threshold", type=float, default=-5.00)
    parser.add_argument("--drift-eps", type=float, default=0.80)
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
    parser.add_argument("--ci-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--ci-seed", type=int, default=12345)
    parser.add_argument("--write-token-rows", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_rows(path: str | Path, max_examples_per_condition: int) -> List[Dict]:
    rows_by_condition: Dict[str, List[Dict]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            condition = str(row.get("condition", ""))
            bucket = rows_by_condition.setdefault(condition, [])
            if max_examples_per_condition > 0 and len(bucket) >= max_examples_per_condition:
                continue
            bucket.append(row)
    out: List[Dict] = []
    for condition in sorted(rows_by_condition):
        out.extend(rows_by_condition[condition])
    return out


def _config(args: argparse.Namespace) -> ProcessScoreConfig:
    return ProcessScoreConfig(
        hazard_threshold=float(args.hazard_threshold),
        closure_threshold=float(args.closure_threshold),
        closure_eps=float(args.closure_eps),
        answer_logprob_threshold=float(args.answer_logprob_threshold),
        answer_eps=float(args.answer_eps),
        answer_survival_mode=str(args.answer_survival_mode),
        verify_logprob_threshold=float(args.verify_logprob_threshold),
        verify_eps=float(args.verify_eps),
        verify_mode=str(args.verify_mode),
        verify_relative_weight=float(args.verify_relative_weight),
        verify_relative_eps=float(args.verify_relative_eps),
        reasoning_verify_offset=float(args.reasoning_verify_offset),
        drift_logprob_threshold=float(args.drift_logprob_threshold),
        drift_eps=float(args.drift_eps),
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


def _condition_summary(rows: Sequence[Dict]) -> List[Dict]:
    fields = [
        "generated_tokens",
        "correct_numeric",
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
    summaries = []
    for condition in sorted({row["condition"] for row in rows}):
        use = [row for row in rows if row["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "family": next((row.get("family") for row in use if row.get("family")), ""),
                "n": len(use),
                **{f"{field}_mean": _mean([row.get(field) for row in use]) for field in fields},
            }
        )
    return summaries


def _condition_summary_ci(rows: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    fields = [
        "jump_count",
        "plateau_longest",
        "multi_step_count",
        "local_reset_count",
        "rise_reset_cycle_count",
        "vpcg_sum",
    ]
    summaries = []
    for condition in sorted({row["condition"] for row in rows}):
        use = [row for row in rows if row["condition"] == condition]
        row = {
            "condition": condition,
            "family": next((item.get("family") for item in use if item.get("family")), ""),
            "n": len(use),
        }
        for field in fields:
            row.update(
                summarize_field(
                    use,
                    field,
                    digits=3 if field == "vpcg_sum" else 2,
                    n_bootstrap=int(args.ci_bootstrap_samples),
                    seed=int(args.ci_seed),
                )
            )
        summaries.append(row)
    return summaries


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.examples_csv, int(args.max_examples_per_condition))
    if not rows:
        raise RuntimeError("No examples loaded.")
    model = LocalCausalLM(args.model_path, device=args.device)
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=model.device)
    head.eval()
    config = _config(args)

    out_rows: List[Dict] = []
    token_rows: List[Dict] = []
    for idx, row in enumerate(rows, start=1):
        response_ids = model.tokenizer(str(row.get("response_text", "")), add_special_tokens=False)["input_ids"]
        response_ids = [int(token_id) for token_id in response_ids]
        summary, tokens = score_response_process(
            model,
            head,
            str(row.get("prompt", "")),
            str(row.get("suffix", "")),
            response_ids,
            config,
            include_token_rows=bool(args.write_token_rows),
        )
        out_row = {
            "id": row.get("id"),
            "condition": row.get("condition"),
            "family": row.get("family"),
            "suffix": row.get("suffix"),
            "generated_tokens": row.get("generated_tokens"),
            "correct": row.get("correct"),
            "correct_numeric": 1.0 if str(row.get("correct")).lower() == "true" else 0.0,
            **summary,
        }
        out_rows.append(out_row)
        if args.write_token_rows:
            for token_row in tokens:
                token_rows.append(
                    {
                        "id": row.get("id"),
                        "condition": row.get("condition"),
                        "family": row.get("family"),
                        **token_row,
                    }
                )
        if idx % 10 == 0:
            print(f"progress {idx}/{len(rows)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "suffix_closure_dynamics_rows.csv", out_rows)
    write_csv(out_dir / "suffix_closure_dynamics_summary.csv", _condition_summary(out_rows))
    write_csv(out_dir / "suffix_closure_dynamics_summary_ci.csv", _condition_summary_ci(out_rows, args))
    if args.write_token_rows:
        write_csv(out_dir / "suffix_token_process_rows.csv", token_rows)
    write_json(
        out_dir / "suffix_closure_dynamics_report.json",
        {
            "created_at": now_iso(),
            "examples_csv": str(args.examples_csv),
            "hazard_head_json": str(args.hazard_head_json),
            "config": vars(args),
            "n_rows": len(out_rows),
            "wrote_token_rows": bool(args.write_token_rows),
        },
    )
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
