import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.branching import branching_summary
from closure_delay.exit_hazard_torch import DifferentiableExitHazardHead
from closure_delay.model import LocalCausalLM
from closure_delay.process import ProcessScoreConfig, score_response_process
from closure_delay.rag import (
    DEFAULT_RAG_INSTRUCTION,
    evidence_closure_metrics,
    format_rag_prompt,
    load_rag_conditions,
    load_rag_records,
    rag_stage_summary,
)
from closure_delay.repetition import repetition_summary
from closure_delay.runtime import now_iso, set_seed, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled generator-only RAG suffix benchmarks.")
    parser.add_argument("--dataset-path", required=True, help="JSON or JSONL RAG records with fixed retrieved contexts.")
    parser.add_argument("--conditions-json", default="data/rag_suffix_conditions.json")
    parser.add_argument("--optimized-suffix-json")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-contexts", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=1600)
    parser.add_argument("--instruction", default=DEFAULT_RAG_INSTRUCTION)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--input-cost-per-1k", type=float, default=0.0)
    parser.add_argument("--output-cost-per-1k", type=float, default=0.0)
    _add_process_args(parser)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _add_process_args(parser: argparse.ArgumentParser) -> None:
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


def _process_config(args: argparse.Namespace) -> ProcessScoreConfig:
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


def _mean(values: Sequence[Any]) -> float | None:
    clean = []
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            clean.append(1.0 if value else 0.0)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            clean.append(number)
    return float(np.mean(clean)) if clean else None


def _estimated_cost(input_tokens: int, output_tokens: int, args: argparse.Namespace) -> float:
    return (
        float(input_tokens) * float(args.input_cost_per_1k)
        + float(output_tokens) * float(args.output_cost_per_1k)
    ) / 1000.0


def _input_token_count(model: LocalCausalLM, prompt: str, suffix: str) -> int:
    prompt_text = model.build_prompt_text(prompt, suffix)
    return int(len(model.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]))


def _add_baseline_deltas(rows: List[Dict]) -> None:
    baseline_by_id = {row["id"]: row for row in rows if row["condition"] == "no_suffix"}
    for row in rows:
        baseline = baseline_by_id.get(row["id"])
        if baseline is None:
            row.update(
                {
                    "baseline_tokens": None,
                    "delta_tokens": None,
                    "length_ratio": None,
                    "latency_ratio": None,
                    "estimated_cost_delta": None,
                    "cost_amplification_ratio": None,
                    "support_drop": None,
                    "answer_correct_drop": None,
                    "drift_delta": None,
                    "repetition_delta": None,
                    "truncation_delta": None,
                    "risk_score": None,
                }
            )
            continue
        base_tokens = float(baseline.get("generated_tokens") or 0.0)
        base_latency = float(baseline.get("latency_sec") or 0.0)
        base_cost = float(baseline.get("estimated_cost") or 0.0)
        base_truncated = 1.0 if _bool_or_none(baseline.get("truncated")) else 0.0
        row_truncated = 1.0 if _bool_or_none(row.get("truncated")) else 0.0
        row["baseline_tokens"] = int(base_tokens)
        row["delta_tokens"] = float(row["generated_tokens"]) - base_tokens
        row["length_ratio"] = float(row["generated_tokens"]) / base_tokens if base_tokens > 0 else None
        row["latency_ratio"] = float(row["latency_sec"]) / base_latency if base_latency > 0 else None
        row["estimated_cost_delta"] = float(row["estimated_cost"]) - base_cost
        row["cost_amplification_ratio"] = float(row["estimated_cost"]) / base_cost if base_cost > 0 else None
        row["support_drop"] = _drop_delta(baseline.get("answer_supported"), row.get("answer_supported"))
        row["answer_correct_drop"] = _drop_delta(baseline.get("answer_correct_proxy"), row.get("answer_correct_proxy"))
        row["drift_delta"] = _numeric_delta(row.get("drift_mean"), baseline.get("drift_mean"))
        row["repetition_delta"] = _numeric_delta(row.get("repeat_4gram_rate"), baseline.get("repeat_4gram_rate"))
        row["truncation_delta"] = float(row_truncated - base_truncated)
        row["risk_score"] = sum(
            max(float(value or 0.0), 0.0)
            for value in [
                row["support_drop"],
                row["drift_delta"],
                row["repetition_delta"],
                row["truncation_delta"],
                row.get("uncited_tail_sentence_rate"),
            ]
        )


def _drop_delta(baseline_value: Any, current_value: Any) -> float | None:
    baseline_bool = _bool_or_none(baseline_value)
    current_bool = _bool_or_none(current_value)
    if baseline_bool is None or current_bool is None:
        return None
    return (1.0 if baseline_bool else 0.0) - (1.0 if current_bool else 0.0)


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _numeric_delta(value: Any, baseline_value: Any) -> float | None:
    if value in (None, "") or baseline_value in (None, ""):
        return None
    try:
        return float(value) - float(baseline_value)
    except (TypeError, ValueError):
        return None


def _condition_summary(rows: Sequence[Dict], conditions: Sequence[Dict]) -> List[Dict]:
    fields = [
        "generated_tokens",
        "delta_tokens",
        "length_ratio",
        "latency_sec",
        "latency_ratio",
        "input_token_count",
        "estimated_cost",
        "estimated_cost_delta",
        "cost_amplification_ratio",
        "risk_score",
        "support_drop",
        "answer_correct_drop",
        "drift_delta",
        "repetition_delta",
        "truncation_delta",
        "answer_correct_proxy",
        "answer_exact_match",
        "answer_f1",
        "answer_contains",
        "answer_supported",
        "citation_precision",
        "citation_recall",
        "support_coverage",
        "post_evidence_tokens",
        "uncited_tail_sentence_rate",
        "repeat_4gram_rate",
        "max_repeated_line_count",
        "branch_marker_count",
        "mean_raw_hazard",
        "mean_cumlogit",
        "max_cumprob",
        "first_cross_token",
        "post_cross_tokens",
        "time_above_threshold",
        "auc_qc",
        "pcg_sum",
        "vpcg_sum",
        "verify_mean",
        "drift_mean",
        "jump_count",
        "plateau_token_total",
        "plateau_longest",
        "multi_step_count",
        "local_reset_count",
        "rise_reset_cycle_count",
        "evidence_stage_count",
        "cross_check_stage_count",
        "conflict_check_stage_count",
        "citation_switch_count",
        "truncated",
    ]
    summary_rows = []
    for condition in conditions:
        use = [row for row in rows if row["condition"] == condition["condition"]]
        row = {
            "condition": condition["condition"],
            "family": condition.get("family", ""),
            "n": len(use),
            "suffix_token_count": condition.get("suffix_token_count"),
        }
        row.update({f"{field}_mean": _mean([item.get(field) for item in use]) for field in fields})
        summary_rows.append(row)
    return summary_rows


def _delta_summary(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        if row["condition"] == "no_suffix":
            continue
        out.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "family": row["family"],
                "delta_tokens": row.get("delta_tokens"),
                "length_ratio": row.get("length_ratio"),
                "latency_ratio": row.get("latency_ratio"),
                "estimated_cost_delta": row.get("estimated_cost_delta"),
                "cost_amplification_ratio": row.get("cost_amplification_ratio"),
                "risk_score": row.get("risk_score"),
                "support_drop": row.get("support_drop"),
                "answer_correct_drop": row.get("answer_correct_drop"),
                "drift_delta": row.get("drift_delta"),
                "repetition_delta": row.get("repetition_delta"),
                "truncation_delta": row.get("truncation_delta"),
                "answer_contains": row.get("answer_contains"),
                "answer_correct_proxy": row.get("answer_correct_proxy"),
                "answer_f1": row.get("answer_f1"),
                "answer_supported": row.get("answer_supported"),
                "citation_precision": row.get("citation_precision"),
                "drift_mean": row.get("drift_mean"),
                "repeat_4gram_rate": row.get("repeat_4gram_rate"),
                "vpcg_sum": row.get("vpcg_sum"),
                "jump_count": row.get("jump_count"),
                "rise_reset_cycle_count": row.get("rise_reset_cycle_count"),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    records = load_rag_records(
        args.dataset_path,
        n_samples=args.n_samples,
        seed=int(args.seed),
        top_k_contexts=int(args.top_k_contexts) if args.top_k_contexts else None,
        max_context_chars=int(args.max_context_chars) if args.max_context_chars else None,
    )
    if not records:
        raise RuntimeError("No RAG records loaded.")
    conditions = load_rag_conditions(args.conditions_json, optimized_suffix_json=args.optimized_suffix_json)

    model = LocalCausalLM(args.model_path, device=args.device)
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=model.device)
    head.eval()
    score_config = _process_config(args)
    for condition in conditions:
        suffix = str(condition.get("suffix", ""))
        condition["suffix_token_count"] = len(model.tokenizer(suffix, add_special_tokens=False)["input_ids"]) if suffix else 0

    rows: List[Dict] = []
    token_rows: List[Dict] = []
    for record_index, record in enumerate(records, start=1):
        prompt = format_rag_prompt(
            record,
            instruction=str(args.instruction),
            top_k_contexts=int(args.top_k_contexts) if args.top_k_contexts else None,
        )
        for condition in conditions:
            suffix = str(condition.get("suffix", ""))
            started = time.perf_counter()
            trace = model.generate_trace(
                prompt=prompt,
                suffix=suffix,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=args.top_p,
            )
            latency_sec = time.perf_counter() - started
            process_summary, per_token = score_response_process(
                model,
                head,
                prompt,
                suffix,
                trace.generated_ids,
                score_config,
                include_token_rows=True,
            )
            evidence = evidence_closure_metrics(
                trace.response_text,
                trace.generated_ids,
                model.tokenizer,
                answer=record.answer,
                supporting_doc_ids=record.supporting_doc_ids,
                answer_aliases=record.answer_aliases,
            )
            input_tokens = _input_token_count(model, prompt, suffix)
            row = {
                "id": record.id,
                "dataset_path": str(args.dataset_path),
                "record_index": int(record_index),
                "condition": condition["condition"],
                "family": condition.get("family", ""),
                "question": record.question,
                "answer": record.answer,
                "answer_aliases_json": json.dumps(record.answer_aliases, ensure_ascii=False),
                "supporting_doc_ids_json": json.dumps(record.supporting_doc_ids, ensure_ascii=False),
                "context_doc_ids_json": json.dumps([context.doc_id for context in record.contexts], ensure_ascii=False),
                "context_titles_json": json.dumps([context.title for context in record.contexts], ensure_ascii=False),
                "suffix": suffix,
                "input_token_count": input_tokens,
                "generated_tokens": int(trace.generated_token_count),
                "latency_sec": float(latency_sec),
                "estimated_cost": _estimated_cost(input_tokens, int(trace.generated_token_count), args),
                "truncated": int(trace.generated_token_count) >= int(args.max_new_tokens),
                "response_text": trace.response_text,
                "response_token_ids_json": json.dumps(trace.generated_ids),
                **repetition_summary(trace.response_text),
                **branching_summary(trace.response_text, trace.generated_token_count),
                **rag_stage_summary(trace.response_text),
                **evidence,
                **process_summary,
            }
            rows.append(row)
            for token_row in per_token:
                token_rows.append(
                    {
                        "id": record.id,
                        "condition": condition["condition"],
                        "family": condition.get("family", ""),
                        **token_row,
                    }
                )
            print(
                f"{record_index}/{len(records)} {condition['condition']} "
                f"tokens={row['generated_tokens']} support={row['answer_supported']} "
                f"vpcg={row.get('vpcg_sum')}",
                flush=True,
            )

    _add_baseline_deltas(rows)
    condition_rows = _condition_summary(rows, conditions)
    delta_rows = _delta_summary(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "rag_generation_rows.csv", rows)
    write_csv(out_dir / "rag_token_process_rows.csv", token_rows)
    write_csv(out_dir / "rag_condition_summary.csv", condition_rows)
    write_csv(out_dir / "rag_delta_vs_baseline.csv", delta_rows)
    write_json(
        out_dir / "rag_benchmark_report.json",
        {
            "created_at": now_iso(),
            "dataset_path": str(args.dataset_path),
            "hazard_head_json": str(args.hazard_head_json),
            "config": vars(args),
            "conditions": conditions,
            "n_records": len(records),
            "metric_definitions": {
                "answer_supported": "Generated answer is supported by cited retrieved evidence; with gold support docs this requires citing all gold supporting document IDs.",
                "citation_precision": "Number of cited gold supporting document IDs divided by all cited document IDs.",
                "citation_recall": "Number of cited gold supporting document IDs divided by all gold supporting document IDs.",
                "post_evidence_tokens": "Generated tokens after the first prefix that cites sufficient supporting evidence before final answer emission.",
                "sufficient_evidence_hotpotqa": "All gold supporting document IDs are cited. If an LLM judge is used later, record the fixed prompt and audit a manual sample.",
                "risk_score": "support_drop + max(drift_delta,0) + max(repetition_delta,0) + max(truncation_delta,0) + uncited_tail_sentence_rate.",
            },
            "outputs": {
                "generation_rows": str(out_dir / "rag_generation_rows.csv"),
                "token_process_rows": str(out_dir / "rag_token_process_rows.csv"),
                "condition_summary": str(out_dir / "rag_condition_summary.csv"),
                "delta_vs_baseline": str(out_dir / "rag_delta_vs_baseline.csv"),
            },
        },
    )
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
