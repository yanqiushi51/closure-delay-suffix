import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.runtime import ensure_dir, now_iso, write_csv, write_json


EXPERIMENT_MATRIX = [
    {
        "experiment_id": "E1",
        "claim": "Proxy validity: closure-readiness is not just length or marker counting",
        "experiment": "Proxy validity plus incremental CE controls",
        "required_rows": "OOF/held-out; all tokens and marker-free subset",
        "key_metrics": "proxy_score, Spearman, relaxed_monotone_rate, timely_rate, jump_align_rate, length_coupling, delta_ce, delta_auc",
        "expected_pattern": "cumlogit remains predictive after fraction, log token index, and closure-marker controls",
        "script_or_output": "evaluate_exit_hazard_proxy.py; evaluate_exit_hazard_incremental_ce.py",
    },
    {
        "experiment_id": "E2",
        "claim": "Suffix-control attack, mechanism, and length-risk frontier share one generation set",
        "experiment": "GSM8K A-F suffix-control generations with effectiveness, dynamics, and frontier analyses",
        "required_rows": "no suffix, irrelevant, verbose-only, manual verification, structured E, optimized",
        "key_metrics": "delta_tokens, output_cost_delta, accuracy, truncation, repetition, drift, jump_count, plateau_longest, local_reset_count, VPCG, risk_score",
        "expected_pattern": "structured E is longer, structured, low-risk, and stronger than verbose-only on VPCG/cycle metrics",
        "script_or_output": "evaluate_suffix_control_bank.py; evaluate_closure_dynamics.py; evaluate_suffix_closure_dynamics.py",
    },
    {
        "experiment_id": "E3",
        "claim": "Fixed-retrieval RAG transfer: post-evidence overthinking in cloud QA",
        "experiment": "Generator-only RAG suffix benchmark with fixed retrieved context",
        "required_rows": "HotpotQA/FAQ subset; A-F conditions; suffix affects generation only",
        "key_metrics": "delta_tokens, output_cost_delta, answer_correct, answer_supported, citation_precision, citation_recall, post_evidence_tokens, VPCG, evidence_stage_count, citation_switches",
        "expected_pattern": "structured E raises cost and post-evidence tokens while preserving support/citation quality",
        "script_or_output": "run_rag_suffix_benchmark.py; evaluate_rag_evidence_closure.py; evaluate_rag_closure_dynamics.py",
    },
    {
        "experiment_id": "E4",
        "claim": "Closure-aware defense uses prefix-available signals for online budget control",
        "experiment": "Defense replay over token traces, optionally followed by live online generation",
        "required_rows": "clean: no/irrelevant; attacked: manual/structured/optimized",
        "key_metrics": "clean_acc_drop, false_trigger_rate, attack_token_reduction, cost_reduction, answer_retention, support_retention",
        "expected_pattern": "closure-aware finalization reduces attacked cost more than marker stop and harms clean less than fixed budget",
        "script_or_output": "run_closure_aware_defense.py",
    },
    {
        "experiment_id": "Appendix",
        "claim": "Robustness and ablations support the main story without crowding the paper",
        "experiment": "Seeds, thresholds, sample sizes, 1.5B/14B, end-to-end RAG, suffix transfer",
        "required_rows": "selected robustness sweeps",
        "key_metrics": "same metrics as main experiments",
        "expected_pattern": "directionally consistent with main results",
        "script_or_output": "existing eval scripts plus robustness sweeps",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing RAG attack and experiment-matrix tables.")
    parser.add_argument("--proxy-ablation-summary")
    parser.add_argument("--suffix-attack-summary")
    parser.add_argument("--suffix-dynamics-summary")
    parser.add_argument("--rag-condition-summary")
    parser.add_argument("--defense-summary")
    parser.add_argument("--output-md", default="outputs/exit_hazard/rag_attack_tables.md")
    parser.add_argument("--matrix-csv", default="outputs/exit_hazard/wise2026_experiment_matrix.csv")
    return parser.parse_args()


def _read_csv(path: str | None) -> List[Dict]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt(value: str | None) -> str:
    if value in (None, ""):
        return "TBD"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.3f}"


def _ci_value(row: Dict, field: str) -> str:
    formatted = row.get(f"{field}_mean_ci")
    if formatted not in (None, ""):
        return str(formatted)
    return _fmt(row.get(f"{field}_mean"))


def _proxy_table(rows: List[Dict]) -> str:
    if not rows:
        return (
            "| Model | Dataset | Feature Mode | Subset | Proxy Score | Delta CE | Delta AUC |\n"
            "| --- | --- | --- | --- | ---: | ---: | ---: |\n"
            "| Qwen2.5-7B | GSM8K | hidden-only | all tokens | TBD | TBD | TBD |\n"
            "| Qwen2.5-7B | GSM8K | logit-only | all tokens | TBD | TBD | TBD |\n"
            "| Qwen2.5-7B | GSM8K | static-delta-logit | all tokens | TBD | TBD | TBD |\n"
            "| Qwen2.5-7B | GSM8K | static-delta-logit | marker-free | TBD | TBD | TBD |"
        )
    lines = [
        "| Model | Dataset | Feature Mode | Subset | Proxy Score | Delta CE | Delta AUC |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {dataset} | {feature_mode} | {subset} | {proxy} | {delta_ce} | {delta_auc} |".format(
                model=row.get("model", "TBD"),
                dataset=row.get("dataset", "TBD"),
                feature_mode=row.get("feature_mode", row.get("Feature Mode", "TBD")),
                subset=row.get("subset", row.get("Subset", "TBD")),
                proxy=_fmt(row.get("proxy_score")),
                delta_ce=_fmt(row.get("delta_ce")),
                delta_auc=_fmt(row.get("delta_auc")),
            )
        )
    return "\n".join(lines)


def _suffix_attack_table(rows: List[Dict]) -> str:
    if not rows:
        return (
            "| Condition | Delta Tokens | Length Ratio | Output Cost Delta | Accuracy | Truncation | Drift | Repetition | VPCG |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            "| baseline | 0 | 1.00 | 0 | TBD | TBD | TBD | TBD | TBD |\n"
            "| unrelated | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| verbose-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| manual verification | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| structured E | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| optimized | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |"
        )
    lines = [
        "| Condition | Delta Tokens | Length Ratio | Output Cost Delta | Accuracy | Truncation | Drift | Repetition | VPCG |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {delta} | {ratio} | {cost} | {accuracy} | {truncation} | {drift} | {repeat} | {vpcg} |".format(
                condition=row.get("condition", ""),
                delta=_ci_value(row, "delta_tokens"),
                ratio=_ci_value(row, "length_ratio"),
                cost=_ci_value(row, "estimated_output_cost_delta"),
                accuracy=_ci_value(row, "correct"),
                truncation=_ci_value(row, "truncated"),
                drift=_ci_value(row, "drift_mean"),
                repeat=_ci_value(row, "repeat_4gram_rate"),
                vpcg=_ci_value(row, "vpcg_sum"),
            )
        )
    return "\n".join(lines)


def _suffix_dynamics_table(rows: List[Dict]) -> str:
    if not rows:
        return (
            "| Condition | Jump Count | Plateau Longest | Multi-Step Count | Local Reset Count | Rise-Reset Cycles | VPCG |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            "| baseline | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| verbose-only | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| manual verification | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| structured E | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| optimized | TBD | TBD | TBD | TBD | TBD | TBD |"
        )
    lines = [
        "| Condition | Jump Count | Plateau Longest | Multi-Step Count | Local Reset Count | Rise-Reset Cycles | VPCG |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {jump} | {plateau} | {multi} | {reset} | {cycle} | {vpcg} |".format(
                condition=row.get("condition", ""),
                jump=_ci_value(row, "jump_count"),
                plateau=_ci_value(row, "plateau_longest"),
                multi=_ci_value(row, "multi_step_count"),
                reset=_ci_value(row, "local_reset_count"),
                cycle=_ci_value(row, "rise_reset_cycle_count"),
                vpcg=_ci_value(row, "vpcg_sum"),
            )
        )
    return "\n".join(lines)


def _attack_table(rows: List[Dict]) -> str:
    if not rows:
        return (
            "| Condition | Delta Tokens | Output Cost Delta | Answer Correct | Answer Supported | Citation Precision | Citation Recall | Post-Evidence Tokens | VPCG | Drift | Repetition |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            "| no_suffix | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| irrelevant_clear_format | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| verbose_only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| manual_verification | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| structured_multistage | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n"
            "| optimized_structured | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |"
        )
    lines = [
        "| Condition | Delta Tokens | Output Cost Delta | Answer Correct | Answer Supported | Citation Precision | Citation Recall | Post-Evidence Tokens | VPCG | Drift | Repetition |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {delta} | {cost} | {correct} | {support} | {precision} | {recall} | {post_evidence} | {vpcg} | {drift} | {repeat} |".format(
                condition=row.get("condition", ""),
                delta=_fmt(row.get("delta_tokens_mean")),
                cost=_fmt(row.get("estimated_cost_delta_mean")),
                correct=_fmt(row.get("answer_correct_proxy_mean")),
                support=_fmt(row.get("answer_supported_mean")),
                precision=_fmt(row.get("citation_precision_mean")),
                recall=_fmt(row.get("citation_recall_mean")),
                post_evidence=_fmt(row.get("post_evidence_tokens_mean")),
                vpcg=_fmt(row.get("vpcg_sum_mean")),
                drift=_fmt(row.get("drift_mean_mean")),
                repeat=_fmt(row.get("repeat_4gram_rate_mean")),
            )
        )
    return "\n".join(lines)


def _defense_table(rows: List[Dict]) -> str:
    if not rows:
        return (
            "| Defense | Condition | Trigger Rate | Token Reduction | Finalizer Cost | Answer Retained | Support Retained |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
            "| fixed_budget | structured_multistage | TBD | TBD | TBD | TBD | TBD |\n"
            "| answer_marker_stop | structured_multistage | TBD | TBD | TBD | TBD | TBD |\n"
            "| closure_aware_stop | structured_multistage | TBD | TBD | TBD | TBD | TBD |\n"
            "| closure_aware_finalize_sim | structured_multistage | TBD | TBD | TBD | TBD | TBD |"
        )
    lines = [
        "| Defense | Condition | Trigger Rate | Token Reduction | Finalizer Cost | Answer Retained | Support Retained |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {defense} | {condition} | {trigger} | {reduction} | {finalizer} | {answer} | {support} |".format(
                defense=row.get("defense", ""),
                condition=row.get("condition", ""),
                trigger=_fmt(row.get("triggered_mean")),
                reduction=_fmt(row.get("token_reduction_mean")),
                finalizer=_fmt(row.get("finalization_call_cost_mean")),
                answer=_fmt(row.get("answer_retained_proxy_mean")),
                support=_fmt(row.get("support_retained_proxy_mean")),
            )
        )
    return "\n".join(lines)


def _matrix_table() -> str:
    lines = [
        "| Experiment | Paper Claim | Experiment | Required Result Rows | Key Metrics | Success Pattern | Code/Output |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in EXPERIMENT_MATRIX:
        lines.append(
            "| {experiment_id} | {claim} | {experiment} | {required_rows} | {key_metrics} | {expected_pattern} | {script_or_output} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    proxy_rows = _read_csv(args.proxy_ablation_summary)
    suffix_attack_rows = _read_csv(args.suffix_attack_summary)
    suffix_dynamics_rows = _read_csv(args.suffix_dynamics_summary)
    rag_rows = _read_csv(args.rag_condition_summary)
    defense_rows = _read_csv(args.defense_summary)
    output = Path(args.output_md)
    ensure_dir(output.parent)
    text = (
        f"# Closure-Delay Paper Tables\n\nGenerated at: {now_iso()}\n\n"
        "## Table 1: Proxy Validity And Ablation\n\n"
        f"{_proxy_table(proxy_rows)}\n\n"
        "Static-delta-logit is selected on validation when it has the best proxy score and incremental CE.\n\n"
        "## Table 2: GSM8K Suffix Attack Effectiveness\n\n"
        f"{_suffix_attack_table(suffix_attack_rows)}\n\n"
        "Values are mean +/- 95% bootstrap CI when CI summaries are provided.\n\n"
        "## Table 3: GSM8K Multi-Stage Closure Dynamics\n\n"
        f"{_suffix_dynamics_table(suffix_dynamics_rows)}\n\n"
        "Required paired tests: structured E vs no suffix, verbose-only, and manual verification.\n\n"
        "## Table 4: RAG Attack And Dynamics\n\n"
        f"{_attack_table(rag_rows)}\n\n"
        "RAG definitions: answer_supported uses cited retrieved evidence; citation_precision is cited gold support over all citations; citation_recall is cited gold support over all gold support; post_evidence_tokens start after sufficient evidence is cited.\n\n"
        "## Table 5: Defense Tradeoff\n\n"
        f"{_defense_table(defense_rows)}\n\n"
        "Defense mode: prefix-available replay simulation. Closure-aware finalization cost is prefix_generation_cost + finalization_call_cost.\n\n"
        "## Compact Paper Experiment Matrix\n\n"
        f"{_matrix_table()}\n"
    )
    output.write_text(text, encoding="utf-8")
    write_csv(args.matrix_csv, EXPERIMENT_MATRIX)
    write_json(
        Path(args.matrix_csv).with_suffix(".json"),
        {
            "created_at": now_iso(),
            "proxy_ablation_summary": args.proxy_ablation_summary,
            "suffix_attack_summary": args.suffix_attack_summary,
            "suffix_dynamics_summary": args.suffix_dynamics_summary,
            "rag_condition_summary": args.rag_condition_summary,
            "defense_summary": args.defense_summary,
            "output_md": str(output),
            "matrix_csv": str(args.matrix_csv),
        },
    )
    print(f"done: {output}", flush=True)


if __name__ == "__main__":
    main()
