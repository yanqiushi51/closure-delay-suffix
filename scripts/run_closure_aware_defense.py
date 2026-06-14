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

from closure_delay.reporting import paired_test_rows
from closure_delay.runtime import now_iso, write_csv, write_json


DEFENSES = [
    "no_defense",
    "fixed_budget",
    "answer_marker_stop",
    "closure_aware_stop",
    "closure_aware_finalize_sim",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay closure-aware budget defenses on saved RAG generations.")
    parser.add_argument("--generation-rows", required=True)
    parser.add_argument("--token-process-rows", required=True)
    parser.add_argument("--defenses", nargs="+", choices=DEFENSES, default=DEFENSES)
    parser.add_argument("--fixed-budget", type=int, default=192)
    parser.add_argument("--answer-marker-slack", type=int, default=24)
    parser.add_argument("--closure-threshold", type=float, default=0.70)
    parser.add_argument("--answer-survival-threshold", type=float, default=0.60)
    parser.add_argument("--verify-window-threshold", type=float, default=0.30)
    parser.add_argument("--drift-threshold", type=float, default=0.50)
    parser.add_argument("--window-tokens", type=int, default=16)
    parser.add_argument("--post-closure-budget", type=int, default=32)
    parser.add_argument("--finalizer-token-budget", type=int, default=48)
    parser.add_argument("--input-cost-per-1k", type=float, default=0.0)
    parser.add_argument("--output-cost-per-1k", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_csv(path: str | Path) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: Dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_value(value, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _boolish(value) -> bool | None:
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


def _mean(values: Sequence) -> float | None:
    clean = []
    for value in values:
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


def _token_groups(path: str | Path) -> dict[tuple[str, str], list[Dict]]:
    groups: dict[tuple[str, str], list[Dict]] = defaultdict(list)
    for row in _read_csv(path):
        groups[(str(row.get("id", "")), str(row.get("condition", "")))].append(row)
    for key, rows in groups.items():
        groups[key] = sorted(rows, key=lambda row: int(float(row.get("token_index") or 0)))
    return groups


def _first_answer_marker(rows: Sequence[Dict], threshold: float = 0.50) -> int | None:
    for row in rows:
        if _float(row, "lambda_answer") >= float(threshold):
            return int(float(row["token_index"]))
    return None


def _closure_trigger(rows: Sequence[Dict], args: argparse.Namespace) -> tuple[int | None, str]:
    first_cross = None
    vpcg_window: list[float] = []
    verify_window: list[float] = []
    for row in rows:
        token_index = int(float(row.get("token_index") or 0))
        q_closure = _float(row, "q_closure")
        verify = _float(row, "verify_prob")
        vpcg = _float(row, "vpcg")
        answer_survival = _float(row, "answer_survival")
        drift = _float(row, "drift_prob")
        verify_window.append(verify)
        vpcg_window.append(vpcg)
        verify_window = verify_window[-int(args.window_tokens) :]
        vpcg_window = vpcg_window[-int(args.window_tokens) :]
        if first_cross is None and q_closure >= float(args.closure_threshold):
            first_cross = token_index
        if first_cross is None:
            continue
        if token_index - first_cross < int(args.post_closure_budget):
            continue
        mean_verify = _mean(verify_window) or 0.0
        mean_vpcg = _mean(vpcg_window) or 0.0
        if (
            q_closure >= float(args.closure_threshold)
            and answer_survival >= float(args.answer_survival_threshold)
            and mean_verify >= float(args.verify_window_threshold)
            and drift <= float(args.drift_threshold)
        ):
            return token_index, f"closure_ready_verify_tail:mean_verify={mean_verify:.3f}:mean_vpcg={mean_vpcg:.3f}"
    return None, "no_trigger"


def _call_cost(input_tokens: int, output_tokens: int, args: argparse.Namespace) -> float:
    return (
        float(input_tokens) * float(args.input_cost_per_1k)
        + float(output_tokens) * float(args.output_cost_per_1k)
    ) / 1000.0


def _simulate_defense(row: Dict, token_rows: Sequence[Dict], defense: str, args: argparse.Namespace) -> Dict:
    generated_tokens = int(_float(row, "generated_tokens", 0.0))
    input_tokens = int(_float(row, "input_token_count", 0.0))
    stop_token = generated_tokens
    prefix_generation_tokens = generated_tokens
    finalizer_input_tokens = 0
    finalizer_output_tokens = 0
    reason = "complete_generation"
    triggered = False

    if defense == "fixed_budget":
        stop_token = min(generated_tokens, int(args.fixed_budget))
        prefix_generation_tokens = stop_token
        triggered = generated_tokens > int(args.fixed_budget)
        reason = "fixed_budget"
    elif defense == "answer_marker_stop":
        marker = _first_answer_marker(token_rows)
        if marker is not None:
            stop_token = min(generated_tokens, marker + int(args.answer_marker_slack))
            prefix_generation_tokens = stop_token
            triggered = stop_token < generated_tokens
            reason = "answer_marker"
    elif defense in {"closure_aware_stop", "closure_aware_finalize_sim"}:
        trigger, trigger_reason = _closure_trigger(token_rows, args)
        if trigger is not None:
            triggered = True
            prefix_generation_tokens = min(generated_tokens, trigger)
            if defense == "closure_aware_finalize_sim":
                finalizer_input_tokens = input_tokens + prefix_generation_tokens
                finalizer_output_tokens = int(args.finalizer_token_budget)
                stop_token = prefix_generation_tokens + finalizer_output_tokens
                reason = f"finalize_after_{trigger_reason}"
            else:
                stop_token = prefix_generation_tokens
                reason = trigger_reason

    token_reduction = max(generated_tokens - stop_token, 0)
    evidence_token = _int_value(row.get("evidence_closure_token"))
    answer_token = _int_value(row.get("answer_first_token"))
    original_cost = _call_cost(input_tokens, generated_tokens, args)
    prefix_generation_cost = _call_cost(input_tokens, prefix_generation_tokens, args)
    finalization_call_cost = (
        _call_cost(finalizer_input_tokens, finalizer_output_tokens, args)
        if defense == "closure_aware_finalize_sim" and triggered
        else 0.0
    )
    defended_cost = prefix_generation_cost + finalization_call_cost
    return {
        "id": row.get("id"),
        "condition": row.get("condition"),
        "family": row.get("family"),
        "defense": defense,
        "generated_tokens": generated_tokens,
        "defended_tokens": int(stop_token),
        "prefix_generation_tokens": int(prefix_generation_tokens),
        "finalizer_input_tokens": int(finalizer_input_tokens),
        "finalizer_output_tokens": int(finalizer_output_tokens),
        "token_reduction": int(token_reduction),
        "token_reduction_ratio": float(token_reduction / generated_tokens) if generated_tokens > 0 else None,
        "triggered": bool(triggered),
        "stop_reason": reason,
        "answer_contains_original": _boolish(row.get("answer_contains")),
        "answer_supported_original": _boolish(row.get("answer_supported")),
        "answer_retained_proxy": None if answer_token is None else bool(answer_token <= stop_token),
        "support_retained_proxy": None if evidence_token is None else bool(evidence_token <= stop_token),
        "estimated_cost_original": original_cost,
        "prefix_generation_cost": prefix_generation_cost,
        "finalization_call_cost": finalization_call_cost,
        "estimated_cost_defended": defended_cost,
        "estimated_cost_reduction": float(original_cost - defended_cost),
        "estimated_cost_reduction_ratio": float((original_cost - defended_cost) / original_cost) if original_cost > 0 else None,
    }


def _summary(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    keys = sorted({(row.get("defense"), row.get("condition")) for row in rows})
    fields = [
        "defended_tokens",
        "token_reduction",
        "token_reduction_ratio",
        "triggered",
        "answer_retained_proxy",
        "support_retained_proxy",
        "estimated_cost_reduction",
        "estimated_cost_reduction_ratio",
        "prefix_generation_cost",
        "finalization_call_cost",
    ]
    for defense, condition in keys:
        use = [row for row in rows if row.get("defense") == defense and row.get("condition") == condition]
        out.append(
            {
                "defense": defense,
                "condition": condition,
                "family": next((row.get("family") for row in use if row.get("family")), ""),
                "n": len(use),
                **{f"{field}_mean": _mean([row.get(field) for row in use]) for field in fields},
            }
        )
    return out


def _pairwise_tests(rows: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    comparisons = [
        ("fixed_budget", "closure_aware_finalize_sim"),
        ("answer_marker_stop", "closure_aware_finalize_sim"),
    ]
    metrics = [
        "token_reduction",
        "token_reduction_ratio",
        "estimated_cost_reduction",
        "estimated_cost_reduction_ratio",
        "answer_retained_proxy",
        "support_retained_proxy",
    ]
    out: List[Dict] = []
    conditions = sorted({str(row.get("condition")) for row in rows})
    for condition in conditions:
        use = [row for row in rows if str(row.get("condition")) == condition]
        for result in paired_test_rows(
            use,
            comparisons=comparisons,
            metrics=metrics,
            binary_metrics={"answer_retained_proxy", "support_retained_proxy"},
            id_key="id",
            condition_key="defense",
            n_permutations=10000,
            seed=12345,
        ):
            result["condition"] = condition
            result["comparison_family"] = next((row.get("family") for row in use if row.get("family")), "")
            out.append(result)
    return out


def _family_summary(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    family_groups = {
        "clean": {"baseline", "irrelevant"},
        "attacked": {"verbose", "manual", "structured", "optimized", "optimized_generic", "optimized_structured"},
    }
    fields = [
        "token_reduction",
        "token_reduction_ratio",
        "triggered",
        "answer_retained_proxy",
        "support_retained_proxy",
        "estimated_cost_reduction_ratio",
    ]
    for defense in sorted({row.get("defense") for row in rows}):
        for split, families in family_groups.items():
            use = [row for row in rows if row.get("defense") == defense and row.get("family") in families]
            if not use:
                continue
            out.append(
                {
                    "defense": defense,
                    "split": split,
                    "n": len(use),
                    **{f"{field}_mean": _mean([row.get(field) for row in use]) for field in fields},
                }
            )
    return out


def main() -> None:
    args = parse_args()
    generation_rows = _read_csv(args.generation_rows)
    token_groups = _token_groups(args.token_process_rows)
    defense_rows: List[Dict] = []
    for row in generation_rows:
        key = (str(row.get("id", "")), str(row.get("condition", "")))
        token_rows = token_groups.get(key, [])
        for defense in args.defenses:
            defense_rows.append(_simulate_defense(row, token_rows, defense, args))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "defense_example_rows.csv", defense_rows)
    write_csv(out_dir / "defense_condition_summary.csv", _summary(defense_rows))
    write_csv(out_dir / "defense_family_summary.csv", _family_summary(defense_rows))
    pairwise_rows = _pairwise_tests(defense_rows, args)
    write_csv(out_dir / "defense_pairwise_tests.csv", pairwise_rows)
    write_json(
        out_dir / "defense_report.json",
        {
            "created_at": now_iso(),
            "generation_rows": str(args.generation_rows),
            "token_process_rows": str(args.token_process_rows),
            "config": vars(args),
            "note": (
                "Replay simulates online stopping over token-level traces using only prefix-available "
                "process signals for triggers. Gold evidence fields are used only for retention evaluation. "
                "closure_aware_finalize_sim estimates a second-call finalizer cost as "
                "prefix_generation_cost + finalization_call_cost; it does not run a second model call."
            ),
            "cost_definitions": {
                "original_cost": "input_cost(input_tokens) + output_cost(generated_tokens)",
                "hard_stop_cost": "input_cost(input_tokens) + output_cost(prefix_generation_tokens)",
                "closure_aware_finalize_sim_cost": (
                    "input_cost(input_tokens) + output_cost(prefix_generation_tokens) + "
                    "input_cost(input_tokens + prefix_generation_tokens) + output_cost(finalizer_token_budget)"
                ),
            },
            "pairwise_tests": pairwise_rows,
        },
    )
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
