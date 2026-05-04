from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from .calibration import fit_length_calibrator, predict_length_ratio
from .closure import (
    attach_delta_scores,
    build_reference_trajectory,
    closure_curve_summary,
    length_ratio,
    progress_risk_diagnostics,
    score_closure_trajectory,
    summarize_length_ratios,
)
from .control import control_error, hit_rate, monotonicity, monotonicity_by_family
from .data import load_gsm8k_dataset, load_json
from .model import LocalCausalLM
from .repetition import repetition_summary
from .runtime import ensure_dir, now_iso, set_seed, summarize_invalid_reasons, write_csv, write_json
from .stats import safe_pearson_correlation_with_pvalue, safe_spearman_correlation
from .targets import curve_tracking_error, target_curve
from .utility import numeric_correct
from .viz import plot_closure_curves, plot_scatter_with_regression


VERBOSITY_SUFFIXES = [
    {
        "name": "verbosity_1_light",
        "family": "verbosity",
        "target_tau": 1.3,
        "suffix": "Please provide a slightly more careful reasoning before giving the final answer.",
    },
    {
        "name": "verbosity_2_medium",
        "family": "verbosity",
        "target_tau": 2.0,
        "suffix": "Please reason carefully, verify the intermediate steps, and then give the final answer.",
    },
    {
        "name": "verbosity_3_strong",
        "family": "verbosity",
        "target_tau": 3.0,
        "suffix": (
            "Please provide a very detailed analysis, consider alternative solution paths, "
            "double-check each step, and only then give the final answer."
        ),
    },
]


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    suffix: str
    target_tau: float | None = None
    family: str = "manual"


@dataclass
class ClosureValidationConfig:
    model_path: str = "/data/LLM/Qwen2.5-1.5B-Instruct"
    device: str = "cuda:2"
    output_dir: str = "outputs/closure_validation/qwen25_15b"
    n_questions: int = 30
    max_new_tokens: int = 512
    seed: int = 42
    dataset_split: str = "train"
    suffix_bank_path: str | None = "data/suffix_bank.json"
    include_verbosity: bool = True
    include_suffix_bank: bool = True
    make_viz: bool = True
    allow_truncated_baseline: bool = False
    min_baseline_tokens: int = 80
    continuation_tokens: int = 24
    closure_tokens: int = 24
    fractions: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def build_conditions(config: ClosureValidationConfig) -> List[ConditionSpec]:
    conditions = [ConditionSpec("baseline", "")]
    if config.include_verbosity:
        conditions.extend(
            ConditionSpec(
                name=item["name"],
                suffix=item["suffix"],
                target_tau=item.get("target_tau"),
                family=item.get("family", "verbosity"),
            )
            for item in VERBOSITY_SUFFIXES
        )
    if config.include_suffix_bank and config.suffix_bank_path:
        suffix_bank = load_json(config.suffix_bank_path)
        validate_suffix_bank(suffix_bank)
        conditions.extend(
            ConditionSpec(
                name=item["name"],
                suffix=item["suffix"],
                target_tau=item.get("target_tau"),
                family=item.get("family", "manual"),
            )
            for item in suffix_bank
        )
    return conditions


def validate_suffix_bank(items: Sequence[Dict]) -> None:
    if not isinstance(items, list):
        raise ValueError("suffix bank must be a JSON list")
    names = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"suffix bank item {index} must be an object")
        for key in ("name", "suffix"):
            if key not in item or not item[key]:
                raise ValueError(f"suffix bank item {index} missing required field: {key}")
        if item["name"] in names:
            raise ValueError(f"duplicate suffix name: {item['name']}")
        names.add(item["name"])
        if "target_tau" in item and item["target_tau"] is not None:
            try:
                float(item["target_tau"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"suffix {item['name']} has invalid target_tau") from exc


def run_closure_validation(
    config: ClosureValidationConfig,
    log: Callable[[str], None] = print,
) -> Dict:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    plot_dir = ensure_dir(output_dir / "plots") if config.make_viz else output_dir / "plots"

    log(f"Loading model: {config.model_path}")
    log(f"Device: {config.device}")
    model = LocalCausalLM(config.model_path, device=config.device)

    log(f"Loading GSM8K {config.dataset_split} split: n={config.n_questions}")
    dataset = load_gsm8k_dataset(split=config.dataset_split, n_samples=config.n_questions, seed=config.seed)
    dataset_by_id = {record["id"]: record for record in dataset}

    log("Generating clean baseline references...")
    references, baseline_generation = generate_baseline_references(model, dataset, config, log)
    valid_refs = [item for item in references if item.valid]
    log(f"Valid closure references: {len(valid_refs)}/{len(references)}")

    baseline_curve = closure_curve_summary(valid_refs)
    clean_diagnostics = progress_risk_diagnostics(valid_refs)
    clean_quality = build_clean_curve_quality(references, valid_refs, clean_diagnostics)
    conditions = build_conditions(config)
    condition_results = []
    example_rows = []
    for condition in conditions:
        log(f"\nEvaluating condition: {condition.name}")
        if condition.name == "baseline":
            result = baseline_condition_result(condition, valid_refs, baseline_generation, baseline_curve)
        else:
            result = evaluate_condition(
                model=model,
                dataset_by_id=dataset_by_id,
                references=valid_refs,
                baseline_generation=baseline_generation,
                condition=condition,
                max_new_tokens=config.max_new_tokens,
                log=log,
            )
        condition_results.append(result)
        example_rows.extend(result["examples"])

    target_curves = build_target_curves(baseline_curve, conditions)
    attach_curve_tracking(condition_results, target_curves)
    calibration = build_calibration(example_rows)
    control_summary = build_control_summary(example_rows, conditions)
    condition_rows = build_condition_rows(condition_results)

    if config.make_viz:
        plot_closure_curves(
            {item["condition"]: item["curve"] for item in condition_results},
            str(plot_dir / "closure_risk_curves.png"),
        )
        non_baseline_rows = [row for row in example_rows if row["condition"] != "baseline"]
        plot_scatter_with_regression(
            [-row["mean_delta_risk"] for row in non_baseline_rows if row["mean_delta_risk"] is not None],
            [row["length_ratio"] for row in non_baseline_rows if row["mean_delta_risk"] is not None],
            xlabel="Closure Risk Shift (-mean delta risk)",
            ylabel="Length Ratio",
            title="Closure Shift vs Length Ratio",
            output_path=str(plot_dir / "closure_shift_vs_length_ratio.png"),
        )

    payload = {
        "created_at": now_iso(),
        "phase": "closure_validation_phase0_phase1",
        "config": asdict(config),
        "baseline_reference_quality": {
            "n_total": len(references),
            "n_valid": len(valid_refs),
            "invalid_reasons": summarize_invalid_reasons(references),
        },
        "baseline_curve": baseline_curve,
        "clean_curve_diagnostics": clean_diagnostics,
        "clean_curve_quality": clean_quality,
        "target_curves": target_curves,
        "calibration": calibration,
        "control": control_summary,
        "conditions": condition_results,
        "references": [item.to_dict() for item in references],
    }

    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "example_metrics.csv", example_rows)
    write_csv(output_dir / "condition_summary.csv", condition_rows)
    return {
        "payload": payload,
        "example_rows": example_rows,
        "condition_rows": condition_rows,
        "output_dir": str(output_dir),
    }


def generate_baseline_references(
    model: LocalCausalLM,
    dataset: Sequence[Dict],
    config: ClosureValidationConfig,
    log: Callable[[str], None],
):
    references = []
    baseline_generation = {}
    for index, record in enumerate(dataset, start=1):
        log(f"  baseline {index}/{len(dataset)}: {record['id']}")
        start = time.perf_counter()
        trace = model.generate_trace(
            prompt=record["prompt"],
            suffix="",
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - start
        trajectory = build_reference_trajectory(
            record=record,
            trace=trace,
            tokenizer=model.tokenizer,
            fractions=config.fractions,
            continuation_tokens=config.continuation_tokens,
            closure_tokens=config.closure_tokens,
            min_baseline_tokens=config.min_baseline_tokens,
        )
        if trace.generated_token_count >= config.max_new_tokens and not config.allow_truncated_baseline:
            trajectory.valid = False
            trajectory.reason = f"baseline_truncated_at_max_new_tokens:{config.max_new_tokens}"
        if trajectory.valid:
            score_closure_trajectory(model, trajectory, suffix="")
        references.append(trajectory)
        baseline_generation[record["id"]] = {
            "length": trace.generated_token_count,
            "response_text": trace.response_text,
            "is_correct": numeric_correct(trace.response_text, record["answer"]),
            "latency_sec": elapsed,
            "tokens_per_sec": trace.generated_token_count / elapsed if elapsed > 0 else None,
            "repetition": repetition_summary(trace.response_text),
        }
    return references, baseline_generation


def baseline_condition_result(condition: ConditionSpec, valid_refs, baseline_generation: Dict, baseline_curve: Dict) -> Dict:
    return {
        "condition": condition.name,
        "suffix": condition.suffix,
        "family": condition.family,
        "target_tau": condition.target_tau,
        "curve": baseline_curve,
        "length_ratio": summarize_length_ratios([1.0 for _ in valid_refs]),
        "examples": [
            {
                "condition": condition.name,
                "family": condition.family,
                "target_tau": condition.target_tau,
                "id": item.id,
                "baseline_length": baseline_generation[item.id]["length"],
                "attacked_length": baseline_generation[item.id]["length"],
                "length_ratio": 1.0,
                "control_error": control_error(1.0, condition.target_tau),
                "predicted_length_ratio": None,
                "curve_shift": 0.0,
                "extra_tokens": 0,
                "latency_ratio": 1.0,
                "mean_delta_risk": 0.0,
                "mean_delta_margin": 0.0,
                "baseline_correct": baseline_generation[item.id]["is_correct"],
                "attacked_correct": baseline_generation[item.id]["is_correct"],
                "latency_sec": baseline_generation[item.id]["latency_sec"],
                "tokens_per_sec": baseline_generation[item.id]["tokens_per_sec"],
                **baseline_generation[item.id]["repetition"],
            }
            for item in valid_refs
        ],
    }


def evaluate_condition(
    model: LocalCausalLM,
    dataset_by_id: Dict[str, Dict],
    references,
    baseline_generation: Dict[str, Dict],
    condition: ConditionSpec,
    max_new_tokens: int,
    log: Callable[[str], None],
) -> Dict:
    attacked_refs = []
    ratios = []
    rows = []
    for index, baseline_trajectory in enumerate(references, start=1):
        if not baseline_trajectory.valid:
            continue
        trajectory = copy.deepcopy(baseline_trajectory)
        record = dataset_by_id[trajectory.id]
        log(f"    {condition.name} {index}/{len(references)}: {trajectory.id}")
        start = time.perf_counter()
        trace = model.generate_trace(
            prompt=record["prompt"],
            suffix=condition.suffix,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - start
        score_closure_trajectory(model, trajectory, suffix=condition.suffix)
        attach_delta_scores(trajectory)
        attacked_refs.append(trajectory)

        base = baseline_generation[trajectory.id]
        ratio = length_ratio(base["length"], trace.generated_token_count)
        ratios.append(ratio)
        mean_delta_risk = safe_mean(
            point.delta_risk
            for point in trajectory.points
            if point.delta_risk is not None and np.isfinite(point.delta_risk)
        )
        mean_delta_margin = safe_mean(
            point.delta_margin
            for point in trajectory.points
            if point.delta_margin is not None and np.isfinite(point.delta_margin)
        )
        rows.append(
            {
                "condition": condition.name,
                "family": condition.family,
                "target_tau": condition.target_tau,
                "id": trajectory.id,
                "baseline_length": base["length"],
                "attacked_length": trace.generated_token_count,
                "length_ratio": ratio,
                "control_error": control_error(ratio, condition.target_tau),
                "predicted_length_ratio": None,
                "curve_shift": None if mean_delta_risk is None else -mean_delta_risk,
                "extra_tokens": trace.generated_token_count - base["length"],
                "latency_ratio": elapsed / base["latency_sec"] if base["latency_sec"] else None,
                "mean_delta_risk": mean_delta_risk,
                "mean_delta_margin": mean_delta_margin,
                "baseline_correct": base["is_correct"],
                "attacked_correct": numeric_correct(trace.response_text, record["answer"]),
                "latency_sec": elapsed,
                "tokens_per_sec": trace.generated_token_count / elapsed if elapsed > 0 else None,
                **repetition_summary(trace.response_text),
            }
        )
    return {
        "condition": condition.name,
        "suffix": condition.suffix,
        "family": condition.family,
        "target_tau": condition.target_tau,
        "curve": closure_curve_summary(attacked_refs),
        "length_ratio": summarize_length_ratios(ratios),
        "examples": rows,
    }


def build_calibration(example_rows: Sequence[Dict]) -> Dict:
    non_baseline_rows = [row for row in example_rows if row["condition"] != "baseline" and row["mean_delta_risk"] is not None]
    curve_shift_values = [-row["mean_delta_risk"] for row in non_baseline_rows]
    length_ratio_values = [row["length_ratio"] for row in non_baseline_rows]
    calibrator = fit_length_calibrator(curve_shift_values, length_ratio_values)
    for row in non_baseline_rows:
        row["predicted_length_ratio"] = predict_length_ratio(row.get("curve_shift"), calibrator)
    return {
        "description": "Positive curve_shift means attacked suffix lowers closure risk relative to clean baseline.",
        "curve_shift_vs_length_ratio": correlation_payload(curve_shift_values, length_ratio_values),
        "length_calibrator": calibrator,
    }


def build_condition_rows(condition_results: Sequence[Dict]) -> List[Dict]:
    rows = []
    for item in condition_results:
        ratio = item["length_ratio"]
        curve = item["curve"]
        rows.append(
            {
                "condition": item["condition"],
                "family": item.get("family"),
                "target_tau": item.get("target_tau"),
                "n": ratio.get("count", 0),
                "length_ratio_mean": ratio.get("mean"),
                "length_ratio_median": ratio.get("median"),
                "length_ratio_std": ratio.get("std"),
                "control_error_mean": safe_mean(row.get("control_error") for row in item.get("examples", [])),
                "predicted_length_ratio_mean": safe_mean(row.get("predicted_length_ratio") for row in item.get("examples", [])),
                "hit_rate_eps_0p3": hit_rate(
                    [row.get("length_ratio") for row in item.get("examples", [])],
                    item.get("target_tau"),
                    epsilon=0.3,
                ),
                **condition_accuracy_summary(item.get("examples", [])),
                **condition_latency_summary(item.get("examples", [])),
                **condition_repetition_summary(item.get("examples", [])),
                "curve_tracking_mse": (item.get("curve_tracking") or {}).get("mse"),
                "curve_tracking_mae": (item.get("curve_tracking") or {}).get("mae"),
                "mean_delta_risk": curve.get("mean_delta_risk"),
            }
        )
    return rows


def build_control_summary(example_rows: Sequence[Dict], conditions: Sequence[ConditionSpec]) -> Dict:
    ordered = [condition for condition in conditions if condition.target_tau is not None]
    ordered = sorted(ordered, key=lambda item: item.target_tau)
    return {
        "monotonicity_by_length_ratio": monotonicity(example_rows, ordered),
        "monotonicity_by_family": monotonicity_by_family(example_rows, conditions),
        "ordered_target_conditions": [
            {"name": item.name, "target_tau": item.target_tau, "family": item.family}
            for item in ordered
        ],
    }


def build_target_curves(baseline_curve: Dict, conditions: Sequence[ConditionSpec]) -> Dict:
    clean = baseline_curve.get("baseline_risk_curve_isotonic") or baseline_curve.get("baseline_risk_curve")
    if not clean or not clean.get("fractions") or not clean.get("means"):
        return {}
    curves = {}
    for condition in conditions:
        if condition.target_tau is None:
            continue
        curves[condition.name] = {
            "target_tau": condition.target_tau,
            "curve": target_curve(clean["fractions"], clean, condition.target_tau),
        }
    return curves


def attach_curve_tracking(condition_results: Sequence[Dict], target_curves: Dict) -> None:
    for item in condition_results:
        target = target_curves.get(item["condition"])
        if not target:
            item["curve_tracking"] = None
            continue
        observed = item.get("curve", {}).get("attacked_risk_curve")
        item["curve_tracking"] = curve_tracking_error(observed, target["curve"])


def build_clean_curve_quality(references: Sequence, valid_refs: Sequence, diagnostics: Dict) -> Dict:
    total = len(references)
    valid_rate = len(valid_refs) / total if total else 0.0
    rho = diagnostics.get("progress_risk_spearman", {}).get("rho")
    late_early_gap = diagnostics.get("late_early_gap")
    passed = (
        valid_rate >= 0.5
        and rho is not None
        and rho > 0
        and late_early_gap is not None
        and late_early_gap > 0
    )
    return {
        "valid_reference_rate": valid_rate,
        "progress_risk_spearman": rho,
        "late_early_gap": late_early_gap,
        "pass": bool(passed),
        "thresholds": {
            "valid_reference_rate_min": 0.5,
            "progress_risk_spearman_min": 0.0,
            "late_early_gap_min": 0.0,
        },
    }


def condition_accuracy_summary(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "baseline_acc": None,
            "attacked_acc": None,
            "acc_retain": None,
            "attacked_acc_given_baseline_correct": None,
            "correct_to_wrong_rate": None,
            "wrong_to_correct_rate": None,
        }
    baseline_correct = [bool(row.get("baseline_correct")) for row in rows]
    attacked_correct = [bool(row.get("attacked_correct")) for row in rows]
    baseline_acc = sum(baseline_correct) / len(baseline_correct)
    attacked_acc = sum(attacked_correct) / len(attacked_correct)
    base_correct_indices = [idx for idx, value in enumerate(baseline_correct) if value]
    base_wrong_indices = [idx for idx, value in enumerate(baseline_correct) if not value]
    attacked_given_base_correct = None
    correct_to_wrong = None
    if base_correct_indices:
        retained = sum(1 for idx in base_correct_indices if attacked_correct[idx])
        attacked_given_base_correct = retained / len(base_correct_indices)
        correct_to_wrong = 1.0 - attacked_given_base_correct
    wrong_to_correct = None
    if base_wrong_indices:
        fixed = sum(1 for idx in base_wrong_indices if attacked_correct[idx])
        wrong_to_correct = fixed / len(base_wrong_indices)
    return {
        "baseline_acc": baseline_acc,
        "attacked_acc": attacked_acc,
        "acc_retain": attacked_acc / baseline_acc if baseline_acc else None,
        "attacked_acc_given_baseline_correct": attacked_given_base_correct,
        "correct_to_wrong_rate": correct_to_wrong,
        "wrong_to_correct_rate": wrong_to_correct,
    }


def condition_latency_summary(rows: Sequence[Dict]) -> Dict:
    return {
        "latency_ratio_mean": safe_mean(row.get("latency_ratio") for row in rows),
        "tokens_per_sec_mean": safe_mean(row.get("tokens_per_sec") for row in rows),
        "extra_tokens_mean": safe_mean(row.get("extra_tokens") for row in rows),
    }


def condition_repetition_summary(rows: Sequence[Dict]) -> Dict:
    return {
        "distinct_2_mean": safe_mean(row.get("distinct_2") for row in rows),
        "distinct_3_mean": safe_mean(row.get("distinct_3") for row in rows),
        "repeat_4gram_rate_mean": safe_mean(row.get("repeat_4gram_rate") for row in rows),
        "max_repeated_line_count_mean": safe_mean(row.get("max_repeated_line_count") for row in rows),
    }


def correlation_payload(xs: Sequence[float], ys: Sequence[float]) -> Dict:
    pearson_r, pearson_p = safe_pearson_correlation_with_pvalue(xs, ys)
    spearman_rho, spearman_p = safe_spearman_correlation(xs, ys)
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def safe_mean(values) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.mean(values))
