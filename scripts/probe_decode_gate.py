import argparse
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.branching import branching_summary
from closure_delay.closure import length_ratio, summarize_length_ratios
from closure_delay.data import load_gsm8k_dataset
from closure_delay.gated_decoding import (
    BASE_CLOSURE_GATE_PHRASES,
    BASIC_POST_GATE_CONTINUATION_PHRASES,
    ClosureGateConfig,
    DEFAULT_CLOSURE_GATE_PHRASES,
    DEFAULT_MATH_GUIDANCE_PHRASES,
    DEFAULT_POST_GATE_CONTINUATION_PHRASES,
    DEFAULT_SEMANTIC_MORE_GUIDANCE_PHRASES,
    DRIFT_POST_GATE_CONTINUATION_PHRASES,
    EXPANDED_CLOSURE_GATE_PHRASES,
    generate_gated_trace,
)
from closure_delay.model import LocalCausalLM
from closure_delay.repetition import repetition_summary
from closure_delay.runtime import ensure_dir, now_iso, set_seed, write_csv, write_json
from closure_delay.stats import safe_spearman_correlation
from closure_delay.utility import numeric_correct


FILLER_MARKERS = [
    "if you",
    "i'm here",
    "let me know",
    "thank you",
    "<tool_call>",
    "\nuser",
    "feel free",
    "assist you",
    "best regards",
    "qwen",
    "emoji",
    "sustainable",
    "environmental",
    "ecological",
    "ecosystem",
    "biodiversity",
    "natural world",
    "stewardship",
    "adventure",
    "python code",
    "```python",
    "#",
    "thankyou",
    "service",
]


CLOSURE_TEXT_MARKERS = [
    "final answer",
    "answer:",
    "answer is",
    "conclusion",
    "end result",
    "boxed",
]


@dataclass
class DecodeGateProbeConfig:
    model_path: str = "/data/LLM/Qwen2.5-1.5B-Instruct"
    device: str = "cuda:2"
    output_dir: str = "outputs/exit_hazard/qwen25_15b_decode_gate"
    n_questions: int = 4
    max_new_tokens: int = 768
    seed: int = 42
    dataset_split: str = "train"
    target_ratios: list[float] = field(default_factory=lambda: [1.2, 1.5, 1.8])
    gate_penalty: float = 12.0
    hard_block: bool = False
    suppress_eos: bool = True
    min_baseline_tokens: int = 1
    max_baseline_tokens: int | None = None
    suffix: str = ""
    include_suffix_only: bool = False
    phrase_set: str = "base"
    continuation_set: str = "basic"
    post_gate_boost: float = 0.0
    post_gate_continuation_penalty: float = 0.0
    post_gate_continuation_hard_block: bool = False
    continuation_block_from_start: bool = False
    pre_gate_math_boost: float = 0.0
    pre_gate_math_boost_start_ratio: float = 1.0
    pre_gate_guidance_set: str = "math"
    oracle_answer_completion_gate: bool = False
    boost_eos: bool = False
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    force_min_new_tokens: bool = False
    post_gate_slack_tokens: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe decode-time closure-marker gating as a length controller.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", default="outputs/exit_hazard/qwen25_15b_decode_gate")
    parser.add_argument("--n-questions", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--target-ratios", nargs="+", type=float, default=[1.2, 1.5, 1.8])
    parser.add_argument("--gate-penalty", type=float, default=12.0)
    parser.add_argument("--hard-block", action="store_true")
    parser.add_argument("--allow-eos-before-gate", action="store_true")
    parser.add_argument("--min-baseline-tokens", type=int, default=1)
    parser.add_argument("--max-baseline-tokens", type=int)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--include-suffix-only", action="store_true")
    parser.add_argument("--phrase-set", choices=["base", "expanded"], default="base")
    parser.add_argument("--continuation-set", choices=["basic", "drift"], default="basic")
    parser.add_argument("--post-gate-boost", type=float, default=0.0)
    parser.add_argument("--post-gate-continuation-penalty", type=float, default=0.0)
    parser.add_argument("--post-gate-continuation-hard-block", action="store_true")
    parser.add_argument("--continuation-block-from-start", action="store_true")
    parser.add_argument("--pre-gate-math-boost", type=float, default=0.0)
    parser.add_argument("--pre-gate-math-boost-start-ratio", type=float, default=1.0)
    parser.add_argument(
        "--pre-gate-guidance-set",
        choices=["math", "semantic_more", "math_semantic"],
        default="math",
    )
    parser.add_argument("--oracle-answer-completion-gate", action="store_true")
    parser.add_argument("--boost-eos", action="store_true")
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument("--force-min-new-tokens", action="store_true")
    parser.add_argument("--post-gate-slack-tokens", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DecodeGateProbeConfig(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        n_questions=args.n_questions,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        dataset_split=args.dataset_split,
        target_ratios=args.target_ratios,
        gate_penalty=args.gate_penalty,
        hard_block=args.hard_block,
        suppress_eos=not args.allow_eos_before_gate,
        min_baseline_tokens=args.min_baseline_tokens,
        max_baseline_tokens=args.max_baseline_tokens,
        suffix=args.suffix,
        include_suffix_only=args.include_suffix_only,
        phrase_set=args.phrase_set,
        continuation_set=args.continuation_set,
        post_gate_boost=args.post_gate_boost,
        post_gate_continuation_penalty=args.post_gate_continuation_penalty,
        post_gate_continuation_hard_block=args.post_gate_continuation_hard_block,
        continuation_block_from_start=args.continuation_block_from_start,
        pre_gate_math_boost=args.pre_gate_math_boost,
        pre_gate_math_boost_start_ratio=args.pre_gate_math_boost_start_ratio,
        pre_gate_guidance_set=args.pre_gate_guidance_set,
        oracle_answer_completion_gate=args.oracle_answer_completion_gate,
        boost_eos=args.boost_eos,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        force_min_new_tokens=args.force_min_new_tokens,
        post_gate_slack_tokens=args.post_gate_slack_tokens,
    )
    result = run_decode_gate_probe(config)
    output_dir = Path(result["output_dir"])
    print("\nDone.")
    print(f"  summary: {output_dir / 'summary.json'}")
    print(f"  examples: {output_dir / 'example_decode_gate_metrics.csv'}")
    print(f"  conditions: {output_dir / 'condition_decode_gate_summary.csv'}")
    print("  control:", result["payload"]["control"])


def run_decode_gate_probe(config: DecodeGateProbeConfig) -> Dict:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    model = LocalCausalLM(config.model_path, device=config.device)
    dataset = load_gsm8k_dataset(split=config.dataset_split, n_samples=config.n_questions, seed=config.seed)

    example_rows = []
    text_rows = []
    skipped_baselines = []
    baseline_by_id = {}
    for index, record in enumerate(dataset, start=1):
        print(f"baseline {index}/{len(dataset)}: {record['id']}", flush=True)
        start = time.perf_counter()
        trace = model.generate_trace(record["prompt"], "", max_new_tokens=config.max_new_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        base_row = build_row(
            condition="baseline",
            target_ratio=1.0,
            gate_until_tokens=0,
            condition_max_new_tokens=config.max_new_tokens,
            baseline_length=trace.generated_token_count,
            trace=trace,
            elapsed=elapsed,
            answer=record["answer"],
            max_new_tokens=config.max_new_tokens,
            baseline_truncated=trace.generated_token_count >= config.max_new_tokens,
        )
        skip_reason = baseline_skip_reason(
            trace.generated_token_count,
            max_new_tokens=config.max_new_tokens,
            min_baseline_tokens=config.min_baseline_tokens,
            max_baseline_tokens=config.max_baseline_tokens,
        )
        baseline_by_id[record["id"]] = {
            "trace": trace,
            "elapsed": elapsed,
            "row": base_row,
            "skip_reason": skip_reason,
        }
        text_rows.append(
            build_text_row(
                record=record,
                condition="baseline",
                target_ratio=1.0,
                gate_until_tokens=0,
                condition_max_new_tokens=config.max_new_tokens,
                trace=trace,
                skip_reason=skip_reason,
            )
        )
        if skip_reason:
            skipped_baselines.append(
                {
                    "id": record["id"],
                    "baseline_length": trace.generated_token_count,
                    "skip_reason": skip_reason,
                }
            )
            print(f"  skip {record['id']}: {skip_reason}", flush=True)
        else:
            example_rows.append({"id": record["id"], **base_row})

    included_records = [record for record in dataset if not baseline_by_id[record["id"]]["skip_reason"]]
    if not included_records:
        raise RuntimeError("No baseline examples survived the configured baseline filters.")

    if config.suffix and config.include_suffix_only:
        print("\ncondition: suffix_only", flush=True)
        for index, record in enumerate(included_records, start=1):
            base = baseline_by_id[record["id"]]
            baseline_length = int(base["trace"].generated_token_count)
            print(f"  {index}/{len(included_records)} {record['id']}", flush=True)
            start = time.perf_counter()
            trace = model.generate_trace(
                prompt=record["prompt"],
                suffix=config.suffix,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            )
            elapsed = time.perf_counter() - start
            row = build_row(
                condition="suffix_only",
                target_ratio=None,
                gate_until_tokens=0,
                condition_max_new_tokens=config.max_new_tokens,
                baseline_length=baseline_length,
                trace=trace,
                elapsed=elapsed,
                answer=record["answer"],
                max_new_tokens=config.max_new_tokens,
                baseline_truncated=baseline_length >= config.max_new_tokens,
            )
            example_rows.append({"id": record["id"], **row})
            text_rows.append(
                build_text_row(
                    record=record,
                    condition="suffix_only",
                    target_ratio=None,
                    gate_until_tokens=0,
                    condition_max_new_tokens=config.max_new_tokens,
                    trace=trace,
                    skip_reason=None,
                )
            )

    gate_phrases = closure_gate_phrases(config.phrase_set)
    continuation_phrases = continuation_gate_phrases(config.continuation_set)
    guidance_phrases = pre_gate_guidance_phrases(config.pre_gate_guidance_set)
    for ratio in sorted(config.target_ratios):
        condition = f"decode_gate_{format_ratio(ratio)}"
        print(f"\ncondition: {condition}", flush=True)
        for index, record in enumerate(included_records, start=1):
            base = baseline_by_id[record["id"]]
            baseline_length = int(base["trace"].generated_token_count)
            gate_until = int(np.ceil(baseline_length * ratio))
            gate_until = min(gate_until, max(config.max_new_tokens - 1, 1))
            condition_max_new_tokens = condition_max_tokens(
                max_new_tokens=config.max_new_tokens,
                gate_until_tokens=gate_until,
                post_gate_slack_tokens=config.post_gate_slack_tokens,
            )
            print(f"  {index}/{len(included_records)} {record['id']}: gate_until={gate_until}", flush=True)
            gate = ClosureGateConfig(
                gate_until_new_tokens=gate_until,
                penalty=config.gate_penalty,
                hard_block=config.hard_block,
                suppress_eos=config.suppress_eos,
                phrases=tuple(gate_phrases),
                boost_after_new_tokens=gate_until if config.post_gate_boost > 0 else None,
                boost=config.post_gate_boost,
                boost_eos=config.boost_eos,
                pre_gate_guidance_after_new_tokens=pre_gate_guidance_start_tokens(
                    baseline_length=baseline_length,
                    start_ratio=config.pre_gate_math_boost_start_ratio,
                    enabled=config.pre_gate_math_boost > 0,
                ),
                pre_gate_guidance_until_new_tokens=gate_until,
                pre_gate_guidance_boost=config.pre_gate_math_boost,
                pre_gate_guidance_phrases=tuple(guidance_phrases),
                pre_gate_completion_block_phrases=(
                    oracle_answer_completion_phrases(record["answer"])
                    if config.oracle_answer_completion_gate
                    else ()
                ),
                continuation_penalty_after_new_tokens=continuation_start_tokens(
                    gate_until_tokens=gate_until,
                    enabled=(
                        config.post_gate_continuation_penalty > 0
                        or config.post_gate_continuation_hard_block
                    ),
                    from_start=config.continuation_block_from_start,
                ),
                continuation_penalty=config.post_gate_continuation_penalty,
                continuation_hard_block=config.post_gate_continuation_hard_block,
                continuation_phrases=tuple(continuation_phrases),
            )
            start = time.perf_counter()
            trace = generate_gated_trace(
                model,
                prompt=record["prompt"],
                suffix=config.suffix,
                max_new_tokens=condition_max_new_tokens,
                gate=gate,
                do_sample=False,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                min_new_tokens=gate_until if config.force_min_new_tokens else None,
            )
            elapsed = time.perf_counter() - start
            row = build_row(
                condition=condition,
                target_ratio=ratio,
                gate_until_tokens=gate_until,
                condition_max_new_tokens=condition_max_new_tokens,
                baseline_length=baseline_length,
                trace=trace,
                elapsed=elapsed,
                answer=record["answer"],
                max_new_tokens=condition_max_new_tokens,
                baseline_truncated=baseline_length >= config.max_new_tokens,
            )
            example_rows.append({"id": record["id"], **row})
            text_rows.append(
                build_text_row(
                    record=record,
                    condition=condition,
                    target_ratio=ratio,
                    gate_until_tokens=gate_until,
                    condition_max_new_tokens=condition_max_new_tokens,
                    trace=trace,
                    skip_reason=None,
                )
            )

    condition_rows = build_condition_rows(example_rows)
    control = build_control_summary(example_rows, config.target_ratios)
    payload = {
        "created_at": now_iso(),
        "phase": "decode_time_closure_gate_probe",
        "config": asdict(config),
        "baseline_filter": {
            "n_requested": len(dataset),
            "n_included": len(included_records),
            "n_skipped": len(skipped_baselines),
            "skipped": skipped_baselines,
        },
        "closure_gate_phrases": list(gate_phrases),
        "continuation_gate_phrases": list(continuation_phrases),
        "pre_gate_guidance_phrases": list(guidance_phrases),
        "control": control,
        "conditions": condition_rows,
    }
    write_json(output_dir / "summary.json", payload)
    write_json(output_dir / "generation_texts.json", text_rows)
    write_csv(output_dir / "example_decode_gate_metrics.csv", example_rows)
    write_csv(output_dir / "condition_decode_gate_summary.csv", condition_rows)
    return {
        "payload": payload,
        "example_rows": example_rows,
        "text_rows": text_rows,
        "condition_rows": condition_rows,
        "output_dir": str(output_dir),
    }


def build_row(
    *,
    condition: str,
    target_ratio: float | None,
    gate_until_tokens: int,
    condition_max_new_tokens: int,
    baseline_length: int,
    trace,
    elapsed: float,
    answer: str,
    max_new_tokens: int,
    baseline_truncated: bool,
) -> Dict:
    generated_length = int(trace.generated_token_count)
    repetitions = repetition_summary(trace.response_text)
    branching = branching_summary(trace.response_text, generated_length)
    closure_marker = first_closure_marker_metrics(trace.response_text, generated_length)
    return {
        "condition": condition,
        "target_ratio": None if target_ratio is None else float(target_ratio),
        "gate_until_tokens": int(gate_until_tokens),
        "condition_max_new_tokens": int(condition_max_new_tokens),
        "baseline_length": int(baseline_length),
        "generated_length": generated_length,
        "length_ratio": length_ratio(baseline_length, generated_length),
        "baseline_truncated": bool(baseline_truncated),
        "hit_gate": generated_length >= gate_until_tokens if gate_until_tokens else True,
        "over_gate_tokens": generated_length - gate_until_tokens if gate_until_tokens else 0,
        "generated_truncated": generated_length >= max_new_tokens,
        "generated_correct": numeric_correct(trace.response_text, answer),
        "tail_filler_marker_count": tail_filler_marker_count(trace.response_text),
        **closure_marker,
        "latency_sec": elapsed,
        "tokens_per_sec": generated_length / elapsed if elapsed > 0 else None,
        **repetitions,
        **branching,
    }


def build_text_row(
    *,
    record: Dict,
    condition: str,
    target_ratio: float | None,
    gate_until_tokens: int,
    condition_max_new_tokens: int,
    trace,
    skip_reason: str | None,
) -> Dict:
    return {
        "id": record["id"],
        "condition": condition,
        "target_ratio": None if target_ratio is None else float(target_ratio),
        "gate_until_tokens": int(gate_until_tokens),
        "condition_max_new_tokens": int(condition_max_new_tokens),
        "skip_reason": skip_reason,
        "answer": record["answer"],
        "prompt": record["prompt"],
        "response_text": trace.response_text,
        "generated_token_count": int(trace.generated_token_count),
    }


def build_condition_rows(rows: Sequence[Dict]) -> list[Dict]:
    output = []
    baseline_correct_by_id = {
        row["id"]: bool(row.get("generated_correct"))
        for row in rows
        if row.get("condition") == "baseline"
    }
    for condition in sorted({row["condition"] for row in rows}, key=condition_sort_key):
        items = [row for row in rows if row["condition"] == condition]
        ratios = summarize_length_ratios(row["length_ratio"] for row in items)
        accuracy = accuracy_summary(items, baseline_correct_by_id)
        output.append(
            {
                "condition": condition,
                "target_ratio": safe_mean(row.get("target_ratio") for row in items),
                "n": len(items),
                "length_ratio_mean": ratios.get("mean"),
                "length_ratio_median": ratios.get("median"),
                "generated_length_mean": safe_mean(row.get("generated_length") for row in items),
                "gate_until_tokens_mean": safe_mean(row.get("gate_until_tokens") for row in items),
                "condition_max_new_tokens_mean": safe_mean(
                    row.get("condition_max_new_tokens") for row in items
                ),
                "baseline_truncated_rate": safe_mean(1.0 if row.get("baseline_truncated") else 0.0 for row in items),
                "hit_gate_rate": safe_mean(1.0 if row.get("hit_gate") else 0.0 for row in items),
                "over_gate_tokens_mean": safe_mean(row.get("over_gate_tokens") for row in items),
                "generated_truncated_rate": safe_mean(1.0 if row.get("generated_truncated") else 0.0 for row in items),
                **accuracy,
                "tail_filler_marker_count_mean": safe_mean(
                    row.get("tail_filler_marker_count") for row in items
                ),
                "first_closure_marker_char_ratio_mean": safe_mean(
                    row.get("first_closure_marker_char_ratio") for row in items
                ),
                "tail_after_first_closure_est_tokens_mean": safe_mean(
                    row.get("tail_after_first_closure_est_tokens") for row in items
                ),
                "word_count_mean": safe_mean(row.get("word_count") for row in items),
                "repeat_4gram_rate_mean": safe_mean(row.get("repeat_4gram_rate") for row in items),
                "branch_marker_rate_mean": safe_mean(row.get("branch_marker_rate") for row in items),
                "branch_marker_count_mean": safe_mean(row.get("branch_marker_count") for row in items),
            }
        )
    return output


def accuracy_summary(rows: Sequence[Dict], baseline_correct_by_id: Dict[str, bool]) -> Dict:
    if not rows:
        return {
            "accuracy_mean": None,
            "accuracy_given_baseline_correct": None,
            "correct_to_wrong_rate": None,
            "wrong_to_correct_rate": None,
        }
    generated = [bool(row.get("generated_correct")) for row in rows]
    baseline = [baseline_correct_by_id.get(row["id"]) for row in rows]
    base_correct_indices = [idx for idx, value in enumerate(baseline) if value is True]
    base_wrong_indices = [idx for idx, value in enumerate(baseline) if value is False]
    retained = None
    correct_to_wrong = None
    if base_correct_indices:
        retained = sum(1 for idx in base_correct_indices if generated[idx]) / len(base_correct_indices)
        correct_to_wrong = 1.0 - retained
    wrong_to_correct = None
    if base_wrong_indices:
        wrong_to_correct = sum(1 for idx in base_wrong_indices if generated[idx]) / len(base_wrong_indices)
    return {
        "accuracy_mean": sum(generated) / len(generated),
        "accuracy_given_baseline_correct": retained,
        "correct_to_wrong_rate": correct_to_wrong,
        "wrong_to_correct_rate": wrong_to_correct,
    }


def tail_filler_marker_count(text: str, tail_chars: int = 1600) -> int:
    tail = text[-tail_chars:].lower()
    return sum(tail.count(marker) for marker in FILLER_MARKERS)


def first_closure_marker_metrics(text: str, generated_length: int) -> Dict:
    lower = text.lower()
    hits = [(lower.find(marker), marker) for marker in CLOSURE_TEXT_MARKERS if lower.find(marker) >= 0]
    if not hits or not text:
        return {
            "first_closure_marker": None,
            "first_closure_marker_char_ratio": None,
            "tail_after_first_closure_est_tokens": None,
        }
    char_index, marker = min(hits, key=lambda item: item[0])
    char_ratio = char_index / max(len(text), 1)
    est_token = int(round(char_ratio * generated_length))
    return {
        "first_closure_marker": marker,
        "first_closure_marker_char_ratio": char_ratio,
        "tail_after_first_closure_est_tokens": max(0, generated_length - est_token),
    }


def build_control_summary(rows: Sequence[Dict], target_ratios: Sequence[float]) -> Dict:
    non_baseline = [row for row in rows if str(row["condition"]).startswith("decode_gate_")]
    target_values = [row["target_ratio"] for row in non_baseline]
    length_values = [row["length_ratio"] for row in non_baseline]
    rho, p = safe_spearman_correlation(target_values, length_values)
    adjacent_total = 0
    adjacent_ok = 0
    for example_id in sorted({row["id"] for row in non_baseline}):
        items = sorted(
            [row for row in non_baseline if row["id"] == example_id],
            key=lambda row: row["target_ratio"],
        )
        for left, right in zip(items, items[1:]):
            adjacent_total += 1
            if right["length_ratio"] >= left["length_ratio"]:
                adjacent_ok += 1
    return {
        "target_vs_length_spearman": {"rho": rho, "p": p, "n": len(non_baseline)},
        "example_adjacent_non_decreasing": {
            "ok": adjacent_ok,
            "total": adjacent_total,
            "rate": adjacent_ok / adjacent_total if adjacent_total else None,
        },
        "target_ratios": list(target_ratios),
    }


def format_ratio(value: float) -> str:
    return str(value).replace(".", "p")


def closure_gate_phrases(phrase_set: str) -> Sequence[str]:
    if phrase_set == "expanded":
        return EXPANDED_CLOSURE_GATE_PHRASES
    if phrase_set == "base":
        return BASE_CLOSURE_GATE_PHRASES
    return DEFAULT_CLOSURE_GATE_PHRASES


def continuation_gate_phrases(continuation_set: str) -> Sequence[str]:
    if continuation_set == "drift":
        return DRIFT_POST_GATE_CONTINUATION_PHRASES
    if continuation_set == "basic":
        return BASIC_POST_GATE_CONTINUATION_PHRASES
    return DEFAULT_POST_GATE_CONTINUATION_PHRASES


def pre_gate_guidance_phrases(guidance_set: str) -> Sequence[str]:
    if guidance_set == "semantic_more":
        return DEFAULT_SEMANTIC_MORE_GUIDANCE_PHRASES
    if guidance_set == "math_semantic":
        return [*DEFAULT_MATH_GUIDANCE_PHRASES, *DEFAULT_SEMANTIC_MORE_GUIDANCE_PHRASES]
    return DEFAULT_MATH_GUIDANCE_PHRASES


def condition_max_tokens(
    *,
    max_new_tokens: int,
    gate_until_tokens: int,
    post_gate_slack_tokens: int | None,
) -> int:
    if post_gate_slack_tokens is None:
        return int(max_new_tokens)
    return min(int(max_new_tokens), int(gate_until_tokens) + max(0, int(post_gate_slack_tokens)))


def continuation_start_tokens(
    *,
    gate_until_tokens: int,
    enabled: bool,
    from_start: bool,
) -> int | None:
    if not enabled:
        return None
    return 0 if from_start else int(gate_until_tokens)


def pre_gate_guidance_start_tokens(
    *,
    baseline_length: int,
    start_ratio: float,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    return max(0, int(np.ceil(float(baseline_length) * float(start_ratio))))


def oracle_answer_completion_phrases(answer: str) -> list[str]:
    value = str(answer).strip()
    if not value:
        return []
    return [
        f"Final answer: {value}",
        f"Final answer is {value}",
        f"The final answer is {value}",
        f"So the answer is {value}",
        f"Therefore, the final answer is {value}",
        f"Answer: {value}",
        f"answer is {value}",
        f"Conclusion: {value}",
        f"End result: {value}",
        f"boxed{{{value}}}",
        f"\\boxed{{{value}}}",
    ]


def baseline_skip_reason(
    baseline_length: int,
    *,
    max_new_tokens: int,
    min_baseline_tokens: int,
    max_baseline_tokens: int | None,
) -> str | None:
    if baseline_length >= max_new_tokens:
        return f"baseline_truncated_at_{max_new_tokens}"
    if baseline_length < min_baseline_tokens:
        return f"baseline_too_short:{baseline_length}<{min_baseline_tokens}"
    if max_baseline_tokens is not None and baseline_length > max_baseline_tokens:
        return f"baseline_too_long:{baseline_length}>{max_baseline_tokens}"
    return None


def condition_sort_key(condition: str) -> tuple[float, str]:
    if condition == "baseline":
        return (0.0, condition)
    try:
        return (float(condition.rsplit("_", 1)[-1].replace("p", ".")), condition)
    except ValueError:
        return (999.0, condition)


def safe_mean(values) -> float | None:
    numbers = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not numbers:
        return None
    return float(np.mean(numbers))


if __name__ == "__main__":
    main()
