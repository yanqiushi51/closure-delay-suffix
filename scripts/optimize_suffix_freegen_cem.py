import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.branching import branching_summary
from closure_delay.data import load_gsm8k_dataset
from closure_delay.exit_hazard_torch import DifferentiableExitHazardHead
from closure_delay.model import LocalCausalLM
from closure_delay.repetition import repetition_summary
from closure_delay.runtime import now_iso, write_csv, write_json
from closure_delay.utility import numeric_correct
from scripts.evaluate_suffix_overthinking import _score_response
from scripts.optimize_suffix_phrase_cem import DEFAULT_SUFFIX_PREFIX, _format_suffix, load_phrase_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy/free-generation phrase CEM for fixed suffix search.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--phrases-per-suffix", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--suffix-prefix", default=DEFAULT_SUFFIX_PREFIX)
    parser.add_argument("--phrase-bank-json", help="Optional JSON phrase bank. Accepts list[str], list[dict], or dict[str, list].")
    parser.add_argument("--exclude-phrase-regex", default="", help="Regex for phrases to exclude from the loaded bank.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--reward-mode", choices=["suppress-cumlogit"], default="suppress-cumlogit")
    parser.add_argument("--cumlogit-weight", type=float, default=1.0)
    parser.add_argument("--length-weight", type=float, default=0.40)
    parser.add_argument("--correctness-weight", type=float, default=2.0)
    parser.add_argument("--drift-weight", type=float, default=2.0)
    parser.add_argument("--repeat-weight", type=float, default=4.0)
    parser.add_argument("--max-length-ratio-bonus", type=float, default=1.0)
    parser.add_argument("--hazard-threshold", type=float, default=0.30)
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
    parser.add_argument("--output-dir", default="outputs/exit_hazard/freegen_cem_suppress_cumlogit")
    return parser.parse_args()


def _random_candidate(rng: random.Random, phrase_bank: Sequence[str], k: int) -> tuple[str, ...]:
    return tuple(rng.sample(list(phrase_bank), k=min(k, len(phrase_bank))))


def _mutate(
    rng: random.Random,
    candidate: tuple[str, ...],
    phrase_bank: Sequence[str],
    mutation_rate: float,
) -> tuple[str, ...]:
    output = list(candidate)
    for idx in range(len(output)):
        if rng.random() < mutation_rate:
            choices = [item for item in phrase_bank if item not in output]
            if choices:
                output[idx] = rng.choice(choices)
    if rng.random() < mutation_rate:
        rng.shuffle(output)
    return tuple(output)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _score_generation(
    model: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    item: Dict,
    suffix: str,
    args: argparse.Namespace,
) -> Dict:
    trace = model.generate_trace(
        prompt=item["prompt"],
        suffix=suffix,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=False,
    )
    score = _score_response(model, head, item["prompt"], suffix, trace.generated_ids, args)
    repeat = repetition_summary(trace.response_text)
    branch = branching_summary(trace.response_text, trace.generated_token_count)
    return {
        "id": item["id"],
        "answer": item["answer"],
        "suffix": suffix,
        "generated_tokens": int(trace.generated_token_count),
        "correct": bool(numeric_correct(trace.response_text, item["answer"])),
        "response_text": trace.response_text,
        **score,
        **repeat,
        **branch,
    }


def _baseline_rows(
    model: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    dataset: Sequence[Dict],
    args: argparse.Namespace,
) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    for item in dataset:
        row = _score_generation(model, head, item, "", args)
        rows[str(item["id"])] = row
        print(
            f"baseline {item['id']} len={row['generated_tokens']} "
            f"cum={row['mean_cumlogit']:.3f} correct={row['correct']}",
            flush=True,
        )
    return rows


def _candidate_reward(row: Dict, baseline: Dict, args: argparse.Namespace) -> Dict:
    baseline_len = max(float(baseline.get("generated_tokens") or 1), 1.0)
    length_delta_ratio = (float(row["generated_tokens"]) - baseline_len) / baseline_len
    length_bonus = min(max(length_delta_ratio, -1.0), float(args.max_length_ratio_bonus))
    cumlogit_delta = float(row["mean_cumlogit"]) - float(baseline["mean_cumlogit"])
    drift = float(row.get("drift_mean") or 0.0)
    repeat = float(row.get("repeat_4gram_rate") or 0.0)
    correct = 1.0 if row.get("correct") else 0.0
    reward = (
        -float(args.cumlogit_weight) * cumlogit_delta
        + float(args.length_weight) * length_bonus
        + float(args.correctness_weight) * correct
        - float(args.drift_weight) * drift
        - float(args.repeat_weight) * repeat
    )
    return {
        "reward": float(reward),
        "cumlogit_delta": float(cumlogit_delta),
        "length_delta": float(float(row["generated_tokens"]) - baseline_len),
        "length_delta_ratio": float(length_delta_ratio),
        "length_bonus": float(length_bonus),
        "correct_reward": float(correct),
        "drift_penalty_value": float(drift),
        "repeat_penalty_value": float(repeat),
    }


def _score_candidate(
    model: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    dataset: Sequence[Dict],
    baselines: Dict[str, Dict],
    phrases: Sequence[str],
    args: argparse.Namespace,
) -> tuple[Dict, List[Dict]]:
    suffix = _format_suffix(phrases, args.suffix_prefix)
    rows: List[Dict] = []
    rewards: List[float] = []
    for item in dataset:
        row = _score_generation(model, head, item, suffix, args)
        reward_parts = _candidate_reward(row, baselines[str(item["id"])], args)
        row.update(reward_parts)
        rows.append(row)
        rewards.append(float(reward_parts["reward"]))
    summary = {
        "suffix": suffix,
        "phrases": list(phrases),
        "reward_mean": _mean(rewards),
        "generated_tokens_mean": _mean([float(row["generated_tokens"]) for row in rows]),
        "length_delta_mean": _mean([float(row["length_delta"]) for row in rows]),
        "length_delta_ratio_mean": _mean([float(row["length_delta_ratio"]) for row in rows]),
        "correct_rate": _mean([1.0 if row["correct"] else 0.0 for row in rows]),
        "mean_cumlogit": _mean([float(row["mean_cumlogit"]) for row in rows]),
        "cumlogit_delta_mean": _mean([float(row["cumlogit_delta"]) for row in rows]),
        "mean_raw_hazard": _mean([float(row["mean_raw_hazard"]) for row in rows]),
        "post_exit_tokens_mean": _mean([float(row.get("post_exit_tokens") or 0.0) for row in rows]),
        "drift_mean": _mean([float(row.get("drift_mean") or 0.0) for row in rows]),
        "repeat_4gram_rate_mean": _mean([float(row.get("repeat_4gram_rate") or 0.0) for row in rows]),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    dataset = load_gsm8k_dataset(
        split=str(args.dataset_split),
        n_samples=int(args.train_size),
        seed=int(args.seed),
    )
    if not dataset:
        raise RuntimeError("No dataset examples loaded.")

    model = LocalCausalLM(args.model_path, device=args.device)
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=model.device)
    head.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = _baseline_rows(model, head, dataset, args)
    baseline_summary = {
        "generated_tokens_mean": _mean([float(row["generated_tokens"]) for row in baselines.values()]),
        "correct_rate": _mean([1.0 if row["correct"] else 0.0 for row in baselines.values()]),
        "mean_cumlogit": _mean([float(row["mean_cumlogit"]) for row in baselines.values()]),
        "mean_raw_hazard": _mean([float(row["mean_raw_hazard"]) for row in baselines.values()]),
        "post_exit_tokens_mean": _mean([float(row.get("post_exit_tokens") or 0.0) for row in baselines.values()]),
        "drift_mean": _mean([float(row.get("drift_mean") or 0.0) for row in baselines.values()]),
        "repeat_4gram_rate_mean": _mean([float(row.get("repeat_4gram_rate") or 0.0) for row in baselines.values()]),
    }

    phrase_bank = load_phrase_bank(args.phrase_bank_json, args.exclude_phrase_regex)
    if len(phrase_bank) < int(args.phrases_per_suffix):
        raise RuntimeError("Phrase bank is smaller than --phrases-per-suffix after filtering.")
    print(f"loaded phrase bank: {len(phrase_bank)} phrases", flush=True)
    population = [
        _random_candidate(rng, phrase_bank, int(args.phrases_per_suffix))
        for _ in range(int(args.population))
    ]

    history: List[Dict] = []
    detail_rows: List[Dict] = []
    best_row: Dict | None = None
    best_phrases: tuple[str, ...] | None = None
    for round_idx in range(int(args.rounds)):
        scored: List[Dict] = []
        seen = set()
        for phrases in population:
            if phrases in seen:
                continue
            seen.add(phrases)
            summary, rows = _score_candidate(model, head, dataset, baselines, phrases, args)
            summary["round"] = int(round_idx)
            scored.append(summary)
            history.append(summary)
            for row in rows:
                detail_rows.append(
                    {
                        "round": int(round_idx),
                        "suffix": summary["suffix"],
                        "phrases": " | ".join(phrases),
                        **{key: value for key, value in row.items() if key != "response_text"},
                    }
                )
            if best_row is None or float(summary["reward_mean"]) > float(best_row["reward_mean"]):
                best_row = dict(summary)
                best_phrases = tuple(phrases)
        scored.sort(key=lambda row: float(row["reward_mean"]), reverse=True)
        elites = [tuple(row["phrases"]) for row in scored[: max(1, int(args.elite))]]
        top = scored[0]
        print(
            f"round={round_idx} reward={top['reward_mean']:.4f} "
            f"cum_delta={top['cumlogit_delta_mean']:.4f} len_delta={top['length_delta_mean']:.2f} "
            f"acc={top['correct_rate']:.3f} repeat={top['repeat_4gram_rate_mean']:.4f} "
            f"suffix={top['suffix']!r}",
            flush=True,
        )
        next_population = list(elites)
        while len(next_population) < int(args.population):
            parent = rng.choice(elites)
            next_population.append(_mutate(rng, parent, phrase_bank, float(args.mutation_rate)))
        population = next_population

    write_csv(out_dir / "freegen_cem_history.csv", history)
    write_csv(out_dir / "freegen_cem_details.csv", detail_rows)
    write_csv(out_dir / "freegen_cem_baseline_rows.csv", baselines.values())
    write_json(
        out_dir / "freegen_cem_report.json",
        {
            "created_at": now_iso(),
            "objective": "on-policy/free-generation CEM reward B: suppress cumlogit while preserving length and correctness",
            "config": vars(args),
            "baseline_summary": baseline_summary,
            "best": best_row,
            "best_phrases": list(best_phrases or []),
            "suffix": None if best_row is None else best_row["suffix"],
        },
    )
    if best_row is not None:
        suffix_payload = {
            "optimizer": "freegen_phrase_cem",
            "objective": "suppress-cumlogit free-generation black-box reward",
            "hazard_head_json": str(args.hazard_head_json),
            "suffix": best_row["suffix"],
            "phrases": list(best_phrases or []),
            "baseline_summary": baseline_summary,
            "best": best_row,
            "config": vars(args),
        }
        write_json(out_dir / "best_suffix.json", suffix_payload)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
