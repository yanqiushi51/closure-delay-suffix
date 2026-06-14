import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.branching import branching_summary
from closure_delay.data import load_gsm8k_dataset
from closure_delay.exit_hazard_torch import DifferentiableExitHazardHead, exit_logit_features_from_logits
from closure_delay.model import LocalCausalLM
from closure_delay.runtime import now_iso, write_csv, write_json
from closure_delay.utility import numeric_correct
from scripts.optimize_suffix_phrase_cem import DEFAULT_SUFFIX_PREFIX, _format_suffix, load_phrase_bank


@dataclass
class RollingExample:
    example_id: str
    prompt: str
    answer: str
    response_text: str
    response_ids: List[int]
    generated_tokens: int
    correct: bool
    score: Dict[str, float | int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling teacher-forced CEM with trajectory refresh.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--phrases-per-suffix", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--random-immigrant-frac", type=float, default=0.25)
    parser.add_argument("--suffix-prefix", default=DEFAULT_SUFFIX_PREFIX)
    parser.add_argument("--phrase-bank-json", help="Optional JSON phrase bank. Accepts list[str], list[dict], or dict[str, list].")
    parser.add_argument("--exclude-phrase-regex", default="", help="Regex for phrases to exclude from the loaded bank.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--hazard-start-frac", type=float, default=0.20)
    parser.add_argument("--hazard-end-frac", type=float, default=0.80)
    parser.add_argument("--answer-loss-weight", type=float, default=0.25)
    parser.add_argument("--answer-nll-margin", type=float, default=6.0)
    parser.add_argument("--answer-template", default=" Final answer: {answer}")
    parser.add_argument("--hazard-threshold", type=float, default=0.30)
    parser.add_argument("--accept-non-improving", action="store_true")
    parser.add_argument(
        "--seed-candidate",
        action="append",
        default=[],
        help="Initial phrase candidate, using 'phrase 1|phrase 2|phrase 3'. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _random_candidate(rng: random.Random, phrase_bank: Sequence[str], k: int) -> tuple[str, ...]:
    return tuple(rng.sample(list(phrase_bank), k=min(k, len(phrase_bank))))


def _parse_seed_candidates(items: Sequence[str]) -> List[tuple[str, ...]]:
    candidates: List[tuple[str, ...]] = []
    for item in items:
        phrases = tuple(part.strip() for part in str(item).split("|") if part.strip())
        if phrases:
            candidates.append(phrases)
    return candidates


def _mutate(
    rng: random.Random,
    candidate: tuple[str, ...],
    phrase_bank: Sequence[str],
    mutation_rate: float,
) -> tuple[str, ...]:
    if not candidate:
        return _random_candidate(rng, phrase_bank, 3)
    output = list(candidate)
    for idx in range(len(output)):
        if rng.random() < mutation_rate:
            choices = [item for item in phrase_bank if item not in output]
            if choices:
                output[idx] = rng.choice(choices)
    if rng.random() < mutation_rate:
        rng.shuffle(output)
    return tuple(output)


def _window_bounds(n_tokens: int, start_frac: float, end_frac: float) -> tuple[int, int]:
    n = int(n_tokens)
    start = int(max(0, min(n - 1, round(float(start_frac) * n))))
    stop = int(max(start + 1, min(n, round(float(end_frac) * n))))
    return start, stop


def _score_sequence(
    lm: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    prompt: str,
    suffix: str,
    response_ids: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, float | int | None]:
    response_ids = [int(tok) for tok in response_ids[: int(args.max_response_tokens)]]
    if len(response_ids) < 2:
        return {
            "mean_raw_hazard": None,
            "mean_cumlogit": None,
            "window_cumlogit_mean": None,
            "max_cumprob": None,
            "first_cross_token": None,
            "post_exit_tokens": None,
        }
    tokenizer = lm.tokenizer
    prompt_text = lm.build_prompt_text(prompt, suffix)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    full_ids = list(prompt_ids) + list(response_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=lm.device)
    attention_mask = torch.ones_like(input_ids, device=lm.device)
    with torch.no_grad():
        outputs = lm.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        start = len(prompt_ids)
        end = start + len(response_ids)
        hidden = outputs.hidden_states[head.config.layer][0, start:end, :].float()
        logits = outputs.logits[0, start:end, :].float()
        logit_features = exit_logit_features_from_logits(logits, tokenizer)
        raw = head(hidden, logit_features)
        cumprob, cumlogit = head.cumulative_scores(raw)
    win_start, win_stop = _window_bounds(len(response_ids), args.hazard_start_frac, args.hazard_end_frac)
    cumprob_list = cumprob.detach().cpu().tolist()
    crossing = next((idx + 1 for idx, value in enumerate(cumprob_list) if value >= float(args.hazard_threshold)), None)
    return {
        "mean_raw_hazard": float(raw.mean().detach().cpu()),
        "mean_cumlogit": float(cumlogit.mean().detach().cpu()),
        "window_cumlogit_mean": float(cumlogit[win_start:win_stop].mean().detach().cpu()),
        "max_cumprob": float(cumprob.max().detach().cpu()),
        "first_cross_token": crossing,
        "post_exit_tokens": int(len(response_ids) - crossing) if crossing is not None else 0,
    }


def _answer_nll(lm: LocalCausalLM, prompt: str, suffix: str, answer: str, args: argparse.Namespace) -> float:
    tokenizer = lm.tokenizer
    prompt_text = lm.build_prompt_text(prompt, suffix)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    target_text = str(args.answer_template).format(answer=answer)
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return 0.0
    input_ids = torch.tensor([list(prompt_ids) + list(target_ids)], dtype=torch.long, device=lm.device)
    attention_mask = torch.ones_like(input_ids, device=lm.device)
    labels = torch.tensor(target_ids, dtype=torch.long, device=lm.device)
    with torch.no_grad():
        outputs = lm.model(input_ids=input_ids, attention_mask=attention_mask)
        prompt_len = len(prompt_ids)
        logits = outputs.logits[0, prompt_len - 1 : prompt_len + len(target_ids) - 1, :].float()
        loss = F.cross_entropy(logits, labels, reduction="mean")
    return float(loss.detach().cpu())


def _generate_examples(
    lm: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    records: Sequence[Dict],
    suffix: str,
    args: argparse.Namespace,
) -> List[RollingExample]:
    examples: List[RollingExample] = []
    for item in records:
        trace = lm.generate_trace(
            prompt=str(item["prompt"]),
            suffix=suffix,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
        )
        score = _score_sequence(lm, head, str(item["prompt"]), suffix, trace.generated_ids, args)
        examples.append(
            RollingExample(
                example_id=str(item["id"]),
                prompt=str(item["prompt"]),
                answer=str(item["answer"]),
                response_text=trace.response_text,
                response_ids=[int(tok) for tok in trace.generated_ids],
                generated_tokens=int(trace.generated_token_count),
                correct=bool(numeric_correct(trace.response_text, str(item["answer"]))),
                score=score,
            )
        )
        print(
            f"generate {item['id']} len={trace.generated_token_count} "
            f"correct={examples[-1].correct} mean_raw={score['mean_raw_hazard']}",
            flush=True,
        )
    return examples


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _summarize_examples(round_idx: int, suffix: str, examples: Sequence[RollingExample]) -> Dict:
    return {
        "round": int(round_idx),
        "suffix": suffix,
        "n": len(examples),
        "generated_tokens_mean": _mean([ex.generated_tokens for ex in examples]),
        "correct_rate": _mean([1.0 if ex.correct else 0.0 for ex in examples]),
        "mean_raw_hazard": _mean([ex.score.get("mean_raw_hazard") for ex in examples]),
        "mean_cumlogit": _mean([ex.score.get("mean_cumlogit") for ex in examples]),
        "window_cumlogit_mean": _mean([ex.score.get("window_cumlogit_mean") for ex in examples]),
        "post_exit_tokens_mean": _mean([ex.score.get("post_exit_tokens") for ex in examples]),
    }


def _score_candidate(
    lm: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    child_suffix: str,
    parent_examples: Sequence[RollingExample],
    args: argparse.Namespace,
) -> Dict:
    child_means = []
    parent_means = [example.score.get("window_cumlogit_mean") for example in parent_examples]
    answer_nlls = []
    for example in parent_examples:
        child_score = _score_sequence(lm, head, example.prompt, child_suffix, example.response_ids, args)
        child_means.append(child_score["window_cumlogit_mean"])
        answer_nlls.append(_answer_nll(lm, example.prompt, child_suffix, example.answer, args))
    child_mean = _mean(child_means)
    parent_mean = _mean(parent_means)
    delta = None if child_mean is None or parent_mean is None else float(child_mean - parent_mean)
    answer_nll = _mean(answer_nlls)
    answer_penalty = max(0.0, float(answer_nll or 0.0) - float(args.answer_nll_margin))
    loss = float(delta or 0.0) + float(args.answer_loss_weight) * answer_penalty
    return {
        "loss": loss,
        "delta_cumlogit": delta,
        "child_window_cumlogit_mean": child_mean,
        "parent_window_cumlogit_mean": parent_mean,
        "answer_nll": answer_nll,
        "answer_penalty": answer_penalty,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lm = LocalCausalLM(args.model_path, device=args.device)
    for parameter in lm.model.parameters():
        parameter.requires_grad_(False)
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=lm.device)
    head.eval()

    records = load_gsm8k_dataset(split=args.dataset_split, n_samples=int(args.n_samples), seed=int(args.seed))
    phrase_bank = load_phrase_bank(args.phrase_bank_json, args.exclude_phrase_regex)
    if len(phrase_bank) < int(args.phrases_per_suffix):
        raise RuntimeError("Phrase bank is smaller than --phrases-per-suffix after filtering.")
    print(f"loaded phrase bank: {len(phrase_bank)} phrases", flush=True)
    parent_phrases: tuple[str, ...] = tuple()
    parent_suffix = ""
    parent_examples = _generate_examples(lm, head, records, parent_suffix, args)

    round_rows = [_summarize_examples(0, parent_suffix, parent_examples)]
    candidate_rows: List[Dict] = []
    example_rows: List[Dict] = [
        {
            "round": 0,
            "id": example.example_id,
            "suffix": parent_suffix,
            "generated_tokens": example.generated_tokens,
            "correct": example.correct,
            "response_text": example.response_text,
            **branching_summary(example.response_text, example.generated_tokens),
            **example.score,
        }
        for example in parent_examples
    ]

    population = _parse_seed_candidates(args.seed_candidate)
    while len(population) < int(args.population):
        population.append(_random_candidate(rng, phrase_bank, int(args.phrases_per_suffix)))
    for round_idx in range(1, int(args.rounds) + 1):
        scored = []
        seen = set()
        for candidate in population:
            if candidate in seen:
                continue
            seen.add(candidate)
            suffix = _format_suffix(candidate, args.suffix_prefix)
            metrics = _score_candidate(lm, head, suffix, parent_examples, args)
            row = {
                "round": round_idx,
                "phrases": " | ".join(candidate),
                "suffix": suffix,
                **metrics,
            }
            scored.append(row)
            candidate_rows.append(row)
        scored.sort(key=lambda item: float(item["loss"]))
        best = scored[0]
        parent_summary = round_rows[-1]
        print(
            f"round={round_idx} best_loss={best['loss']:.4f} "
            f"delta={best['delta_cumlogit']:.4f} answer_nll={best['answer_nll']:.4f} "
            f"parent_len={parent_summary['generated_tokens_mean']:.2f} suffix={best['suffix']!r}",
            flush=True,
        )

        accepted = bool(args.accept_non_improving) or float(best["delta_cumlogit"] or 0.0) < 0.0
        if not accepted:
            rejected_summary = dict(parent_summary)
            rejected_summary["round"] = int(round_idx)
            rejected_summary["teacher_forced_delta_cumlogit"] = best["delta_cumlogit"]
            rejected_summary["teacher_forced_loss"] = best["loss"]
            rejected_summary["teacher_forced_answer_nll"] = best["answer_nll"]
            rejected_summary["length_delta_from_parent"] = 0.0
            rejected_summary["raw_hazard_delta_from_parent"] = 0.0
            rejected_summary["cumlogit_delta_from_parent"] = 0.0
            rejected_summary["accepted"] = False
            round_rows.append(rejected_summary)
            print(f"round={round_idx} rejected non-improving child; keeping parent suffix.", flush=True)
            elites = [tuple(row["phrases"].split(" | ")) for row in scored[: max(1, int(args.elite))]]
            population = list(elites)
            while len(population) < int(args.population):
                if rng.random() < float(args.random_immigrant_frac):
                    population.append(_random_candidate(rng, phrase_bank, int(args.phrases_per_suffix)))
                else:
                    population.append(
                        _mutate(
                            rng,
                            rng.choice(elites or [parent_phrases]),
                            phrase_bank,
                            float(args.mutation_rate),
                        )
                    )
            continue

        child_suffix = str(best["suffix"])
        child_examples = _generate_examples(lm, head, records, child_suffix, args)
        child_summary = _summarize_examples(round_idx, child_suffix, child_examples)
        child_summary["teacher_forced_delta_cumlogit"] = best["delta_cumlogit"]
        child_summary["teacher_forced_loss"] = best["loss"]
        child_summary["teacher_forced_answer_nll"] = best["answer_nll"]
        child_summary["length_delta_from_parent"] = (
            float(child_summary["generated_tokens_mean"]) - float(parent_summary["generated_tokens_mean"])
        )
        child_summary["raw_hazard_delta_from_parent"] = (
            float(child_summary["mean_raw_hazard"]) - float(parent_summary["mean_raw_hazard"])
        )
        child_summary["cumlogit_delta_from_parent"] = (
            float(child_summary["mean_cumlogit"]) - float(parent_summary["mean_cumlogit"])
        )
        child_summary["accepted"] = True
        round_rows.append(child_summary)

        for example in child_examples:
            example_rows.append(
                {
                    "round": round_idx,
                    "id": example.example_id,
                    "suffix": child_suffix,
                    "generated_tokens": example.generated_tokens,
                    "correct": example.correct,
                    "response_text": example.response_text,
                    **branching_summary(example.response_text, example.generated_tokens),
                    **example.score,
                }
            )

        parent_suffix = child_suffix
        parent_phrases = tuple(best["phrases"].split(" | "))
        parent_examples = child_examples
        elites = [tuple(row["phrases"].split(" | ")) for row in scored[: max(1, int(args.elite))]]
        population = list(elites)
        while len(population) < int(args.population):
            if rng.random() < float(args.random_immigrant_frac):
                population.append(_random_candidate(rng, phrase_bank, int(args.phrases_per_suffix)))
            else:
                population.append(_mutate(rng, rng.choice(elites or [parent_phrases]), phrase_bank, float(args.mutation_rate)))

    write_csv(out_dir / "round_summary.csv", round_rows)
    write_csv(out_dir / "candidate_scores.csv", candidate_rows)
    write_csv(out_dir / "generation_examples.csv", example_rows)
    write_json(
        out_dir / "rolling_teacher_forced_cem_report.json",
        {
            "created_at": now_iso(),
            "config": vars(args),
            "round_summary": round_rows,
            "final_suffix": parent_suffix,
        },
    )
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
