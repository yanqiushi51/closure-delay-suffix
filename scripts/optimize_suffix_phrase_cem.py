import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard_torch import DifferentiableExitHazardHead
from closure_delay.model import LocalCausalLM
from scripts.optimize_suffix import _batch_loss, _load_examples


PHRASE_BANK = [
    "continue the reasoning a little further",
    "verify each arithmetic step",
    "check for hidden assumptions",
    "try an alternative derivation",
    "compare the result with another method",
    "delay the final answer until the consistency check is complete",
    "make sure no case has been missed",
    "recompute the key quantities",
    "inspect the boundary cases",
    "cross-check the intermediate values",
    "look for possible off-by-one errors",
    "validate the final number before stating it",
    "keep the reasoning explicit",
    "do one more independent check",
    "audit the calculation carefully",
    "avoid rushing to the final answer",
    "continue only with relevant verification",
    "resolve any uncertainty before concluding",
    "test whether the answer is self-consistent",
    "review the equation setup",
    "double-check unit conversions",
    "confirm the interpretation of the question",
    "trace the solution from the beginning",
    "summarize and then verify the conclusion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phrase-level CEM search for natural overthinking suffixes.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--require-correct", action="store_true")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--phrases-per-suffix", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--hazard-start-frac", type=float, default=0.55)
    parser.add_argument("--hazard-end-frac", type=float, default=0.95)
    parser.add_argument("--hazard-loss-scale", type=float, default=0.001)
    parser.add_argument("--direct-margin-weight", type=float, default=0.0)
    parser.add_argument("--answer-loss-weight", type=float, default=2.0)
    parser.add_argument("--answer-nll-margin", type=float, default=4.0)
    parser.add_argument("--answer-template", default=" Final answer: {answer}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="outputs/learned_suffixes/phrase_cem_exit_hazard_suffix.json")
    return parser.parse_args()


def _format_suffix(phrases: Sequence[str]) -> str:
    body = "; ".join(phrase.strip().rstrip(".") for phrase in phrases)
    return f"Before giving the final answer, {body}."


def _token_ids(tokenizer, suffix: str) -> List[int]:
    ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    return [int(tok) for tok in ids]


def _candidate_loss(
    lm: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    examples: Sequence,
    suffix: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    suffix_ids = _token_ids(lm.tokenizer, suffix)
    _, metrics, _ = _batch_loss(
        lm,
        lm.tokenizer,
        head,
        examples,
        suffix_ids,
        args,
        require_grad=False,
    )
    metrics["suffix_token_count"] = float(len(suffix_ids))
    return metrics


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


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    lm = LocalCausalLM(args.model_path, device=args.device)
    for parameter in lm.model.parameters():
        parameter.requires_grad_(False)
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=lm.device)
    head.eval()
    examples = _load_examples(args)
    if not examples:
        raise RuntimeError("No optimization examples found.")

    phrase_bank = list(PHRASE_BANK)
    population = [_random_candidate(rng, phrase_bank, int(args.phrases_per_suffix)) for _ in range(int(args.population))]
    history = []
    best_row = None
    for round_idx in range(int(args.rounds)):
        batch = rng.sample(examples, k=min(int(args.batch_size), len(examples)))
        scored = []
        seen = set()
        for candidate in population:
            if candidate in seen:
                continue
            seen.add(candidate)
            suffix = _format_suffix(candidate)
            metrics = _candidate_loss(lm, head, batch, suffix, args)
            row = {
                "round": round_idx,
                "phrases": list(candidate),
                "suffix": suffix,
                **metrics,
            }
            scored.append(row)
            if best_row is None or row["loss"] < best_row["loss"]:
                best_row = dict(row)
        scored.sort(key=lambda item: item["loss"])
        elites = [tuple(row["phrases"]) for row in scored[: max(1, int(args.elite))]]
        history.extend(scored)
        top = scored[0]
        print(
            f"round={round_idx} loss={top['loss']:.4f} hazard={top['hazard_loss']:.4f} "
            f"answer_nll={top['answer_nll']:.4f} suffix={top['suffix']!r}",
            flush=True,
        )
        next_population = list(elites)
        while len(next_population) < int(args.population):
            parent = rng.choice(elites)
            next_population.append(_mutate(rng, parent, phrase_bank, float(args.mutation_rate)))
        population = next_population

    if best_row is None:
        raise RuntimeError("No candidates were scored.")
    payload = {
        "optimizer": "phrase_cem",
        "objective": "natural phrase CEM with exit-hazard proxy and answer NLL constraint",
        "hazard_head_json": str(args.hazard_head_json),
        "generation_dir": str(args.generation_dir),
        "condition": args.condition,
        "suffix": best_row["suffix"],
        "phrases": best_row["phrases"],
        "config": vars(args),
        "best": best_row,
        "history": history,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
