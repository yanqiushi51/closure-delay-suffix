import argparse
import json
import random
import re
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


DEFAULT_SUFFIX_PREFIX = "Use the following reasoning strategy:"


PHRASE_BANK = [
    # Non-exit continuation: extend the reasoning without pointing at the final answer.
    "develop the intermediate structure",
    "enumerate the constraints",
    "map the quantities before solving",
    "keep deriving relationships among the quantities",
    "expand the intermediate derivation before numeric substitution",
    "describe how each quantity is connected to the others",
    "carry the symbolic relation forward before calculating",
    "make the dependency chain explicit",
    # Exploratory reasoning: ask for alternate structure, not post-answer checking.
    "try a symbolic formulation",
    "build a small table of cases",
    "derive the relation in two independent ways",
    "inspect how each variable changes",
    "reason from the units before calculating",
    "test a simple example before generalizing",
    "compare the direct and inverse calculation",
    "translate the wording into equations before solving",
    # Maintain uncertainty: avoid early commitment to one interpretation/path.
    "avoid committing to a single interpretation too early",
    "keep multiple candidate interpretations active",
    "separate assumptions from derived facts",
    "state what is known before deciding what follows",
    "distinguish givens, unknowns, and derived quantities",
    "keep the calculation path provisional until the quantities are mapped",
    "mark which steps are assumptions and which are consequences",
    "consider whether another quantity could be the target",
    # Structured reasoning: increase useful intermediate work.
    "list the known quantities",
    "define variables explicitly",
    "construct the equation step by step",
    "organize the reasoning into setup, relation, and computation",
    "track each variable through the derivation",
    "show the transformation from words to mathematical form",
    "write the intermediate equation before simplifying",
    "connect each arithmetic operation to a stated quantity",
    # Verification is retained as a minority baseline direction.
    "check for hidden assumptions",
    "recompute the key quantities",
    "inspect the boundary cases",
    "cross-check the intermediate values",
    "look for possible off-by-one errors",
    "keep the reasoning explicit",
    "review the equation setup",
    "double-check unit conversions",
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
    parser.add_argument("--suffix-prefix", default=DEFAULT_SUFFIX_PREFIX)
    parser.add_argument("--phrase-bank-json", help="Optional JSON phrase bank. Accepts list[str], list[dict], or dict[str, list].")
    parser.add_argument("--include-default-phrase-bank", action="store_true")
    parser.add_argument("--exclude-phrase-regex", default="", help="Regex for phrases to exclude from the loaded bank.")
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--hazard-start-frac", type=float, default=0.55)
    parser.add_argument("--hazard-end-frac", type=float, default=0.95)
    parser.add_argument("--hazard-loss-scale", type=float, default=0.001)
    parser.add_argument("--direct-margin-weight", type=float, default=0.0)
    parser.add_argument(
        "--shape-objective",
        choices=["mean-hazard", "mean-cumlogit", "delta-cumlogit", "rise-control", "rise-redistribute", "vpcg"],
        default="vpcg",
    )
    parser.add_argument("--closure-threshold", type=float, default=0.30)
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
    parser.add_argument("--plateau-weight", type=float, default=1.0)
    parser.add_argument("--pcg-weight", type=float, default=0.25)
    parser.add_argument("--early-closure-weight", type=float, default=0.25)
    parser.add_argument("--jump-weight", type=float, default=0.10)
    parser.add_argument("--jump-margin", type=float, default=0.08)
    parser.add_argument("--jump-eps", type=float, default=0.05)
    parser.add_argument("--rise-bin-count", type=int, default=4)
    parser.add_argument("--target-rise-count", type=int, default=2)
    parser.add_argument("--target-rise-bins", default="")
    parser.add_argument("--target-rise-magnitude", type=float, default=0.20)
    parser.add_argument("--rise-target-weight", type=float, default=1.0)
    parser.add_argument("--rise-offtarget-weight", type=float, default=0.50)
    parser.add_argument("--rise-total-weight", type=float, default=0.25)
    parser.add_argument("--rise-initial-weight", type=float, default=0.25)
    parser.add_argument("--rise-suppress-weight", type=float, default=1.0)
    parser.add_argument("--rise-transport-weight", type=float, default=1.0)
    parser.add_argument("--rise-overlap-weight", type=float, default=0.25)
    parser.add_argument("--drift-weight", type=float, default=0.25)
    parser.add_argument("--answer-loss-weight", type=float, default=2.0)
    parser.add_argument("--answer-nll-margin", type=float, default=4.0)
    parser.add_argument("--answer-template", default=" Final answer: {answer}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="outputs/learned_suffixes/phrase_cem_exit_hazard_suffix.json")
    return parser.parse_args()


def _format_suffix(phrases: Sequence[str], prefix: str = DEFAULT_SUFFIX_PREFIX) -> str:
    body = "; ".join(phrase.strip().rstrip(".") for phrase in phrases if phrase.strip())
    if not body:
        return ""
    clean_prefix = str(prefix).strip()
    if not clean_prefix:
        return f"{body}."
    if clean_prefix[-1] in ":,;":
        return f"{clean_prefix} {body}."
    return f"{clean_prefix} {body}."


def load_phrase_bank(path: str | None = None, exclude_regex: str = "") -> List[str]:
    if not path:
        phrases = list(PHRASE_BANK)
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        phrases = _flatten_phrase_payload(payload)
    if exclude_regex:
        pattern = re.compile(exclude_regex, flags=re.IGNORECASE)
        phrases = [phrase for phrase in phrases if not pattern.search(phrase)]
    return _dedupe_phrases(phrases)


def merge_default_phrase_bank(phrases: Sequence[str]) -> List[str]:
    return _dedupe_phrases(list(PHRASE_BANK) + [str(phrase) for phrase in phrases])


def _dedupe_phrases(phrases: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for phrase in phrases:
        clean = str(phrase).strip().rstrip(".")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _flatten_phrase_payload(payload) -> List[str]:
    if isinstance(payload, list):
        phrases = []
        for item in payload:
            if isinstance(item, str):
                phrases.append(item)
            elif isinstance(item, dict):
                phrase = item.get("phrase") or item.get("text")
                if phrase:
                    phrases.append(str(phrase))
            else:
                raise ValueError("phrase bank list items must be strings or objects with a phrase/text field")
        return phrases
    if isinstance(payload, dict):
        phrases = []
        for value in payload.values():
            if not isinstance(value, list):
                raise ValueError("phrase bank dict values must be lists")
            phrases.extend(_flatten_phrase_payload(value))
        return phrases
    raise ValueError("phrase bank JSON must be a list or dict")


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

    phrase_bank = load_phrase_bank(args.phrase_bank_json, args.exclude_phrase_regex)
    if args.include_default_phrase_bank and args.phrase_bank_json:
        phrase_bank = merge_default_phrase_bank(phrase_bank)
        if args.exclude_phrase_regex:
            pattern = re.compile(args.exclude_phrase_regex, flags=re.IGNORECASE)
            phrase_bank = [phrase for phrase in phrase_bank if not pattern.search(phrase)]
    if len(phrase_bank) < int(args.phrases_per_suffix):
        raise RuntimeError("Phrase bank is smaller than --phrases-per-suffix after filtering.")
    print(f"loaded phrase bank: {len(phrase_bank)} phrases", flush=True)
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
            suffix = _format_suffix(candidate, args.suffix_prefix)
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
            f"round={round_idx} loss={top['loss']:.4f} shape={top['hazard_loss']:.4f} "
            f"vpcg={top.get('vpcg_mean', 0.0):.4f} pcg={top.get('pcg_mean', 0.0):.4f} "
            f"rise={top.get('rise_total', 0.0):.4f} base_rise={top.get('baseline_rise_total', 0.0):.4f} "
            f"transport={top.get('rise_transport_loss', 0.0):.4f} suppress={top.get('rise_suppress_loss', 0.0):.4f} "
            f"verify={top.get('verify_mean', 0.0):.4f} "
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
        "objective": "natural phrase CEM with post-closure verification shape loss and answer NLL constraint",
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
