import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard import build_prompt_text, load_metrics, load_text_rows
from closure_delay.exit_hazard_torch import (
    DifferentiableExitHazardHead,
    exit_logit_features_from_logits,
    exit_process_scores,
)
from closure_delay.model import LocalCausalLM


SLOT = "<EXIT_HAZARD_SUFFIX_SLOT>"


@dataclass
class OptimizationExample:
    example_id: str
    prompt: str
    answer: str
    response_text: str


def parse_args():
    parser = argparse.ArgumentParser(description="GCG search for suffixes that shape the exit-hazard process.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-dir", required=True, help="Directory containing generation_texts.json.")
    parser.add_argument("--condition", default="decode_gate_2p4")
    parser.add_argument("--require-correct", action="store_true")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--suffix-length", type=int, default=12)
    parser.add_argument("--init-suffix", default="Please verify carefully before finalizing.")
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--candidates-per-step", type=int, default=32)
    parser.add_argument("--candidate-token-filter", choices=["none", "printable-ascii", "natural-ascii"], default="none")
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--hazard-start-frac", type=float, default=0.20)
    parser.add_argument("--hazard-end-frac", type=float, default=0.70)
    parser.add_argument("--hazard-loss-scale", type=float, default=1.0)
    parser.add_argument("--direct-margin-weight", type=float, default=0.0)
    parser.add_argument("--shape-objective", choices=["mean-hazard", "vpcg"], default="vpcg")
    parser.add_argument("--closure-threshold", type=float, default=0.30)
    parser.add_argument("--closure-eps", type=float, default=0.08)
    parser.add_argument("--answer-logprob-threshold", type=float, default=-3.50)
    parser.add_argument("--answer-eps", type=float, default=0.60)
    parser.add_argument("--verify-logprob-threshold", type=float, default=-4.50)
    parser.add_argument("--verify-eps", type=float, default=0.80)
    parser.add_argument("--drift-logprob-threshold", type=float, default=-5.00)
    parser.add_argument("--drift-eps", type=float, default=0.80)
    parser.add_argument("--plateau-weight", type=float, default=1.0)
    parser.add_argument("--pcg-weight", type=float, default=0.25)
    parser.add_argument("--early-closure-weight", type=float, default=0.25)
    parser.add_argument("--jump-weight", type=float, default=0.10)
    parser.add_argument("--jump-margin", type=float, default=0.08)
    parser.add_argument("--jump-eps", type=float, default=0.05)
    parser.add_argument("--drift-weight", type=float, default=0.25)
    parser.add_argument("--answer-loss-weight", type=float, default=0.20)
    parser.add_argument("--answer-nll-margin", type=float, default=4.0)
    parser.add_argument("--answer-template", default=" Final answer: {answer}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="outputs/learned_suffixes/gcg_exit_hazard_suffix.json")
    return parser.parse_args()


def _load_examples(args: argparse.Namespace) -> List[OptimizationExample]:
    generation_dir = Path(args.generation_dir)
    rows = load_text_rows(generation_dir / "generation_texts.json", args.condition)
    metrics = {}
    if args.require_correct:
        metrics = load_metrics(generation_dir / "example_decode_gate_metrics.csv", args.condition)
    examples: List[OptimizationExample] = []
    for example_id, row in rows.items():
        correct_value = str(metrics.get(example_id, {}).get("generated_correct", "")).lower()
        if args.require_correct and correct_value not in {"true", "1", "yes"}:
            continue
        response_text = str(row.get("response_text", ""))
        answer = str(row.get("answer", ""))
        prompt = str(row.get("prompt", ""))
        if prompt and answer and response_text:
            examples.append(
                OptimizationExample(
                    example_id=str(row["id"]),
                    prompt=prompt,
                    answer=answer,
                    response_text=response_text,
                )
            )
    return examples[: int(args.train_size)]


def _tokenize_suffix(tokenizer, text: str, length: int) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(" verify", add_special_tokens=False)["input_ids"]
    pad_id = ids[-1]
    if len(ids) < length:
        ids = ids + [pad_id] * (length - len(ids))
    return [int(tok) for tok in ids[:length]]


def _slot_prompt_parts(tokenizer, prompt: str, suffix_ids: Sequence[int]) -> tuple[List[int], int, int]:
    prompt_text = build_prompt_text(tokenizer, f"{prompt}\n\n{SLOT}")
    slot_start = prompt_text.index(SLOT)
    slot_end = slot_start + len(SLOT)
    encoded = tokenizer(prompt_text, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = [int(tok) for tok in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    slot_positions = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > slot_start and start < slot_end
    ]
    if not slot_positions:
        raise RuntimeError("Could not locate suffix slot in tokenized prompt.")
    first = min(slot_positions)
    last = max(slot_positions) + 1
    full_ids = input_ids[:first] + list(suffix_ids) + input_ids[last:]
    return full_ids, first, first + len(suffix_ids)


def _replace_suffix_embeds(
    model,
    input_ids: Sequence[int],
    suffix_start: int,
    suffix_embeds: torch.Tensor,
) -> torch.Tensor:
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=model.device)
    embeds = model.model.get_input_embeddings()(ids)
    embeds = embeds.clone()
    embeds[:, suffix_start : suffix_start + suffix_embeds.shape[0], :] = suffix_embeds.unsqueeze(0)
    return embeds


def _answer_nll(
    model,
    tokenizer,
    example: OptimizationExample,
    suffix_ids: Sequence[int],
    suffix_embeds: torch.Tensor,
    answer_template: str,
) -> torch.Tensor:
    prompt_ids, suffix_start, _ = _slot_prompt_parts(tokenizer, example.prompt, suffix_ids)
    target_text = answer_template.format(answer=example.answer)
    target_ids = [int(tok) for tok in tokenizer(target_text, add_special_tokens=False)["input_ids"]]
    if not target_ids:
        return suffix_embeds.new_tensor(0.0)
    input_ids = prompt_ids + target_ids
    embeds = _replace_suffix_embeds(model, input_ids, suffix_start, suffix_embeds)
    attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=model.device)
    outputs = model.model(inputs_embeds=embeds, attention_mask=attention_mask)
    prompt_len = len(prompt_ids)
    logits = outputs.logits[0, prompt_len - 1 : prompt_len + len(target_ids) - 1, :].float()
    labels = torch.tensor(target_ids, dtype=torch.long, device=model.device)
    return F.cross_entropy(logits, labels, reduction="mean")


def _shape_loss(
    model,
    tokenizer,
    head: DifferentiableExitHazardHead,
    example: OptimizationExample,
    suffix_ids: Sequence[int],
    suffix_embeds: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    prompt_ids, suffix_start, _ = _slot_prompt_parts(tokenizer, example.prompt, suffix_ids)
    response_ids = tokenizer(example.response_text, add_special_tokens=False)["input_ids"]
    response_ids = [int(tok) for tok in response_ids[: int(args.max_response_tokens)]]
    if len(response_ids) < 8:
        return suffix_embeds.new_tensor(0.0)
    input_ids = prompt_ids + response_ids
    embeds = _replace_suffix_embeds(model, input_ids, suffix_start, suffix_embeds)
    attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=model.device)
    outputs = model.model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    prompt_len = len(prompt_ids)
    end = prompt_len + len(response_ids)
    hidden = outputs.hidden_states[head.config.layer][0, prompt_len:end, :].float()
    logits = outputs.logits[0, prompt_len:end, :].float()
    logit_features = exit_logit_features_from_logits(logits, tokenizer)
    raw_hazard = head(hidden, logit_features)
    n = raw_hazard.shape[0]
    start = int(max(0, min(n - 1, round(float(args.hazard_start_frac) * n))))
    stop = int(max(start + 1, min(n, round(float(args.hazard_end_frac) * n))))
    if str(getattr(args, "shape_objective", "vpcg")) == "mean-hazard":
        direct_margin = logit_features[start:stop, 0].mean()
        loss = raw_hazard[start:stop].mean() + float(getattr(args, "direct_margin_weight", 0.0)) * direct_margin
        zero = loss.new_tensor(0.0)
        return loss, {
            "closure_mean": zero,
            "pcg_mean": zero,
            "vpcg_mean": zero,
            "early_closure": zero,
            "jump_penalty": zero,
            "drift_mean": zero,
            "answer_survival_mean": zero,
            "verify_mean": zero,
        }

    cumprob, _ = head.cumulative_scores(raw_hazard)
    process = exit_process_scores(
        logits,
        tokenizer,
        cumprob,
        closure_threshold=float(getattr(args, "closure_threshold", 0.30)),
        closure_eps=float(getattr(args, "closure_eps", 0.08)),
        answer_logprob_threshold=float(getattr(args, "answer_logprob_threshold", -3.50)),
        answer_eps=float(getattr(args, "answer_eps", 0.60)),
        verify_logprob_threshold=float(getattr(args, "verify_logprob_threshold", -4.50)),
        verify_eps=float(getattr(args, "verify_eps", 0.80)),
        drift_logprob_threshold=float(getattr(args, "drift_logprob_threshold", -5.00)),
        drift_eps=float(getattr(args, "drift_eps", 0.80)),
    )
    q_closure = process["q_closure"]
    window = slice(start, stop)
    early_stop = max(2, start) if start > 1 else max(2, int(round(0.30 * n)))
    early_stop = min(max(2, early_stop), n)
    early_closure = q_closure[:early_stop].mean()
    if early_stop > 2:
        jumps = q_closure[1:early_stop] - q_closure[: early_stop - 1]
        jump_penalty = F.softplus(
            (jumps - float(args.jump_margin)) / float(args.jump_eps)
        ).mean()
    else:
        jump_penalty = q_closure.new_tensor(0.0)

    pcg = process["pcg"][window].mean()
    vpcg = process["vpcg"][window].mean()
    drift = (process["answer_survival"] * process["drift_prob"])[window].mean()
    loss = (
        -float(getattr(args, "plateau_weight", 1.0)) * vpcg
        -float(getattr(args, "pcg_weight", 0.25)) * pcg
        + float(getattr(args, "early_closure_weight", 0.25)) * early_closure
        + float(getattr(args, "jump_weight", 0.10)) * jump_penalty
        + float(getattr(args, "drift_weight", 0.25)) * drift
    )
    return loss, {
        "closure_mean": q_closure[window].mean(),
        "pcg_mean": pcg,
        "vpcg_mean": vpcg,
        "early_closure": early_closure,
        "jump_penalty": jump_penalty,
        "drift_mean": drift,
        "answer_survival_mean": process["answer_survival"][window].mean(),
        "verify_mean": process["verify_prob"][window].mean(),
    }


def _hazard_loss(
    model,
    tokenizer,
    head: DifferentiableExitHazardHead,
    example: OptimizationExample,
    suffix_ids: Sequence[int],
    suffix_embeds: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    loss, _ = _shape_loss(model, tokenizer, head, example, suffix_ids, suffix_embeds, args)
    return loss


def _batch_loss(
    model,
    tokenizer,
    head: DifferentiableExitHazardHead,
    examples: Sequence[OptimizationExample],
    suffix_ids: Sequence[int],
    args: argparse.Namespace,
    require_grad: bool,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor | None]:
    embedding = model.model.get_input_embeddings()
    suffix_tensor = torch.tensor(list(suffix_ids), dtype=torch.long, device=model.device)
    suffix_embeds = embedding(suffix_tensor).detach().clone()
    suffix_embeds.requires_grad_(require_grad)
    context = nullcontext() if require_grad else torch.no_grad()
    with context:
        hazard_terms = []
        process_metric_terms: Dict[str, List[torch.Tensor]] = {}
        answer_terms = []
        for example in examples:
            shape_term, shape_metrics = _shape_loss(model, tokenizer, head, example, suffix_ids, suffix_embeds, args)
            hazard_terms.append(shape_term)
            for key, value in shape_metrics.items():
                process_metric_terms.setdefault(key, []).append(value)
            answer_terms.append(_answer_nll(model, tokenizer, example, suffix_ids, suffix_embeds, args.answer_template))
        hazard = torch.stack(hazard_terms).mean()
        answer_nll = torch.stack(answer_terms).mean()
        answer_penalty = F.relu(answer_nll - float(args.answer_nll_margin))
        scaled_hazard = float(args.hazard_loss_scale) * hazard
        loss = scaled_hazard + float(args.answer_loss_weight) * answer_penalty
    metrics = {
        "loss": float(loss.detach().cpu()),
        "hazard_loss": float(hazard.detach().cpu()),
        "scaled_hazard_loss": float(scaled_hazard.detach().cpu()),
        "answer_nll": float(answer_nll.detach().cpu()),
        "answer_penalty": float(answer_penalty.detach().cpu()),
    }
    for key, values in process_metric_terms.items():
        metrics[key] = float(torch.stack(values).mean().detach().cpu())
    if require_grad:
        loss.backward()
        grad = suffix_embeds.grad.detach().clone()
    else:
        grad = None
    return loss.detach(), metrics, grad


def _token_allowed(tokenizer, token_id: int, mode: str) -> bool:
    if mode == "none":
        return True
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    if not text:
        return False
    if mode == "printable-ascii":
        return all(32 <= ord(ch) <= 126 or ch in "\n\t" for ch in text)
    if mode == "natural-ascii":
        if any(ord(ch) < 32 or ord(ch) > 126 for ch in text):
            return False
        if not any(ch.isalpha() for ch in text):
            return text in {".", ",", ";", ":", "-", " "}
        blocked = set("{}[]()<>/\\|_=*`~@#$%^&")
        if any(ch in blocked for ch in text):
            return False
        return True
    raise ValueError(f"Unsupported candidate token filter: {mode}")


def _candidate_token_ids(
    tokenizer,
    model,
    grad: torch.Tensor,
    current_ids: Sequence[int],
    topk: int,
    token_filter: str,
) -> List[tuple[int, int]]:
    embedding_matrix = model.model.get_input_embeddings().weight.detach().float()
    special = set(int(tok) for tok in tokenizer.all_special_ids)
    candidates: List[tuple[int, int]] = []
    for pos in range(grad.shape[0]):
        scores = grad[pos].float() @ embedding_matrix.T
        order = torch.argsort(scores, descending=False).detach().cpu().tolist()
        added = 0
        for token_id in order:
            token_id = int(token_id)
            if token_id in special or token_id == int(current_ids[pos]):
                continue
            if not _token_allowed(tokenizer, token_id, token_filter):
                continue
            candidates.append((pos, token_id))
            added += 1
            if added >= int(topk):
                break
    return candidates


def _decode_suffix(tokenizer, suffix_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(suffix_ids), skip_special_tokens=True)


def main():
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    lm = LocalCausalLM(args.model_path, device=args.device)
    for parameter in lm.model.parameters():
        parameter.requires_grad_(False)
    tokenizer = lm.tokenizer
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=lm.device)
    head.eval()
    examples = _load_examples(args)
    if not examples:
        raise RuntimeError("No optimization examples found.")
    suffix_ids = _tokenize_suffix(tokenizer, args.init_suffix, int(args.suffix_length))
    history = []

    for step in range(int(args.steps)):
        batch = random.sample(examples, k=min(int(args.batch_size), len(examples)))
        _, metrics, grad = _batch_loss(lm, tokenizer, head, batch, suffix_ids, args, require_grad=True)
        candidate_pairs = _candidate_token_ids(
            tokenizer,
            lm,
            grad,
            suffix_ids,
            int(args.topk),
            str(args.candidate_token_filter),
        )
        random.shuffle(candidate_pairs)
        candidate_pairs = candidate_pairs[: int(args.candidates_per_step)]

        best_ids = list(suffix_ids)
        best_metrics = metrics
        best_loss = metrics["loss"]
        for pos, token_id in candidate_pairs:
            trial = list(suffix_ids)
            trial[pos] = int(token_id)
            _, trial_metrics, _ = _batch_loss(lm, tokenizer, head, batch, trial, args, require_grad=False)
            if trial_metrics["loss"] < best_loss:
                best_loss = trial_metrics["loss"]
                best_metrics = trial_metrics
                best_ids = trial
        suffix_ids = best_ids
        row = {
            "step": step,
            **best_metrics,
            "suffix": _decode_suffix(tokenizer, suffix_ids),
            "suffix_ids": list(suffix_ids),
        }
        history.append(row)
        print(
            f"step={step} loss={row['loss']:.4f} shape={row['hazard_loss']:.4f} "
            f"vpcg={row.get('vpcg_mean', 0.0):.4f} pcg={row.get('pcg_mean', 0.0):.4f} "
            f"scaled_hazard={row['scaled_hazard_loss']:.4f} "
            f"answer_nll={row['answer_nll']:.4f} suffix={row['suffix']!r}",
            flush=True,
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimizer": "gcg",
        "objective": "hazard_loss_scale * process_shape_loss + answer_nll_hinge",
        "hazard_head_json": str(args.hazard_head_json),
        "generation_dir": str(args.generation_dir),
        "condition": args.condition,
        "suffix": _decode_suffix(tokenizer, suffix_ids),
        "suffix_ids": list(suffix_ids),
        "config": vars(args),
        "history": history,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
