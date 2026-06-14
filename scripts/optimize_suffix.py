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


def _target_rise_bins(args: argparse.Namespace, bin_count: int) -> List[int]:
    spec = str(getattr(args, "target_rise_bins", "") or "").strip()
    if spec:
        bins = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            bins.append(max(0, min(int(item), int(bin_count) - 1)))
        return sorted(set(bins))
    target_count = max(0, min(int(getattr(args, "target_rise_count", 2)), int(bin_count)))
    if target_count <= 0:
        return []
    step = float(bin_count) / float(target_count)
    return sorted({max(0, min(int((idx + 0.5) * step), int(bin_count) - 1)) for idx in range(target_count)})


def _rise_bin_sums(window: torch.Tensor, bin_count: int) -> torch.Tensor:
    if window.numel() < 2:
        return window.new_zeros((max(1, int(bin_count)),))
    deltas = torch.relu(window[1:] - window[:-1])
    n_deltas = int(deltas.shape[0])
    bin_rises = []
    for bin_idx in range(max(1, int(bin_count))):
        bin_start = int(round(bin_idx * n_deltas / max(1, int(bin_count))))
        bin_stop = int(round((bin_idx + 1) * n_deltas / max(1, int(bin_count))))
        if bin_stop > bin_start:
            bin_rises.append(deltas[bin_start:bin_stop].sum())
        else:
            bin_rises.append(window.new_tensor(0.0))
    return torch.stack(bin_rises)


def _target_rise_bins_from_baseline(
    args: argparse.Namespace,
    baseline_rises: torch.Tensor,
) -> List[int]:
    bin_count = int(baseline_rises.shape[0])
    spec = str(getattr(args, "target_rise_bins", "") or "").strip()
    if spec:
        return _target_rise_bins(args, bin_count)
    target_count = max(0, min(int(getattr(args, "target_rise_count", 2)), bin_count))
    if target_count <= 0:
        return []
    order = torch.argsort(baseline_rises.detach().float(), descending=False).detach().cpu().tolist()
    return sorted(int(idx) for idx in order[:target_count])


def _rise_control_loss(
    q_closure: torch.Tensor,
    start: int,
    stop: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    window = q_closure[start:stop]
    if window.numel() < 2:
        zero = q_closure.new_tensor(0.0)
        return zero, {
            "rise_control_loss": zero,
            "rise_target_loss": zero,
            "rise_offtarget_loss": zero,
            "rise_total_loss": zero,
            "rise_initial_loss": zero,
            "rise_total": zero,
            "rise_target_total": zero,
            "rise_target_count": zero,
        }

    deltas = torch.relu(window[1:] - window[:-1])
    bin_count = max(1, int(getattr(args, "rise_bin_count", 4)))
    target_bins = set(_target_rise_bins(args, bin_count))
    target_magnitude = float(getattr(args, "target_rise_magnitude", 0.20))
    bin_rises = []
    n_deltas = int(deltas.shape[0])
    for bin_idx in range(bin_count):
        bin_start = int(round(bin_idx * n_deltas / bin_count))
        bin_stop = int(round((bin_idx + 1) * n_deltas / bin_count))
        if bin_stop > bin_start:
            bin_rises.append(deltas[bin_start:bin_stop].sum())
        else:
            bin_rises.append(window.new_tensor(0.0))
    rises = torch.stack(bin_rises)

    target_losses = []
    offtarget_losses = []
    for bin_idx, rise in enumerate(bin_rises):
        if bin_idx in target_bins:
            target_losses.append((rise - target_magnitude).pow(2))
        else:
            offtarget_losses.append(rise.pow(2))
    zero = window.new_tensor(0.0)
    target_loss = torch.stack(target_losses).mean() if target_losses else zero
    offtarget_loss = torch.stack(offtarget_losses).mean() if offtarget_losses else zero
    target_total = window.new_tensor(float(len(target_bins)) * target_magnitude)
    total_loss = (rises.sum() - target_total).pow(2)
    initial_loss = window[0].pow(2)
    loss = (
        float(getattr(args, "rise_target_weight", 1.0)) * target_loss
        + float(getattr(args, "rise_offtarget_weight", 0.50)) * offtarget_loss
        + float(getattr(args, "rise_total_weight", 0.25)) * total_loss
        + float(getattr(args, "rise_initial_weight", 0.25)) * initial_loss
    )
    metrics: Dict[str, torch.Tensor] = {
        "rise_control_loss": loss,
        "rise_target_loss": target_loss,
        "rise_offtarget_loss": offtarget_loss,
        "rise_total_loss": total_loss,
        "rise_initial_loss": initial_loss,
        "rise_total": rises.sum(),
        "rise_target_total": target_total,
        "rise_target_count": window.new_tensor(float(len(target_bins))),
    }
    for bin_idx, rise in enumerate(bin_rises):
        metrics[f"rise_bin_{bin_idx}"] = rise
        metrics[f"rise_target_bin_{bin_idx}"] = window.new_tensor(
            target_magnitude if bin_idx in target_bins else 0.0
        )
    return loss, metrics


def _rise_redistribution_loss(
    q_closure: torch.Tensor,
    baseline_q_closure: torch.Tensor,
    start: int,
    stop: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    window = q_closure[start:stop]
    baseline_window = baseline_q_closure[start:stop].detach()
    bin_count = max(1, int(getattr(args, "rise_bin_count", 4)))
    rises = _rise_bin_sums(window, bin_count)
    baseline_rises = _rise_bin_sums(baseline_window, bin_count).detach()
    target_bins = set(_target_rise_bins_from_baseline(args, baseline_rises))
    target_magnitude = float(getattr(args, "target_rise_magnitude", 0.20))
    eps = window.new_tensor(1e-6)

    target = window.new_zeros((bin_count,))
    for bin_idx in target_bins:
        target[bin_idx] = 1.0
    target = target / (target.sum() + eps)
    candidate_dist = rises / (rises.sum() + eps)
    baseline_dist = baseline_rises / (baseline_rises.sum() + eps)

    transport_loss = (torch.cumsum(candidate_dist, dim=0) - torch.cumsum(target, dim=0)).pow(2).mean()
    suppress_loss = (baseline_dist * rises.pow(2)).sum()
    overlap_loss = (candidate_dist * baseline_dist).sum()
    target_loss = (
        torch.stack([(rises[bin_idx] - target_magnitude).pow(2) for bin_idx in target_bins]).mean()
        if target_bins
        else window.new_tensor(0.0)
    )
    offtarget = [bin_idx for bin_idx in range(bin_count) if bin_idx not in target_bins]
    offtarget_loss = (
        torch.stack([rises[bin_idx].pow(2) for bin_idx in offtarget]).mean()
        if offtarget
        else window.new_tensor(0.0)
    )
    target_total = window.new_tensor(float(len(target_bins)) * target_magnitude)
    total_loss = (rises.sum() - target_total).pow(2)
    initial_loss = window[0].pow(2) if window.numel() else window.new_tensor(0.0)
    loss = (
        float(getattr(args, "rise_transport_weight", 1.0)) * transport_loss
        + float(getattr(args, "rise_suppress_weight", 1.0)) * suppress_loss
        + float(getattr(args, "rise_overlap_weight", 0.25)) * overlap_loss
        + float(getattr(args, "rise_target_weight", 1.0)) * target_loss
        + float(getattr(args, "rise_offtarget_weight", 0.50)) * offtarget_loss
        + float(getattr(args, "rise_total_weight", 0.25)) * total_loss
        + float(getattr(args, "rise_initial_weight", 0.25)) * initial_loss
    )
    metrics: Dict[str, torch.Tensor] = {
        "rise_redistribute_loss": loss,
        "rise_transport_loss": transport_loss,
        "rise_suppress_loss": suppress_loss,
        "rise_overlap_loss": overlap_loss,
        "rise_target_loss": target_loss,
        "rise_offtarget_loss": offtarget_loss,
        "rise_total_loss": total_loss,
        "rise_initial_loss": initial_loss,
        "rise_total": rises.sum(),
        "baseline_rise_total": baseline_rises.sum(),
        "rise_target_total": target_total,
        "rise_target_count": window.new_tensor(float(len(target_bins))),
    }
    for bin_idx in range(bin_count):
        metrics[f"rise_bin_{bin_idx}"] = rises[bin_idx]
        metrics[f"baseline_rise_bin_{bin_idx}"] = baseline_rises[bin_idx]
        metrics[f"rise_delta_bin_{bin_idx}"] = rises[bin_idx] - baseline_rises[bin_idx]
        metrics[f"rise_target_bin_{bin_idx}"] = window.new_tensor(
            target_magnitude if bin_idx in target_bins else 0.0
        )
    return loss, metrics


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
    shape_objective = str(getattr(args, "shape_objective", "vpcg"))
    if shape_objective == "mean-hazard":
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
    if shape_objective == "mean-cumlogit":
        _, cumlogit = head.cumulative_scores(raw_hazard)
        direct_margin = logit_features[start:stop, 0].mean()
        loss = cumlogit[start:stop].mean() + float(getattr(args, "direct_margin_weight", 0.0)) * direct_margin
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
            "cumlogit_mean": cumlogit[start:stop].mean(),
        }
    if shape_objective == "delta-cumlogit":
        _, cumlogit = head.cumulative_scores(raw_hazard)
        candidate_mean = cumlogit[start:stop].mean()
        base_prompt_text = build_prompt_text(tokenizer, example.prompt)
        base_prompt_ids = tokenizer(base_prompt_text, add_special_tokens=True)["input_ids"]
        base_full_ids = list(base_prompt_ids) + list(response_ids)
        base_input_ids = torch.tensor([base_full_ids], dtype=torch.long, device=model.device)
        base_attention_mask = torch.ones_like(base_input_ids, device=model.device)
        with torch.no_grad():
            base_outputs = model.model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                output_hidden_states=True,
            )
            base_start = len(base_prompt_ids)
            base_end = base_start + len(response_ids)
            base_hidden = base_outputs.hidden_states[head.config.layer][0, base_start:base_end, :].float()
            base_logits = base_outputs.logits[0, base_start:base_end, :].float()
            base_logit_features = exit_logit_features_from_logits(base_logits, tokenizer)
            base_raw_hazard = head(base_hidden, base_logit_features)
            _, base_cumlogit = head.cumulative_scores(base_raw_hazard)
            base_mean = base_cumlogit[start:stop].mean()
        direct_margin = logit_features[start:stop, 0].mean()
        delta = candidate_mean - base_mean
        loss = delta + float(getattr(args, "direct_margin_weight", 0.0)) * direct_margin
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
            "cumlogit_mean": candidate_mean,
            "baseline_cumlogit_mean": base_mean,
            "cumlogit_delta": delta,
        }
    if shape_objective == "rise-control":
        cumprob, cumlogit = head.cumulative_scores(raw_hazard)
        q_closure = torch.sigmoid(
            (cumprob - float(getattr(args, "closure_threshold", 0.30))) / float(getattr(args, "closure_eps", 0.08))
        )
        loss, rise_metrics = _rise_control_loss(q_closure, start, stop, args)
        zero = loss.new_tensor(0.0)
        return loss, {
            "closure_mean": q_closure[start:stop].mean(),
            "pcg_mean": zero,
            "vpcg_mean": zero,
            "early_closure": q_closure[: max(1, start)].mean() if start > 0 else q_closure[:1].mean(),
            "jump_penalty": zero,
            "drift_mean": zero,
            "answer_survival_mean": zero,
            "verify_mean": zero,
            "cumlogit_mean": cumlogit[start:stop].mean(),
            **rise_metrics,
        }
    if shape_objective == "rise-redistribute":
        cumprob, cumlogit = head.cumulative_scores(raw_hazard)
        q_closure = torch.sigmoid(
            (cumprob - float(getattr(args, "closure_threshold", 0.30))) / float(getattr(args, "closure_eps", 0.08))
        )
        base_prompt_text = build_prompt_text(tokenizer, example.prompt)
        base_prompt_ids = tokenizer(base_prompt_text, add_special_tokens=True)["input_ids"]
        base_full_ids = list(base_prompt_ids) + list(response_ids)
        base_input_ids = torch.tensor([base_full_ids], dtype=torch.long, device=model.device)
        base_attention_mask = torch.ones_like(base_input_ids, device=model.device)
        with torch.no_grad():
            base_outputs = model.model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                output_hidden_states=True,
            )
            base_start = len(base_prompt_ids)
            base_end = base_start + len(response_ids)
            base_hidden = base_outputs.hidden_states[head.config.layer][0, base_start:base_end, :].float()
            base_logits = base_outputs.logits[0, base_start:base_end, :].float()
            base_logit_features = exit_logit_features_from_logits(base_logits, tokenizer)
            base_raw_hazard = head(base_hidden, base_logit_features)
            base_cumprob, base_cumlogit = head.cumulative_scores(base_raw_hazard)
            base_q_closure = torch.sigmoid(
                (base_cumprob - float(getattr(args, "closure_threshold", 0.30)))
                / float(getattr(args, "closure_eps", 0.08))
            )
        loss, rise_metrics = _rise_redistribution_loss(q_closure, base_q_closure, start, stop, args)
        zero = loss.new_tensor(0.0)
        return loss, {
            "closure_mean": q_closure[start:stop].mean(),
            "baseline_closure_mean": base_q_closure[start:stop].mean(),
            "pcg_mean": zero,
            "vpcg_mean": zero,
            "early_closure": q_closure[: max(1, start)].mean() if start > 0 else q_closure[:1].mean(),
            "jump_penalty": zero,
            "drift_mean": zero,
            "answer_survival_mean": zero,
            "verify_mean": zero,
            "cumlogit_mean": cumlogit[start:stop].mean(),
            "baseline_cumlogit_mean": base_cumlogit[start:stop].mean(),
            "cumlogit_delta": cumlogit[start:stop].mean() - base_cumlogit[start:stop].mean(),
            **rise_metrics,
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
        answer_survival_mode=str(getattr(args, "answer_survival_mode", "local")),
        verify_logprob_threshold=float(getattr(args, "verify_logprob_threshold", -4.50)),
        verify_eps=float(getattr(args, "verify_eps", 0.80)),
        verify_mode=str(getattr(args, "verify_mode", "hybrid")),
        verify_relative_weight=float(getattr(args, "verify_relative_weight", 0.50)),
        verify_relative_eps=float(getattr(args, "verify_relative_eps", 0.75)),
        reasoning_verify_offset=float(getattr(args, "reasoning_verify_offset", 0.75)),
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
        "verify_abs_mean": process["verify_abs"][window].mean(),
        "verify_relative_mean": process["verify_relative"][window].mean(),
        "verify_evidence_mean": process["verify_evidence"][window].mean(),
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
            f"verify={row.get('verify_mean', 0.0):.4f} "
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
