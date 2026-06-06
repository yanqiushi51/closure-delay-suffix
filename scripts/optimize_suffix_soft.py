import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard_torch import DifferentiableExitHazardHead
from closure_delay.model import LocalCausalLM
from scripts.optimize_suffix import (
    _answer_nll,
    _decode_suffix,
    _hazard_loss,
    _load_examples,
    _token_allowed,
    _tokenize_suffix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous soft-suffix optimization for exit-hazard suppression.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--require-correct", action="store_true")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--suffix-length", type=int, default=12)
    parser.add_argument("--init-suffix", default="Please verify carefully before finalizing.")
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--hazard-start-frac", type=float, default=0.55)
    parser.add_argument("--hazard-end-frac", type=float, default=0.95)
    parser.add_argument("--hazard-loss-scale", type=float, default=0.001)
    parser.add_argument("--hazard-ramp-steps", type=int, default=20)
    parser.add_argument("--answer-loss-weight", type=float, default=2.0)
    parser.add_argument("--answer-nll-margin", type=float, default=4.0)
    parser.add_argument("--embedding-l2-weight", type=float, default=0.02)
    parser.add_argument("--candidate-token-filter", choices=["none", "printable-ascii", "natural-ascii"], default="printable-ascii")
    parser.add_argument("--direct-margin-weight", type=float, default=0.0)
    parser.add_argument("--answer-template", default=" Final answer: {answer}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="outputs/learned_suffixes/soft_exit_hazard_suffix.json")
    return parser.parse_args()


def _allowed_token_ids(tokenizer, mode: str) -> torch.Tensor:
    ids = []
    special = set(int(tok) for tok in tokenizer.all_special_ids)
    for token_id in range(int(getattr(tokenizer, "vocab_size", len(tokenizer)))):
        if token_id in special:
            continue
        if _token_allowed(tokenizer, token_id, mode):
            ids.append(int(token_id))
    if not ids:
        raise RuntimeError("No candidate tokens survived the token filter.")
    return torch.tensor(ids, dtype=torch.long)


def _project_suffix_ids(
    tokenizer,
    embedding_matrix: torch.Tensor,
    suffix_embeds: torch.Tensor,
    allowed_ids: torch.Tensor,
) -> list[int]:
    allowed_ids = allowed_ids.to(device=suffix_embeds.device)
    allowed = embedding_matrix[allowed_ids].float()
    allowed = F.normalize(allowed, dim=-1)
    query = F.normalize(suffix_embeds.detach().float(), dim=-1)
    scores = query @ allowed.T
    nearest = torch.argmax(scores, dim=-1)
    return [int(allowed_ids[idx].detach().cpu()) for idx in nearest]


def _soft_batch_loss(
    model,
    tokenizer,
    head: DifferentiableExitHazardHead,
    examples: Sequence,
    suffix_ids: Sequence[int],
    suffix_embeds: torch.Tensor,
    init_embeds: torch.Tensor,
    args: argparse.Namespace,
    hazard_scale: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    hazard_terms = []
    answer_terms = []
    for example in examples:
        hazard_terms.append(_hazard_loss(model, tokenizer, head, example, suffix_ids, suffix_embeds, args))
        answer_terms.append(_answer_nll(model, tokenizer, example, suffix_ids, suffix_embeds, args.answer_template))
    hazard = torch.stack(hazard_terms).mean()
    answer_nll = torch.stack(answer_terms).mean()
    answer_penalty = F.relu(answer_nll - float(args.answer_nll_margin))
    l2 = torch.mean((suffix_embeds - init_embeds) ** 2)
    scaled_hazard = float(hazard_scale) * hazard
    loss = scaled_hazard + float(args.answer_loss_weight) * answer_penalty + float(args.embedding_l2_weight) * l2
    metrics = {
        "loss": float(loss.detach().cpu()),
        "hazard_loss": float(hazard.detach().cpu()),
        "scaled_hazard_loss": float(scaled_hazard.detach().cpu()),
        "answer_nll": float(answer_nll.detach().cpu()),
        "answer_penalty": float(answer_penalty.detach().cpu()),
        "embedding_l2": float(l2.detach().cpu()),
    }
    return loss, metrics


def main() -> None:
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
    embedding = lm.model.get_input_embeddings()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=lm.device)
    init_embeds = embedding(suffix_tensor).detach()
    suffix_embeds = torch.nn.Parameter(init_embeds.clone())
    optimizer = torch.optim.AdamW([suffix_embeds], lr=float(args.lr))
    allowed_ids = _allowed_token_ids(tokenizer, str(args.candidate_token_filter))

    history = []
    best = None
    for step in range(int(args.steps)):
        batch = random.sample(examples, k=min(int(args.batch_size), len(examples)))
        ramp = min(1.0, float(step + 1) / max(float(args.hazard_ramp_steps), 1.0))
        hazard_scale = float(args.hazard_loss_scale) * ramp
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _soft_batch_loss(
            lm,
            tokenizer,
            head,
            batch,
            suffix_ids,
            suffix_embeds,
            init_embeds,
            args,
            hazard_scale,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([suffix_embeds], max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            projected_ids = _project_suffix_ids(tokenizer, embedding.weight.detach(), suffix_embeds, allowed_ids)
        row = {
            "step": int(step),
            "hazard_scale": float(hazard_scale),
            **metrics,
            "projected_suffix": _decode_suffix(tokenizer, projected_ids),
            "projected_suffix_ids": projected_ids,
        }
        history.append(row)
        if best is None or row["loss"] < best["loss"]:
            best = dict(row)
        print(
            f"step={step} loss={row['loss']:.4f} hazard={row['hazard_loss']:.4f} "
            f"scaled_hazard={row['scaled_hazard_loss']:.4f} answer_nll={row['answer_nll']:.4f} "
            f"suffix={row['projected_suffix']!r}",
            flush=True,
        )

    final_ids = best["projected_suffix_ids"] if best else _project_suffix_ids(
        tokenizer,
        embedding.weight.detach(),
        suffix_embeds,
        allowed_ids,
    )
    payload = {
        "optimizer": "soft_suffix_adamw",
        "objective": "ramped hazard_loss_scale * mean_raw_exit_hazard + answer_nll_hinge + embedding_l2",
        "hazard_head_json": str(args.hazard_head_json),
        "generation_dir": str(args.generation_dir),
        "condition": args.condition,
        "suffix": _decode_suffix(tokenizer, final_ids),
        "suffix_ids": list(final_ids),
        "best_step": None if best is None else int(best["step"]),
        "config": vars(args),
        "history": history,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
