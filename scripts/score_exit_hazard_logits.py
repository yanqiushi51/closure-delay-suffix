import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard import (
    CONTINUE_MARKER_PROBE_PHRASES,
    EXIT_PROBE_PHRASES,
    EXIT_MARKER_PROBE_PHRASES,
    REASONING_PROBE_PHRASES,
    build_prompt_text,
    first_token_ids,
    load_metrics,
    load_text_rows,
    parse_float,
    running_max,
    running_min,
)
from closure_delay.model import LocalCausalLM
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score per-prefix logit features used by the exit-hazard proxy.")
    parser.add_argument("output_dir", help="Merged generation directory.")
    parser.add_argument("--condition", default="decode_gate_2p4")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-subdir", default="exit_hazard_logits")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _score_sequence(
    model: LocalCausalLM,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    exit_ids: Sequence[int],
    reasoning_ids: Sequence[int],
    exit_marker_ids: Sequence[int],
    continue_marker_ids: Sequence[int],
    eos_id: int | None,
) -> Dict[str, List[float]]:
    full_ids = list(prompt_ids) + list(response_ids)
    if not prompt_ids or not response_ids:
        return {}

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    start = len(prompt_ids)
    end = len(prompt_ids) + len(response_ids)

    with torch.inference_mode():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, start:end, :].detach().float()
        log_denom = torch.logsumexp(logits, dim=-1)
        max_logit = torch.max(logits, dim=-1).values
        pmax = torch.exp(max_logit - log_denom)

        if eos_id is not None and 0 <= int(eos_id) < logits.shape[-1]:
            eos_prob = torch.exp(logits[:, int(eos_id)] - log_denom)
        else:
            eos_prob = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)

        if exit_ids:
            exit_log_mass = torch.logsumexp(logits[:, list(exit_ids)], dim=-1)
        else:
            exit_log_mass = torch.full((logits.shape[0],), -30.0, dtype=torch.float32, device=logits.device)
        if reasoning_ids:
            reasoning_log_mass = torch.logsumexp(logits[:, list(reasoning_ids)], dim=-1)
        else:
            reasoning_log_mass = torch.full((logits.shape[0],), -30.0, dtype=torch.float32, device=logits.device)
        margin = exit_log_mass - reasoning_log_mass
        if exit_marker_ids:
            exit_marker_log_mass = torch.logsumexp(logits[:, list(exit_marker_ids)], dim=-1)
        else:
            exit_marker_log_mass = torch.full((logits.shape[0],), -30.0, dtype=torch.float32, device=logits.device)
        if continue_marker_ids:
            continue_marker_log_mass = torch.logsumexp(logits[:, list(continue_marker_ids)], dim=-1)
        else:
            continue_marker_log_mass = torch.full((logits.shape[0],), -30.0, dtype=torch.float32, device=logits.device)
        marker_margin = exit_marker_log_mass - continue_marker_log_mass

    return {
        "exit_logit_pmax": pmax.cpu().numpy().astype(float).tolist(),
        "exit_logit_eos_prob": eos_prob.cpu().numpy().astype(float).tolist(),
        "exit_logit_exit_logmass": exit_log_mass.cpu().numpy().astype(float).tolist(),
        "exit_logit_reasoning_logmass": reasoning_log_mass.cpu().numpy().astype(float).tolist(),
        "exit_logit_margin": margin.cpu().numpy().astype(float).tolist(),
        "exit_marker_logit_margin": marker_margin.cpu().numpy().astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    root = Path(args.output_dir)
    out_dir = root / args.eval_subdir
    metrics_by_id = load_metrics(root / "example_decode_gate_metrics.csv", args.condition)
    texts_by_id = load_text_rows(root / "generation_texts.json", args.condition)
    example_ids = sorted(set(metrics_by_id) & set(texts_by_id))
    if args.max_samples > 0:
        example_ids = example_ids[: int(args.max_samples)]
    example_ids = example_ids[int(args.shard_index) :: int(args.num_shards)]
    if not example_ids:
        raise RuntimeError("No examples to score.")

    model = LocalCausalLM(args.model_path, device=args.device)
    tokenizer = model.tokenizer
    exit_ids = first_token_ids(tokenizer, EXIT_PROBE_PHRASES)
    reasoning_ids = first_token_ids(tokenizer, REASONING_PROBE_PHRASES)
    exit_marker_ids = first_token_ids(tokenizer, EXIT_MARKER_PROBE_PHRASES)
    continue_marker_ids = first_token_ids(tokenizer, CONTINUE_MARKER_PROBE_PHRASES)
    eos_id = tokenizer.eos_token_id

    rows: List[Dict] = []
    scored_examples = 0
    skipped_examples = 0
    for local_idx, example_id in enumerate(example_ids, start=1):
        text_row = texts_by_id[example_id]
        prompt = str(text_row.get("prompt", ""))
        response_text = str(text_row.get("response_text", ""))
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        if not response_ids:
            skipped_examples += 1
            continue

        prompt_ids = tokenizer(build_prompt_text(tokenizer, prompt), add_special_tokens=True)["input_ids"]
        scored = _score_sequence(
            model,
            prompt_ids,
            response_ids,
            exit_ids,
            reasoning_ids,
            exit_marker_ids,
            continue_marker_ids,
            eos_id,
        )
        if not scored:
            skipped_examples += 1
            continue

        margins = [float(value) for value in scored["exit_logit_margin"]]
        runmax = running_max(margins)
        runmin = running_min(margins)
        marker_margins = [float(value) for value in scored["exit_marker_logit_margin"]]
        marker_runmax = running_max(marker_margins)
        pos_cum: List[float] = []
        neg_cum: List[float] = []
        marker_pos_cum: List[float] = []
        marker_neg_cum: List[float] = []
        pos_total = 0.0
        neg_total = 0.0
        prev = margins[0]
        marker_pos_total = 0.0
        marker_neg_total = 0.0
        marker_prev = marker_margins[0]
        for margin, marker_margin in zip(margins, marker_margins):
            pos_total += max(0.0, float(margin) - float(prev))
            neg_total += max(0.0, float(prev) - float(margin))
            pos_cum.append(float(pos_total))
            neg_cum.append(float(neg_total))
            prev = float(margin)
            marker_pos_total += max(0.0, float(marker_margin) - float(marker_prev))
            marker_neg_total += max(0.0, float(marker_prev) - float(marker_margin))
            marker_pos_cum.append(float(marker_pos_total))
            marker_neg_cum.append(float(marker_neg_total))
            marker_prev = float(marker_margin)

        closure_fraction = parse_float(metrics_by_id[example_id].get("first_closure_marker_char_ratio"))
        generated_count = len(response_ids)
        for idx in range(generated_count):
            token_index = idx + 1
            fraction = float(token_index / max(generated_count, 1))
            rows.append(
                {
                    "id": example_id,
                    "fraction": fraction,
                    "token_index": float(token_index),
                    "generated_token_count": float(generated_count),
                    "closure_marker_hit": (
                        1.0 if closure_fraction is not None and fraction >= float(closure_fraction) else 0.0
                    ),
                    "exit_logit_pmax": float(scored["exit_logit_pmax"][idx]),
                    "exit_logit_eos_prob": float(scored["exit_logit_eos_prob"][idx]),
                    "exit_logit_margin": float(margins[idx]),
                    "exit_logit_exit_logmass": float(scored["exit_logit_exit_logmass"][idx]),
                    "exit_logit_reasoning_logmass": float(scored["exit_logit_reasoning_logmass"][idx]),
                    "exit_logit_margin_runmax": float(runmax[idx]),
                    "exit_logit_margin_runmin": float(runmin[idx]),
                    "exit_logit_margin_pos_cumsum": float(pos_cum[idx]),
                    "exit_logit_margin_neg_cumsum": float(neg_cum[idx]),
                    "exit_marker_logit_margin": float(marker_margins[idx]),
                    "exit_marker_logit_margin_runmax": float(marker_runmax[idx]),
                    "exit_marker_logit_margin_pos_cumsum": float(marker_pos_cum[idx]),
                    "exit_marker_logit_margin_neg_cumsum": float(marker_neg_cum[idx]),
                }
            )
        scored_examples += 1
        if args.progress_every > 0 and local_idx % int(args.progress_every) == 0:
            print(
                f"progress shard={args.shard_index}/{args.num_shards} "
                f"examples={local_idx}/{len(example_ids)} rows={len(rows)}",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "exit_hazard_logit_points.csv", rows)
    write_json(
        out_dir / "exit_hazard_logit_report.json",
        {
            "created_at": now_iso(),
            "condition": args.condition,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "n_requested_examples": len(example_ids),
            "n_scored_examples": scored_examples,
            "n_skipped_examples": skipped_examples,
            "n_points": len(rows),
            "exit_first_token_ids": list(exit_ids),
            "reasoning_first_token_ids": list(reasoning_ids),
            "exit_marker_first_token_ids": list(exit_marker_ids),
            "continue_marker_first_token_ids": list(continue_marker_ids),
        },
    )
    print(f"done: {out_dir} examples={scored_examples} points={len(rows)}")


if __name__ == "__main__":
    main()
