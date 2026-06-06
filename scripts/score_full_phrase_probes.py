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
    EXIT_PROBE_PHRASES,
    REASONING_PROBE_PHRASES,
    build_prompt_text,
    load_metrics,
    load_text_rows,
    parse_float,
    running_max,
)
from closure_delay.model import LocalCausalLM
from closure_delay.probes import logmeanexp
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score non-approximate full-phrase probe margins on sparse prefixes.")
    parser.add_argument("output_dir", help="Merged generation directory.")
    parser.add_argument("--condition", default="decode_gate_2p4")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-subdir", default="full_phrase_probes")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def _target_logprobs_for_context(
    model: LocalCausalLM,
    context_text: str,
    target_texts: Sequence[str],
    probe_batch_size: int,
) -> List[float]:
    tokenizer = model.tokenizer
    context_ids = tokenizer(context_text, add_special_tokens=True)["input_ids"]
    encoded_targets = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in target_texts]
    encoded_targets = [[int(tok) for tok in ids] for ids in encoded_targets]
    scores: List[float] = []
    batch_size = max(1, int(probe_batch_size))
    for batch_start in range(0, len(encoded_targets), batch_size):
        batch_targets = encoded_targets[batch_start : batch_start + batch_size]
        rows = [list(context_ids) + ids for ids in batch_targets]
        scores.extend(_target_logprobs_for_rows(model, rows, len(context_ids), batch_targets))
    return scores


def _target_logprobs_for_rows(
    model: LocalCausalLM,
    rows: Sequence[Sequence[int]],
    context_len: int,
    encoded_targets: Sequence[Sequence[int]],
) -> List[float]:
    tokenizer = model.tokenizer
    max_len = max(len(row) for row in rows)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros_like(input_ids, device=model.device)
    for row_idx, row in enumerate(rows):
        input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=model.device)
        attention_mask[row_idx, : len(row)] = 1

    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    scores: List[float] = []
    for row_idx, target_ids in enumerate(encoded_targets):
        if not target_ids:
            scores.append(float("-inf"))
            continue
        start = max(context_len - 1, 0)
        end = start + len(target_ids)
        values = token_log_probs[row_idx, start:end]
        scores.append(float(values.mean().detach().cpu().item()) if values.numel() else float("-inf"))
    return scores


def _prefix_indices(n_tokens: int, stride: int) -> List[int]:
    if n_tokens <= 0:
        return []
    stride = max(1, int(stride))
    indices = list(range(stride, n_tokens + 1, stride))
    if 1 not in indices:
        indices.insert(0, 1)
    if n_tokens not in indices:
        indices.append(n_tokens)
    return sorted(set(max(1, min(n_tokens, idx)) for idx in indices))


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    out_dir = root / args.eval_subdir
    metrics_by_id = load_metrics(root / "example_decode_gate_metrics.csv", args.condition)
    texts_by_id = load_text_rows(root / "generation_texts.json", args.condition)
    example_ids = sorted(set(metrics_by_id) & set(texts_by_id))
    if args.max_samples > 0:
        example_ids = example_ids[: int(args.max_samples)]
    if not example_ids:
        raise RuntimeError("No examples to score.")

    model = LocalCausalLM(args.model_path, device=args.device)
    tokenizer = model.tokenizer
    rows: List[Dict] = []
    scored = 0
    skipped = 0
    all_probes = list(EXIT_PROBE_PHRASES) + list(REASONING_PROBE_PHRASES)

    for local_idx, example_id in enumerate(example_ids, start=1):
        text_row = texts_by_id[example_id]
        prompt = str(text_row.get("prompt", ""))
        response_text = str(text_row.get("response_text", ""))
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        response_ids = [int(tok) for tok in response_ids]
        if not response_ids:
            skipped += 1
            continue
        prompt_text = build_prompt_text(tokenizer, prompt)
        margins = []
        point_rows = []
        closure_fraction = parse_float(metrics_by_id[example_id].get("first_closure_marker_char_ratio"))
        for token_index in _prefix_indices(len(response_ids), int(args.stride)):
            prefix_text = tokenizer.decode(response_ids[:token_index], skip_special_tokens=True)
            context = prompt_text + prefix_text
            scores = _target_logprobs_for_context(model, context, all_probes, int(args.probe_batch_size))
            n_exit = len(EXIT_PROBE_PHRASES)
            exit_score = logmeanexp(scores[:n_exit])
            reasoning_score = logmeanexp(scores[n_exit:])
            margin = float(exit_score - reasoning_score)
            margins.append(margin)
            fraction = float(token_index / max(len(response_ids), 1))
            point_rows.append(
                {
                    "id": example_id,
                    "fraction": fraction,
                    "token_index": float(token_index),
                    "generated_token_count": float(len(response_ids)),
                    "closure_marker_hit": (
                        1.0 if closure_fraction is not None and fraction >= float(closure_fraction) else 0.0
                    ),
                    "full_probe_margin": margin,
                    "full_probe_risk": float(1.0 / (1.0 + np.exp(-np.clip(margin, -30.0, 30.0)))),
                }
            )
        runmax = running_max(margins)
        for row, value in zip(point_rows, runmax):
            row["full_probe_margin_runmax"] = float(value)
            rows.append(row)
        scored += 1
        if args.progress_every > 0 and local_idx % int(args.progress_every) == 0:
            print(f"progress examples={local_idx}/{len(example_ids)} scored={scored} rows={len(rows)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "full_phrase_probe_points.csv", rows)
    write_json(
        out_dir / "full_phrase_probe_report.json",
        {
            "created_at": now_iso(),
            "condition": args.condition,
            "stride": int(args.stride),
            "probe_batch_size": int(args.probe_batch_size),
            "n_requested_examples": len(example_ids),
            "n_scored_examples": scored,
            "n_skipped_examples": skipped,
            "n_points": len(rows),
            "exit_probe_phrases": list(EXIT_PROBE_PHRASES),
            "reasoning_probe_phrases": list(REASONING_PROBE_PHRASES),
        },
    )
    print(f"done: {out_dir} examples={scored} points={len(rows)}")


if __name__ == "__main__":
    main()
