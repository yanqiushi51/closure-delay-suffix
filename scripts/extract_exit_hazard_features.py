import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard import build_prompt_text, event_fractions, load_metrics, load_text_rows
from closure_delay.model import LocalCausalLM
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hidden states used by the exit-hazard proxy.")
    parser.add_argument("output_dir", help="Merged generation directory.")
    parser.add_argument("--condition", default="decode_gate_2p4")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[24])
    parser.add_argument("--horizon", type=float, default=0.15)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-subdir", default="exit_hazard_features")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _append_layer_states(
    hidden_states: Sequence[torch.Tensor],
    layer_buffers: Dict[int, List[np.ndarray]],
    layers: Sequence[int],
    start: int,
    end: int,
) -> None:
    for layer in layers:
        if 0 <= int(layer) < len(hidden_states):
            states = hidden_states[int(layer)][0, start:end, :].detach().to(torch.float16).cpu().numpy()
            layer_buffers[int(layer)].append(states)


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
        raise RuntimeError("No examples to extract.")

    model = LocalCausalLM(args.model_path, device=args.device)
    tokenizer = model.tokenizer

    rows: List[Dict] = []
    layer_buffers: Dict[int, List[np.ndarray]] = {int(layer): [] for layer in args.layers}
    scored = 0
    skipped_no_event = 0
    skipped_empty = 0

    for local_idx, example_id in enumerate(example_ids, start=1):
        text_row = texts_by_id[example_id]
        prompt = str(text_row.get("prompt", ""))
        response_text = str(text_row.get("response_text", ""))
        closure_fraction, drift_fraction, exit_fraction = event_fractions(metrics_by_id[example_id], response_text)
        if exit_fraction is None:
            skipped_no_event += 1
            continue

        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        if not response_ids:
            skipped_empty += 1
            continue

        prompt_text = build_prompt_text(tokenizer, prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        full_ids = list(prompt_ids) + list(response_ids)
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
        with torch.inference_mode():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            _append_layer_states(
                hidden_states=outputs.hidden_states,
                layer_buffers=layer_buffers,
                layers=args.layers,
                start=prompt_len,
                end=prompt_len + response_len,
            )

        for idx in range(response_len):
            token_index = idx + 1
            fraction = float(token_index / max(response_len, 1))
            rows.append(
                {
                    "id": example_id,
                    "fraction": fraction,
                    "token_index": float(token_index),
                    "generated_token_count": float(response_len),
                    "closure_fraction": closure_fraction,
                    "drift_fraction": drift_fraction,
                    "exit_fraction": exit_fraction,
                    "exit_horizon_label": 1 if fraction >= float(exit_fraction) - float(args.horizon) else 0,
                    "already_exit_label": 1 if fraction >= float(exit_fraction) else 0,
                    "closure_marker_hit": (
                        1.0 if closure_fraction is not None and fraction >= float(closure_fraction) else 0.0
                    ),
                }
            )
        scored += 1
        if args.progress_every > 0 and local_idx % int(args.progress_every) == 0:
            print(
                f"progress shard={args.shard_index}/{args.num_shards} "
                f"examples={local_idx}/{len(example_ids)} scored={scored} rows={len(rows)}",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_shapes = {}
    for layer, chunks in layer_buffers.items():
        if not chunks:
            continue
        array = np.concatenate(chunks, axis=0).astype(np.float16, copy=False)
        np.save(out_dir / f"hidden_layer_{layer}.npy", array)
        feature_shapes[f"layer_{layer}"] = list(array.shape)

    write_csv(out_dir / "exit_hazard_feature_rows.csv", rows)
    write_json(
        out_dir / "exit_hazard_feature_report.json",
        {
            "created_at": now_iso(),
            "condition": args.condition,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "layers": [int(layer) for layer in args.layers],
            "horizon": float(args.horizon),
            "n_requested_examples": len(example_ids),
            "n_scored_examples": scored,
            "n_skipped_no_event": skipped_no_event,
            "n_skipped_empty": skipped_empty,
            "n_rows": len(rows),
            "feature_shapes": feature_shapes,
        },
    )
    print(f"done: {out_dir} examples={scored} rows={len(rows)}")


if __name__ == "__main__":
    main()
