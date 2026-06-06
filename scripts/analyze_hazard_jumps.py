import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard_torch import DifferentiableExitHazardHead, exit_logit_features_from_logits
from closure_delay.model import LocalCausalLM
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect semantic text around major exit-hazard jumps.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--examples-csv", required=True)
    parser.add_argument("--condition", default="suffix")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--top-jumps-per-example", type=int, default=3)
    parser.add_argument("--window-tokens", type=int, default=48)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_rows(path: Path, condition: str, max_examples: int) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("condition")) != condition:
                continue
            rows.append(row)
            if max_examples > 0 and len(rows) >= max_examples:
                break
    return rows


def _score_curve(model: LocalCausalLM, head: DifferentiableExitHazardHead, prompt: str, suffix: str, response_text: str):
    tokenizer = model.tokenizer
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    response_ids = [int(tok) for tok in response_ids]
    prompt_text = model.build_prompt_text(prompt, suffix)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    full_ids = list(prompt_ids) + response_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        start = len(prompt_ids)
        end = start + len(response_ids)
        hidden = outputs.hidden_states[head.config.layer][0, start:end, :].float()
        logits = outputs.logits[0, start:end, :].float()
        logit_features = exit_logit_features_from_logits(logits, tokenizer)
        raw = head(hidden, logit_features)
        cumprob, cumlogit = head.cumulative_scores(raw)
    return response_ids, raw.detach().cpu().numpy(), cumlogit.detach().cpu().numpy()


def _window_text(tokenizer, response_ids: List[int], token_index: int, window_tokens: int) -> str:
    start = max(0, int(token_index) - int(window_tokens))
    end = min(len(response_ids), int(token_index) + int(window_tokens))
    return tokenizer.decode(response_ids[start:end], skip_special_tokens=True).replace("\n", "\\n")


def _phase_heuristic(text: str) -> str:
    low = text.lower()
    if any(x in low for x in ["final answer", "answer:", "therefore", "so the answer", "the answer is"]):
        return "finalization_or_answer"
    if any(x in low for x in ["check", "verify", "confirm", "review", "trace", "summarize"]):
        return "verification_or_review"
    if any(x in low for x in ["let", "suppose", "set ", "equation", "calculate", "compute", "total"]):
        return "calculation_or_setup"
    if any(x in low for x in ["wait", "mistake", "actually", "instead", "correct"]):
        return "correction_or_backtrack"
    return "other"


def main() -> None:
    args = parse_args()
    rows = _read_rows(Path(args.examples_csv), args.condition, int(args.max_examples))
    if not rows:
        raise RuntimeError("No matching examples.")
    model = LocalCausalLM(args.model_path, device=args.device)
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=model.device)
    head.eval()
    out_rows: List[Dict] = []
    for idx, row in enumerate(rows, start=1):
        response_text = str(row.get("response_text", ""))
        response_ids, raw, cumlogit = _score_curve(
            model,
            head,
            str(row.get("prompt", "")),
            str(row.get("suffix", "")),
            response_text,
        )
        if len(cumlogit) < 3:
            continue
        deltas = np.diff(cumlogit)
        top_indices = np.argsort(deltas)[::-1][: int(args.top_jumps_per_example)]
        for rank, delta_idx in enumerate(top_indices, start=1):
            token_index = int(delta_idx + 2)
            text = _window_text(model.tokenizer, response_ids, token_index, int(args.window_tokens))
            out_rows.append(
                {
                    "id": row.get("id"),
                    "condition": args.condition,
                    "rank": rank,
                    "token_index": token_index,
                    "fraction": float(token_index / max(len(response_ids), 1)),
                    "jump_delta": float(deltas[delta_idx]),
                    "raw_hazard": float(raw[token_index - 1]),
                    "cumlogit": float(cumlogit[token_index - 1]),
                    "generated_tokens": len(response_ids),
                    "correct": row.get("correct"),
                    "phase_heuristic": _phase_heuristic(text),
                    "text_window": text,
                }
            )
        if idx % 10 == 0:
            print(f"progress {idx}/{len(rows)} jumps={len(out_rows)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "hazard_jump_windows.csv", out_rows)
    phase_counts: Dict[str, int] = {}
    for item in out_rows:
        phase_counts[item["phase_heuristic"]] = phase_counts.get(item["phase_heuristic"], 0) + 1
    write_json(
        out_dir / "hazard_jump_report.json",
        {
            "created_at": now_iso(),
            "examples_csv": args.examples_csv,
            "condition": args.condition,
            "n_examples": len(rows),
            "n_jumps": len(out_rows),
            "phase_counts": phase_counts,
        },
    )
    print(f"done: {out_dir} jumps={len(out_rows)}")


if __name__ == "__main__":
    main()
