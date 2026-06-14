import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.data import load_gsm8k_dataset
from closure_delay.exit_hazard_torch import (
    DifferentiableExitHazardHead,
    exit_logit_features_from_logits,
    exit_process_scores,
)
from closure_delay.model import LocalCausalLM
from closure_delay.runtime import now_iso, write_json
from closure_delay.utility import numeric_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one reasoning trace and export its exit-hazard curve.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--prompt-id", default="gsm8k_train_0")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--condition", choices=["baseline", "suffix"], required=True)
    parser.add_argument("--suffix-json")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--hazard-threshold", type=float, default=0.30)
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
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_suffix(args: argparse.Namespace) -> str:
    if args.condition == "baseline":
        return ""
    if args.suffix:
        return str(args.suffix)
    if args.suffix_json:
        payload = json.loads(Path(args.suffix_json).read_text(encoding="utf-8"))
        return str(payload.get("suffix", ""))
    return ""


def _select_prompt(prompt_id: str, split: str) -> Dict:
    for item in load_gsm8k_dataset(split=split):
        if item["id"] == prompt_id:
            return item
    raise RuntimeError(f"Prompt id not found: {prompt_id}")


def _score_curve(
    model: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    prompt: str,
    suffix: str,
    response_ids: List[int],
    args: argparse.Namespace,
) -> Dict:
    tokenizer = model.tokenizer
    prompt_text = model.build_prompt_text(prompt, suffix)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    full_ids = list(prompt_ids) + list(response_ids)
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
        process = exit_process_scores(
            logits,
            tokenizer,
            cumprob,
            closure_threshold=float(args.hazard_threshold),
            closure_eps=float(args.closure_eps),
            answer_logprob_threshold=float(args.answer_logprob_threshold),
            answer_eps=float(args.answer_eps),
            answer_survival_mode=str(args.answer_survival_mode),
            verify_logprob_threshold=float(args.verify_logprob_threshold),
            verify_eps=float(args.verify_eps),
            verify_mode=str(args.verify_mode),
            verify_relative_weight=float(args.verify_relative_weight),
            verify_relative_eps=float(args.verify_relative_eps),
            reasoning_verify_offset=float(args.reasoning_verify_offset),
            drift_logprob_threshold=float(args.drift_logprob_threshold),
            drift_eps=float(args.drift_eps),
        )

    raw_np = raw.detach().cpu().numpy().astype(float)
    cumprob_np = cumprob.detach().cpu().numpy().astype(float)
    cumlogit_np = cumlogit.detach().cpu().numpy().astype(float)
    q_closure_np = process["q_closure"].detach().cpu().numpy().astype(float)
    answer_survival_np = process["answer_survival"].detach().cpu().numpy().astype(float)
    verify_np = process["verify_prob"].detach().cpu().numpy().astype(float)
    drift_np = process["drift_prob"].detach().cpu().numpy().astype(float)

    crossing = next((idx + 1 for idx, value in enumerate(cumprob_np) if value >= float(args.hazard_threshold)), None)
    return {
        "tokens": list(range(1, len(response_ids) + 1)),
        "exit_hazard": raw_np.tolist(),
        "exit_hazard_cumprob": cumprob_np.tolist(),
        "exit_hazard_cumlogit": cumlogit_np.tolist(),
        "closure_prob": q_closure_np.tolist(),
        "answer_survival": answer_survival_np.tolist(),
        "verify_prob": verify_np.tolist(),
        "drift_prob": drift_np.tolist(),
        "first_cross_token": crossing,
        "mean_raw_hazard": float(np.mean(raw_np)) if raw_np.size else None,
        "mean_cumlogit": float(np.mean(cumlogit_np)) if cumlogit_np.size else None,
        "max_cumprob": float(np.max(cumprob_np)) if cumprob_np.size else None,
        "closure_mean": float(np.mean(q_closure_np)) if q_closure_np.size else None,
        "answer_survival_mean": float(np.mean(answer_survival_np)) if answer_survival_np.size else None,
        "verify_mean": float(np.mean(verify_np)) if verify_np.size else None,
        "drift_mean": float(np.mean(drift_np)) if drift_np.size else None,
    }


def main() -> None:
    args = parse_args()
    suffix = _load_suffix(args)
    item = _select_prompt(str(args.prompt_id), str(args.dataset_split))

    model = LocalCausalLM(str(args.model_path), device=str(args.device))
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=model.device)
    head.eval()

    trace = model.generate_trace(
        prompt=item["prompt"],
        suffix=suffix,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=False,
    )
    curve = _score_curve(model, head, item["prompt"], suffix, trace.generated_ids, args)
    payload = {
        "created_at": now_iso(),
        "condition": args.condition,
        "device": str(args.device),
        "model_path": str(args.model_path),
        "hazard_head_json": str(args.hazard_head_json),
        "prompt_id": item["id"],
        "prompt": item["prompt"],
        "answer": item["answer"],
        "suffix": suffix,
        "response_text": trace.response_text,
        "generated_tokens": trace.generated_token_count,
        "correct": numeric_correct(trace.response_text, item["answer"]),
        **curve,
    }
    write_json(args.output_json, payload)
    print(
        f"{args.condition} {item['id']} tokens={trace.generated_token_count} "
        f"correct={payload['correct']} cross={payload['first_cross_token']} "
        f"mean_raw={payload['mean_raw_hazard']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
