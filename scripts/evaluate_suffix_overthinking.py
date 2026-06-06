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
from closure_delay.branching import branching_summary
from closure_delay.exit_hazard_torch import (
    DifferentiableExitHazardHead,
    exit_logit_features_from_logits,
    exit_process_scores,
)
from closure_delay.model import LocalCausalLM
from closure_delay.runtime import now_iso, write_csv, write_json
from closure_delay.utility import numeric_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate whether a suffix lowers exit hazard and induces overthinking.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hazard-head-json", required=True)
    parser.add_argument("--suffix-json", help="JSON produced by scripts/optimize_suffix.py")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=768)
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
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", default="outputs/exit_hazard/suffix_overthinking_eval")
    return parser.parse_args()


def _load_suffix(args: argparse.Namespace) -> str:
    if args.suffix:
        return str(args.suffix)
    if args.suffix_json:
        payload = json.loads(Path(args.suffix_json).read_text(encoding="utf-8"))
        return str(payload.get("suffix", ""))
    return ""


def _score_response(
    model: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    prompt: str,
    suffix: str,
    response_ids: List[int],
    args: argparse.Namespace,
) -> Dict[str, float | int | None]:
    if not response_ids:
        return {
            "mean_raw_hazard": None,
            "first_cross_token": None,
            "post_exit_tokens": None,
            "max_cumprob": None,
        }
    tokenizer = model.tokenizer
    prompt_text = model.build_prompt_text(prompt, suffix)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    full_ids = list(prompt_ids) + list(response_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
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
    crossing = next(
        (idx + 1 for idx, value in enumerate(cumprob.detach().cpu().tolist()) if value >= float(args.hazard_threshold)),
        None,
    )
    return {
        "mean_raw_hazard": float(torch.mean(raw).detach().cpu()),
        "mean_cumlogit": float(torch.mean(cumlogit).detach().cpu()),
        "max_cumprob": float(torch.max(cumprob).detach().cpu()),
        "first_cross_token": crossing,
        "post_exit_tokens": int(len(response_ids) - crossing) if crossing is not None else 0,
        "closure_mean": float(process["q_closure"].mean().detach().cpu()),
        "answer_survival_mean": float(process["answer_survival"].mean().detach().cpu()),
        "verify_abs_mean": float(process["verify_abs"].mean().detach().cpu()),
        "verify_relative_mean": float(process["verify_relative"].mean().detach().cpu()),
        "verify_mean": float(process["verify_prob"].mean().detach().cpu()),
        "verify_evidence_mean": float(process["verify_evidence"].mean().detach().cpu()),
        "drift_mean": float(process["drift_prob"].mean().detach().cpu()),
        "pcg_sum": float(process["pcg"].sum().detach().cpu()),
        "pcg_mean": float(process["pcg"].mean().detach().cpu()),
        "vpcg_sum": float(process["vpcg"].sum().detach().cpu()),
        "vpcg_mean": float(process["vpcg"].mean().detach().cpu()),
    }


def _condition_row(rows: List[Dict], condition: str) -> Dict:
    use = [row for row in rows if row["condition"] == condition]
    lengths = [float(row["generated_tokens"]) for row in use]
    post_exit = [float(row["post_exit_tokens"] or 0) for row in use]
    correct = [1.0 if row["correct"] else 0.0 for row in use]
    mean_hazard = [float(row["mean_raw_hazard"]) for row in use if row["mean_raw_hazard"] is not None]
    pcg_sum = [float(row["pcg_sum"]) for row in use if row.get("pcg_sum") is not None]
    vpcg_sum = [float(row["vpcg_sum"]) for row in use if row.get("vpcg_sum") is not None]
    drift = [float(row["drift_mean"]) for row in use if row.get("drift_mean") is not None]
    verify = [float(row["verify_mean"]) for row in use if row.get("verify_mean") is not None]
    verify_abs = [float(row["verify_abs_mean"]) for row in use if row.get("verify_abs_mean") is not None]
    verify_relative = [float(row["verify_relative_mean"]) for row in use if row.get("verify_relative_mean") is not None]
    branch_rate = [float(row["branch_marker_rate"]) for row in use if row.get("branch_marker_rate") is not None]
    branch_count = [float(row["branch_marker_count"]) for row in use if row.get("branch_marker_count") is not None]
    return {
        "condition": condition,
        "n": len(use),
        "generated_tokens_mean": float(np.mean(lengths)) if lengths else None,
        "post_exit_tokens_mean": float(np.mean(post_exit)) if post_exit else None,
        "correct_rate": float(np.mean(correct)) if correct else None,
        "mean_raw_hazard": float(np.mean(mean_hazard)) if mean_hazard else None,
        "pcg_sum_mean": float(np.mean(pcg_sum)) if pcg_sum else None,
        "vpcg_sum_mean": float(np.mean(vpcg_sum)) if vpcg_sum else None,
        "verify_mean": float(np.mean(verify)) if verify else None,
        "verify_abs_mean": float(np.mean(verify_abs)) if verify_abs else None,
        "verify_relative_mean": float(np.mean(verify_relative)) if verify_relative else None,
        "drift_mean": float(np.mean(drift)) if drift else None,
        "branch_marker_count_mean": float(np.mean(branch_count)) if branch_count else None,
        "branch_marker_rate_mean": float(np.mean(branch_rate)) if branch_rate else None,
    }


def main() -> None:
    args = parse_args()
    suffix = _load_suffix(args)
    model = LocalCausalLM(args.model_path, device=args.device)
    head = DifferentiableExitHazardHead.from_files(args.hazard_head_json, device=model.device)
    head.eval()
    dataset = load_gsm8k_dataset(split=args.dataset_split, n_samples=int(args.n_samples), seed=int(args.seed))
    rows: List[Dict] = []
    for item in dataset:
        for condition, condition_suffix in [("baseline", ""), ("suffix", suffix)]:
            trace = model.generate_trace(
                prompt=item["prompt"],
                suffix=condition_suffix,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
            )
            score = _score_response(
                model,
                head,
                item["prompt"],
                condition_suffix,
                trace.generated_ids,
                args,
            )
            rows.append(
                {
                    "id": item["id"],
                    "condition": condition,
                    "answer": item["answer"],
                    "suffix": condition_suffix,
                    "generated_tokens": trace.generated_token_count,
                    "correct": numeric_correct(trace.response_text, item["answer"]),
                    "response_text": trace.response_text,
                    **branching_summary(trace.response_text, trace.generated_token_count),
                    **score,
                }
            )
            print(
                f"{condition} {item['id']} len={trace.generated_token_count} "
                f"post_exit={score['post_exit_tokens']} correct={rows[-1]['correct']}",
                flush=True,
            )
    summary = [_condition_row(rows, "baseline"), _condition_row(rows, "suffix")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "suffix_overthinking_examples.csv", rows)
    write_csv(out_dir / "suffix_overthinking_summary.csv", summary)
    write_json(
        out_dir / "suffix_overthinking_report.json",
        {
            "created_at": now_iso(),
            "hazard_head_json": str(args.hazard_head_json),
            "suffix": suffix,
            "config": vars(args),
            "summary": summary,
        },
    )
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
