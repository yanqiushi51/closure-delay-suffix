import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.data import load_gsm8k_dataset
from closure_delay.model import LocalCausalLM
from closure_delay.runtime import ensure_dir, now_iso, set_seed, write_csv, write_json
from closure_delay.utility import numeric_correct


@dataclass
class BaselineGenerationConfig:
    model_path: str = "/data/LLM/Qwen2.5-7B-Instruct"
    device: str = "cuda:0"
    output_dir: str = "outputs/exit_hazard/qwen25_7b_baseline_n300"
    n_questions: int = 300
    max_new_tokens: int = 1024
    seed: int = 42
    dataset_split: str = "train"
    progress_every: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect baseline-only generations for suffix optimization.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exit_hazard/qwen25_7b_baseline_n300")
    parser.add_argument("--n-questions", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def _metric_row(record: Dict, trace, elapsed: float, max_new_tokens: int) -> Dict:
    return {
        "id": record["id"],
        "condition": "baseline",
        "target_ratio": 1.0,
        "gate_until_tokens": 0,
        "condition_max_new_tokens": int(max_new_tokens),
        "baseline_length": int(trace.generated_token_count),
        "generated_length": int(trace.generated_token_count),
        "length_ratio": 1.0,
        "baseline_truncated": bool(trace.generated_token_count >= max_new_tokens),
        "hit_gate": True,
        "over_gate_tokens": 0,
        "generated_truncated": bool(trace.generated_token_count >= max_new_tokens),
        "generated_correct": bool(numeric_correct(trace.response_text, record["answer"])),
        "latency_sec": float(elapsed),
        "tokens_per_sec": float(trace.generated_token_count / elapsed) if elapsed > 0 else None,
    }


def _text_row(record: Dict, trace, max_new_tokens: int) -> Dict:
    return {
        "id": record["id"],
        "condition": "baseline",
        "target_ratio": 1.0,
        "gate_until_tokens": 0,
        "condition_max_new_tokens": int(max_new_tokens),
        "skip_reason": None,
        "answer": record["answer"],
        "prompt": record["prompt"],
        "response_text": trace.response_text,
        "generated_token_count": int(trace.generated_token_count),
    }


def main() -> None:
    args = parse_args()
    config = BaselineGenerationConfig(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        n_questions=int(args.n_questions),
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        dataset_split=args.dataset_split,
        progress_every=int(args.progress_every),
    )
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    model = LocalCausalLM(config.model_path, device=config.device)
    dataset = load_gsm8k_dataset(split=config.dataset_split, n_samples=config.n_questions, seed=config.seed)

    metric_rows: List[Dict] = []
    text_rows: List[Dict] = []
    for idx, record in enumerate(dataset, start=1):
        start = time.perf_counter()
        trace = model.generate_trace(record["prompt"], "", max_new_tokens=config.max_new_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        metric_rows.append(_metric_row(record, trace, elapsed, config.max_new_tokens))
        text_rows.append(_text_row(record, trace, config.max_new_tokens))
        if config.progress_every > 0 and idx % config.progress_every == 0:
            correct = sum(1 for row in metric_rows if row["generated_correct"])
            print(f"progress {idx}/{len(dataset)} correct={correct}", flush=True)

    correct = sum(1 for row in metric_rows if row["generated_correct"])
    truncated = sum(1 for row in metric_rows if row["generated_truncated"])
    write_csv(output_dir / "example_decode_gate_metrics.csv", metric_rows)
    write_json(output_dir / "generation_texts.json", text_rows)
    write_json(
        output_dir / "summary.json",
        {
            "created_at": now_iso(),
            "phase": "baseline_generation_collection",
            "config": asdict(config),
            "n": len(metric_rows),
            "correct": int(correct),
            "correct_rate": float(correct / len(metric_rows)) if metric_rows else None,
            "truncated": int(truncated),
            "truncated_rate": float(truncated / len(metric_rows)) if metric_rows else None,
        },
    )
    print(f"done: {output_dir} n={len(metric_rows)} correct={correct} truncated={truncated}")


if __name__ == "__main__":
    main()
