import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.closure_experiments import ClosureValidationConfig, run_closure_validation


def parse_args():
    parser = argparse.ArgumentParser(description="Validate closure-risk trajectories and length-ratio effects.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", default="outputs/closure_validation/qwen25_15b")
    parser.add_argument("--n-questions", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--suffix-bank-path", default="data/suffix_bank.json")
    parser.add_argument("--no-verbosity", action="store_true")
    parser.add_argument("--no-suffix-bank", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--allow-truncated-baseline", action="store_true")
    parser.add_argument("--min-baseline-tokens", type=int, default=80)
    parser.add_argument("--continuation-tokens", type=int, default=24)
    parser.add_argument("--closure-tokens", type=int, default=24)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ClosureValidationConfig(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        n_questions=args.n_questions,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        dataset_split=args.dataset_split,
        suffix_bank_path=args.suffix_bank_path,
        include_verbosity=not args.no_verbosity,
        include_suffix_bank=not args.no_suffix_bank,
        make_viz=not args.no_viz,
        allow_truncated_baseline=args.allow_truncated_baseline,
        min_baseline_tokens=args.min_baseline_tokens,
        continuation_tokens=args.continuation_tokens,
        closure_tokens=args.closure_tokens,
        fractions=args.fractions,
    )
    result = run_closure_validation(config)
    output_dir = Path(result["output_dir"])
    print("\nDone.")
    print(f"  summary: {output_dir / 'summary.json'}")
    print(f"  examples: {output_dir / 'example_metrics.csv'}")
    print(f"  conditions: {output_dir / 'condition_summary.csv'}")
    print("  calibration:", result["payload"]["calibration"]["curve_shift_vs_length_ratio"])


if __name__ == "__main__":
    main()
