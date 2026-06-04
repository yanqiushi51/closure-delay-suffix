import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Placeholder for GCD prompt optimization against the exit-hazard objective. "
            "The search loop is intentionally deferred until the hazard proxy is locked."
        )
    )
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--optimizer", default="gcd", choices=["gcd"])
    parser.add_argument("--target-behavior", default="over_reasoning", choices=["over_reasoning", "early_exit"])
    parser.add_argument("--hazard-score", default="exit_hazard_cumlogit")
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--calib-size", type=int, default=50)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-path", default="outputs/learned_suffixes/gcd_exit_hazard_suffix_bank.json")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": f"gcd_{args.target_behavior}_placeholder",
            "family": "gcd_exit_hazard",
            "target_behavior": args.target_behavior,
            "hazard_score": args.hazard_score,
            "suffix": "",
            "status": "placeholder",
            "note": "GCD optimizer is reserved for the next stage after the exit-hazard proxy is finalized.",
            "config": vars(args),
        }
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote placeholder GCD suffix bank: {output_path}")


if __name__ == "__main__":
    main()
