import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Placeholder for CTS learned suffix optimization. The optimizer is intentionally not implemented until closure calibration passes."
    )
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--target-tau", type=float, required=True)
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--calib-size", type=int, default=50)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-path", default="outputs/learned_suffixes/suffix_bank_learned.json")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": f"learned_tau_{str(args.target_tau).replace('.', 'p')}_placeholder",
            "family": "learned",
            "target_tau": args.target_tau,
            "suffix": "",
            "status": "placeholder",
            "note": "CTS optimizer is not implemented yet. Run closure calibration first, then replace this placeholder with a learned suffix.",
            "config": vars(args),
        }
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote placeholder learned suffix bank: {output_path}")


if __name__ == "__main__":
    main()
