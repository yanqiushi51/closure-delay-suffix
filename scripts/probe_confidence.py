import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.closure import (
    build_reference_trajectory,
    closure_curve_summary as legacy_closure_summary,
    progress_risk_diagnostics,
    score_closure_trajectory,
)
from closure_delay.confidence import (
    confidence_curve_diagnostics,
    confidence_curve_for_trajectory,
    confidence_curve_summary,
)
from closure_delay.data import load_gsm8k_dataset
from closure_delay.model import LocalCausalLM
from closure_delay.repetition import repetition_summary
from closure_delay.runtime import ensure_dir, now_iso, set_seed, write_csv, write_json
from closure_delay.utility import numeric_correct
from closure_delay.viz import plot_closure_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Probe direction certainty and confidence readiness on clean baseline.")
    parser.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", default="outputs/confidence_probe/qwen25_15b")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--min-baseline-tokens", type=int, default=80)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--closure-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    plot_dir = ensure_dir(output_dir / "plots") if not args.no_viz else None

    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Questions: {args.n_questions}, max_new_tokens: {args.max_new_tokens}")

    model = LocalCausalLM(args.model_path, device=args.device)
    dataset = load_gsm8k_dataset(split=args.dataset_split, n_samples=args.n_questions, seed=args.seed)

    print(f"\nGenerating {len(dataset)} clean baselines...")
    trajectories = []
    confidence_results = []
    example_rows = []

    for idx, record in enumerate(dataset, start=1):
        print(f"  [{idx}/{len(dataset)}] {record['id']}")
        start = time.perf_counter()
        trace = model.generate_trace(
            prompt=record["prompt"],
            suffix="",
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - start

        trajectory = build_reference_trajectory(
            record=record,
            trace=trace,
            tokenizer=model.tokenizer,
            fractions=args.fractions,
            continuation_tokens=args.continuation_tokens,
            closure_tokens=args.closure_tokens,
            min_baseline_tokens=args.min_baseline_tokens,
        )

        if not trajectory.valid:
            print(f"    SKIP: {trajectory.reason}")
            trajectories.append(trajectory)
            continue

        score_closure_trajectory(model, trajectory, suffix="")

        curve_rows = confidence_curve_for_trajectory(model, trajectory, suffix="", temperature=args.temperature)

        confidence_results.append(curve_rows)
        trajectories.append(trajectory)

        is_correct = numeric_correct(trace.response_text, record["answer"])
        rep = repetition_summary(trace.response_text)

        for row in curve_rows:
            example_rows.append({
                "id": record["id"],
                "fraction": row["fraction"],
                "token_index": row["token_index"],
                "c_dir": row["c_dir"],
                "c_conf": row["c_conf"],
                "e_dir": row["e_dir"],
                "p_finalize": row["p_finalize"],
                "direction_entropy": row["direction_entropy"],
                "top_direction": row["top_direction"],
                "top_prob": row["top_prob"],
                "u_t": row["u_t"],
                "z_pos": row["z_pos"],
                "z_neg": row["z_neg"],
                "baseline_length": trajectory.baseline_length,
                "baseline_correct": is_correct,
                "latency_sec": elapsed,
                "tokens_per_sec": trace.generated_token_count / elapsed if elapsed > 0 else None,
                **rep,
            })

    valid_trajectories = [t for t in trajectories if t.valid]
    print(f"\nValid: {len(valid_trajectories)}/{len(trajectories)}")

    curve_summary = confidence_curve_summary(confidence_results)
    diagnostics = confidence_curve_diagnostics(curve_summary)

    legacy_curve = legacy_closure_summary(valid_trajectories)
    legacy_diag = progress_risk_diagnostics(valid_trajectories)

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print("\n=== C_dir Curve ===")
    for f, m in zip(curve_summary["c_dir_curve"]["fractions"], curve_summary["c_dir_curve"]["means"]):
        print(f"  fraction={f:.1f}: C_dir={_fmt(m)}")
    print(f"  spearman rho={_fmt(diagnostics['c_dir']['spearman_rho'])}, p={_fmt(diagnostics['c_dir']['spearman_p'])}")
    print(f"  late_early_gap={_fmt(diagnostics['c_dir']['late_early_gap'])}")

    print("\n=== C_conf Curve ===")
    for f, m in zip(curve_summary["c_conf_curve"]["fractions"], curve_summary["c_conf_curve"]["means"]):
        print(f"  fraction={f:.1f}: C_conf={_fmt(m)}")
    print(f"  spearman rho={_fmt(diagnostics['c_conf']['spearman_rho'])}, p={_fmt(diagnostics['c_conf']['spearman_p'])}")
    print(f"  late_early_gap={_fmt(diagnostics['c_conf']['late_early_gap'])}")

    print("\n=== E_dir Curve ===")
    for f, m in zip(curve_summary["e_dir_curve"]["fractions"], curve_summary["e_dir_curve"]["means"]):
        print(f"  fraction={f:.1f}: E_dir={_fmt(m)}")

    print("\n=== Legacy Clean Curve (reference) ===")
    print(f"  progress_risk_spearman: {legacy_diag['progress_risk_spearman']}")
    print(f"  late_early_gap: {legacy_diag['late_early_gap']}")

    payload = {
        "created_at": now_iso(),
        "phase": "confidence_probe_clean_baseline",
        "config": vars(args),
        "n_total": len(trajectories),
        "n_valid": len(valid_trajectories),
        "c_dir_diagnostics": diagnostics["c_dir"],
        "c_conf_diagnostics": diagnostics["c_conf"],
        "e_dir_diagnostics": diagnostics["e_dir"],
        "curve_summary": curve_summary,
        "legacy_closure_curve": legacy_curve,
        "legacy_diagnostics": legacy_diag,
    }

    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "example_metrics.csv", example_rows)

    if plot_dir:
        c_dir_plot = {
            "c_dir": {"attacked_risk_curve": curve_summary["c_dir_curve"]},
        }
        plot_closure_curves(c_dir_plot, str(plot_dir / "c_dir_curve.png"))

        c_conf_plot = {
            "c_conf": {"attacked_risk_curve": curve_summary["c_conf_curve"]},
        }
        plot_closure_curves(c_conf_plot, str(plot_dir / "c_conf_curve.png"))

        e_dir_plot = {
            "e_dir": {"attacked_risk_curve": curve_summary["e_dir_curve"]},
        }
        plot_closure_curves(e_dir_plot, str(plot_dir / "e_dir_curve.png"))

    print(f"\nDone. Output: {output_dir}")
    print(f"  summary: {output_dir / 'summary.json'}")
    print(f"  examples: {output_dir / 'example_metrics.csv'}")


if __name__ == "__main__":
    main()
