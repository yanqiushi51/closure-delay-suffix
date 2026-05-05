"""Train and evaluate internal state probes for direction certainty and readiness.

Pipeline:
  1. Generate clean baselines on GSM8K
  2. Detect final-answer onset positions
  3. Extract hidden states at each checkpoint position
  4. Build direction and readiness labels
  5. Train linear probes with leave-one-out cross-validation
  6. Report C_dir and C_conf curves
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.data import load_gsm8k_dataset
from closure_delay.internals import collect_all_positions, extract_hidden_trajectory
from closure_delay.model import LocalCausalLM
from closure_delay.onset import find_closure_onset, onset_token_index
from closure_delay.probes_internal import (
    DirectionProbe,
    ReadinessProbe,
    fraction_curve,
    DIRECTION_NAMES,
)
from closure_delay.repetition import repetition_summary
from closure_delay.runtime import ensure_dir, now_iso, set_seed, write_csv, write_json
from closure_delay.state_labels import (
    build_direction_labels,
    build_ready_labels,
)
from closure_delay.stats import safe_spearman_correlation
from closure_delay.utility import numeric_correct


def parse_args():
    p = argparse.ArgumentParser(description="Train internal state probes on hidden states")
    p.add_argument("--model-path", default="/data/LLM/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--output-dir", default="outputs/state_probes/qwen25_15b")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--layers", nargs="+", type=int, default=[8, 12, 16, 20, 24, 27])
    p.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--delta", type=int, default=32, help="ready label: onset within delta tokens")
    p.add_argument("--future-window", type=int, default=32, help="direction label: future token window")
    p.add_argument("--probe-C", type=float, default=1.0, help="LogisticRegression regularization")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    device = args.device
    print(f"Model: {args.model_path}\nDevice: {device}")
    model = LocalCausalLM(args.model_path, device=device)

    # ---- Step 1: Generate clean baselines ----
    print(f"\nLoading GSM8K train: n={args.n_questions}")
    dataset = load_gsm8k_dataset(split="train", n_samples=args.n_questions, seed=args.seed)
    print(f"Generating {len(dataset)} clean baselines...")

    records = []  # per-sample dicts

    for idx, record in enumerate(dataset, start=1):
        rid = record["id"]
        print(f"  [{idx}/{len(dataset)}] {rid}", end="", flush=True)

        start = time.perf_counter()
        trace = model.generate_trace(
            prompt=record["prompt"],
            suffix="",
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - start

        response_text = trace.response_text
        response_ids = trace.generated_ids
        resp_len = len(response_ids)

        # Detect onset
        onset_info = find_closure_onset(response_text)
        onset_tok = onset_token_index(model.tokenizer, response_text, onset_info)

        # Extract hidden states at stride positions
        hidden = extract_hidden_trajectory(
            model,
            prompt=record["prompt"],
            suffix="",
            response_ids=response_ids,
            layers=args.layers,
            stride=args.stride,
        )
        positions = hidden["positions"]

        # Build labels
        if onset_tok is not None and onset_tok > 0:
            ready_y = build_ready_labels(onset_tok, positions, delta=args.delta)
        else:
            ready_y = np.zeros(len(positions), dtype=np.float32)
        dir_y = build_direction_labels(
            response_text, positions,
            onset_token_idx=onset_tok, future_window=args.future_window, delta=args.delta,
        )

        records.append({
            "id": rid,
            "prompt": record["prompt"],
            "answer": record["answer"],
            "response_text": response_text,
            "response_ids": response_ids,
            "response_length": resp_len,
            "onset_found": onset_info["found"],
            "onset_token": onset_tok,
            "onset_marker": onset_info.get("marker"),
            "hidden": hidden,
            "positions": positions,
            "ready_labels": ready_y,
            "direction_labels": dir_y,
            "is_correct": numeric_correct(response_text, record["answer"]),
            "latency_sec": elapsed,
            "tokens_per_sec": resp_len / elapsed if elapsed > 0 else None,
            "repetition": repetition_summary(response_text),
        })
        n_ready = int(ready_y.sum())
        n_fin = int((dir_y == 4).sum())
        print(f"  len={resp_len} onset={onset_tok} ready={n_ready} finalize={n_fin}")

    # ---- Step 2: Report label stats ----
    print("\n=== Label Statistics ===")
    onset_found_rate = sum(r["onset_found"] for r in records) / len(records)
    print(f"  onset found: {onset_found_rate:.2%}")
    all_dir = np.concatenate([r["direction_labels"] for r in records])
    n_fin = int((all_dir == 1).sum())
    print(f"  continue: {len(all_dir) - n_fin}, finalize: {n_fin}")
    all_ready = np.concatenate([r["ready_labels"] for r in records])
    print(f"  ready=1: {int(all_ready.sum())}/{len(all_ready)} ({all_ready.mean():.2%})")

    # ---- Step 3: Leave-one-out probe evaluation ----
    print("\n=== Probe Evaluation (leave-one-out) ===")

    results_by_layer = {}
    for layer_idx in args.layers:
        layer_key = f"layer_{layer_idx}"

        # Collect all states and labels for this layer
        X_all = []
        dir_y_all = []
        ready_y_all = []
        pos_all = []
        sample_ids = []
        sample_lengths = []  # for fraction curve alignment

        for r in records:
            hidden = r["hidden"]
            if layer_key not in hidden or len(hidden[layer_key]) == 0:
                continue
            states = hidden[layer_key]
            positions = np.array(r["positions"])
            n = len(states)
            X_all.append(states)
            dir_y_all.append(r["direction_labels"])
            ready_y_all.append(r["ready_labels"])
            pos_all.append(positions)
            sample_ids.extend([r["id"]] * n)
            sample_lengths.append(r["response_length"])

        if len(X_all) == 0:
            print(f"  {layer_key}: no data")
            continue

        # Leave-one-out
        n_samples = len(X_all)
        dir_preds_list = []
        ready_preds_list = []
        dir_test_ids = []
        ready_test_ids = []

        for holdout_idx in range(n_samples):
            # Train on all but one
            X_train_parts = [X_all[i] for i in range(n_samples) if i != holdout_idx]
            dir_train_parts = [dir_y_all[i] for i in range(n_samples) if i != holdout_idx]
            ready_train_parts = [ready_y_all[i] for i in range(n_samples) if i != holdout_idx]

            X_train = np.concatenate(X_train_parts, axis=0)
            dir_train = np.concatenate(dir_train_parts, axis=0)
            ready_train = np.concatenate(ready_train_parts, axis=0)

            X_test = X_all[holdout_idx]
            dir_test = dir_y_all[holdout_idx]
            ready_test = ready_y_all[holdout_idx]

            # Train direction probe
            if len(np.unique(dir_train)) >= 2 and len(dir_test) > 0:
                dp = DirectionProbe(C=args.probe_C)
                try:
                    dp.fit(X_train, dir_train)
                    probs = dp.predict_proba(X_test)
                    dir_preds_list.append(probs)
                    dir_test_ids.extend([records[holdout_idx]["id"]] * len(X_test))
                except Exception:
                    pass

            # Train readiness probe
            if len(np.unique(ready_train)) >= 2 and ready_train.sum() > 1 and (1 - ready_train).sum() > 1 and len(ready_test) > 0:
                rp = ReadinessProbe(C=args.probe_C)
                try:
                    rp.fit(X_train, ready_train)
                    probs = rp.predict_proba(X_test)
                    ready_preds_list.append(probs)
                    ready_test_ids.extend([records[holdout_idx]["id"]] * len(X_test))
                except Exception:
                    pass

        if not dir_preds_list:
            print(f"  {layer_key}: probe training failed")
            continue

        # Aggregate predictions
        all_dir_probs = np.concatenate(dir_preds_list, axis=0)
        all_ready_probs = np.concatenate(ready_preds_list, axis=0) if ready_preds_list else None

        c_dir_values = DirectionProbe.c_dir(all_dir_probs)
        e_dir_values = DirectionProbe.e_dir(all_dir_probs)
        c_conf_values = ReadinessProbe.c_conf(all_ready_probs) if all_ready_probs is not None else None

        # Per-sample C_dir and C_conf curves
        dir_curves = []
        conf_curves = []
        e_dir_curves = []
        offset = 0
        for r in records:
            hidden_r = r["hidden"]
            if layer_key not in hidden_r or len(hidden_r[layer_key]) == 0:
                continue
            n = len(hidden_r[layer_key])
            if offset + n > len(c_dir_values):
                break
            dir_curves.append((r["positions"], c_dir_values[offset:offset + n]))
            e_dir_curves.append((r["positions"], e_dir_values[offset:offset + n]))
            if c_conf_values is not None and offset + n <= len(c_conf_values):
                conf_curves.append((r["positions"], c_conf_values[offset:offset + n]))
            offset += n

        # Aggregate into fraction-based curve
        dir_positions_list = [np.array(p) for p, _ in dir_curves]
        dir_values_list = [np.array(v) for _, v in dir_curves]
        dir_curve = fraction_curve(dir_positions_list, sample_lengths, dir_values_list, args.fractions)

        e_dir_positions_list = [np.array(p) for p, _ in e_dir_curves]
        e_dir_values_list = [np.array(v) for _, v in e_dir_curves]
        e_dir_curve = fraction_curve(e_dir_positions_list, sample_lengths, e_dir_values_list, args.fractions)

        conf_curve = None
        if conf_curves and c_conf_values is not None:
            conf_positions_list = [np.array(p) for p, _ in conf_curves]
            conf_values_list = [np.array(v) for _, v in conf_curves]
            conf_curve = fraction_curve(conf_positions_list, sample_lengths, conf_values_list, args.fractions)

        # Spearman correlation
        dir_flat_fractions = []
        dir_flat_values = []
        for p, v in dir_curves:
            dir_flat_fractions.extend(p)
            dir_flat_values.extend(v)
        dir_spearman = safe_spearman_correlation(dir_flat_fractions, dir_flat_values)

        conf_spearman = (None, None)
        if conf_curves:
            conf_flat_fractions = []
            conf_flat_values = []
            for p, v in conf_curves:
                conf_flat_fractions.extend(p)
                conf_flat_values.extend(v)
            conf_spearman = safe_spearman_correlation(conf_flat_fractions, conf_flat_values)

        results_by_layer[layer_key] = {
            "c_dir_curve": dir_curve,
            "c_dir_spearman": {"rho": dir_spearman[0], "p": dir_spearman[1]},
            "c_dir_late_early_gap": _late_early_gap(dir_curve),
            "c_conf_curve": conf_curve,
            "c_conf_spearman": {"rho": conf_spearman[0], "p": conf_spearman[1]},
            "c_conf_late_early_gap": _late_early_gap(conf_curve) if conf_curve else None,
            "e_dir_curve": e_dir_curve,
            "n_samples_valid": n_samples,
            "n_total_states": len(all_dir_probs),
        }

    # ---- Step 4: Print results ----
    for layer_key in sorted(results_by_layer, key=lambda k: int(k.split("_")[1])):
        r = results_by_layer[layer_key]
        print(f"\n--- {layer_key} ---")
        print(f"  C_dir curve:")
        c = r["c_dir_curve"]
        for f, m in zip(c["fractions"], c["means"]):
            print(f"    fraction={f:.1f}: C_dir={_fmt(m)}")
        print(f"    spearman rho={_fmt(r['c_dir_spearman']['rho'])}, p={_fmt(r['c_dir_spearman']['p'])}")
        print(f"    late_early_gap={_fmt(r['c_dir_late_early_gap'])}")

        if r["c_conf_curve"] is not None:
            print(f"  C_conf curve:")
            cc = r["c_conf_curve"]
            for f, m in zip(cc["fractions"], cc["means"]):
                print(f"    fraction={f:.1f}: C_conf={_fmt(m)}")
            print(f"    spearman rho={_fmt(r['c_conf_spearman']['rho'])}, p={_fmt(r['c_conf_spearman']['p'])}")
            print(f"    late_early_gap={_fmt(r['c_conf_late_early_gap'])}")

    # ---- Step 5: Output ----
    example_rows = _build_example_rows(records, results_by_layer, args)

    def _serializable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {kk: _serializable(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_serializable(vv) for vv in v]
        return v

    payload = {
        "created_at": now_iso(),
        "phase": "state_probe_clean_baseline",
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "label_stats": {
            "onset_found_rate": onset_found_rate,
            "direction_distribution": {"continue": int(len(all_dir) - n_fin), "finalize": int(n_fin)},
            "ready_positive_rate": float(all_ready.mean()),
        },
        "results_by_layer": {
            k: _serializable({kk: vv for kk, vv in v.items()})
            for k, v in results_by_layer.items()
        },
        "samples": [{k: v for k, v in r.items() if k not in ("hidden", "response_ids")} for r in records],
    }
    write_json(output_dir / "summary.json", _serializable(payload))
    write_csv(output_dir / "example_metrics.csv", example_rows)

    print(f"\nDone. Output: {output_dir}")
    print(f"  summary: {output_dir / 'summary.json'}")


def _late_early_gap(curve: dict) -> float | None:
    """Mean difference between late (fraction >= 0.7) and early (fraction <= 0.3)."""
    if not curve or not curve.get("fractions"):
        return None
    early_vals = [m for f, m in zip(curve["fractions"], curve["means"]) if f <= 0.3 and not np.isnan(m)]
    late_vals = [m for f, m in zip(curve["fractions"], curve["means"]) if f >= 0.7 and not np.isnan(m)]
    if not early_vals or not late_vals:
        return None
    return float(np.mean(late_vals) - np.mean(early_vals))


def _build_example_rows(records, results_by_layer, args):
    """Build per-sample per-fraction rows for CSV output."""
    rows = []
    for layer_key, layer_result in results_by_layer.items():
        if layer_result["c_dir_curve"] is None:
            continue
        dir_curve = layer_result["c_dir_curve"]
        conf_curve = layer_result.get("c_conf_curve")
        e_dir_curve = layer_result.get("e_dir_curve")
        for i, frac in enumerate(dir_curve["fractions"]):
            row = {
                "layer": layer_key,
                "fraction": frac,
                "c_dir_mean": dir_curve["means"][i] if i < len(dir_curve["means"]) else None,
                "c_dir_count": dir_curve["counts"][i] if i < len(dir_curve["counts"]) else None,
            }
            if conf_curve and i < len(conf_curve["means"]):
                row["c_conf_mean"] = conf_curve["means"][i]
                row["c_conf_count"] = conf_curve["counts"][i]
            if e_dir_curve and i < len(e_dir_curve["means"]):
                row["e_dir_mean"] = e_dir_curve["means"][i]
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
