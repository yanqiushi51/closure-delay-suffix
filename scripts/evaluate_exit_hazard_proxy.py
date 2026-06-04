import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard import (
    HAZARD_SCORE_COLUMNS,
    event_fractions,
    load_hazard_points,
    load_metrics,
    load_text_rows,
    normalize,
    normalized_auc,
    robust_minmax,
    safe_float,
)
from closure_delay.runtime import now_iso, write_csv, write_json
from closure_delay.stats import safe_spearman_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the online exit-hazard proxy.")
    parser.add_argument("output_dir", help="Merged generation directory.")
    parser.add_argument("--condition", default="decode_gate_2p4")
    parser.add_argument("--hazard-points-csv", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--jump-window", type=float, default=0.15)
    parser.add_argument("--min-lead", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-subdir", default="exit_hazard_eval")
    return parser.parse_args()


def _candidate_columns(rows: Sequence[Dict]) -> List[str]:
    available = []
    for name in HAZARD_SCORE_COLUMNS:
        if any(name in row and row[name] not in (None, "") for row in rows):
            available.append(name)
    if "exit_hazard_cumlogit" in available:
        return ["exit_hazard_cumlogit", *[name for name in available if name != "exit_hazard_cumlogit"]]
    return available


def _threshold_metrics(
    normalized_by_id: Dict[str, Dict[str, List[float]]],
    closure_fraction_by_id: Dict[str, float | None],
    drift_fraction_by_id: Dict[str, float | None],
    threshold: float,
    min_lead: float,
) -> Dict[str, float]:
    total_closure = 0
    crossed_closure = 0
    timely_closure = 0
    total_drift = 0
    crossed_drift = 0
    timely_drift = 0
    for example_id, packed in normalized_by_id.items():
        fractions = packed["fractions"]
        values = packed["norm_values"]
        crossing_idx = next((idx for idx, value in enumerate(values) if value >= threshold), None)
        closure_fraction = closure_fraction_by_id.get(example_id)
        if closure_fraction is not None:
            total_closure += 1
            if crossing_idx is not None:
                crossed_closure += 1
                lead = float(closure_fraction) - float(fractions[crossing_idx])
                if lead >= min_lead:
                    timely_closure += 1
        drift_fraction = drift_fraction_by_id.get(example_id)
        if drift_fraction is not None:
            total_drift += 1
            if crossing_idx is not None:
                crossed_drift += 1
                lead = float(drift_fraction) - float(fractions[crossing_idx])
                if lead >= min_lead:
                    timely_drift += 1
    timely_rate_closure = float(timely_closure / total_closure) if total_closure else 0.0
    timely_rate_drift = float(timely_drift / total_drift) if total_drift else 0.0
    return {
        "crossing_rate_eligible": float(crossed_closure / total_closure) if total_closure else 0.0,
        "crossing_rate_drift": float(crossed_drift / total_drift) if total_drift else 0.0,
        "timely_rate_closure": timely_rate_closure,
        "timely_rate_drift": timely_rate_drift,
        "timely_rate": float(np.mean([timely_rate_closure, timely_rate_drift]))
        if total_closure and total_drift
        else timely_rate_closure,
    }


def _summarize_candidate(
    name: str,
    per_example_points: Dict[str, List[Dict]],
    closure_fraction_by_id: Dict[str, float | None],
    drift_fraction_by_id: Dict[str, float | None],
    length_ratio_by_id: Dict[str, float | None],
    args: argparse.Namespace,
) -> Dict:
    series_by_id = {
        example_id: {
            "fractions": [safe_float(row.get("fraction")) for row in rows],
            "values": [safe_float(row.get(name)) for row in rows],
        }
        for example_id, rows in per_example_points.items()
        if all(name in row for row in rows)
    }

    changes = [series["values"][-1] - series["values"][0] for series in series_by_id.values() if len(series["values"]) >= 2]
    direction = 1.0 if (np.median(changes) if changes else 0.0) >= 0 else -1.0
    all_adj_values = [direction * float(value) for series in series_by_id.values() for value in series["values"]]
    norm_lo, norm_hi = robust_minmax(all_adj_values)

    normalized_by_id: Dict[str, Dict[str, List[float]]] = {}
    pooled_fracs: List[float] = []
    pooled_values: List[float] = []
    violation_rates: List[float] = []
    drop_magnitudes: List[float] = []
    monotone_flags: List[float] = []
    relaxed_monotone_flags: List[float] = []
    jump_offsets: List[float] = []
    jump_hits: List[float] = []
    aucs: List[float] = []
    length_ratios: List[float] = []

    for example_id, series in series_by_id.items():
        fractions = series["fractions"]
        values = series["values"]
        if len(fractions) < 2:
            continue
        adj_vals = [direction * float(value) for value in values]
        deltas = np.diff(np.asarray(adj_vals, dtype=float))
        violation = deltas < -float(args.tol)
        violation_rate = float(np.mean(violation)) if deltas.size else 0.0
        drop_magnitude = float(np.mean(-deltas[violation])) if np.any(violation) else 0.0
        violation_rates.append(violation_rate)
        drop_magnitudes.append(drop_magnitude)
        monotone_flags.append(1.0 if not np.any(violation) else 0.0)
        relaxed_monotone_flags.append(1.0 if violation_rate <= 0.1 else 0.0)
        pooled_fracs.extend(fractions)
        pooled_values.extend(adj_vals)

        norm_values = normalize(adj_vals, norm_lo, norm_hi)
        normalized_by_id[example_id] = {"fractions": fractions, "norm_values": norm_values}

        closure_fraction = closure_fraction_by_id.get(example_id)
        if closure_fraction is not None and deltas.size:
            jump_fraction = float(fractions[int(np.argmax(deltas)) + 1])
            jump_offset = abs(jump_fraction - float(closure_fraction))
            jump_offsets.append(float(jump_offset))
            jump_hits.append(1.0 if jump_offset <= float(args.jump_window) else 0.0)

        length_ratio = length_ratio_by_id.get(example_id)
        if length_ratio is not None:
            aucs.append(normalized_auc(fractions, norm_values))
            length_ratios.append(float(length_ratio))

    rho, rho_p = safe_spearman_correlation(pooled_fracs, pooled_values)
    len_rho, len_rho_p = safe_spearman_correlation(aucs, length_ratios)
    threshold_grid = [round(float(value), 2) for value in np.arange(0.30, 0.86, 0.05)]
    best_threshold = float(args.threshold)
    best = _threshold_metrics(
        normalized_by_id,
        closure_fraction_by_id,
        drift_fraction_by_id,
        best_threshold,
        float(args.min_lead),
    )
    for threshold in threshold_grid:
        candidate = _threshold_metrics(
            normalized_by_id,
            closure_fraction_by_id,
            drift_fraction_by_id,
            float(threshold),
            float(args.min_lead),
        )
        current_obj = candidate["timely_rate"] + 0.08 * candidate["crossing_rate_eligible"] + 0.05 * candidate["crossing_rate_drift"]
        best_obj = best["timely_rate"] + 0.08 * best["crossing_rate_eligible"] + 0.05 * best["crossing_rate_drift"]
        if current_obj > best_obj:
            best_threshold = float(threshold)
            best = candidate

    monotone_rate = float(np.mean(monotone_flags)) if monotone_flags else 0.0
    relaxed_monotone_rate = float(np.mean(relaxed_monotone_flags)) if relaxed_monotone_flags else 0.0
    mean_violation = float(np.mean(violation_rates)) if violation_rates else 1.0
    jump_align_rate = float(np.mean(jump_hits)) if jump_hits else 0.0
    length_coupling = abs(float(len_rho or 0.0))
    accepted = (
        relaxed_monotone_rate >= 0.85
        and mean_violation <= 0.10
        and best["timely_rate_closure"] >= 0.35
        and best["timely_rate_drift"] >= 0.35
        and best["crossing_rate_eligible"] >= 0.70
        and jump_align_rate >= 0.50
        and length_coupling <= 0.85
    )
    proxy_score = (
        0.30 * relaxed_monotone_rate
        + 0.20 * (1.0 - mean_violation)
        + 0.20 * best["timely_rate"]
        + 0.20 * jump_align_rate
        + 0.10 * (1.0 - min(length_coupling, 1.0))
    )

    return {
        "candidate": name,
        "proxy_score": float(proxy_score),
        "direction": "increasing" if direction > 0 else "decreasing",
        "n_examples": len(monotone_flags),
        "monotone_sample_rate": monotone_rate,
        "relaxed_monotone_rate": relaxed_monotone_rate,
        "mean_violation_rate": mean_violation,
        "p90_violation_rate": float(np.quantile(violation_rates, 0.9)) if violation_rates else 1.0,
        "mean_drop_magnitude": float(np.mean(drop_magnitudes)) if drop_magnitudes else 0.0,
        "global_fraction_spearman_rho": rho,
        "global_fraction_spearman_p": rho_p,
        "best_threshold": best_threshold,
        "best_crossing_rate_eligible": best["crossing_rate_eligible"],
        "best_crossing_rate_drift": best["crossing_rate_drift"],
        "best_timely_rate": best["timely_rate"],
        "best_timely_rate_closure": best["timely_rate_closure"],
        "best_timely_rate_drift": best["timely_rate_drift"],
        "jump_align_rate": jump_align_rate,
        "jump_offset_mean": float(np.mean(jump_offsets)) if jump_offsets else None,
        "length_ratio_auc_spearman_rho": len_rho,
        "length_ratio_auc_spearman_p": len_rho_p,
        "accepted": accepted,
        "failure_reasons": "通过当前验收阈值" if accepted else "未达当前验收阈值",
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    out_dir = root / args.eval_subdir

    metrics_by_id = load_metrics(root / "example_decode_gate_metrics.csv", args.condition)
    texts_by_id = load_text_rows(root / "generation_texts.json", args.condition)
    points_by_id = load_hazard_points(args.hazard_points_csv)
    common_ids = sorted(set(metrics_by_id) & set(texts_by_id) & set(points_by_id))
    if args.max_samples > 0:
        common_ids = common_ids[: int(args.max_samples)]
    if not common_ids:
        raise RuntimeError("No overlapping examples found.")

    per_example_points = {example_id: points_by_id[example_id] for example_id in common_ids}
    point_rows = [row for example_id in common_ids for row in per_example_points[example_id]]
    candidate_columns = _candidate_columns(point_rows)
    if not candidate_columns:
        raise RuntimeError("No exit-hazard score columns found.")

    closure_fraction_by_id: Dict[str, float | None] = {}
    drift_fraction_by_id: Dict[str, float | None] = {}
    length_ratio_by_id: Dict[str, float | None] = {}
    for example_id in common_ids:
        response_text = str(texts_by_id[example_id].get("response_text", ""))
        closure_fraction, drift_fraction, _ = event_fractions(metrics_by_id[example_id], response_text)
        closure_fraction_by_id[example_id] = closure_fraction
        drift_fraction_by_id[example_id] = drift_fraction
        length_ratio_by_id[example_id] = safe_float(metrics_by_id[example_id].get("length_ratio"), float("nan"))
        if not np.isfinite(length_ratio_by_id[example_id]):
            length_ratio_by_id[example_id] = None

    summary_rows = [
        _summarize_candidate(
            name,
            per_example_points,
            closure_fraction_by_id,
            drift_fraction_by_id,
            length_ratio_by_id,
            args,
        )
        for name in candidate_columns
    ]
    summary_rows = sorted(summary_rows, key=lambda row: float(row["proxy_score"]), reverse=True)
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "exit_hazard_eval_points.csv", point_rows)
    write_csv(out_dir / "exit_hazard_eval_summary.csv", summary_rows)
    write_json(
        out_dir / "exit_hazard_eval_report.json",
        {
            "created_at": now_iso(),
            "phase": "exit_hazard_eval",
            "condition": args.condition,
            "n_examples": len(common_ids),
            "active_scores": candidate_columns,
            "top_score": summary_rows[0] if summary_rows else None,
            "n_accepted": sum(1 for row in summary_rows if bool(row.get("accepted"))),
        },
    )

    print(f"done: {out_dir}")
    for row in summary_rows:
        print(
            f"  rank={row['rank']} {row['candidate']} score={row['proxy_score']:.4f} "
            f"mono={row['monotone_sample_rate']:.3f} timely={row['best_timely_rate']:.3f} "
            f"jump_align={row['jump_align_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
