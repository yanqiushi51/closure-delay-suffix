import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard import (
    HAZARD_SCORE_COLUMNS,
    event_fractions,
    load_hazard_points,
    load_metrics,
    load_text_rows,
    safe_float,
)
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure exit-hazard information beyond length and marker controls.")
    parser.add_argument("output_dir", help="Merged generation directory.")
    parser.add_argument("--condition", default="decode_gate_2p4")
    parser.add_argument("--hazard-points-csv", nargs="+", required=True)
    parser.add_argument("--eval-subdir", default="exit_hazard_incremental_ce")
    parser.add_argument("--horizon", type=float, default=0.15)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def _matrix(rows: Sequence[Dict], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([[safe_float(row.get(key), 0.0) for key in keys] for row in rows], dtype=float)


def _evaluate_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int) -> Dict[str, float | int | None]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(np.unique(y)) < 2:
        return {"log_loss": None, "auc": None, "n_eval": 0}
    splits = min(max(2, int(n_splits)), len(unique_groups))
    cv = GroupKFold(n_splits=splits)
    pred = np.full(y.shape[0], np.nan, dtype=float)
    used = np.zeros(y.shape[0], dtype=bool)
    for train_idx, test_idx in cv.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        used[test_idx] = True
    if not np.any(used):
        return {"log_loss": None, "auc": None, "n_eval": 0}
    y_eval = y[used]
    pred_eval = np.clip(pred[used], 1e-6, 1.0 - 1e-6)
    auc = float(roc_auc_score(y_eval, pred_eval)) if len(np.unique(y_eval)) >= 2 else None
    return {"log_loss": float(log_loss(y_eval, pred_eval, labels=[0, 1])), "auc": auc, "n_eval": int(y_eval.shape[0])}


def _build_rows(args: argparse.Namespace) -> List[Dict]:
    root = Path(args.output_dir)
    metrics_by_id = load_metrics(root / "example_decode_gate_metrics.csv", args.condition)
    texts_by_id = load_text_rows(root / "generation_texts.json", args.condition)
    points_by_id = load_hazard_points(args.hazard_points_csv)
    rows: List[Dict] = []
    for example_id in sorted(set(metrics_by_id) & set(texts_by_id) & set(points_by_id)):
        response_text = str(texts_by_id[example_id].get("response_text", ""))
        _, _, exit_fraction = event_fractions(metrics_by_id[example_id], response_text)
        if exit_fraction is None:
            continue
        for row in points_by_id[example_id]:
            fraction = safe_float(row.get("fraction"), float("nan"))
            token_index = safe_float(row.get("token_index"), float("nan"))
            if not np.isfinite(fraction) or not np.isfinite(token_index):
                continue
            marker = safe_float(row.get("closure_marker_hit"), 0.0)
            enriched = dict(row)
            enriched["exit_fraction"] = exit_fraction
            enriched["exit_horizon_label"] = 1 if fraction >= exit_fraction - float(args.horizon) else 0
            enriched["nuisance_fraction"] = fraction
            enriched["nuisance_fraction_sq"] = fraction * fraction
            enriched["nuisance_log_token_index"] = float(np.log1p(token_index))
            enriched["nuisance_closure_marker_hit"] = float(marker)
            rows.append(enriched)
    return rows


def _summarize_candidate(candidate: str, rows: Sequence[Dict], n_splits: int, marker_free: bool) -> Dict:
    use_rows = [
        row
        for row in rows
        if candidate in row
        and row.get(candidate) not in (None, "")
        and (not marker_free or safe_float(row.get("nuisance_closure_marker_hit"), 0.0) < 0.5)
    ]
    subset = "marker_free" if marker_free else "all"
    if not use_rows:
        return {"candidate": candidate, "subset": subset, "n_points": 0, "delta_ce": None, "delta_auc": None}

    nuisance_keys = [
        "nuisance_fraction",
        "nuisance_fraction_sq",
        "nuisance_log_token_index",
        "nuisance_closure_marker_hit",
    ]
    y = _matrix(use_rows, ["exit_horizon_label"]).ravel().astype(int)
    groups = np.asarray([str(row["id"]) for row in use_rows])
    base = _evaluate_cv(_matrix(use_rows, nuisance_keys), y, groups, n_splits)
    full = _evaluate_cv(_matrix(use_rows, [*nuisance_keys, candidate]), y, groups, n_splits)
    base_loss = base["log_loss"]
    full_loss = full["log_loss"]
    base_auc = base["auc"]
    full_auc = full["auc"]
    return {
        "candidate": candidate,
        "subset": subset,
        "n_points": len(use_rows),
        "n_examples": len(set(groups.tolist())),
        "positive_rate": float(np.mean(y)) if y.size else None,
        "base_log_loss": base_loss,
        "full_log_loss": full_loss,
        "delta_ce": float(base_loss - full_loss) if base_loss is not None and full_loss is not None else None,
        "base_auc": base_auc,
        "full_auc": full_auc,
        "delta_auc": float(full_auc - base_auc) if base_auc is not None and full_auc is not None else None,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    out_dir = root / args.eval_subdir
    rows = _build_rows(args)
    candidate_names = [name for name in HAZARD_SCORE_COLUMNS if any(name in row for row in rows)]
    if "exit_hazard_cumlogit" in candidate_names:
        candidate_names = ["exit_hazard_cumlogit", *[name for name in candidate_names if name != "exit_hazard_cumlogit"]]

    summary_rows: List[Dict] = []
    for candidate in candidate_names:
        summary_rows.append(_summarize_candidate(candidate, rows, args.n_splits, marker_free=False))
        summary_rows.append(_summarize_candidate(candidate, rows, args.n_splits, marker_free=True))
    summary_rows.sort(key=lambda row: (1.0 if row.get("delta_ce") is None else -float(row["delta_ce"]), row["candidate"], row["subset"]))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "exit_hazard_incremental_ce_points.csv", rows)
    write_csv(out_dir / "exit_hazard_incremental_ce_summary.csv", summary_rows)
    write_json(
        out_dir / "exit_hazard_incremental_ce_report.json",
        {
            "created_at": now_iso(),
            "phase": "exit_hazard_incremental_ce",
            "condition": args.condition,
            "horizon": float(args.horizon),
            "n_rows": len(rows),
            "n_examples": len(set(str(row["id"]) for row in rows)),
            "active_scores": candidate_names,
        },
    )
    print(f"done: {out_dir}")
    for row in summary_rows[:6]:
        print(f"  {row['candidate']} subset={row['subset']} delta_ce={row.get('delta_ce')} delta_auc={row.get('delta_auc')}")


if __name__ == "__main__":
    main()
