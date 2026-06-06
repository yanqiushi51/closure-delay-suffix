import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure_delay.exit_hazard import LOGIT_FEATURE_KEYS, load_candidate_points, load_csv_rows, safe_float
from closure_delay.runtime import now_iso, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the internal-state exit-hazard proxy.")
    parser.add_argument("output_dir", help="Merged generation directory.")
    parser.add_argument("--feature-dir", nargs="+", required=True)
    parser.add_argument("--logit-points-csv", nargs="*", default=None)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--lag", type=int, default=8)
    parser.add_argument("--pre-window-frac", type=float, default=0.03)
    parser.add_argument("--post-window-frac", type=float, default=0.01)
    parser.add_argument("--min-window-tokens", type=int, default=8)
    parser.add_argument("--max-window-tokens", type=int, default=32)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-train-points-per-fold", type=int, default=160000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-subdir", default="exit_hazard_proxy")
    parser.add_argument(
        "--feature-mode",
        default="static-delta-logit",
        choices=["static", "delta", "static-delta", "static-delta-logit"],
    )
    return parser.parse_args()


def _feature_rows_path(feature_dir: str | Path) -> Path:
    feature_dir = Path(feature_dir)
    path = feature_dir / "exit_hazard_feature_rows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_feature_rows(feature_dirs: Sequence[str | Path]) -> List[Dict]:
    rows: List[Dict] = []
    for feature_dir in feature_dirs:
        rows.extend(load_csv_rows(_feature_rows_path(feature_dir)))
    return rows


def _load_layer_features(feature_dirs: Sequence[str | Path], layer: int) -> np.ndarray:
    arrays = []
    for feature_dir in feature_dirs:
        path = Path(feature_dir) / f"hidden_layer_{int(layer)}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.load(path, mmap_mode=None))
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def _attach_logit_features(rows: List[Dict], paths: Sequence[str | Path] | None) -> None:
    if not paths:
        return
    by_id = load_candidate_points(paths)
    indexed: Dict[str, Dict[int, Dict]] = {}
    for example_id, source_rows in by_id.items():
        indexed[example_id] = {int(round(safe_float(row.get("token_index")))): row for row in source_rows}
    for row in rows:
        source = indexed.get(str(row["id"]), {}).get(int(round(safe_float(row.get("token_index")))), {})
        for key in LOGIT_FEATURE_KEYS:
            row[key] = safe_float(source.get(key), 0.0)


def _group_indices(rows: Sequence[Dict]) -> Dict[str, List[int]]:
    by_id: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        by_id.setdefault(str(row["id"]), []).append(idx)
    for indices in by_id.values():
        indices.sort(key=lambda item: safe_float(rows[item].get("token_index")))
    return by_id


def _event_index(row: Dict) -> int:
    generated = max(1, int(round(safe_float(row.get("generated_token_count"), 1.0))))
    exit_fraction = min(max(safe_float(row.get("exit_fraction"), 1.0), 0.0), 1.0)
    return max(1, min(generated, int(round(exit_fraction * generated))))


def _build_hazard_targets(rows: Sequence[Dict], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(len(rows), dtype=np.int64)
    eligible = np.zeros(len(rows), dtype=bool)
    for indices in _group_indices(rows).values():
        first = rows[indices[0]]
        generated = max(1, int(round(safe_float(first.get("generated_token_count"), len(indices)))))
        event_idx = _event_index(first)
        pre = int(round(float(args.pre_window_frac) * generated))
        post = int(round(float(args.post_window_frac) * generated))
        pre = max(int(args.min_window_tokens), min(int(args.max_window_tokens), pre))
        post = max(1, min(int(args.max_window_tokens), post))
        lo = max(1, event_idx - pre)
        hi = min(generated, event_idx + post)
        for idx in indices:
            token_index = int(round(safe_float(rows[idx].get("token_index"))))
            if token_index <= hi:
                eligible[idx] = True
            if lo <= token_index <= hi:
                y[idx] = 1
    return y, eligible


def _hidden_delta(X: np.ndarray, rows: Sequence[Dict], lag: int) -> np.ndarray:
    delta = np.zeros_like(X, dtype=np.float32)
    for indices in _group_indices(rows).values():
        for local_pos, idx in enumerate(indices):
            ref_idx = indices[max(0, local_pos - int(lag))]
            delta[idx] = X[idx] - X[ref_idx]
    return delta


def _logit_matrix(rows: Sequence[Dict]) -> np.ndarray:
    return np.asarray([[safe_float(row.get(key)) for key in LOGIT_FEATURE_KEYS] for row in rows], dtype=np.float32)


def _features_for_mode(mode: str, X: np.ndarray, delta: np.ndarray, logit_features: np.ndarray) -> np.ndarray:
    if mode == "static":
        return X
    if mode == "delta":
        return delta
    if mode == "static-delta":
        return np.concatenate([X, delta], axis=1)
    if mode == "static-delta-logit":
        return np.concatenate([X, delta, logit_features], axis=1)
    raise ValueError(f"Unsupported feature mode: {mode}")


def _sample_balanced(
    train_idx: np.ndarray,
    y: np.ndarray,
    eligible: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    train_idx = train_idx[eligible[train_idx]]
    positives = train_idx[y[train_idx] == 1]
    negatives = train_idx[y[train_idx] == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return train_idx
    if max_points <= 0 or len(train_idx) <= max_points:
        return train_idx
    n_pos = min(len(positives), max_points // 2)
    n_neg = min(len(negatives), max_points - n_pos)
    sampled = np.concatenate(
        [
            rng.choice(positives, size=n_pos, replace=False),
            rng.choice(negatives, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(sampled)
    return sampled


def _fit_predict_oof(
    X: np.ndarray,
    y: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    splits = min(max(2, int(args.n_splits)), len(unique_groups))
    cv = GroupKFold(n_splits=splits)
    out = np.full(len(y), np.nan, dtype=np.float32)
    rng = np.random.default_rng(int(args.seed))
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        fit_idx = _sample_balanced(train_idx, y, eligible, int(args.max_train_points_per_fold), rng)
        if len(fit_idx) < 10 or len(np.unique(y[fit_idx])) < 2:
            continue
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X[fit_idx])
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(args.alpha),
            max_iter=int(args.max_iter),
            tol=1e-3,
            average=True,
            random_state=int(args.seed) + fold_idx,
        )
        model.fit(X_fit, y[fit_idx])
        out[test_idx] = model.predict_proba(scaler.transform(X[test_idx]))[:, 1].astype(np.float32)
        print(
            f"  fold={fold_idx}/{splits} train={len(fit_idx)} test={len(test_idx)} "
            f"pos_train={float(np.mean(y[fit_idx])):.4f}",
            flush=True,
        )
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def _fit_final_head(
    X: np.ndarray,
    y: np.ndarray,
    eligible: np.ndarray,
    args: argparse.Namespace,
) -> tuple[StandardScaler, SGDClassifier, np.ndarray]:
    rng = np.random.default_rng(int(args.seed) + 1009)
    all_idx = np.arange(X.shape[0])
    fit_idx = _sample_balanced(all_idx, y, eligible, int(args.max_train_points_per_fold), rng)
    if len(fit_idx) < 10 or len(np.unique(y[fit_idx])) < 2:
        raise RuntimeError("Not enough labeled points to fit final exit-hazard head.")
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X[fit_idx])
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(args.alpha),
        max_iter=int(args.max_iter),
        tol=1e-3,
        average=True,
        random_state=int(args.seed) + 2003,
    )
    model.fit(X_fit, y[fit_idx])
    return scaler, model, fit_idx


def _logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _softplus(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x64))) + np.maximum(x64, 0.0)


def _attach_cumulative_scores(rows: List[Dict]) -> None:
    for indices in _group_indices(rows).values():
        raw = np.asarray([safe_float(rows[idx].get("exit_hazard")) for idx in indices], dtype=np.float64)
        center = float(np.quantile(raw[np.isfinite(raw)], 0.75)) if np.any(np.isfinite(raw)) else 0.0
        intensity = _softplus(raw - center)
        if len(intensity) > 0:
            scale = max(float(np.quantile(intensity, 0.95)), 1e-6)
            intensity = np.clip(intensity / scale, 0.0, 10.0)
        cumulative = np.cumsum(intensity)
        state = np.clip(1.0 - np.exp(-cumulative), 1e-6, 1.0 - 1e-6)
        state_logit = np.log(state / (1.0 - state))
        for idx, prob_value, logit_value in zip(indices, state, state_logit):
            rows[idx]["exit_hazard_cumprob"] = float(prob_value)
            rows[idx]["exit_hazard_cumlogit"] = float(logit_value)


def _metric_row(name: str, y: np.ndarray, eligible: np.ndarray, score: np.ndarray) -> Dict:
    valid = eligible & np.isfinite(score)
    if not np.any(valid) or len(np.unique(y[valid])) < 2:
        return {"candidate": name, "n_points": int(np.sum(valid)), "auc": None, "log_loss": None}
    prob = 1.0 / (1.0 + np.exp(-np.clip(score[valid], -30.0, 30.0)))
    return {
        "candidate": name,
        "n_points": int(np.sum(valid)),
        "positive_rate": float(np.mean(y[valid])),
        "auc": float(roc_auc_score(y[valid], prob)),
        "log_loss": float(log_loss(y[valid], np.clip(prob, 1e-6, 1.0 - 1e-6), labels=[0, 1])),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    out_dir = root / args.eval_subdir

    rows = _load_feature_rows(args.feature_dir)
    if not rows:
        raise RuntimeError("No feature rows found.")
    _attach_logit_features(rows, args.logit_points_csv)

    groups = np.asarray([str(row["id"]) for row in rows])
    y, eligible = _build_hazard_targets(rows, args)
    print(
        f"rows={len(rows)} examples={len(np.unique(groups))} "
        f"eligible={int(np.sum(eligible))} pos={int(np.sum(y[eligible]))} "
        f"pos_rate={float(np.mean(y[eligible])):.4f}",
        flush=True,
    )

    X_hidden = _load_layer_features(args.feature_dir, int(args.layer))
    if X_hidden.shape[0] != len(rows):
        raise RuntimeError(f"Feature/row mismatch: {X_hidden.shape[0]} != {len(rows)}")
    X_delta = _hidden_delta(X_hidden, rows, int(args.lag))
    X_logit = _logit_matrix(rows)
    X = _features_for_mode(args.feature_mode, X_hidden, X_delta, X_logit)
    print(f"feature_mode={args.feature_mode} X={X.shape}", flush=True)

    prob = _fit_predict_oof(X, y, eligible, groups, args)
    raw_score = _logit(prob)
    for row, label, is_eligible, value in zip(rows, y, eligible, raw_score):
        row["hazard_label"] = int(label)
        row["hazard_train_eligible"] = int(is_eligible)
        row["exit_hazard"] = float(value)
    _attach_cumulative_scores(rows)

    cum_score = np.asarray([safe_float(row.get("exit_hazard_cumlogit")) for row in rows], dtype=np.float32)
    metric_rows = [
        _metric_row("exit_hazard", y, eligible, raw_score),
        _metric_row("exit_hazard_cumlogit", y, eligible, cum_score),
    ]
    final_scaler, final_model, final_fit_idx = _fit_final_head(X, y, eligible, args)

    out_dir.mkdir(parents=True, exist_ok=True)
    head_npz_path = out_dir / "exit_hazard_head.npz"
    head_json_path = out_dir / "exit_hazard_head.json"
    np.savez(
        head_npz_path,
        scaler_mean=final_scaler.mean_.astype(np.float32),
        scaler_scale=final_scaler.scale_.astype(np.float32),
        coef=final_model.coef_.astype(np.float32),
        intercept=final_model.intercept_.astype(np.float32),
    )
    write_csv(out_dir / "exit_hazard_points.csv", rows)
    write_csv(out_dir / "exit_hazard_metrics.csv", metric_rows)
    write_json(
        head_json_path,
        {
            "created_at": now_iso(),
            "format": "exit_hazard_linear_head_v1",
            "head_npz": str(head_npz_path),
            "layer": int(args.layer),
            "lag": int(args.lag),
            "feature_mode": args.feature_mode,
            "hidden_dim": int(X_hidden.shape[1]),
            "logit_feature_keys": list(LOGIT_FEATURE_KEYS),
            "n_features": int(X.shape[1]),
            "n_fit_points": int(len(final_fit_idx)),
            "positive_rate_fit": float(np.mean(y[final_fit_idx])),
            "alpha": float(args.alpha),
            "max_iter": int(args.max_iter),
        },
    )
    write_json(
        out_dir / "exit_hazard_report.json",
        {
            "created_at": now_iso(),
            "feature_dirs": [str(path) for path in args.feature_dir],
            "logit_points_csv": [str(path) for path in (args.logit_points_csv or [])],
            "layer": int(args.layer),
            "lag": int(args.lag),
            "feature_mode": args.feature_mode,
            "pre_window_frac": float(args.pre_window_frac),
            "post_window_frac": float(args.post_window_frac),
            "n_points": len(rows),
            "n_examples": len(np.unique(groups)),
            "n_eligible": int(np.sum(eligible)),
            "n_positive": int(np.sum(y[eligible])),
            "positive_rate": float(np.mean(y[eligible])) if np.any(eligible) else None,
            "head_json": str(head_json_path),
            "head_npz": str(head_npz_path),
        },
    )
    print(f"done: {out_dir}")
    for row in metric_rows:
        print(row)


if __name__ == "__main__":
    main()
