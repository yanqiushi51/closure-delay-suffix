from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .closure import ClosureTrajectory, average_logprob, sigmoid
from .probes import logmeanexp

DIRECTION_PROBES: Dict[str, List[str]] = {
    "continue": [
        "Next, we continue the current calculation by",
        "Following the current reasoning, we get",
    ],
    "verify": [
        "Before finalizing, we should verify",
        "Let's check whether this step is correct",
    ],
    "alternative": [
        "Another way to solve the problem is",
        "We can also approach it by",
    ],
    "compare": [
        "Now we compare the two results",
        "We need to reconcile these values",
    ],
    "finalize": [
        "Therefore, the final answer is",
        "So the answer is",
    ],
}

POSITIVE_CONFIDENCE_PROBES = [
    "The reasoning is consistent, so we can proceed.",
    "The calculation checks out.",
    "This result is verified.",
    "The steps are consistent with the answer.",
]

NEGATIVE_CONFIDENCE_PROBES = [
    "I should verify this step again.",
    "There may still be an error in the reasoning.",
    "I need to double-check the calculation.",
    "We should check whether the previous step is correct.",
]


def compute_direction_certainty(model, context: str, temperature: float = 1.0) -> dict:
    """Compute C_dir(t) from direction probe log-probabilities."""
    direction_scores: Dict[str, float] = {}
    for direction, probes in DIRECTION_PROBES.items():
        scores = [average_logprob(model, context, probe) for probe in probes]
        direction_scores[direction] = logmeanexp(scores)

    z_values = np.asarray(list(direction_scores.values()), dtype=float)
    max_z = float(np.max(z_values))
    exp_z = np.exp((z_values - max_z) / temperature)
    probs = exp_z / exp_z.sum()

    direction_probs = dict(zip(DIRECTION_PROBES.keys(), probs.tolist()))

    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))) / np.log(len(probs)))

    c_dir = float(1.0 - entropy)
    p_fin = direction_probs.get("finalize", 0.0)

    return {
        "direction_scores": direction_scores,
        "direction_probs": direction_probs,
        "direction_entropy": entropy,
        "c_dir": c_dir,
        "p_finalize": p_fin,
        "top_direction": max(direction_probs, key=direction_probs.get),
        "top_prob": float(max(probs)),
    }


def compute_exploration_pressure(dir_result: dict) -> float:
    """E_dir(t): pressure from non-finalize direction dispersion."""
    probs = dir_result["direction_probs"]
    p_fin = probs.get("finalize", 0.0)
    if p_fin >= 1.0:
        return 0.0

    nonfin_dirs = [d for d in DIRECTION_PROBES if d != "finalize"]
    nonfin_probs = np.asarray([probs[d] for d in nonfin_dirs], dtype=float)
    nonfin_probs = nonfin_probs / nonfin_probs.sum()

    n_nonfin = len(nonfin_dirs)
    h_nonfin = float(-np.sum(nonfin_probs * np.log(np.maximum(nonfin_probs, 1e-12))) / np.log(n_nonfin))

    return float((1.0 - p_fin) * h_nonfin)


def compute_confidence_readiness(model, context: str, temperature: float = 1.0) -> dict:
    """Compute C_conf(t) from positive/negative confidence probe margin."""
    pos_scores = [average_logprob(model, context, probe) for probe in POSITIVE_CONFIDENCE_PROBES]
    neg_scores = [average_logprob(model, context, probe) for probe in NEGATIVE_CONFIDENCE_PROBES]

    z_pos = logmeanexp(pos_scores)
    z_neg = logmeanexp(neg_scores)

    u_t = float(z_pos - z_neg)
    c_conf = sigmoid(u_t / temperature)

    return {
        "z_pos": z_pos,
        "z_neg": z_neg,
        "u_t": u_t,
        "c_conf": c_conf,
    }


def confidence_curve_for_trajectory(
    model,
    trajectory: ClosureTrajectory,
    suffix: str = "",
    temperature: float = 1.0,
) -> List[dict]:
    """Compute C_dir, E_dir, and C_conf at each fraction point of a trajectory."""
    base_context = model.build_prompt_text(trajectory.prompt, suffix)
    rows = []
    for point in trajectory.points:
        context = base_context + point.prefix_text
        dir_result = compute_direction_certainty(model, context, temperature)
        conf_result = compute_confidence_readiness(model, context, temperature)
        e_dir = compute_exploration_pressure(dir_result)
        rows.append({
            "fraction": point.fraction,
            "token_index": point.token_index,
            **dir_result,
            "e_dir": e_dir,
            **conf_result,
        })
    return rows


def confidence_curve_summary(trajectory_results: Sequence[Sequence[dict]]) -> dict:
    """Aggregate C_dir, C_conf, E_dir curves across samples."""
    def _aggregate(field: str) -> dict:
        buckets: Dict[float, List[float]] = {}
        for sample_rows in trajectory_results:
            for row in sample_rows:
                value = row.get(field)
                if value is None or not np.isfinite(value):
                    continue
                buckets.setdefault(float(row["fraction"]), []).append(float(value))
        fractions = sorted(buckets)
        means = [float(np.mean(buckets[f])) for f in fractions]
        stds = [float(np.std(buckets[f])) for f in fractions]
        counts = [len(buckets[f]) for f in fractions]
        return {"fractions": fractions, "means": means, "stds": stds, "counts": counts}

    return {
        "c_dir_curve": _aggregate("c_dir"),
        "c_conf_curve": _aggregate("c_conf"),
        "e_dir_curve": _aggregate("e_dir"),
        "p_finalize_curve": _aggregate("p_finalize"),
        "u_t_curve": _aggregate("u_t"),
    }


def confidence_curve_diagnostics(curve_summary: dict) -> dict:
    """Spearman correlation and late-early gap for C_dir and C_conf."""
    from .stats import safe_spearman_correlation

    def _diagnose(curve: dict) -> dict:
        fractions = curve.get("fractions", [])
        means = curve.get("means", [])
        if len(fractions) < 3:
            return {"spearman_rho": None, "spearman_p": None, "late_early_gap": None}
        rho, p = safe_spearman_correlation(fractions, means)
        early = [m for f, m in zip(fractions, means) if f <= 0.3]
        late = [m for f, m in zip(fractions, means) if f >= 0.7]
        gap = None
        if early and late:
            gap = float(np.mean(late) - np.mean(early))
        return {"spearman_rho": rho, "spearman_p": p, "late_early_gap": gap}

    return {
        "c_dir": _diagnose(curve_summary.get("c_dir_curve", {})),
        "c_conf": _diagnose(curve_summary.get("c_conf_curve", {})),
        "e_dir": _diagnose(curve_summary.get("e_dir_curve", {})),
    }
