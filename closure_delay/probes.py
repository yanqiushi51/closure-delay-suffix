from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .closure import ClosureTrajectory, average_logprob


CLOSURE_PROBES = [
    " Therefore, the final answer is",
    " So the answer is",
    " The final answer is",
    " Final answer:",
]

CONTINUATION_PROBES = [
    " Next, we need to",
    " Let's verify this step",
    " Now consider another way",
    " We should check whether",
]


def logmeanexp(values: Sequence[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    max_value = float(np.max(values))
    return float(max_value + np.log(np.mean(np.exp(values - max_value))))


def probe_margin_for_context(
    model,
    context: str,
    closure_probes: Sequence[str] = CLOSURE_PROBES,
    continuation_probes: Sequence[str] = CONTINUATION_PROBES,
) -> float:
    closure_scores = [average_logprob(model, context, probe) for probe in closure_probes]
    continuation_scores = [average_logprob(model, context, probe) for probe in continuation_probes]
    return float(logmeanexp(closure_scores) - logmeanexp(continuation_scores))


def probe_margins_for_trajectory(model, trajectory: ClosureTrajectory, suffix: str = "") -> List[Dict]:
    base_context = model.build_prompt_text(trajectory.prompt, suffix)
    rows = []
    for point in trajectory.points:
        context = base_context + point.prefix_text
        margin = probe_margin_for_context(model, context)
        rows.append({"fraction": point.fraction, "probe_margin": margin})
    return rows


def probe_shift_metrics(baseline_rows: Sequence[Dict], attacked_rows: Sequence[Dict]) -> Dict:
    attacked_by_fraction = {float(row["fraction"]): row for row in attacked_rows}
    deltas = []
    late_deltas = []
    weighted = []
    for base in baseline_rows:
        fraction = float(base["fraction"])
        attacked = attacked_by_fraction.get(fraction)
        if not attacked:
            continue
        delta = float(attacked["probe_margin"]) - float(base["probe_margin"])
        if not np.isfinite(delta):
            continue
        deltas.append(delta)
        weighted.append(fraction * -delta)
        if fraction >= 0.6:
            late_deltas.append(delta)

    return {
        "probe_margin_mean_shift": float(-np.mean(deltas)) if deltas else None,
        "probe_margin_late_shift": float(-np.mean(late_deltas)) if late_deltas else None,
        "probe_weighted_margin_auc": float(np.mean(weighted)) if weighted else None,
    }
