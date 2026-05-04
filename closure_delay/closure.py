from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


@dataclass
class ClosurePoint:
    fraction: float
    token_index: int
    prefix_ids: List[int]
    continue_ids: List[int]
    close_ids: List[int]
    prefix_text: str
    continue_text: str
    close_text: str
    baseline_margin: float | None = None
    attacked_margin: float | None = None
    baseline_risk: float | None = None
    attacked_risk: float | None = None
    delta_margin: float | None = None
    delta_risk: float | None = None


@dataclass
class ClosureTrajectory:
    id: str
    prompt: str
    answer: str
    baseline_response: str
    baseline_length: int
    points: List[ClosurePoint]
    valid: bool
    reason: str = ""

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["points"] = [asdict(point) for point in self.points]
        return payload


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def logit(value: float, eps: float = 1e-5) -> float:
    value = min(max(float(value), eps), 1.0 - eps)
    return float(np.log(value / (1.0 - value)))


def isotonic_non_decreasing(values: Sequence[float]) -> List[float]:
    """Pool adjacent violators algorithm for unweighted isotonic smoothing."""
    if not values:
        return []
    levels: List[float] = []
    weights: List[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            pooled = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2:] = [pooled]
            weights[-2:] = [total_weight]
    smoothed: List[float] = []
    for level, weight in zip(levels, weights):
        smoothed.extend([float(level)] * weight)
    return smoothed


def build_reference_trajectory(
    record: Dict,
    trace,
    tokenizer,
    fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    continuation_tokens: int = 24,
    closure_tokens: int = 24,
    min_baseline_tokens: int = 80,
) -> ClosureTrajectory:
    generated_ids = list(trace.generated_ids)
    baseline_length = len(generated_ids)
    if baseline_length < min_baseline_tokens:
        return ClosureTrajectory(
            id=record["id"],
            prompt=record["prompt"],
            answer=str(record["answer"]),
            baseline_response=trace.response_text,
            baseline_length=baseline_length,
            points=[],
            valid=False,
            reason=f"baseline_too_short:{baseline_length}<{min_baseline_tokens}",
        )

    close_start = max(0, baseline_length - closure_tokens)
    close_ids = generated_ids[close_start:]
    close_text = tokenizer.decode(generated_ids[close_start:], skip_special_tokens=True)
    points: List[ClosurePoint] = []
    seen_indices = set()
    for fraction in fractions:
        token_index = int(round(fraction * baseline_length))
        token_index = max(1, min(token_index, baseline_length - 1))
        if token_index in seen_indices:
            continue
        if token_index + continuation_tokens >= close_start:
            continue
        seen_indices.add(token_index)
        prefix_ids = generated_ids[:token_index]
        continue_ids = generated_ids[token_index : token_index + continuation_tokens]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        continue_text = tokenizer.decode(
            continue_ids,
            skip_special_tokens=True,
        )
        if not prefix_text.strip() or not continue_text.strip() or not close_text.strip():
            continue
        points.append(
            ClosurePoint(
                fraction=float(fraction),
                token_index=token_index,
                prefix_ids=list(prefix_ids),
                continue_ids=list(continue_ids),
                close_ids=list(close_ids),
                prefix_text=prefix_text,
                continue_text=continue_text,
                close_text=close_text,
            )
        )

    if len(points) < 3:
        return ClosureTrajectory(
            id=record["id"],
            prompt=record["prompt"],
            answer=str(record["answer"]),
            baseline_response=trace.response_text,
            baseline_length=baseline_length,
            points=points,
            valid=False,
            reason=f"too_few_closure_points:{len(points)}",
        )

    return ClosureTrajectory(
        id=record["id"],
        prompt=record["prompt"],
        answer=str(record["answer"]),
        baseline_response=trace.response_text,
        baseline_length=baseline_length,
        points=points,
        valid=True,
    )


@torch.no_grad()
def average_logprob(model, context_text: str, target_text: str) -> float:
    """Mean teacher-forced log probability of target_text after context_text."""
    if not target_text:
        return float("-inf")
    tokenizer = model.tokenizer
    context_ids = tokenizer(context_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.device)
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    if target_ids.numel() == 0:
        return float("-inf")
    context_len = int(context_ids.shape[1])
    input_ids = torch.cat([context_ids, target_ids], dim=1)
    outputs = model.model(input_ids=input_ids)
    log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    target_start = max(context_len - 1, 0)
    target_end = target_start + int(target_ids.shape[1])
    target_log_probs = token_log_probs[:, target_start:target_end]
    if target_log_probs.numel() == 0:
        return float("-inf")
    return float(target_log_probs.mean().detach().cpu().item())


@torch.no_grad()
def average_logprob_ids(model, context_text: str, target_ids: Sequence[int]) -> float:
    """Mean teacher-forced log probability of target token ids after context_text."""
    if not target_ids:
        return float("-inf")
    tokenizer = model.tokenizer
    context_ids = tokenizer(context_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.device)
    target_tensor = torch.tensor([list(target_ids)], dtype=torch.long, device=model.device)
    context_len = int(context_ids.shape[1])
    input_ids = torch.cat([context_ids, target_tensor], dim=1)
    outputs = model.model(input_ids=input_ids)
    log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    target_start = max(context_len - 1, 0)
    target_end = target_start + int(target_tensor.shape[1])
    target_log_probs = token_log_probs[:, target_start:target_end]
    if target_log_probs.numel() == 0:
        return float("-inf")
    return float(target_log_probs.mean().detach().cpu().item())


def closure_margin_for_point(model, prompt: str, suffix: str, point: ClosurePoint) -> float:
    base_context = model.build_prompt_text(prompt, suffix)
    context = base_context + point.prefix_text
    close_lp = average_logprob_ids(model, context, point.close_ids)
    continue_lp = average_logprob_ids(model, context, point.continue_ids)
    return float(close_lp - continue_lp)


def progress_risk_diagnostics(trajectories: Sequence[ClosureTrajectory]) -> Dict:
    fractions = []
    risks = []
    for trajectory in trajectories:
        if not trajectory.valid:
            continue
        for point in trajectory.points:
            if point.baseline_risk is None or not np.isfinite(point.baseline_risk):
                continue
            fractions.append(point.fraction)
            risks.append(point.baseline_risk)
    if not fractions:
        return {
            "progress_risk_spearman": {"rho": None, "p": None},
            "late_early_gap": None,
            "n_points": 0,
        }
    from .stats import safe_spearman_correlation

    rho, p = safe_spearman_correlation(fractions, risks)
    early = [risk for fraction, risk in zip(fractions, risks) if fraction <= 0.3]
    late = [risk for fraction, risk in zip(fractions, risks) if fraction >= 0.7]
    late_early_gap = None
    if early and late:
        late_early_gap = float(np.mean(late) - np.mean(early))
    return {
        "progress_risk_spearman": {"rho": rho, "p": p},
        "late_early_gap": late_early_gap,
        "n_points": len(fractions),
    }


def score_closure_trajectory(
    model,
    trajectory: ClosureTrajectory,
    suffix: str = "",
    temperature: float = 1.0,
) -> ClosureTrajectory:
    for point in trajectory.points:
        margin = closure_margin_for_point(model, trajectory.prompt, suffix, point)
        risk = sigmoid(margin / temperature)
        if suffix:
            point.attacked_margin = margin
            point.attacked_risk = risk
        else:
            point.baseline_margin = margin
            point.baseline_risk = risk
    return trajectory


def attach_delta_scores(trajectory: ClosureTrajectory) -> ClosureTrajectory:
    for point in trajectory.points:
        if point.baseline_margin is not None and point.attacked_margin is not None:
            point.delta_margin = point.attacked_margin - point.baseline_margin
        if point.baseline_risk is not None and point.attacked_risk is not None:
            point.delta_risk = point.attacked_risk - point.baseline_risk
    return trajectory


def mean_curve(points_by_example: Sequence[ClosureTrajectory], field: str) -> Dict:
    buckets: Dict[float, List[float]] = {}
    for trajectory in points_by_example:
        if not trajectory.valid:
            continue
        for point in trajectory.points:
            value = getattr(point, field)
            if value is None or not np.isfinite(value):
                continue
            buckets.setdefault(point.fraction, []).append(float(value))
    fractions = sorted(buckets)
    means = [float(np.mean(buckets[f])) for f in fractions]
    stds = [float(np.std(buckets[f])) for f in fractions]
    counts = [len(buckets[f]) for f in fractions]
    return {"fractions": fractions, "means": means, "stds": stds, "counts": counts}


def closure_curve_summary(trajectories: Sequence[ClosureTrajectory]) -> Dict:
    baseline = mean_curve(trajectories, "baseline_risk")
    attacked = mean_curve(trajectories, "attacked_risk")
    delta = mean_curve(trajectories, "delta_risk")
    smoothed = isotonic_non_decreasing(baseline["means"])
    return {
        "n_total": len(trajectories),
        "n_valid": sum(1 for item in trajectories if item.valid),
        "baseline_risk_curve": baseline,
        "baseline_risk_curve_isotonic": {
            "fractions": baseline["fractions"],
            "means": smoothed,
        },
        "attacked_risk_curve": attacked,
        "delta_risk_curve": delta,
        "mean_delta_risk": float(np.mean(delta["means"])) if delta["means"] else None,
    }


def length_ratio(baseline_length: int, attacked_length: int) -> float:
    return float(attacked_length / max(baseline_length, 1))


def summarize_length_ratios(ratios: Iterable[float]) -> Dict:
    values = np.asarray(list(ratios), dtype=float)
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }
