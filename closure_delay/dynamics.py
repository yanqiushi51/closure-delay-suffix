from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DynamicsConfig:
    closure_threshold: float = 0.70
    jump_threshold: float = 0.05
    jump_quantile: float = 0.90
    plateau_high_threshold: float = 0.60
    plateau_slope_threshold: float = 0.01
    min_plateau_tokens: int = 5
    min_stage_gap: int = 8
    smooth_window: int = 5
    local_peak_quantile: float = 0.75
    local_valley_quantile: float = 0.35
    local_reset_margin: float = 0.50
    answer_onset_threshold: float = 0.50


def finite_array(values: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def sequence_array(values: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return np.where(np.isfinite(arr), arr, np.nan)


def safe_mean(values: Sequence[float] | np.ndarray | None) -> float | None:
    arr = finite_array(values)
    return float(np.mean(arr)) if arr.size else None


def safe_sum(values: Sequence[float] | np.ndarray | None) -> float | None:
    arr = finite_array(values)
    return float(np.sum(arr)) if arr.size else None


def safe_max(values: Sequence[float] | np.ndarray | None) -> float | None:
    arr = finite_array(values)
    return float(np.max(arr)) if arr.size else None


def first_crossing(values: Sequence[float] | np.ndarray | None, threshold: float) -> int | None:
    arr = sequence_array(values)
    for index, value in enumerate(arr):
        if np.isfinite(value) and float(value) >= float(threshold):
            return int(index + 1)
    return None


def moving_average(values: Sequence[float] | np.ndarray, window: int) -> np.ndarray:
    arr = sequence_array(values)
    if arr.size == 0:
        return arr
    clean = np.where(np.isfinite(arr), arr, np.nanmedian(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0)
    width = max(int(window), 1)
    if width == 1 or clean.size < 3:
        return clean.astype(float)
    width = min(width, clean.size)
    kernel = np.ones(width, dtype=float) / float(width)
    left = width // 2
    right = width - 1 - left
    padded = np.pad(clean, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(float)


def normalized_auc(values: Sequence[float] | np.ndarray | None) -> float | None:
    arr = finite_array(values)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return float(arr[0])
    xs = np.arange(arr.size, dtype=float)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(arr, xs) / max(float(xs[-1] - xs[0]), 1.0))


def _segments(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        segments.append((start, previous))
        start = current
        previous = current
    segments.append((start, previous))
    return segments


def _slice_before_stop(values: np.ndarray, stop_token: int | None) -> np.ndarray:
    if stop_token is None:
        return values
    end = max(int(stop_token) - 1, 0)
    return values[:end]


def cumulative_dynamics(
    q_closure: Sequence[float] | np.ndarray | None,
    *,
    config: DynamicsConfig = DynamicsConfig(),
    stop_token: int | None = None,
) -> dict[str, float | int | None]:
    arr = _slice_before_stop(sequence_array(q_closure), stop_token)
    if arr.size == 0:
        return {
            "time_above_threshold": 0,
            "auc_qc": None,
            "jump_count": 0,
            "jump_magnitude_mean": None,
            "jump_magnitude_max": None,
            "multi_step_count": 0,
            "multi_step_transition_count": 0,
            "plateau_count": 0,
            "plateau_token_total": 0,
            "plateau_longest": 0,
        }

    smooth = moving_average(arr, int(config.smooth_window))
    deltas = np.diff(smooth, prepend=smooth[0])
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    adaptive = float(np.quantile(positive, float(config.jump_quantile))) if positive.size else float(config.jump_threshold)
    jump_threshold = max(float(config.jump_threshold), adaptive)
    jump_indices = np.where(deltas >= jump_threshold)[0]
    jump_segments = _segments(jump_indices)
    jump_magnitudes = [float(np.sum(deltas[start : end + 1])) for start, end in jump_segments]

    plateau_mask = (smooth >= float(config.plateau_high_threshold)) & (
        np.abs(deltas) <= float(config.plateau_slope_threshold)
    )
    plateau_segments = [
        segment
        for segment in _segments(np.where(plateau_mask)[0])
        if segment[1] - segment[0] + 1 >= int(config.min_plateau_tokens)
    ]
    plateau_total = int(sum(end - start + 1 for start, end in plateau_segments))
    plateau_longest = int(max((end - start + 1 for start, end in plateau_segments), default=0))

    stage_segments: list[tuple[int, int]] = []
    for segment in jump_segments:
        if not stage_segments:
            stage_segments.append(segment)
            continue
        previous = stage_segments[-1]
        gap = max(0, segment[0] - previous[1] - 1)
        between = np.abs(deltas[previous[1] + 1 : segment[0]])
        low_slope_tokens = int(np.sum(between <= float(config.plateau_slope_threshold)))
        if gap >= int(config.min_stage_gap) and low_slope_tokens >= int(config.min_plateau_tokens):
            stage_segments.append(segment)
        else:
            stage_segments[-1] = (previous[0], segment[1])

    return {
        "time_above_threshold": int(np.sum(arr >= float(config.closure_threshold))),
        "auc_qc": normalized_auc(arr),
        "jump_count": int(len(jump_segments)),
        "jump_magnitude_mean": float(np.mean(jump_magnitudes)) if jump_magnitudes else None,
        "jump_magnitude_max": float(np.max(jump_magnitudes)) if jump_magnitudes else None,
        "multi_step_count": int(len(stage_segments)),
        "multi_step_transition_count": int(max(len(stage_segments) - 1, 0)),
        "plateau_count": int(len(plateau_segments)),
        "plateau_token_total": plateau_total,
        "plateau_longest": plateau_longest,
    }


def local_reset_dynamics(
    local_evidence: Sequence[float] | np.ndarray | None,
    *,
    config: DynamicsConfig = DynamicsConfig(),
    stop_token: int | None = None,
) -> dict[str, float | int | None]:
    arr = _slice_before_stop(sequence_array(local_evidence), stop_token)
    if arr.size < 3:
        return {
            "local_peak_count": 0,
            "local_valley_count": 0,
            "local_reset_count": 0,
            "rise_reset_cycle_count": 0,
            "second_rise_rate": None,
        }
    smooth = moving_average(arr, int(config.smooth_window))
    finite = smooth[np.isfinite(smooth)]
    if finite.size < 3:
        return {
            "local_peak_count": 0,
            "local_valley_count": 0,
            "local_reset_count": 0,
            "rise_reset_cycle_count": 0,
            "second_rise_rate": None,
        }

    high = float(np.quantile(finite, float(config.local_peak_quantile)))
    low = float(np.quantile(finite, float(config.local_valley_quantile)))
    spread = float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)) if finite.size >= 4 else float(np.std(finite))
    margin = max(float(config.local_reset_margin), 0.50 * max(spread, 1e-6))

    peak_indices: list[int] = []
    valley_indices: list[int] = []
    for index in range(1, smooth.size - 1):
        previous_value = smooth[index - 1]
        value = smooth[index]
        next_value = smooth[index + 1]
        if value >= previous_value and value >= next_value and value >= high:
            peak_indices.append(index)
        if value <= previous_value and value <= next_value and value <= low:
            valley_indices.append(index)

    cycles = 0
    cursor = 0
    for peak in peak_indices:
        if peak < cursor:
            continue
        valley = next((idx for idx in valley_indices if idx > peak and smooth[peak] - smooth[idx] >= margin), None)
        if valley is None:
            continue
        second_peak = next((idx for idx in peak_indices if idx > valley and smooth[idx] - smooth[valley] >= margin), None)
        if second_peak is None:
            continue
        cycles += 1
        cursor = second_peak + 1

    return {
        "local_peak_count": int(len(peak_indices)),
        "local_valley_count": int(len(valley_indices)),
        "local_reset_count": int(cycles),
        "rise_reset_cycle_count": int(cycles),
        "second_rise_rate": float(cycles / max(len(peak_indices), 1)) if peak_indices else None,
    }


def summarize_dynamics(
    *,
    raw_hazard: Sequence[float] | np.ndarray | None,
    cumprob: Sequence[float] | np.ndarray | None,
    cumlogit: Sequence[float] | np.ndarray | None,
    q_closure: Sequence[float] | np.ndarray | None,
    answer_survival: Sequence[float] | np.ndarray | None,
    verify_prob: Sequence[float] | np.ndarray | None,
    drift_prob: Sequence[float] | np.ndarray | None,
    pcg: Sequence[float] | np.ndarray | None,
    vpcg: Sequence[float] | np.ndarray | None,
    lambda_answer: Sequence[float] | np.ndarray | None = None,
    config: DynamicsConfig = DynamicsConfig(),
) -> dict[str, float | int | None]:
    q_arr = sequence_array(q_closure)
    answer_onset = first_crossing(lambda_answer, float(config.answer_onset_threshold))
    first_cross = first_crossing(q_arr, float(config.closure_threshold))
    generated_tokens = int(max(len(sequence_array(raw_hazard)), len(q_arr), len(sequence_array(cumprob))))

    summary: dict[str, float | int | None] = {
        "generated_tokens_scored": generated_tokens,
        "mean_raw_hazard": safe_mean(raw_hazard),
        "mean_cumlogit": safe_mean(cumlogit),
        "max_cumprob": safe_max(cumprob),
        "first_cross_token": first_cross,
        "post_cross_tokens": int(generated_tokens - first_cross) if first_cross is not None else 0,
        "answer_onset_token": answer_onset,
        "closure_mean": safe_mean(q_closure),
        "answer_survival_mean": safe_mean(answer_survival),
        "verify_mean": safe_mean(verify_prob),
        "drift_mean": safe_mean(drift_prob),
        "pcg_sum": safe_sum(pcg),
        "pcg_mean": safe_mean(pcg),
        "vpcg_sum": safe_sum(vpcg),
        "vpcg_mean": safe_mean(vpcg),
    }
    summary.update(cumulative_dynamics(q_arr, config=config, stop_token=answer_onset))
    summary.update(local_reset_dynamics(raw_hazard, config=config, stop_token=answer_onset))
    return summary
