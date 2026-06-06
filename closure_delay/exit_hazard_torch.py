from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .exit_hazard import (
    ANSWER_ONSET_PROBE_PHRASES,
    CONTINUE_MARKER_PROBE_PHRASES,
    DRIFT_BEHAVIOR_PROBE_PHRASES,
    EXIT_MARKER_PROBE_PHRASES,
    EXIT_PROBE_PHRASES,
    LOGIT_FEATURE_KEYS,
    REASONING_PROBE_PHRASES,
    VERIFY_BEHAVIOR_PROBE_PHRASES,
    first_token_ids,
)


@dataclass(frozen=True)
class ExitHazardHeadConfig:
    layer: int
    lag: int
    feature_mode: str
    hidden_dim: int
    n_features: int
    logit_feature_keys: Sequence[str]


class DifferentiableExitHazardHead(torch.nn.Module):
    def __init__(
        self,
        config: ExitHazardHeadConfig,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
        coef: torch.Tensor,
        intercept: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.register_buffer("scaler_mean", scaler_mean.float())
        self.register_buffer("scaler_scale", torch.clamp(scaler_scale.float(), min=1e-6))
        self.register_buffer("coef", coef.reshape(-1).float())
        self.register_buffer("intercept", intercept.reshape(()).float())

    @classmethod
    def from_files(
        cls,
        json_path: str | Path,
        npz_path: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> "DifferentiableExitHazardHead":
        json_path = Path(json_path)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if npz_path is None:
            npz_path = metadata.get("head_npz")
        npz_path = Path(npz_path)
        if not npz_path.is_absolute():
            npz_path = json_path.parent / npz_path.name
        arrays = np.load(npz_path)
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        config = ExitHazardHeadConfig(
            layer=int(metadata["layer"]),
            lag=int(metadata["lag"]),
            feature_mode=str(metadata["feature_mode"]),
            hidden_dim=int(metadata["hidden_dim"]),
            n_features=int(metadata["n_features"]),
            logit_feature_keys=list(metadata.get("logit_feature_keys", LOGIT_FEATURE_KEYS)),
        )
        return cls(
            config=config,
            scaler_mean=torch.tensor(arrays["scaler_mean"], device=target_device),
            scaler_scale=torch.tensor(arrays["scaler_scale"], device=target_device),
            coef=torch.tensor(arrays["coef"], device=target_device),
            intercept=torch.tensor(arrays["intercept"], device=target_device),
        ).to(target_device)

    def forward(self, hidden: torch.Tensor, logit_features: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.dim() != 2:
            raise ValueError("hidden must be [tokens, hidden_dim]")
        delta = hidden - torch.cat([hidden[:1].expand(min(self.config.lag, hidden.shape[0]), -1), hidden[:-self.config.lag]], dim=0) if hidden.shape[0] > self.config.lag else hidden - hidden[:1]
        if self.config.feature_mode == "static":
            features = hidden
        elif self.config.feature_mode == "delta":
            features = delta
        elif self.config.feature_mode == "static-delta":
            features = torch.cat([hidden, delta], dim=-1)
        elif self.config.feature_mode == "static-delta-logit":
            if logit_features is None:
                raise ValueError("logit_features are required for static-delta-logit")
            features = torch.cat([hidden, delta, logit_features], dim=-1)
        else:
            raise ValueError(f"Unsupported feature mode: {self.config.feature_mode}")
        if features.shape[-1] != self.config.n_features:
            raise ValueError(f"Feature width mismatch: {features.shape[-1]} != {self.config.n_features}")
        normalized = (features - self.scaler_mean) / self.scaler_scale
        return normalized @ self.coef + self.intercept

    def cumulative_scores(self, raw_hazard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = raw_hazard[torch.isfinite(raw_hazard)]
        center = torch.quantile(finite, 0.75) if finite.numel() else raw_hazard.new_tensor(0.0)
        intensity = F.softplus(raw_hazard - center)
        scale = torch.clamp(torch.quantile(intensity, 0.95), min=1e-6) if intensity.numel() else raw_hazard.new_tensor(1.0)
        intensity = torch.clamp(intensity / scale, min=0.0, max=10.0)
        cumulative = torch.cumsum(intensity, dim=0)
        cumprob = torch.clamp(1.0 - torch.exp(-cumulative), min=1e-6, max=1.0 - 1e-6)
        cumlogit = torch.logit(cumprob)
        return cumprob, cumlogit


def exit_logit_features_from_logits(
    logits: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("logits must be [tokens, vocab]")
    exit_ids = first_token_ids(tokenizer, EXIT_PROBE_PHRASES)
    reasoning_ids = first_token_ids(tokenizer, REASONING_PROBE_PHRASES)
    exit_marker_ids = first_token_ids(tokenizer, EXIT_MARKER_PROBE_PHRASES)
    continue_marker_ids = first_token_ids(tokenizer, CONTINUE_MARKER_PROBE_PHRASES)
    log_denom = torch.logsumexp(logits, dim=-1)
    pmax = torch.exp(torch.max(logits, dim=-1).values - log_denom)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and 0 <= int(eos_id) < logits.shape[-1]:
        eos_prob = torch.exp(logits[:, int(eos_id)] - log_denom)
    else:
        eos_prob = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    if exit_ids:
        exit_log_mass = torch.logsumexp(logits[:, exit_ids], dim=-1)
    else:
        exit_log_mass = torch.full((logits.shape[0],), -30.0, dtype=logits.dtype, device=logits.device)
    if reasoning_ids:
        reasoning_log_mass = torch.logsumexp(logits[:, reasoning_ids], dim=-1)
    else:
        reasoning_log_mass = torch.full((logits.shape[0],), -30.0, dtype=logits.dtype, device=logits.device)
    margin = exit_log_mass - reasoning_log_mass
    runmax = torch.cummax(margin, dim=0).values
    runmin = torch.cummin(margin, dim=0).values
    deltas = margin[1:] - margin[:-1]
    pos_cum = torch.cat([margin.new_zeros(1), torch.cumsum(torch.relu(deltas), dim=0)])
    neg_cum = torch.cat([margin.new_zeros(1), torch.cumsum(torch.relu(-deltas), dim=0)])
    if exit_marker_ids:
        exit_marker_log_mass = torch.logsumexp(logits[:, exit_marker_ids], dim=-1)
    else:
        exit_marker_log_mass = torch.full((logits.shape[0],), -30.0, dtype=logits.dtype, device=logits.device)
    if continue_marker_ids:
        continue_marker_log_mass = torch.logsumexp(logits[:, continue_marker_ids], dim=-1)
    else:
        continue_marker_log_mass = torch.full((logits.shape[0],), -30.0, dtype=logits.dtype, device=logits.device)
    marker_margin = exit_marker_log_mass - continue_marker_log_mass
    marker_runmax = torch.cummax(marker_margin, dim=0).values
    marker_deltas = marker_margin[1:] - marker_margin[:-1]
    marker_pos_cum = torch.cat([marker_margin.new_zeros(1), torch.cumsum(torch.relu(marker_deltas), dim=0)])
    marker_neg_cum = torch.cat([marker_margin.new_zeros(1), torch.cumsum(torch.relu(-marker_deltas), dim=0)])
    values = {
        "exit_logit_margin": margin,
        "exit_logit_exit_logmass": exit_log_mass,
        "exit_logit_reasoning_logmass": reasoning_log_mass,
        "exit_logit_margin_runmax": runmax,
        "exit_logit_margin_runmin": runmin,
        "exit_logit_margin_pos_cumsum": pos_cum,
        "exit_logit_margin_neg_cumsum": neg_cum,
        "exit_marker_logit_margin": marker_margin,
        "exit_marker_logit_margin_runmax": marker_runmax,
        "exit_marker_logit_margin_pos_cumsum": marker_pos_cum,
        "exit_marker_logit_margin_neg_cumsum": marker_neg_cum,
        "exit_logit_pmax": pmax,
        "exit_logit_eos_prob": eos_prob,
    }
    return torch.stack([values[key] for key in LOGIT_FEATURE_KEYS], dim=-1)


def phrase_logprob_mass(logits: torch.Tensor, tokenizer, phrases: Sequence[str]) -> torch.Tensor:
    """First-token phrase log probability mass for cheap differentiable probes."""
    if logits.dim() != 2:
        raise ValueError("logits must be [tokens, vocab]")
    token_ids = first_token_ids(tokenizer, phrases)
    if not token_ids:
        return torch.full((logits.shape[0],), -30.0, dtype=logits.dtype, device=logits.device)
    log_denom = torch.logsumexp(logits, dim=-1)
    return torch.logsumexp(logits[:, token_ids], dim=-1) - log_denom


def exit_process_scores(
    logits: torch.Tensor,
    tokenizer,
    closure_cumprob: torch.Tensor,
    *,
    closure_threshold: float = 0.30,
    closure_eps: float = 0.08,
    answer_logprob_threshold: float = -3.50,
    answer_eps: float = 0.60,
    verify_logprob_threshold: float = -4.50,
    verify_eps: float = 0.80,
    drift_logprob_threshold: float = -5.00,
    drift_eps: float = 0.80,
) -> dict[str, torch.Tensor]:
    """Build the soft post-closure process channels used by suffix objectives.

    The closure channel comes from the learned exit-readiness process. Answer,
    verification, and drift channels are intentionally cheap first-token probes;
    they can later be replaced by calibrated heads without changing callers.
    """
    q_closure = torch.sigmoid((closure_cumprob - float(closure_threshold)) / float(closure_eps))

    answer_logprob = phrase_logprob_mass(logits, tokenizer, ANSWER_ONSET_PROBE_PHRASES)
    verify_logprob = phrase_logprob_mass(logits, tokenizer, VERIFY_BEHAVIOR_PROBE_PHRASES)
    drift_logprob = phrase_logprob_mass(logits, tokenizer, DRIFT_BEHAVIOR_PROBE_PHRASES)

    lambda_answer = torch.sigmoid((answer_logprob - float(answer_logprob_threshold)) / float(answer_eps))
    answer_survival = torch.exp(
        torch.cumsum(torch.log(torch.clamp(1.0 - lambda_answer, min=1e-6, max=1.0)), dim=0)
    )
    verify_prob = torch.sigmoid((verify_logprob - float(verify_logprob_threshold)) / float(verify_eps))
    drift_prob = torch.sigmoid((drift_logprob - float(drift_logprob_threshold)) / float(drift_eps))

    pcg = q_closure * answer_survival * (1.0 - drift_prob)
    vpcg = pcg * verify_prob
    return {
        "q_closure": q_closure,
        "answer_logprob": answer_logprob,
        "lambda_answer": lambda_answer,
        "answer_survival": answer_survival,
        "verify_logprob": verify_logprob,
        "verify_prob": verify_prob,
        "drift_logprob": drift_logprob,
        "drift_prob": drift_prob,
        "pcg": pcg,
        "vpcg": vpcg,
    }
