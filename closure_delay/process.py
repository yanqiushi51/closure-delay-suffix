from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .dynamics import DynamicsConfig, summarize_dynamics
from .exit_hazard_torch import (
    DifferentiableExitHazardHead,
    exit_logit_features_from_logits,
    exit_process_scores,
)
from .model import LocalCausalLM


@dataclass(frozen=True)
class ProcessScoreConfig:
    hazard_threshold: float = 0.30
    closure_threshold: float = 0.70
    closure_eps: float = 0.08
    answer_logprob_threshold: float = -3.50
    answer_eps: float = 0.60
    answer_survival_mode: str = "local"
    verify_logprob_threshold: float = -4.50
    verify_eps: float = 0.80
    verify_mode: str = "hybrid"
    verify_relative_weight: float = 0.50
    verify_relative_eps: float = 0.75
    reasoning_verify_offset: float = 0.75
    drift_logprob_threshold: float = -5.00
    drift_eps: float = 0.80
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

    def dynamics_config(self) -> DynamicsConfig:
        return DynamicsConfig(
            closure_threshold=float(self.closure_threshold),
            jump_threshold=float(self.jump_threshold),
            jump_quantile=float(self.jump_quantile),
            plateau_high_threshold=float(self.plateau_high_threshold),
            plateau_slope_threshold=float(self.plateau_slope_threshold),
            min_plateau_tokens=int(self.min_plateau_tokens),
            min_stage_gap=int(self.min_stage_gap),
            smooth_window=int(self.smooth_window),
            local_peak_quantile=float(self.local_peak_quantile),
            local_valley_quantile=float(self.local_valley_quantile),
            local_reset_margin=float(self.local_reset_margin),
            answer_onset_threshold=float(self.answer_onset_threshold),
        )


def score_response_process(
    model: LocalCausalLM,
    head: DifferentiableExitHazardHead,
    prompt: str,
    suffix: str,
    response_ids: Sequence[int],
    config: ProcessScoreConfig = ProcessScoreConfig(),
    *,
    include_token_rows: bool = True,
) -> tuple[dict, list[dict]]:
    if not response_ids:
        return _empty_summary(), []

    tokenizer = model.tokenizer
    prompt_text = model.build_prompt_text(prompt, suffix)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    full_ids = list(prompt_ids) + [int(token_id) for token_id in response_ids]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        start = len(prompt_ids)
        end = start + len(response_ids)
        hidden = outputs.hidden_states[head.config.layer][0, start:end, :].float()
        logits = outputs.logits[0, start:end, :].float()
        logit_features = exit_logit_features_from_logits(logits, tokenizer)
        raw = head(hidden, logit_features)
        cumprob, cumlogit = head.cumulative_scores(raw)
        process = exit_process_scores(
            logits,
            tokenizer,
            cumprob,
            closure_threshold=float(config.hazard_threshold),
            closure_eps=float(config.closure_eps),
            answer_logprob_threshold=float(config.answer_logprob_threshold),
            answer_eps=float(config.answer_eps),
            answer_survival_mode=str(config.answer_survival_mode),
            verify_logprob_threshold=float(config.verify_logprob_threshold),
            verify_eps=float(config.verify_eps),
            verify_mode=str(config.verify_mode),
            verify_relative_weight=float(config.verify_relative_weight),
            verify_relative_eps=float(config.verify_relative_eps),
            reasoning_verify_offset=float(config.reasoning_verify_offset),
            drift_logprob_threshold=float(config.drift_logprob_threshold),
            drift_eps=float(config.drift_eps),
        )

    raw_cpu = raw.detach().cpu()
    cumprob_cpu = cumprob.detach().cpu()
    cumlogit_cpu = cumlogit.detach().cpu()
    process_cpu = {key: value.detach().cpu() for key, value in process.items()}
    summary = summarize_dynamics(
        raw_hazard=raw_cpu.numpy(),
        cumprob=cumprob_cpu.numpy(),
        cumlogit=cumlogit_cpu.numpy(),
        q_closure=process_cpu["q_closure"].numpy(),
        answer_survival=process_cpu["answer_survival"].numpy(),
        verify_prob=process_cpu["verify_prob"].numpy(),
        drift_prob=process_cpu["drift_prob"].numpy(),
        pcg=process_cpu["pcg"].numpy(),
        vpcg=process_cpu["vpcg"].numpy(),
        lambda_answer=process_cpu["lambda_answer"].numpy(),
        config=config.dynamics_config(),
    )
    summary["first_hazard_cross_token"] = _first_cross(cumprob_cpu.tolist(), float(config.hazard_threshold))
    summary["post_exit_tokens"] = (
        int(len(response_ids) - int(summary["first_hazard_cross_token"]))
        if summary["first_hazard_cross_token"] is not None
        else 0
    )

    token_rows = []
    if include_token_rows:
        response_ids_list = [int(token_id) for token_id in response_ids]
        for index, token_id in enumerate(response_ids_list):
            token_rows.append(
                {
                    "token_index": int(index + 1),
                    "token_id": int(token_id),
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=True),
                    "exit_hazard": float(raw_cpu[index]),
                    "exit_hazard_cumprob": float(cumprob_cpu[index]),
                    "exit_hazard_cumlogit": float(cumlogit_cpu[index]),
                    "q_closure": float(process_cpu["q_closure"][index]),
                    "lambda_answer": float(process_cpu["lambda_answer"][index]),
                    "answer_survival": float(process_cpu["answer_survival"][index]),
                    "verify_prob": float(process_cpu["verify_prob"][index]),
                    "verify_abs": float(process_cpu["verify_abs"][index]),
                    "verify_relative": float(process_cpu["verify_relative"][index]),
                    "verify_evidence": float(process_cpu["verify_evidence"][index]),
                    "drift_prob": float(process_cpu["drift_prob"][index]),
                    "pcg": float(process_cpu["pcg"][index]),
                    "vpcg": float(process_cpu["vpcg"][index]),
                }
            )
    return summary, token_rows


def _first_cross(values: Sequence[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if float(value) >= float(threshold):
            return int(index + 1)
    return None


def _empty_summary() -> dict:
    return {
        "generated_tokens_scored": 0,
        "mean_raw_hazard": None,
        "mean_cumlogit": None,
        "max_cumprob": None,
        "first_cross_token": None,
        "post_cross_tokens": 0,
        "answer_onset_token": None,
        "closure_mean": None,
        "answer_survival_mean": None,
        "verify_mean": None,
        "drift_mean": None,
        "pcg_sum": None,
        "pcg_mean": None,
        "vpcg_sum": None,
        "vpcg_mean": None,
        "first_hazard_cross_token": None,
        "post_exit_tokens": 0,
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
        "local_peak_count": 0,
        "local_valley_count": 0,
        "local_reset_count": 0,
        "rise_reset_cycle_count": 0,
        "second_rise_rate": None,
    }
