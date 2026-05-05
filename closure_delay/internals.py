"""Extract hidden states from model at specified layers and positions."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import numpy as np


DEFAULT_LAYERS = [8, 12, 16, 20, 24, 27]


@torch.no_grad()
def extract_hidden_trajectory(
    model,
    prompt: str,
    suffix: str,
    response_ids: Sequence[int],
    layers: Sequence[int] | None = None,
    stride: int = 16,
) -> Dict:
    """Extract hidden states at evenly-spaced positions within the response.

    Returns dict with keys:
        layer_{L}: np.array of shape (n_positions, hidden_size)
        positions: list of token indices (0-indexed within response)
        prompt_length: int
        response_length: int
    """
    if layers is None:
        layers = DEFAULT_LAYERS

    # Build full input
    prompt_text = model.build_prompt_text(prompt, suffix)
    prompt_ids = model.tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = int(prompt_ids.shape[1])
    response_tensor = torch.tensor([list(response_ids)], dtype=torch.long, device=model.device)
    input_ids = torch.cat([prompt_ids, response_tensor], dim=1)
    total_len = int(input_ids.shape[1])

    # Collect target positions (within response tokens)
    response_len = len(response_ids)
    target_positions = list(range(0, response_len, stride))
    if not target_positions:
        return _empty_result(layers, prompt_len, response_len)

    # Forward pass
    outputs = model.model(input_ids=input_ids, output_hidden_states=True)
    all_hidden = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)

    result: Dict = {"prompt_length": prompt_len, "response_length": response_len, "positions": target_positions}

    for layer_idx in layers:
        if layer_idx >= len(all_hidden):
            continue
        hidden = all_hidden[layer_idx][0]  # (seq_len, hidden_size)
        # Collect states at target positions (relative to prompt end)
        states = []
        for pos in target_positions:
            abs_pos = prompt_len + pos
            if abs_pos < total_len:
                states.append(hidden[abs_pos].detach().cpu().to(torch.float32).numpy())
        if states:
            result[f"layer_{layer_idx}"] = np.stack(states, axis=0)

    return result


def _empty_result(layers: Sequence[int], prompt_len: int, response_len: int) -> Dict:
    result: Dict = {"prompt_length": prompt_len, "response_length": response_len, "positions": []}
    for layer_idx in layers:
        result[f"layer_{layer_idx}"] = np.array([])
    return result


def collect_all_positions(
    model,
    prompt: str,
    suffix: str,
    response_ids: Sequence[int],
    layers: Sequence[int] | None = None,
    fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> Dict:
    """Extract hidden states at specific fraction positions within the response.

    Like extract_hidden_trajectory but positions are chosen by fraction of response length
    rather than by stride.
    """
    if layers is None:
        layers = DEFAULT_LAYERS

    prompt_text = model.build_prompt_text(prompt, suffix)
    prompt_ids = model.tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = int(prompt_ids.shape[1])
    response_tensor = torch.tensor([list(response_ids)], dtype=torch.long, device=model.device)
    input_ids = torch.cat([prompt_ids, response_tensor], dim=1)
    total_len = int(input_ids.shape[1])
    response_len = len(response_ids)

    # Map fractions to token indices
    target_positions = []
    for frac in fractions:
        idx = int(round(frac * response_len))
        idx = max(0, min(idx, response_len - 1))
        if idx not in target_positions:
            target_positions.append(idx)

    if not target_positions:
        return _empty_result(layers, prompt_len, response_len)

    outputs = model.model(input_ids=input_ids, output_hidden_states=True)
    all_hidden = outputs.hidden_states

    result: Dict = {
        "prompt_length": prompt_len,
        "response_length": response_len,
        "positions": target_positions,
        "fractions": list(fractions),
    }

    for layer_idx in layers:
        if layer_idx >= len(all_hidden):
            continue
        hidden = all_hidden[layer_idx][0]
        states = []
        for pos in target_positions:
            abs_pos = prompt_len + pos
            if abs_pos < total_len:
                states.append(hidden[abs_pos].detach().cpu().to(torch.float32).numpy())
        if states:
            result[f"layer_{layer_idx}"] = np.stack(states, axis=0)

    return result
