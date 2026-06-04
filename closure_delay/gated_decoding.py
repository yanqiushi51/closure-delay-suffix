from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .model import GenerationTrace, LocalCausalLM
from .probes import CLOSURE_PROBES, CONTINUATION_PROBES


BASE_CLOSURE_GATE_PHRASES = [
    *CLOSURE_PROBES,
    " Therefore,",
    " Therefore",
    " So,",
    " So the answer is",
    " Thus,",
    " Thus the answer is",
    " Hence,",
    " Final answer:",
    " final answer",
    " Answer:",
    " answer is",
    "\nFinal answer:",
    "\nTherefore,",
]


EXPANDED_CLOSURE_GATE_PHRASES = [
    *BASE_CLOSURE_GATE_PHRASES,
    " ####",
    " In conclusion,",
    " Conclusion:",
    " Conclusion",
    " **Conclusion",
    " **Final answer",
    " **Answer",
    " End result",
    " **End result",
    " \\boxed",
    " boxed{",
    " This completes",
    " If you have",
    " feel free to ask",
    " I'm here to help",
    " Let me know",
    "\nConclusion:",
    "\n**Answer",
]


DEFAULT_CLOSURE_GATE_PHRASES = BASE_CLOSURE_GATE_PHRASES


DEFAULT_CLOSURE_BOOST_PHRASES = [
    " Final answer:",
    " Therefore, the final answer is",
    " The final answer is",
    " So the answer is",
    " Answer:",
    " Conclusion:",
    "\nFinal answer:",
]


BASIC_POST_GATE_CONTINUATION_PHRASES = [
    *CONTINUATION_PROBES,
    " If you",
    " If you'd",
    " If there",
    " feel free",
    " I'm here",
    " Let me know",
    " Thank you",
    " Please let",
    " Sure,",
    " Could you",
    " I think",
    "<tool_call>",
    "\nuser",
    "\n user",
    "\nassistant",
]


DRIFT_POST_GATE_CONTINUATION_PHRASES = [
    *BASIC_POST_GATE_CONTINUATION_PHRASES,
    " I am here",
    " I am ready",
    " I can",
    " more questions",
    " further questions",
    " additional assistance",
    " need assistance",
    " Good luck",
    " Happy",
    " Have a",
    " Remember,",
    " Thankyou",
    " service",
    " Qwen",
    " tag list",
    " Python code",
    "```python",
    "```",
    "#",
    " #",
    "adventure",
    "Adventure",
    "environmental",
    "ecological",
    "ecosystem",
    "sustainable",
    "sustainability",
    "biodiversity",
    "natural world",
    "stewardship",
    "🌟",
    "✨",
    "🎉",
    "👣",
    "🚀",
    "🌱",
    "🌊",
]


DEFAULT_POST_GATE_CONTINUATION_PHRASES = BASIC_POST_GATE_CONTINUATION_PHRASES


DEFAULT_MATH_GUIDANCE_PHRASES = [
    "\nVerification:",
    "\nCheck:",
    "\nArithmetic check:",
    "\nUnit check:",
    "\nReverse check:",
    "\nRecalculate:",
    "\nNow verify",
    "\nTo verify",
    "\nAnother check:",
    "\nConsistency check:",
    "\nMagnitude check:",
    "\nCompute again:",
]


DEFAULT_SEMANTIC_MORE_GUIDANCE_PHRASES = [
    "\nNext, verify the arithmetic.",
    "\nWe still need to check the calculation.",
    "\nAnother relevant check is needed.",
    "\nContinue by testing the result against the problem statement.",
    "\nBefore answering, do one more consistency check.",
    "\nDetailed verification:",
    "\nAlternative check:",
    "\nConsistency check:",
    "\nCross-check:",
    "\nNow validate the result:",
]


@dataclass(frozen=True)
class ClosureGateConfig:
    gate_until_new_tokens: int
    penalty: float = 12.0
    hard_block: bool = False
    suppress_eos: bool = True
    phrases: Sequence[str] = tuple(DEFAULT_CLOSURE_GATE_PHRASES)
    boost_after_new_tokens: int | None = None
    boost: float = 0.0
    boost_eos: bool = False
    boost_phrases: Sequence[str] = tuple(DEFAULT_CLOSURE_BOOST_PHRASES)
    pre_gate_guidance_after_new_tokens: int | None = None
    pre_gate_guidance_until_new_tokens: int | None = None
    pre_gate_guidance_boost: float = 0.0
    pre_gate_guidance_phrases: Sequence[str] = tuple(DEFAULT_MATH_GUIDANCE_PHRASES)
    pre_gate_completion_block_phrases: Sequence[str] = ()
    continuation_penalty_after_new_tokens: int | None = None
    continuation_penalty: float = 0.0
    continuation_hard_block: bool = False
    continuation_completion_only: bool = True
    continuation_phrases: Sequence[str] = tuple(DEFAULT_POST_GATE_CONTINUATION_PHRASES)


class ClosureGateLogitsProcessor(LogitsProcessor):
    """Suppress closure-marker phrases until a minimum generation length is reached."""

    def __init__(
        self,
        *,
        prompt_length: int,
        gate_until_new_tokens: int,
        phrase_token_ids: Sequence[Sequence[int]],
        boost_phrase_token_ids: Sequence[Sequence[int]] = (),
        guidance_phrase_token_ids: Sequence[Sequence[int]] = (),
        pre_gate_completion_block_token_ids: Sequence[Sequence[int]] = (),
        continuation_phrase_token_ids: Sequence[Sequence[int]] = (),
        eos_token_ids: Iterable[int] = (),
        penalty: float = 12.0,
        hard_block: bool = False,
        boost_after_new_tokens: int | None = None,
        boost: float = 0.0,
        boost_eos: bool = False,
        pre_gate_guidance_after_new_tokens: int | None = None,
        pre_gate_guidance_until_new_tokens: int | None = None,
        pre_gate_guidance_boost: float = 0.0,
        continuation_penalty_after_new_tokens: int | None = None,
        continuation_penalty: float = 0.0,
        continuation_hard_block: bool = False,
        continuation_completion_only: bool = True,
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.gate_until_new_tokens = max(0, int(gate_until_new_tokens))
        self.phrase_token_ids = [tuple(int(token) for token in phrase) for phrase in phrase_token_ids if phrase]
        self.boost_phrase_token_ids = [
            tuple(int(token) for token in phrase)
            for phrase in boost_phrase_token_ids
            if phrase
        ]
        self.guidance_phrase_token_ids = [
            tuple(int(token) for token in phrase)
            for phrase in guidance_phrase_token_ids
            if phrase
        ]
        self.pre_gate_completion_block_token_ids = [
            tuple(int(token) for token in phrase)
            for phrase in pre_gate_completion_block_token_ids
            if phrase
        ]
        self.continuation_phrase_token_ids = [
            tuple(int(token) for token in phrase)
            for phrase in continuation_phrase_token_ids
            if phrase
        ]
        self.eos_token_ids = {int(token) for token in eos_token_ids if token is not None}
        self.penalty = float(penalty)
        self.hard_block = bool(hard_block)
        self.boost_after_new_tokens = (
            None if boost_after_new_tokens is None else max(0, int(boost_after_new_tokens))
        )
        self.boost = float(boost)
        self.boost_eos = bool(boost_eos)
        self.pre_gate_guidance_after_new_tokens = (
            None
            if pre_gate_guidance_after_new_tokens is None
            else max(0, int(pre_gate_guidance_after_new_tokens))
        )
        self.pre_gate_guidance_until_new_tokens = (
            None
            if pre_gate_guidance_until_new_tokens is None
            else max(0, int(pre_gate_guidance_until_new_tokens))
        )
        self.pre_gate_guidance_boost = float(pre_gate_guidance_boost)
        self.continuation_penalty_after_new_tokens = (
            None
            if continuation_penalty_after_new_tokens is None
            else max(0, int(continuation_penalty_after_new_tokens))
        )
        self.continuation_penalty = float(continuation_penalty)
        self.continuation_hard_block = bool(continuation_hard_block)
        self.continuation_completion_only = bool(continuation_completion_only)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_count = int(input_ids.shape[1]) - self.prompt_length
        generated_ids = input_ids[0, self.prompt_length :].detach().cpu().tolist()

        if generated_count < self.gate_until_new_tokens:
            blocked = set(self.eos_token_ids)
            for phrase in self.phrase_token_ids:
                next_token = next_blocked_token(generated_ids, phrase)
                if next_token is not None:
                    blocked.add(next_token)
            for phrase in self.pre_gate_completion_block_token_ids:
                next_token = next_phrase_completion_token(generated_ids, phrase)
                if next_token is not None:
                    blocked.add(next_token)

            if blocked:
                token_ids = torch.tensor(sorted(blocked), dtype=torch.long, device=scores.device)
                if self.hard_block:
                    scores[:, token_ids] = -torch.inf
                else:
                    scores[:, token_ids] = scores[:, token_ids] - self.penalty

        guidance_until = (
            self.gate_until_new_tokens
            if self.pre_gate_guidance_until_new_tokens is None
            else self.pre_gate_guidance_until_new_tokens
        )
        if (
            self.pre_gate_guidance_after_new_tokens is not None
            and generated_count >= self.pre_gate_guidance_after_new_tokens
            and generated_count < guidance_until
            and self.pre_gate_guidance_boost > 0
        ):
            guided = set()
            for phrase in self.guidance_phrase_token_ids:
                next_token = next_blocked_token(generated_ids, phrase)
                if next_token is not None:
                    guided.add(next_token)
            if guided:
                token_ids = torch.tensor(sorted(guided), dtype=torch.long, device=scores.device)
                scores[:, token_ids] = scores[:, token_ids] + self.pre_gate_guidance_boost

        if (
            self.boost_after_new_tokens is not None
            and generated_count >= self.boost_after_new_tokens
            and self.boost > 0
        ):
            boosted = set(self.eos_token_ids) if self.boost_eos else set()
            for phrase in self.boost_phrase_token_ids:
                next_token = next_blocked_token(generated_ids, phrase)
                if next_token is not None:
                    boosted.add(next_token)
            if boosted:
                token_ids = torch.tensor(sorted(boosted), dtype=torch.long, device=scores.device)
                scores[:, token_ids] = scores[:, token_ids] + self.boost

        if (
            self.continuation_penalty_after_new_tokens is not None
            and generated_count >= self.continuation_penalty_after_new_tokens
            and (self.continuation_penalty > 0 or self.continuation_hard_block)
        ):
            penalized = set()
            for phrase in self.continuation_phrase_token_ids:
                if self.continuation_completion_only:
                    next_token = next_phrase_completion_token(generated_ids, phrase)
                else:
                    next_token = next_blocked_token(generated_ids, phrase)
                if next_token is not None:
                    penalized.add(next_token)
            if penalized:
                token_ids = torch.tensor(sorted(penalized), dtype=torch.long, device=scores.device)
                if self.continuation_hard_block:
                    scores[:, token_ids] = -torch.inf
                else:
                    scores[:, token_ids] = scores[:, token_ids] - self.continuation_penalty
        return scores


def next_blocked_token(generated_ids: Sequence[int], phrase: Sequence[int]) -> int | None:
    """Return the next phrase token to block when the generated suffix matches a phrase prefix."""

    if not phrase:
        return None
    if not generated_ids:
        return int(phrase[0])
    max_prefix = min(len(phrase) - 1, len(generated_ids))
    for prefix_len in range(max_prefix, -1, -1):
        if prefix_len == 0:
            return int(phrase[0])
        if list(generated_ids[-prefix_len:]) == list(phrase[:prefix_len]):
            return int(phrase[prefix_len])
    return int(phrase[0])


def next_phrase_completion_token(generated_ids: Sequence[int], phrase: Sequence[int]) -> int | None:
    """Return the token that would complete a seen prefix of a blocked phrase."""

    if not phrase:
        return None
    if len(phrase) == 1:
        return int(phrase[0])
    if not generated_ids:
        return None
    max_prefix = min(len(phrase) - 1, len(generated_ids))
    for prefix_len in range(max_prefix, 0, -1):
        if list(generated_ids[-prefix_len:]) == list(phrase[:prefix_len]):
            return int(phrase[prefix_len])
    return None


def build_phrase_token_ids(tokenizer, phrases: Sequence[str]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    rows: list[list[int]] = []
    for phrase in phrases:
        variants = {phrase, phrase.lstrip()}
        if not phrase.startswith("\n"):
            variants.add("\n" + phrase.lstrip())
        for variant in variants:
            token_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            key = tuple(int(token_id) for token_id in token_ids)
            if key and key not in seen:
                seen.add(key)
                rows.append(list(key))
    return rows


def collect_eos_token_ids(model: LocalCausalLM) -> list[int]:
    token_ids: set[int] = set()
    for value in (
        getattr(model.tokenizer, "eos_token_id", None),
        getattr(model.model.config, "eos_token_id", None),
        getattr(model.model.generation_config, "eos_token_id", None),
    ):
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            token_ids.update(int(item) for item in value if item is not None)
        else:
            token_ids.add(int(value))
    return sorted(token_ids)


@torch.no_grad()
def generate_gated_trace(
    model: LocalCausalLM,
    prompt: str,
    suffix: str,
    max_new_tokens: int,
    gate: ClosureGateConfig,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
    min_new_tokens: int | None = None,
) -> GenerationTrace:
    prompt_text = model.build_prompt_text(prompt, suffix)
    inputs = model.tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_length = int(inputs["input_ids"].shape[1])

    phrase_token_ids = build_phrase_token_ids(model.tokenizer, gate.phrases)
    boost_phrase_token_ids = build_phrase_token_ids(model.tokenizer, gate.boost_phrases)
    guidance_phrase_token_ids = build_phrase_token_ids(model.tokenizer, gate.pre_gate_guidance_phrases)
    pre_gate_completion_block_token_ids = build_phrase_token_ids(
        model.tokenizer,
        gate.pre_gate_completion_block_phrases,
    )
    continuation_phrase_token_ids = build_phrase_token_ids(model.tokenizer, gate.continuation_phrases)
    eos_token_ids = collect_eos_token_ids(model) if gate.suppress_eos else []
    processor = ClosureGateLogitsProcessor(
        prompt_length=prompt_length,
        gate_until_new_tokens=gate.gate_until_new_tokens,
        phrase_token_ids=phrase_token_ids,
        boost_phrase_token_ids=boost_phrase_token_ids,
        guidance_phrase_token_ids=guidance_phrase_token_ids,
        pre_gate_completion_block_token_ids=pre_gate_completion_block_token_ids,
        continuation_phrase_token_ids=continuation_phrase_token_ids,
        eos_token_ids=eos_token_ids,
        penalty=gate.penalty,
        hard_block=gate.hard_block,
        boost_after_new_tokens=gate.boost_after_new_tokens,
        boost=gate.boost,
        boost_eos=gate.boost_eos,
        pre_gate_guidance_after_new_tokens=gate.pre_gate_guidance_after_new_tokens,
        pre_gate_guidance_until_new_tokens=gate.pre_gate_guidance_until_new_tokens,
        pre_gate_guidance_boost=gate.pre_gate_guidance_boost,
        continuation_penalty_after_new_tokens=gate.continuation_penalty_after_new_tokens,
        continuation_penalty=gate.continuation_penalty,
        continuation_hard_block=gate.continuation_hard_block,
        continuation_completion_only=gate.continuation_completion_only,
    )
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": False,
        "pad_token_id": model.tokenizer.pad_token_id,
        "logits_processor": LogitsProcessorList([processor]),
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = max(0, int(min_new_tokens))
    outputs = model.model.generate(**inputs, **gen_kwargs)

    sequence = outputs.sequences[0]
    generated_ids = sequence[prompt_length:]
    full_text = model.tokenizer.decode(sequence, skip_special_tokens=True)
    response_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationTrace(
        prompt=prompt,
        suffix=suffix,
        prompt_text=prompt_text,
        response_text=response_text,
        full_text=full_text,
        generated_token_count=len(generated_ids),
        generated_ids=generated_ids.detach().cpu().tolist(),
    )
