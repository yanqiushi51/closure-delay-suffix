from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationTrace:
    prompt: str
    suffix: str
    prompt_text: str
    response_text: str
    full_text: str
    generated_token_count: int
    generated_ids: List[int]


class LocalCausalLM:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True}
        if self.device.type == "cuda":
            model_kwargs["dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        if self.device.type == "cuda":
            self.model.to(self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt_text(self, prompt: str, suffix: str) -> str:
        user_text = prompt if not suffix else f"{prompt}\n\n{suffix}"
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_text}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_text

    @torch.no_grad()
    def generate_trace(
        self,
        prompt: str,
        suffix: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
    ) -> GenerationTrace:
        prompt_text = self.build_prompt_text(prompt, suffix)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
        outputs = self.model.generate(**inputs, **gen_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        sequence = outputs.sequences[0]
        generated_ids = sequence[prompt_length:]
        full_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationTrace(
            prompt=prompt,
            suffix=suffix,
            prompt_text=prompt_text,
            response_text=response_text,
            full_text=full_text,
            generated_token_count=len(generated_ids),
            generated_ids=generated_ids.detach().cpu().tolist(),
        )
