"""Local LLM client: Qwen3-14B, bitsandbytes 4-bit (NF4), loaded once and reused across
many risk-scoring calls. `enable_thinking=False` is deliberate — Qwen3 defaults to an
extended <think> reasoning trace before its answer, which is unnecessary latency/cost for
a bounded structured-extraction task like this and was observed to blow through a 150-token
budget without ever reaching the JSON answer.

Model choice note: the approved plan named Qwen2.5-14B-Instruct as the default, with an
explicit instruction to check for a newer same-class model at implementation time (~6
months had passed since the plan's knowledge cutoff). Qwen3-14B is that newer generation
and is used here instead — the sizing/quantization logic (4-bit, ~10.6GB VRAM measured)
holds regardless of exact model, per the plan's own reasoning.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-14B"
CACHE_DIR = "/home/user5/Desktop/MS THESIS/models_cache"


class LocalLLMClient:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda:0"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=CACHE_DIR, quantization_config=bnb_config, device_map=device
        )
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False,
        ).to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
        )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
