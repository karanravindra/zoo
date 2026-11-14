"""Loading the Qwen3.5 VLM (4-bit, optionally with a LoRA adapter) and greedy generation for one request."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from common import MAX_PIXELS, MIN_PIXELS, MODEL, build_messages

ADAPTER_META = "ledger.json"  # written next to a saved adapter: base model id + revision


def load_processor(model: str = MODEL, revision: str | None = None):
    return AutoProcessor.from_pretrained(model, revision=revision, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)


def load_model(model: str = MODEL, revision: str | None = None, quantize: bool = True):
    """The model on cuda:0 in bf16, with NF4 double-quantized weights when `quantize`."""
    if not torch.cuda.is_available():
        raise SystemExit("ledger needs a CUDA GPU (bitsandbytes 4-bit + device_map={'': 0}).")
    qcfg = None
    if quantize:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return AutoModelForImageTextToText.from_pretrained(
        model,
        revision=revision,
        quantization_config=qcfg,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )


def load_for_inference(adapter: str | None, quantize: bool | None = None):
    """(model, processor, meta). With an adapter, the base model, revision and quantization come from its
    ledger.json; `quantize` overrides it (default without an adapter: bf16)."""
    meta = {"model": MODEL, "model_revision": None, "adapter": adapter, "quantize": False}
    if adapter is not None:
        meta_path = Path(adapter) / ADAPTER_META
        if meta_path.exists():
            meta.update(json.loads(meta_path.read_text()))
    if quantize is not None:
        meta["quantize"] = quantize
    model = load_model(meta["model"], meta["model_revision"], meta["quantize"])
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, load_processor(meta["model"], meta["model_revision"]), meta


@torch.no_grad()
def generate_batch(model, processor, requests: list[tuple[str, dict, object]], max_new_tokens: int = 512) -> list[tuple[str, int]]:
    """Greedy answers for a batch of (state, fields, image-or-None) requests, left-padded.
    Returns (answer text, prompt length in tokens without padding) per request."""
    texts = [
        processor.apply_chat_template(build_messages(s, f, has_image=img is not None), tokenize=False, add_generation_prompt=True)
        for s, f, img in requests
    ]
    images = [img for _, _, img in requests if img is not None]
    processor.tokenizer.padding_side = "left"
    enc = processor(text=texts, images=images or None, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
    answers = processor.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return list(zip(answers, enc["attention_mask"].sum(1).tolist()))


def generate(model, processor, state: str, fields: dict, image=None, max_new_tokens: int = 512) -> tuple[str, int]:
    """Greedy answer text for one request, and the prompt length in tokens."""
    return generate_batch(model, processor, [(state, fields, image)], max_new_tokens)[0]
