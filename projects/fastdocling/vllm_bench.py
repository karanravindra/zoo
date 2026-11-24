# /// script
# requires-python = ">=3.12"
# dependencies = ["vllm", "datasets", "pillow", "transformers"]
# ///
"""vLLM throughput worker for `cli.py vllm-bench`: batch-1 decoding of the benchmark pages, plain or with vLLM's n-gram drafter.

Runs in its own uv environment (current vLLM pins an older torch than the project) with one engine per invocation; the
PrepareConfig-style JSON argument comes from the CLI. Each page is timed twice, a full request and a 1-token request, so decode
time is the difference (prefill and the vision encoder cancel).
"""

import json
import sys
import time

from pages import load_pages
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def main(cfg: dict) -> None:
    processor = AutoProcessor.from_pretrained(cfg["target"])
    spec = None
    if cfg["spec"] == "ngram":
        spec = {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    llm = LLM(model=cfg["target"], gpu_memory_utilization=cfg["gpu_memory_utilization"], limit_mm_per_prompt={"image": 1},
              max_model_len=cfg["max_new_tokens"] + 4096, speculative_config=spec, enable_prefix_caching=False)
    full = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    one = SamplingParams(temperature=0.0, max_tokens=1)
    records = []
    for task in cfg["tasks"]:
        images, prompt = load_pages(task, cfg["data"], cfg["n_pages"])
        text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                                             add_generation_prompt=True)
        reqs = [{"prompt": text, "multi_modal_data": {"image": im}} for im in images]
        llm.generate(reqs[0], one, use_tqdm=False)  # warm-up
        llm.generate(reqs[0], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
        tokens = prefill = total = 0.0
        for r in reqs:
            t0 = time.perf_counter()
            llm.generate(r, one, use_tqdm=False)
            t1 = time.perf_counter()
            out = llm.generate(r, full, use_tqdm=False)
            t2 = time.perf_counter()
            tokens += len(out[0].outputs[0].token_ids)
            prefill += t1 - t0
            total += t2 - t1
        decode = max(total - prefill, 1e-9)
        records.append(dict(task=task, spec=cfg["spec"], pages=len(reqs), tokens=int(tokens), decode_tok_s=tokens / decode,
                            e2e_tok_s=tokens / total))
        print(records[-1], flush=True)
    with open(cfg["out"], "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
