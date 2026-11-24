"""Speculative-decoding benchmark on the HF target: batch 1 greedy decoding with a KV cache, one page at a time (latency).

Every method decodes the same pages with the same loop style: plain greedy (one token per target pass) is the baseline, a drafter
proposes `k` tokens that one target pass verifies, keeping the longest matching prefix plus the target's own next token.
Timing covers vision encoding, prefill and decoding (processor pre-processing excluded), split into prefill and decode.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoProcessor, DynamicCache

from config import BenchConfig
from module import LitDrafter, load_target
from pages import load_pages

from zoo.training import load_module
from zoo.utils.device import get_device

@dataclass
class Run:
    tokens: list[int]
    prefill: float
    decode: float
    cycles: int = 0  # verification passes (speculative runs)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def decode_plain(target, enc, eos, max_new, device) -> Run:
    cache = DynamicCache()
    sync(device)
    t0 = time.perf_counter()
    out = target.model(**enc, past_key_values=cache, use_cache=True)
    cur = target.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)[:, 0]
    gen = [int(cur)]
    t1 = time.perf_counter()
    while len(gen) < max_new and gen[-1] not in eos:
        out = target.model(input_ids=cur[:, None], past_key_values=cache, use_cache=True)
        cur = target.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)[:, 0]
        gen.append(int(cur))
    sync(device)
    return Run(gen, t1 - t0, time.perf_counter() - t1, len(gen) - 1)


@torch.no_grad()
def decode_spec(target, drafter, embed, layers, enc, eos, max_new, k, device) -> Run:
    cache = DynamicCache()
    sync(device)
    t0 = time.perf_counter()
    out = target.model(**enc, past_key_values=cache, use_cache=True, output_hidden_states=True)
    cur = target.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)[:, 0]
    gen = [int(cur)]
    t1 = time.perf_counter()
    feats = torch.cat([out.hidden_states[i] for i in layers], dim=-1)
    nxt = torch.cat([enc["input_ids"][:, 1:], cur[:, None]], dim=1)  # x[t+1] for every prompt position
    drafts, state = drafter.propose_cached(None, feats, out.last_hidden_state, nxt, embed, k)
    cycles = 0
    while len(gen) < max_new and gen[-1] not in eos:
        kk = drafts.shape[1]
        out = target.model(input_ids=torch.cat([cur[:, None], drafts], dim=1), past_key_values=cache, use_cache=True,
                           output_hidden_states=True)
        g = target.lm_head(out.last_hidden_state).argmax(-1)[0]  # the target's greedy token after each fed one
        m = int((drafts[0] == g[:-1]).int().cumprod(0).sum())  # accepted drafts
        new = torch.cat([drafts[0, :m], g[m : m + 1]])  # accepted drafts + the target's own next token
        if m < kk:
            cache.crop(-(kk - m))  # drop the rejected drafts' keys/values
        gen += new.tolist()
        cur = new[-1:]
        cycles += 1
        if any(t in eos for t in new.tolist()):
            break
        feats = torch.cat([out.hidden_states[i][:, : m + 1] for i in layers], dim=-1)
        drafts, state = drafter.propose_cached(state, feats, out.last_hidden_state[:, : m + 1], new[None], embed, k)
    sync(device)
    return Run(gen, t1 - t0, time.perf_counter() - t1, cycles)


def trim(tokens: list[int], eos: set[int], max_new: int) -> list[int]:
    tokens = tokens[:max_new]
    for i, t in enumerate(tokens):
        if t in eos:
            return tokens[: i + 1]
    return tokens


def run_bench(cfg: BenchConfig) -> list[dict]:
    device = get_device()
    dtype = getattr(torch, cfg.dtype)
    target = load_target(cfg.target, device, dtype)
    processor = AutoProcessor.from_pretrained(cfg.target)
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    embed = target.get_input_embeddings()

    drafters = []  # (label, drafter, feature layers, trained k)
    for path in cfg.ckpts:
        lit = load_module(LitDrafter, path)
        d = lit.drafter.to(device=device, dtype=dtype).eval()
        label = lit.cfg.drafter
        if any(label == x[0] for x in drafters):  # two runs of one method: tell them apart by run id
            label = f"{label}-{Path(path).parts[-3]}"
        drafters.append((label, d, lit.cfg.feature_layers, min(cfg.k, lit.cfg.k)))

    records = []
    for task in cfg.tasks:
        images, prompt = load_pages(task, cfg.data, cfg.n_pages)
        text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                                             add_generation_prompt=True)
        encs = []
        for im in images:
            enc = processor(text=[text], images=[[im]], return_tensors="pt").to(device)
            enc["pixel_values"] = enc["pixel_values"].to(dtype)
            encs.append(dict(enc))
        decode_plain(target, encs[0], eos, 16, device)  # warm-up (kernels, allocator), not timed
        for _, d, layers, kk in drafters:
            decode_spec(target, d, embed, layers, encs[0], eos, 16, kk, device)
        base = [decode_plain(target, e, eos, cfg.max_new_tokens, device) for e in encs]
        base_toks = [trim(r.tokens, eos, cfg.max_new_tokens) for r in base]
        n_tok = sum(len(t) for t in base_toks)
        records.append(summary(task, "baseline", base, base, base_toks, base_toks, n_tok, 1))
        for label, d, layers, kk in drafters:
            runs = [decode_spec(target, d, embed, layers, e, eos, cfg.max_new_tokens, kk, device) for e in encs]
            toks = [trim(r.tokens, eos, cfg.max_new_tokens) for r in runs]
            records.append(summary(task, label, runs, base, toks, base_toks, n_tok, kk))
        print_table([r for r in records if r["task"] == task])
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {cfg.out}")
    return records


def summary(task, method, runs, base, toks, base_toks, n_tok, k) -> dict:
    dec, pre = sum(r.decode for r in runs), sum(r.prefill for r in runs)
    bdec, bpre = sum(r.decode for r in base), sum(r.prefill for r in base)
    n = sum(len(t) for t in toks)
    return dict(task=task, method=method, k=k, pages=len(runs), tokens=n,
                tokens_per_pass=n / max(1, sum(r.cycles for r in runs) + len(runs)),  # + the prefill pass's token
                decode_tok_s=n / dec,
                # throughput ratios, not time ratios: an output that diverges from greedy (bf16) can differ in length
                decode_speedup=(n / dec) / (sum(len(t) for t in base_toks) / bdec),
                e2e_speedup=(n / (pre + dec)) / (sum(len(t) for t in base_toks) / (bpre + bdec)),
                exact_match=sum(a == b for a, b in zip(toks, base_toks)) / len(runs))


def print_table(rows: list[dict]) -> None:
    print(f"\n{rows[0]['task']}: {rows[0]['pages']} pages, {rows[0]['tokens']} baseline tokens")
    print(f"{'method':<12}{'tok/pass':>9}{'decode tok/s':>14}{'decode x':>10}{'e2e x':>8}{'exact':>8}")
    for r in rows:
        print(f"{r['method']:<12}{r['tokens_per_pass']:>9.2f}{r['decode_tok_s']:>14.1f}{r['decode_speedup']:>10.2f}"
              f"{r['e2e_speedup']:>8.2f}{r['exact_match']:>8.0%}")
