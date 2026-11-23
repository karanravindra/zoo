"""Collect what each SAE latent responds to, on the held-out documents the SAE never trained on.

Runs the language model over the first `--num-docs` held-out documents (on the fly, like train_sae.py), encodes the
hook point with the SAE and keeps, per latent:
  - how often it fires (fraction of tokens) and its mean activation when it fires
  - its top-activating tokens (at most one per document, so one long document can't fill the list), with context
  - the output tokens its decoder direction promotes and suppresses through the final norm and the tied unembedding
    (the direct path only, so a rough guide at a middle layer)

Writes next to the checkpoint, to features/: features.json (all latents) and summary.txt (a readable dump of a
random sample).

    uv run projects/lfm_sae/features.py --checkpoint runs/lfm_sae/train/<timestamp>/sae_16384/best.pt
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import tyro

import extract
import train_sae
from common import hook_name
from sae import TopKSAE


@dataclass
class Config:
    checkpoint: Path  # train_sae.py checkpoint (runs/lfm_sae/train/<timestamp>/sae_<n>/best.pt)
    num_docs: int = 8192  # held-out documents to scan (at most the run's --holdout-docs)
    keep: int = 64  # candidate top activations kept per latent while scanning
    examples: int = 12  # examples per latent in the output (distinct documents)
    before: int = 24  # context tokens shown before the activating token
    after: int = 6  # context tokens shown after it
    logits: int = 10  # promoted / suppressed output tokens per latent
    sample: int = 60  # latents written to summary.txt (random, among those that fire)
    seed: int = 0  # seed for the summary sample
    encode_rows: int = 4096  # tokens per SAE encode (bounds the [rows, n_latents] pre-activation memory)
    device: str = "cuda"  # torch device


def merge_top(top_v: torch.Tensor, top_r: torch.Tensor, vals: torch.Tensor, lat: torch.Tensor, ref: torch.Tensor):
    """Merge new (value, latent, ref) triples into the per-latent top lists [L, keep] (empty slots are -inf)."""
    L, keep = top_v.shape
    v = torch.cat([top_v.flatten(), vals])
    r = torch.cat([top_r.flatten(), ref])
    l = torch.cat([torch.arange(L, device=v.device).repeat_interleave(keep), lat])
    o = torch.argsort(v, descending=True)
    o = o[torch.argsort(l[o], stable=True)]  # grouped by latent, descending value within each group
    l, v, r = l[o], v[o], r[o]
    start = torch.searchsorted(l, torch.arange(L, device=l.device))
    rank = torch.arange(len(l), device=l.device) - start[l]
    m = rank < keep
    new_v = torch.full_like(top_v, -torch.inf)
    new_r = torch.zeros_like(top_r)
    new_v[l[m], rank[m]] = v[m]
    new_r[l[m], rank[m]] = r[m]
    return new_v, new_r


def main(cfg: Config):
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    known = train_sae.Config.__dataclass_fields__
    tcfg = train_sae.Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ckpt["config"].items()
                               if k in known})
    tcfg = replace(tcfg, device=cfg.device, compile=False)
    if cfg.num_docs > tcfg.holdout_docs:
        raise SystemExit(f"--num-docs must be <= the run's holdout_docs ({tcfg.holdout_docs})")
    hook = hook_name(tcfg.site, tcfg.layer)
    L = ckpt["sae"]["W_dec"].shape[0]
    sae = TopKSAE(extract.D_MODEL, L, tcfg.k)
    sae.load_state_dict(ckpt["sae"])
    sae = sae.to(cfg.device).eval().requires_grad_(False)

    dcfg = train_sae.data_config(tcfg)
    sources = extract.load_sources(dcfg)
    lm, tok = extract.load_language_model(dcfg, truncate=False)
    norm_w = lm.embedding_norm.weight.float()
    W_U = lm.embed_tokens.weight  # tied head
    lm.layers = lm.layers[: tcfg.layer + 1]
    rec = extract.Recorder(lm, (tcfg.layer,), (tcfg.site,))

    docs: list[list[int]] = []
    doc_src: list[int] = []
    for chunk in extract.iter_chunks(dcfg, tok, sources, 0):
        docs += chunk.ids
        doc_src += chunk.sources
        if len(docs) >= cfg.num_docs:
            break
    docs, doc_src = docs[: cfg.num_docs], doc_src[: cfg.num_docs]
    ctx = tcfg.context

    count = torch.zeros(L, dtype=torch.long, device=cfg.device)
    total = torch.zeros(L, dtype=torch.float64, device=cfg.device)
    top_v = torch.full((L, cfg.keep), -torch.inf, device=cfg.device)
    top_r = torch.zeros((L, cfg.keep), dtype=torch.long, device=cfg.device)
    n_tokens = 0
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    order = [order[i : i + tcfg.lm_batch_size] for i in range(0, len(order), tcfg.lm_batch_size)]
    t0 = time.time()
    with torch.no_grad():
        for b, idx in enumerate(order):
            ids, mask = extract.collate([docs[i] for i in idx], tok.pad_token_id, 8)
            ids, mask = ids.to(cfg.device), mask.to(cfg.device)
            lm(input_ids=ids, attention_mask=mask, use_cache=False)
            keep = mask.bool()
            keep[:, 0] = False
            rows, cols = keep.nonzero(as_tuple=True)
            x = rec.out[hook][keep]
            vals, lat = map(torch.cat, zip(*(sae.latents(x[i : i + cfg.encode_rows])
                                             for i in range(0, len(x), cfg.encode_rows))))
            ref = torch.tensor(idx, device=cfg.device)[rows] * ctx + cols  # doc * context + position
            ref = ref[:, None].expand_as(lat)
            fired = vals > 0
            vals, lat, ref = vals[fired], lat[fired], ref[fired]
            count += torch.bincount(lat, minlength=L)
            total.index_add_(0, lat, vals.double())
            top_v, top_r = merge_top(top_v, top_r, vals.float(), lat, ref)
            n_tokens += len(x)
            if b % 50 == 0:
                print(f"batch {b}/{len(order)}  {n_tokens:,} tokens  {n_tokens / (time.time() - t0):,.0f} tok/s")

        # direct logit effect of each decoder direction: final RMSNorm weight, then the tied unembedding
        promo, supp = [], []
        for s in range(0, L, 1024):
            d = sae.W_dec[s : s + 1024] * norm_w
            lg = d.to(W_U.dtype) @ W_U.T
            promo.append(lg.topk(cfg.logits, dim=-1).indices.cpu())
            supp.append((-lg).topk(cfg.logits, dim=-1).indices.cpu())
        promo, supp = torch.cat(promo), torch.cat(supp)
    print(f"{n_tokens:,} tokens in {time.time() - t0:.0f}s")

    count, total, top_v, top_r = count.cpu(), total.cpu(), top_v.cpu(), top_r.cpu()
    piece = lambda t: tok.decode([t])  # noqa: E731

    def example(ref: int, value: float) -> dict:
        d, p = divmod(ref, ctx)
        toks = docs[d]
        lo, hi = max(1, p - cfg.before), min(len(toks), p + cfg.after + 1)
        return {"doc": d, "pos": p, "act": round(value, 3), "source": extract.SOURCES[doc_src[d]],
                "left": "".join(piece(t) for t in toks[lo:p]), "token": piece(toks[p]),
                "right": "".join(piece(t) for t in toks[p + 1 : hi])}

    features = []
    for i in range(L):
        exs, seen = [], set()
        for v, r in zip(top_v[i].tolist(), top_r[i].tolist()):
            if v == -torch.inf or len(exs) >= cfg.examples:
                break
            if r // ctx in seen:
                continue
            seen.add(r // ctx)
            exs.append(example(r, v))
        n = int(count[i])
        features.append({
            "id": i, "freq": n / n_tokens, "mean_act": float(total[i] / n) if n else 0.0,
            "max_act": float(top_v[i, 0]) if n else 0.0, "examples": exs,
            "promotes": [piece(t) for t in promo[i].tolist()], "suppresses": [piece(t) for t in supp[i].tolist()],
        })

    out = Path(cfg.checkpoint).parent / "features"
    out.mkdir(exist_ok=True)
    (out / "features.json").write_text(json.dumps(
        {"hook": hook, "checkpoint": str(cfg.checkpoint), "n_latents": L, "eval": ckpt.get("eval"), "docs": len(docs), "tokens": n_tokens, "features": features}))

    alive = [f for f in features if f["freq"] > 0]
    freqs = torch.tensor([f["freq"] for f in features])
    lines = [f"{hook}  {n_tokens:,} held-out tokens  {len(alive)}/{L} latents fire",
             "firing frequency quantiles (10/50/90/99%): "
             + ", ".join(f"{q:.2e}" for q in torch.quantile(freqs, torch.tensor([0.1, 0.5, 0.9, 0.99])).tolist()),
             f"latents firing on >10% of tokens: {(freqs > 0.1).sum().item()}", ""]
    for f in sorted(random.Random(cfg.seed).sample(alive, min(cfg.sample, len(alive))), key=lambda f: f["id"]):
        lines.append(f"=== latent {f['id']}  freq {f['freq']:.2e}  max {f['max_act']:.2f}")
        lines.append(f"  promotes:   {f['promotes']}")
        lines.append(f"  suppresses: {f['suppresses']}")
        for e in f["examples"][:8]:
            text = (e["left"] + "[[" + e["token"] + "]]" + e["right"]).replace("\n", "⏎")
            lines.append(f"  {e['act']:6.2f} {e['source'][0]} | {text}")
        lines.append("")
    (out / "summary.txt").write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
