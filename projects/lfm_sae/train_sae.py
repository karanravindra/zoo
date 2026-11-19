"""Train a TopK sparse autoencoder on LFM3-VL-2B activations generated on the fly.

The language model (cut after `--layer` while training) fills a shuffle buffer in CPU RAM. Each round, the trainer
takes random batches from half the buffer and refills those slots with fresh activations, so every token is used
exactly once and nothing goes to disk. The data mix and its defaults are the same as extract.py's.

Held out: the first `--holdout-docs` documents of the stream are never trained on. An extract.py run with the same
data settings and up to ~5M tokens stores only these documents, so it doubles as held-out data for browsing
features. Evaluation (every `--eval-every` steps and at the end) uses the first `--eval-docs` of them:
  fvu              fraction of variance unexplained, ||x - x_hat||^2 / ||x - mean||^2 (lower is better)
  dead_frac        latents that fired on no held-out token
  ce_*             next-token loss of the full model on `--ce-docs` documents: clean, with the SAE reconstruction
                   spliced in at the hook point, and with the activation replaced by its mean
  loss_recovered   (ce_ablate - ce_sae) / (ce_ablate - ce_clean): 1 = the SAE loses nothing

`--n-latents` takes several sizes: those SAEs train side by side on the same batches (the language model, the
expensive part, runs once).

Writes to runs/lfm_sae/train/<timestamp>/: config.json, metrics.csv (per SAE: train rows every `--log-every` steps,
eval rows), sae_<n_latents>/best.pt (lowest eval fvu) and last.pt ({"sae": state_dict, "config", "n_latents",
"hook", "step", "tokens", "eval"}), results.json (final eval per SAE, peak memory) and curves.png.

    uv run projects/lfm_sae/train_sae.py                     # 100M tokens, resid_post.12, 16k latents, k=64
    uv run projects/lfm_sae/train_sae.py --n-latents 16384 32768 65536   # 8x, 16x, 32x together
    uv run projects/lfm_sae/train_sae.py --num-tokens 5000000 --eval-every 500   # quick check
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA initializes

import torch
import torch.nn.functional as F
import tyro

import extract
from common import MODEL, MODEL_REVISION, hook_name, hub_sha
from sae import TopKSAE

from zoo.monitor import GPU_FIELDS, GpuMonitor
from zoo.paths import runs_root
from zoo.plot import plt
from zoo.runs import MetricsLogger, dev_run_dir, git_commit, tee_console
from zoo.train import STAT_FIELDS, ParamStats

D_MODEL = extract.D_MODEL


@dataclass
class Config:
    # what to train on
    layer: int = 12  # decoder layer to hook (0-23)
    site: extract.Site = "resid_post"  # hook point in that layer
    num_tokens: int = 100_000_000  # training tokens
    # SAE
    n_latents: tuple[int, ...] = (16384,)  # dictionary sizes; several SAEs train side by side on the same activations
    k: int = 64  # active latents per token
    k_aux: int = 1024  # dead latents used by the AuxK loss (d_model / 2)
    aux_coef: float = 1 / 32  # weight of the AuxK loss
    dead_tokens: int = 10_000_000  # a latent that hasn't fired for this many training tokens counts as dead for AuxK
    # optimization
    batch_size: int = 4096  # tokens per SAE step
    lr: float = 2e-4  # Adam learning rate at 16384 latents; each SAE uses lr / sqrt(n_latents / 16384) (Gao et al.)
    warmup_steps: int = 500  # linear warmup
    decay_start: float = 0.8  # fraction of training after which the lr decays linearly to 0
    buffer_tokens: int = 2**21  # shuffle buffer in CPU RAM (bf16: 4 KB per token, so 2M tokens = 8.6 GB)
    seed: int = 0  # seed for the data mix, permutations, SAE init and buffer shuffles
    # data (defaults match extract.py)
    model: str = MODEL  # Hub repo of the VLM
    model_revision: str = MODEL_REVISION  # pinned commit (the model ships remote code)
    web_dataset: str = "HuggingFaceFW/fineweb-edu"  # Hub dataset of web documents (`text` column)
    web_files: tuple[str, ...] = ("sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet")  # its files to download
    web_revision: str | None = None  # its commit (default: the current one, resolved and recorded)
    chat_dataset: str = "HuggingFaceTB/smoltalk"  # Hub dataset of conversations (`messages` column)
    chat_files: tuple[str, ...] = ("data/all/train-*.parquet",)  # its files to download
    chat_revision: str | None = None  # its commit (default: the current one, resolved and recorded)
    chat_fraction: float = 0.1  # probability that a document is a chat transcript
    context: int = 1024  # max tokens per document (BOS included)
    lm_batch_size: int = 32  # documents per language-model forward
    chunk_docs: int = 512  # documents per data-worker chunk
    # evaluation
    holdout_docs: int = 8192  # first documents of the stream, never trained on
    eval_docs: int = 512  # held-out documents for fvu and dead latents
    ce_docs: int = 64  # held-out documents for the loss-recovered eval (full-model forwards x3)
    eval_every: int = 2500  # steps between evals (~10M tokens)
    log_every: int = 100  # steps between train log rows
    device: str = "cuda"  # torch device
    compile: bool = False  # torch.compile the language model (pads batches to a multiple of 128; evals recompile)
    dev: bool = False  # dev run: outputs go to a temporary directory deleted on exit, so nothing is logged to runs/


def data_config(cfg: Config) -> extract.Config:
    """The extract.Config that the shared data and model helpers read."""
    return extract.Config(
        model=cfg.model, model_revision=cfg.model_revision, layers=(cfg.layer,), sites=(cfg.site,),
        web_dataset=cfg.web_dataset, web_files=cfg.web_files, web_revision=cfg.web_revision,
        chat_dataset=cfg.chat_dataset, chat_files=cfg.chat_files, chat_revision=cfg.chat_revision,
        chat_fraction=cfg.chat_fraction, context=cfg.context, batch_size=cfg.lm_batch_size,
        chunk_docs=cfg.chunk_docs, seed=cfg.seed, device=cfg.device, compile=cfg.compile,
    )


# ---------------------------------------------------------------- activations


def lm_rows(lm, rec, hook: str, docs: list[list[int]], order: list[list[int]], pad_id: int, device, multiple: int):
    """Activation rows [n, d] (bf16, on device) per language-model batch: real tokens only, BOS excluded."""
    for idx in order:
        ids, mask = extract.collate([docs[i] for i in idx], pad_id, multiple)
        ids, mask = ids.to(device), mask.to(device)
        lm(input_ids=ids, attention_mask=mask, use_cache=False)
        keep = mask.bool()
        keep[:, 0] = False
        yield rec.out[hook][keep]


def activation_stream(cfg: Config, dcfg: extract.Config, lm, rec, tok, sources) -> Iterator[torch.Tensor]:
    """Training activations from document `holdout_docs` on, forever (until a source runs out)."""
    hook = hook_name(cfg.site, cfg.layer)
    multiple = 128 if cfg.compile else 8
    for chunk in extract.chunk_loader(dcfg, tok, sources, cfg.holdout_docs):
        yield from lm_rows(lm, rec, hook, chunk.ids, chunk.batches, tok.pad_token_id, cfg.device, multiple)
    raise SystemExit("ran out of training documents: add --web-files / --chat-files")


class ShuffleBuffer:
    """`size` activation rows in CPU RAM (GPU memory goes to the model and the SAE). `batches` yields random batches
    from half the buffer, moved to `device`, then refills exactly those slots from the stream, so each streamed row
    is yielded once."""

    def __init__(self, stream: Iterator[torch.Tensor], size: int, d: int, device, seed: int):
        self.stream, self.size, self.device = stream, size, device
        self.buf = torch.empty(size, d, dtype=torch.bfloat16)
        self.carry: torch.Tensor | None = None
        self.gen = torch.Generator().manual_seed(seed)
        self.fill(torch.arange(size))

    def fill(self, slots: torch.Tensor):
        done = 0
        while done < len(slots):
            rows = self.carry if self.carry is not None else next(self.stream).cpu()
            take = min(len(rows), len(slots) - done)
            self.buf[slots[done : done + take]] = rows[:take].to(self.buf.dtype)
            self.carry = rows[take:] if take < len(rows) else None
            done += take

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        half = self.size // 2 // batch_size * batch_size
        while True:
            slots = torch.randperm(self.size, generator=self.gen)[:half]
            for i in range(0, half, batch_size):
                yield self.buf[slots[i : i + batch_size]].to(self.device, non_blocking=True)
            self.fill(slots)


@torch.no_grad()
def moments(x: torch.Tensor, chunk: int = 65536) -> tuple[torch.Tensor, float]:
    """(mean [d], mean squared norm) of rows of x, in float32 chunks (no full-size float copy)."""
    total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    sq = 0.0
    for i in range(0, len(x), chunk):
        c = x[i : i + chunk].float()
        total += c.sum(0).double()
        sq += float(c.pow(2).sum())
    return (total / len(x)).float(), sq / len(x)


# ---------------------------------------------------------------- loss


def sae_loss(sae: TopKSAE, x: torch.Tensor, dead: torch.Tensor, k_aux: int):
    """(fvu, aux loss, active indices) for scaled inputs x. AuxK: the residual x - x_hat is reconstructed from the
    top `k_aux` dead latents (decoder without bias), which gives dead latents a gradient (Gao et al., sec. A.2)."""
    x_hat, vals, idx, pre = sae(x)
    err = x - x_hat
    fvu = err.pow(2).sum(-1).mean() / (x - x.mean(0)).pow(2).sum(-1).mean()
    aux = x.new_zeros(())
    n_dead = int(dead.sum())
    if n_dead > 0:
        a_vals, a_idx = sae.topk(pre.masked_fill(~dead, 0.0), min(k_aux, n_dead))
        e = err.detach()
        aux = (sae.decode(a_vals, a_idx, bias=False) - e).pow(2).sum(-1).mean() / e.pow(2).sum(-1).mean()
        aux = torch.nan_to_num(aux)
    return fvu, aux, idx[vals > 0]


def base_lr(cfg: Config, n_latents: int) -> float:
    return cfg.lr / math.sqrt(n_latents / 16384)


def lr_at(cfg: Config, n_latents: int, step: int, total: int) -> float:
    lr = base_lr(cfg, n_latents) * min(1.0, (step + 1) / cfg.warmup_steps)
    start = int(cfg.decay_start * total)
    if step >= start:
        lr *= max(0.0, (total - step) / max(1, total - start))
    return lr


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def eval_fvu(sae: TopKSAE, acts: torch.Tensor, device, batch: int = 2048) -> dict:
    """fvu and dead fraction over held-out activations (raw, bf16, CPU)."""
    mean = moments(acts)[0].to(device) * sae.input_scale
    sse = tot = 0.0
    fired = torch.zeros(sae.n_latents, dtype=torch.bool, device=device)
    for i in range(0, len(acts), batch):
        x = acts[i : i + batch].to(device).float() * sae.input_scale
        x_hat, vals, idx, _ = sae(x)
        sse += float((x - x_hat).pow(2).sum())
        tot += float((x - mean).pow(2).sum())
        fired[idx[vals > 0]] = True
    return {"fvu": sse / tot, "dead_frac": 1 - float(fired.float().mean())}


@torch.no_grad()
def eval_ce(lm, full_layers, short_layers, module, saes: dict[int, TopKSAE], mean: torch.Tensor,
            docs: list[list[int]], pad_id: int, device, batch: int = 8, logit_chunk: int = 1024) -> dict[int, dict]:
    """Next-token loss of the full model: clean, with each SAE's reconstruction spliced in at `module`'s output, and
    with that output replaced by the mean activation. The BOS position is never patched. The layers after the hook
    live in CPU RAM during training; they move to `device` for this eval only. Returns per SAE: ce_clean, ce_sae,
    ce_ablate, loss_recovered."""
    head = lm.embed_tokens.weight  # tied output head

    def patch(fn):
        def hook(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            patched = torch.cat([x[:, :1], fn(x[:, 1:])], 1)
            return (patched, *out[1:]) if isinstance(out, tuple) else patched

        return hook

    def splice(sae):
        return lambda x: sae.reconstruct(x.reshape(-1, x.shape[-1])).reshape(x.shape)

    patches = {"clean": None, "ablate": lambda x: mean.to(x.dtype).expand_as(x)}
    patches |= {n: splice(sae) for n, sae in saes.items()}
    later = full_layers[len(short_layers) :]
    later.to(device)
    lm.layers = full_layers
    ce = {}
    try:
        for name, fn in patches.items():
            handle = module.register_forward_hook(patch(fn)) if fn else None
            loss, n = 0.0, 0
            for i in range(0, len(docs), batch):
                ids, mask = extract.collate(docs[i : i + batch], pad_id, 8)
                ids, mask = ids.to(device), mask.to(device)
                h = lm(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                valid = mask[:, 1:].bool()
                hs, targets = h[:, :-1][valid], ids[:, 1:][valid]
                for j in range(0, len(targets), logit_chunk):
                    logits = (hs[j : j + logit_chunk] @ head.T).float()
                    loss += float(F.cross_entropy(logits, targets[j : j + logit_chunk], reduction="sum"))
                n += len(targets)
            if handle:
                handle.remove()
            ce[name] = loss / n
    finally:
        lm.layers = short_layers
        later.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    gap = ce["ablate"] - ce["clean"]
    return {n: {"ce_clean": ce["clean"], "ce_sae": ce[n], "ce_ablate": ce["ablate"],
                "loss_recovered": (ce["ablate"] - ce[n]) / gap if gap > 0 else float("nan")} for n in saes}


# ---------------------------------------------------------------- plotting


def plot(rows: list[dict], path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for n in sorted({r["n_latents"] for r in rows}):
        train = [r for r in rows if r["split"] == "train" and r["n_latents"] == n]
        ev = [r for r in rows if r["split"] == "eval" and r["n_latents"] == n]
        line, = axes[0].plot([r["tokens"] for r in train], [r["fvu"] for r in train], alpha=0.5, label=f"{n} train")
        axes[0].plot([r["tokens"] for r in ev], [r["fvu"] for r in ev], "o-", color=line.get_color(),
                     label=f"{n} held-out")
        axes[1].plot([r["tokens"] for r in train], [r["dead_frac"] for r in train], color=line.get_color(),
                     label=f"{n} train (AuxK window)")
        axes[1].plot([r["tokens"] for r in ev], [r["dead_frac"] for r in ev], "o--", color=line.get_color(),
                     label=f"{n} held-out (never fired)")
        axes[2].plot([r["tokens"] for r in ev], [r["loss_recovered"] for r in ev], "o-", color=line.get_color(),
                     label=str(n))
    axes[0].set(title="fraction of variance unexplained", xlabel="tokens", yscale="log")
    axes[1].set(title="dead latents (fraction)", xlabel="tokens")
    axes[2].set(title="loss recovered (held-out)", xlabel="tokens")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if cfg.eval_docs > cfg.holdout_docs or cfg.ce_docs > cfg.eval_docs:
        raise SystemExit("need ce_docs <= eval_docs <= holdout_docs")
    if cfg.web_revision is None:
        cfg.web_revision = hub_sha(cfg.web_dataset)
    if cfg.chat_revision is None:
        cfg.chat_revision = hub_sha(cfg.chat_dataset)
    hook = hook_name(cfg.site, cfg.layer)
    if cfg.dev:
        run = dev_run_dir("lfm_sae")
    else:
        run = runs_root("lfm_sae") / "train" / datetime.now().strftime("%Y%m%d-%H%M%S")
        run.mkdir(parents=True)
    tee_console(run)
    device_name = torch.cuda.get_device_name() if cfg.device.startswith("cuda") else cfg.device
    (run / "config.json").write_text(json.dumps(
        {**asdict(cfg), "hook": hook, "git_commit": git_commit(), "device_name": device_name}, indent=2, default=str))
    sizes = sorted(set(cfg.n_latents))
    print(f"{run}\nhook {hook}: SAEs with {', '.join(f'{n:,}' for n in sizes)} latents, k={cfg.k}, "
          f"{cfg.num_tokens:,} tokens")

    dcfg = data_config(cfg)
    sources = extract.load_sources(dcfg)
    lm, tok = extract.load_language_model(dcfg, truncate=False)
    full_layers = lm.layers
    short_layers = lm.layers[: cfg.layer + 1]
    lm.layers = short_layers
    full_layers[cfg.layer + 1 :].to("cpu")  # only the loss-recovered eval needs them
    module = extract.site_module(full_layers[cfg.layer], cfg.site)
    rec = extract.Recorder(lm, (cfg.layer,), (cfg.site,))

    # held-out documents: the first `holdout_docs` of the stream; evaluate on the first `eval_docs`
    eval_ids: list[list[int]] = []
    for chunk in extract.iter_chunks(dcfg, tok, sources, 0):
        eval_ids += chunk.ids
        if len(eval_ids) >= cfg.eval_docs:
            break
    eval_ids = eval_ids[: cfg.eval_docs]
    ce_ids = eval_ids[: cfg.ce_docs]
    order = sorted(range(len(eval_ids)), key=lambda i: len(eval_ids[i]))
    order = [order[i : i + cfg.lm_batch_size] for i in range(0, len(order), cfg.lm_batch_size)]
    with torch.no_grad():
        eval_acts = torch.cat([r.cpu() for r in lm_rows(lm, rec, hook, eval_ids, order, tok.pad_token_id, cfg.device, 8)])
    print(f"held-out: {len(eval_ids)} docs, {len(eval_acts):,} tokens for fvu; {len(ce_ids)} docs for loss recovered")

    if cfg.compile:
        lm.compile(mode="reduce-overhead", dynamic=False)

    with torch.no_grad():
        t = time.time()
        buffer = ShuffleBuffer(activation_stream(cfg, dcfg, lm, rec, tok, sources), cfg.buffer_tokens, D_MODEL,
                               cfg.device, cfg.seed)
        print(f"filled the {cfg.buffer_tokens:,}-token buffer ({cfg.buffer_tokens * D_MODEL * 2 / 1e9:.1f} GB RAM) "
              f"in {time.time() - t:.0f}s")
        mean_raw, sq_norm = moments(buffer.buf)
        mean_raw = mean_raw.to(cfg.device)
        scale = math.sqrt(D_MODEL / sq_norm)

    import torchinfo

    saes: dict[int, TopKSAE] = {}
    opts: dict[int, torch.optim.Adam] = {}
    since_fired: dict[int, torch.Tensor] = {}
    for n in sizes:
        sae = TopKSAE(D_MODEL, n, cfg.k).to(cfg.device)
        with torch.no_grad():
            sae.input_scale.fill_(scale)
            sae.b_dec.copy_(mean_raw * scale)  # pre-encoder bias starts at the data mean
        torchinfo.summary(sae, input_size=(cfg.batch_size, D_MODEL), device=cfg.device, depth=1, verbose=0)
        saes[n] = sae
        opts[n] = torch.optim.Adam(sae.parameters(), lr=base_lr(cfg, n), betas=(0.9, 0.999), eps=6.25e-10)
        since_fired[n] = torch.zeros(n, dtype=torch.long, device=cfg.device)
        (run / f"sae_{n}").mkdir()
        print(f"SAE {n:,} latents ({n // D_MODEL}x): {sum(p.numel() for p in sae.parameters()):,} params, "
              f"lr {base_lr(cfg, n):.3g}")

    total = cfg.num_tokens // cfg.batch_size
    cols = ["n_latents", "step", "tokens", "split", "lr", "fvu", "aux", "dead_frac", "ce_clean", "ce_sae",
            "ce_ablate", "loss_recovered", "tokens_per_s", *STAT_FIELDS, *GPU_FIELDS]
    # Gradient norms per SAE. every=0: no weight-norm / update-ratio / gradient-direction detail, because that clones
    # the gradients and weights, which is gigabytes for the larger dictionaries.
    stats = {n: ParamStats(sae, every=0) for n, sae in saes.items()}
    gpu = GpuMonitor(cfg.device)
    logger = MetricsLogger(run / "metrics.csv", cols, float_fmt=".6g")
    write, rows = logger.log, logger.rows
    best = {n: float("inf") for n in sizes}

    def evaluate(step: int) -> dict[int, dict]:
        evs = {}
        for sae in saes.values():
            sae.eval()
        ces = eval_ce(lm, full_layers, short_layers, module, saes, mean_raw, ce_ids, tok.pad_token_id, cfg.device)
        for n, sae in saes.items():
            ev = eval_fvu(sae, eval_acts, cfg.device) | ces[n]
            sae.train()
            write({"n_latents": n, "step": step, "tokens": step * cfg.batch_size, "split": "eval", **ev})
            print(f"eval {n:,} @ {step * cfg.batch_size:,} tokens: fvu {ev['fvu']:.4f}, dead {ev['dead_frac']:.1%}, "
                  f"ce clean {ev['ce_clean']:.3f} / sae {ev['ce_sae']:.3f} / mean-ablated {ev['ce_ablate']:.3f}, "
                  f"loss recovered {ev['loss_recovered']:.1%}", flush=True)
            ckpt = {"sae": sae.state_dict(), "config": asdict(cfg), "n_latents": n, "hook": hook, "step": step,
                    "tokens": step * cfg.batch_size, "eval": ev}
            torch.save(ckpt, run / f"sae_{n}" / "last.pt")
            if ev["fvu"] < best[n]:
                best[n] = ev["fvu"]
                torch.save(ckpt, run / f"sae_{n}" / "best.pt")
            evs[n] = ev
        return evs

    t0 = t_log = time.time()
    acc = {n: {"fvu": 0.0, "aux": 0.0} for n in sizes}
    batches = buffer.batches(cfg.batch_size)
    for step in range(total):
        with torch.no_grad():
            x = next(batches).float() * scale
        for n, sae in saes.items():
            dead = since_fired[n] > cfg.dead_tokens
            fvu, aux, active = sae_loss(sae, x, dead, cfg.k_aux)
            loss = fvu + cfg.aux_coef * aux
            opts[n].zero_grad(set_to_none=True)
            loss.backward()
            sae.remove_parallel_grads()
            for g in opts[n].param_groups:
                g["lr"] = lr_at(cfg, n, step, total)
            stats[n].grads(step + 1)  # the gradients the optimizer is about to apply
            opts[n].step()
            stats[n].stepped(step + 1)
            sae.normalize_decoder()
            with torch.no_grad():
                since_fired[n] += cfg.batch_size
                since_fired[n][active] = 0
            acc[n]["fvu"] += float(fvu.detach())
            acc[n]["aux"] += float(aux.detach())
            del fvu, aux, active, loss

        if (step + 1) % cfg.log_every == 0:
            dt = time.time() - t_log
            k = cfg.log_every
            show = (step + 1) % (k * 10) == 0
            gpu_stats = gpu.read()  # one reading for the window, shared by every SAE's row
            for n in sizes:
                dead_frac = float((since_fired[n] > cfg.dead_tokens).float().mean())
                write({"n_latents": n, "step": step + 1, "tokens": (step + 1) * cfg.batch_size, "split": "train",
                       "lr": opts[n].param_groups[0]["lr"], "fvu": acc[n]["fvu"] / k, "aux": acc[n]["aux"] / k,
                       "dead_frac": dead_frac, "tokens_per_s": k * cfg.batch_size / dt, **stats[n].read()[0], **gpu_stats})
                if show:
                    print(f"  {n:,}: fvu {acc[n]['fvu'] / k:.4f}, aux {acc[n]['aux'] / k:.3f}, dead {dead_frac:.1%}")
            if show:
                eta = (total - step - 1) * (time.time() - t0) / (step + 1) / 60
                mem = f", GPU {gpu_stats.get('gpu_util', 0):.0f}% peak {gpu_stats['gpu_alloc_gb']:.1f} GiB" if gpu_stats else ""
                print(f"step {step + 1:,}/{total:,}: {k * cfg.batch_size / dt:,.0f} tok/s, ETA {eta:.0f} min{mem}",
                      flush=True)
            acc = {n: {"fvu": 0.0, "aux": 0.0} for n in sizes}
            t_log = time.time()
        if (step + 1) % cfg.eval_every == 0 and step + 1 < total:
            evaluate(step + 1)

    final = evaluate(total)
    logger.close()
    results = {"hook": hook, "tokens": total * cfg.batch_size, "minutes": round((time.time() - t0) / 60, 1),
               "input_scale": scale, "saes": {str(n): ev for n, ev in final.items()}}
    if cfg.device.startswith("cuda"):
        gpu.read()  # fold the last window into the run-wide peak (each read resets torch's peak counter)
        results["peak_gpu_gb"] = round(gpu.summary()["peak_gpu_alloc_gb"] * 2**30 / 1e9, 2)
    (run / "results.json").write_text(json.dumps(results, indent=2))
    plot(rows, run / "curves.png")
    print(json.dumps(results, indent=2))
    print(f"wrote {run}")

if __name__ == "__main__":
    main(tyro.cli(Config))
