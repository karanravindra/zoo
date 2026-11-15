"""Extract LFM3-VL-2B language-model activations from a text corpus, for sparse-autoencoder training.

Runs a seeded mix of web text (FineWeb-Edu) and chat transcripts (SmolTalk, rendered with the model's own chat
template) through through the language model of LiquidAI/LFM3-VL-2B, and records the activations at the chosen hook
points (`--sites` x `--layers`). Only the language model is kept, cut after the deepest hooked layer, so the vision
tower, the later layers and the 128k-vocab lm_head never run.

Hook points (layer i, all [tokens, 2048]):
  resid_post.i  output of decoder layer i (the residual stream after it)
  mixer_out.i   what the conv / attention block adds to the residual (after its post-norm; LFM3 uses sandwich norm)
  mlp_out.i     what the feed-forward block adds to the residual (after its post-norm)

Every document starts with <|startoftext|> (the chat template adds it itself) and is truncated to `--context` tokens.
Padding and (by default) the BOS position are not stored: BOS carries a huge-norm activation that would dominate SAE
training. Documents are processed in chunks of `--chunk-docs`, sorted by length within a chunk to cut padding.

The datasets are downloaded once (`--web-files`, `--chat-files`; about 6 GB into HF_HOME) and memory-mapped, not
streamed: streaming shuffles parquet row groups across many remote files and its read buffers grew to ~30 GB of RAM.
Each source gets a full seeded permutation, and a seeded coin picks the source of each document, so document k is a
pure function of the config and `--resume` jumps straight to it. A DataLoader worker reads and tokenizes, and a
writer thread saves shards, so the GPU keeps running while either is busy.

The run dir lives under `--data-root` (default /mnt/ai/lfm_sae/extract/<timestamp>) and is symlinked from
runs/lfm_sae/extract/<timestamp>. It holds:
  config.json           config, git commit, GPU name, model and dataset revisions
  shards/NNNNN.safetensors
                        `<site>.<layer>` bf16 [n, d] per hook point, plus row-aligned `token_ids` int32, `doc_ids`
                        int64, `positions` int32 and `sources` uint8 (0 = web, 1 = chat) [n]: enough to rebuild each
                        token's context for interpretation. Shards end on chunk boundaries, so no document is split.
  state.json            documents consumed, tokens and shards written: what `--resume` continues from
  stats.safetensors     per hook point: running sums, then (when finished) `<hook>.mean` [d] and
                        `<hook>.sq_norm_mean` [] in float32, for SAE input scaling
  metrics.csv           per shard: tokens, docs, seconds, seconds blocked on data / on the disk writer, tokens/s
  results.json          totals and per-hook norm summary (when finished)

    uv run projects/lfm_sae/extract.py                                  # 5M tokens of resid_post.12 (~20 GB)
    uv run projects/lfm_sae/extract.py --resume runs/lfm_sae/extract/<timestamp>
"""

from __future__ import annotations

import csv
import itertools
import json
import queue
import random
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, IterableDataset

from common import MODEL, MODEL_REVISION, RUNS_ROOT, git_commit, hook_name, hub_sha

Site = Literal["resid_post", "mixer_out", "mlp_out"]
SOURCES = ("web", "chat")
D_MODEL = 2048


@dataclass
class Config:
    model: str = MODEL  # Hub repo of the VLM
    model_revision: str = MODEL_REVISION  # pinned commit (the model ships remote code)
    layers: tuple[int, ...] = (12,)  # decoder layers to hook (0-23)
    sites: tuple[Site, ...] = ("resid_post",)  # hook points per layer
    num_tokens: int = 5_000_000  # stop after this many stored tokens (5M stays inside train_sae.py's held-out documents)
    web_dataset: str = "HuggingFaceFW/fineweb-edu"  # Hub dataset of web documents (`text` column)
    web_files: tuple[str, ...] = ("sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet")  # its files to download (~690k docs each)
    web_revision: str | None = None  # its commit (default: the current one, resolved and recorded)
    chat_dataset: str = "HuggingFaceTB/smoltalk"  # Hub dataset of conversations (`messages` column)
    chat_files: tuple[str, ...] = ("data/all/train-*.parquet",)  # its files to download (1.0M conversations)
    chat_revision: str | None = None  # its commit (default: the current one, resolved and recorded)
    chat_fraction: float = 0.1  # probability that a document is a chat transcript (0 = web only)
    context: int = 1024  # max tokens per document (BOS included); longer documents are truncated
    batch_size: int = 32  # documents per forward pass
    chunk_docs: int = 512  # documents per chunk: sorted by length within a chunk; the unit of resuming
    shard_tokens: int = 2**18  # write a shard once at least this many tokens are buffered (at a chunk boundary)
    skip_bos: bool = True  # don't store the BOS position
    seed: int = 0  # seed for the dataset shuffles and the web/chat mix
    data_root: Path = Path("/mnt/ai/lfm_sae")  # where run dirs physically live (symlinked from runs/lfm_sae/)
    resume: Path | None = None  # run dir to continue with its saved config; only --device and --compile apply
    device: str = "cuda"  # torch device
    compile: bool = False  # torch.compile the language model (pads batches to a multiple of 128 to limit recompiles)


def config_from_json(d: dict) -> Config:
    kw = {}
    for f in fields(Config):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.name in ("layers", "sites", "web_files", "chat_files"):
            v = tuple(v)
        elif f.name in ("data_root", "resume") and v is not None:
            v = Path(v)
        kw[f.name] = v
    return Config(**kw)


# ---------------------------------------------------------------- model


def load_language_model(cfg: Config, truncate: bool = True):
    """(language model, tokenizer). The LFM3 text decoder of the VLM, in bf16 and eval mode. With `truncate`, cut
    after max(cfg.layers) so later layers never run. Its final norm is `embedding_norm` and its output head is tied
    to `embed_tokens`."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    vlm = AutoModelForImageTextToText.from_pretrained(
        cfg.model, revision=cfg.model_revision, dtype=torch.bfloat16, trust_remote_code=True
    )
    lm = vlm.model.language_model
    n_layers = len(lm.layers)
    if not all(0 <= i < n_layers for i in cfg.layers):
        raise SystemExit(f"--layers must be in 0..{n_layers - 1}, got {cfg.layers}")
    # The decoder loops over self.layers[:num_hidden_layers], so slicing the ModuleList ends the forward early.
    if truncate:
        lm.layers = lm.layers[: max(cfg.layers) + 1]
    del vlm
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision, trust_remote_code=True)
    return lm.to(cfg.device).eval().requires_grad_(False), tok


def site_module(layer: torch.nn.Module, site: Site) -> torch.nn.Module:
    if site == "resid_post":
        return layer
    if not getattr(layer, "use_sandwich_norm", False):
        raise SystemExit(f"{site} hooks the post-norm of the block, but this model has no sandwich norm")
    return layer.post_operator_norm if site == "mixer_out" else layer.post_ffn_norm


class Recorder:
    """Forward hooks that keep the latest output of each hook point, keyed by hook name."""

    def __init__(self, lm, layers: tuple[int, ...], sites: tuple[Site, ...]):
        self.out: dict[str, torch.Tensor] = {}
        self.names = [hook_name(s, i) for i in sorted(set(layers)) for s in sites]
        self.handles = []
        for i in sorted(set(layers)):
            for s in sites:
                self.handles.append(site_module(lm.layers[i], s).register_forward_hook(self._hook(hook_name(s, i))))

    def _hook(self, name: str):
        def fn(module, args, output):
            self.out[name] = output[0] if isinstance(output, tuple) else output

        return fn


# ---------------------------------------------------------------- data


def load_source(repo: str, files: tuple[str, ...], revision: str | None, column: str):
    """One source as a memory-mapped Arrow dataset with only `column` (downloaded and converted on first use)."""
    from datasets import load_dataset

    return load_dataset(repo, data_files=list(files), split="train", revision=revision).select_columns([column])


@dataclass
class Sources:
    """The datasets of each source (web, chat) and a seeded permutation of each."""

    data: list
    perms: list[torch.Tensor]


def load_sources(cfg: Config) -> Sources:
    data = [load_source(cfg.web_dataset, cfg.web_files, cfg.web_revision, "text")]
    if cfg.chat_fraction > 0:
        data.append(load_source(cfg.chat_dataset, cfg.chat_files, cfg.chat_revision, "messages"))
    perms = [torch.randperm(len(d), generator=torch.Generator().manual_seed(cfg.seed + i)) for i, d in enumerate(data)]
    print("sources: " + ", ".join(f"{SOURCES[i]} {len(d):,} docs" for i, d in enumerate(data)))
    return Sources(data, perms)


def doc_plan(cfg: Config) -> Iterator[tuple[int, int]]:
    """(source, position in that source's permutation) for documents 0, 1, 2, ...: a seeded coin per document picks
    the source. Pure function of the config, so resuming at document k replays k coin flips."""
    rng = random.Random(cfg.seed)
    taken = [0, 0]
    while True:
        src = 1 if cfg.chat_fraction > 0 and rng.random() < cfg.chat_fraction else 0
        yield src, taken[src]
        taken[src] += 1


def fetch_texts(tok, sources: Sources, plan: list[tuple[int, int]]) -> list[str] | None:
    """Texts for a list of (source, position); chat rows go through the chat template, so they start with BOS and show
    <think></think> before final-turn answers, as the model sees them in use. None if a source ran out."""
    texts: list[str | None] = [None] * len(plan)
    for src in (0, 1):
        slots = [i for i, (s, _) in enumerate(plan) if s == src]
        if not slots:
            continue
        pos = [plan[i][1] for i in slots]
        if max(pos) >= len(sources.perms[src]):
            return None
        rows = sources.data[src][sources.perms[src][pos].tolist()]
        col = rows["text"] if src == 0 else [tok.apply_chat_template(m, tokenize=False) for m in rows["messages"]]
        for i, t in zip(slots, col):
            texts[i] = t
    return texts


def tokenize(tok, texts: list[str], bos: int, context: int) -> list[list[int]]:
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    return [(x if x[:1] == [bos] else [bos, *x])[:context] for x in ids]


@dataclass
class Chunk:
    """`chunk_docs` consecutive documents: doc_ids start at `first_doc`; batches are length-sorted index lists."""

    first_doc: int
    sources: list[int]
    ids: list[list[int]]
    batches: list[list[int]]


def iter_chunks(cfg: Config, tok, sources: Sources, skip_docs: int) -> Iterator[Chunk]:
    plan = doc_plan(cfg)
    for _ in itertools.islice(plan, skip_docs):  # resume: replay the coin flips of the stored documents
        pass
    first = skip_docs
    while True:
        part = list(itertools.islice(plan, cfg.chunk_docs))
        texts = fetch_texts(tok, sources, part)
        if texts is None:
            print("a source ran out of documents: add --web-files / --chat-files")
            return
        ids = tokenize(tok, texts, tok.bos_token_id, cfg.context)
        part_sources = [s for s, _ in part]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        batches = [order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        yield Chunk(first, part_sources, ids, batches)
        first += len(part)


class ChunkStream(IterableDataset):
    """`iter_chunks` in a DataLoader worker process: reading and tokenizing overlap the GPU work without holding the
    main process's GIL, and the worker shuts down cleanly when the main process stops."""

    def __init__(self, cfg: Config, tok, sources: Sources, skip_docs: int):
        self.cfg, self.tok, self.sources, self.skip_docs = cfg, tok, sources, skip_docs

    def __iter__(self):
        return iter_chunks(self.cfg, self.tok, self.sources, self.skip_docs)


def _identity(x):
    return x


def chunk_loader(cfg: Config, tok, sources: Sources, skip_docs: int) -> DataLoader:
    return DataLoader(ChunkStream(cfg, tok, sources, skip_docs), batch_size=None, num_workers=1, prefetch_factor=4,
                      collate_fn=_identity)


def collate(batch: list[list[int]], pad_id: int, multiple: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded (input_ids, attention_mask). Right padding never changes the real tokens' activations in a causal
    model (attention and the causal convolution only look back)."""
    length = max(map(len, batch))
    length = -(-length // multiple) * multiple
    ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), length), dtype=torch.long)
    for r, x in enumerate(batch):
        ids[r, : len(x)] = torch.tensor(x)
        mask[r, : len(x)] = 1
    return ids, mask


# ---------------------------------------------------------------- shards and stats

ROW_KEYS = ("token_ids", "doc_ids", "positions", "sources")


class RunningStats:
    """Per-hook sums (float64, on device) for the mean vector and mean squared norm. Saved with each shard."""

    def __init__(self, names: list[str], device):
        self.names, self.n = names, 0
        self.sum = {k: torch.zeros(D_MODEL, dtype=torch.float64, device=device) for k in names}
        self.sq = {k: torch.zeros((), dtype=torch.float64, device=device) for k in names}
        self.max_norm = {k: torch.zeros((), dtype=torch.float64, device=device) for k in names}

    def update(self, acts: dict[str, torch.Tensor]):
        """Add one batch of rows (on device). Per-batch sums are float32 (at most a few 10k rows), totals float64."""
        n = len(acts[self.names[0]])
        if n == 0:
            return
        for k in self.names:
            x = acts[k].float()
            self.sum[k] += x.sum(0).double()
            sq = x.pow(2).sum(-1)
            self.sq[k] += sq.sum().double()
            self.max_norm[k] = torch.maximum(self.max_norm[k], sq.max().sqrt().double())
        self.n += n

    def merge(self, other: RunningStats):
        for k in self.names:
            self.sum[k] += other.sum[k]
            self.sq[k] += other.sq[k]
            self.max_norm[k] = torch.maximum(self.max_norm[k], other.max_norm[k])
        self.n += other.n

    def reset(self):
        self.n = 0
        for k in self.names:
            self.sum[k].zero_()
            self.sq[k].zero_()
            self.max_norm[k].zero_()

    def state(self, final: bool = False) -> dict[str, torch.Tensor]:
        out = {"n": torch.tensor(self.n, dtype=torch.float64)}
        for k in self.names:
            out |= {f"{k}.sum": self.sum[k].cpu(), f"{k}.sq_sum": self.sq[k].cpu(), f"{k}.max_norm": self.max_norm[k].cpu()}
            if final:
                out[f"{k}.mean"] = (self.sum[k] / self.n).float().cpu()
                out[f"{k}.sq_norm_mean"] = (self.sq[k] / self.n).float().cpu()
        return out

    def load(self, state: dict[str, torch.Tensor]):
        self.n = int(state["n"])
        for k in self.names:
            self.sum[k].copy_(state[f"{k}.sum"])
            self.sq[k].copy_(state[f"{k}.sq_sum"])
            self.max_norm[k].copy_(state[f"{k}.max_norm"])

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            k: {
                "rms_norm": float((self.sq[k] / self.n).sqrt()),
                "max_norm": float(self.max_norm[k]),
                "mean_vector_norm": float((self.sum[k] / self.n).norm()),
            }
            for k in self.names
        }


class ShardWriter:
    """Buffers rows on the CPU (and their stats in `pending`, on the device). `commit` (called at chunk boundaries)
    hands a shard to a writer thread once enough rows are buffered. The thread saves the shard, then the matching
    stats and state.json, in order, so the shards on disk, the stats and the resume point always agree. At most one
    shard waits in the queue: if the disk is slower than the GPU, `commit` blocks (reported as write-wait time)."""

    def __init__(self, run: Path, names: list[str], shard_tokens: int, stats: RunningStats):
        self.run, self.dir = run, run / "shards"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.names, self.shard_tokens, self.stats = names, shard_tokens, stats
        self.pending = RunningStats(names, stats.sum[names[0]].device)
        self.buf: dict[str, list[torch.Tensor]] = {k: [] for k in [*names, *ROW_KEYS]}
        self.n_buf = 0
        self.state = {"docs": 0, "tokens": 0, "shards": 0, "finished": False}
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        while (job := self.queue.get()) is not None:
            if self.error is not None:
                continue
            try:
                buf, name, stats_state, state = job
                tensors = {k: torch.cat(v) for k, v in buf.items()}
                save_file(tensors, self.dir / (name + ".tmp"), metadata={"hooks": json.dumps(self.names)})
                (self.dir / (name + ".tmp")).rename(self.dir / name)
                self._save(stats_state, state)
            except BaseException as e:  # noqa: BLE001 - re-raised in the main thread
                self.error = e

    def _check(self):
        if self.error is not None:
            raise RuntimeError("shard writer failed") from self.error

    def close(self):
        """Finish the queued shards and stop the thread."""
        self.queue.put(None)
        self.thread.join()
        self._check()

    def restore(self):
        """Continue from state.json: drop shards written after it (a crash between shard and state)."""
        path = self.run / "state.json"
        if path.exists():
            self.state = json.loads(path.read_text())
            self.stats.load(load_file(self.run / "stats.safetensors"))
        for p in self.dir.iterdir():
            if p.suffix != ".safetensors" or int(p.stem) >= self.state["shards"]:
                print(f"removing {p.name} (not in state.json)")
                p.unlink()

    def add(self, rows: dict[str, torch.Tensor]):
        """Add one batch of rows (on device): stats are taken there, the rows are moved to the CPU."""
        self.pending.update(rows)
        for k, v in rows.items():
            self.buf[k].append(v.cpu())
        self.n_buf += len(rows["token_ids"])

    def commit(self, docs_done: int, force: bool = False) -> int:
        """Queue the buffer as a shard if it is big enough (or `force`); returns rows queued."""
        self._check()
        if self.n_buf == 0 or (self.n_buf < self.shard_tokens and not force):
            return 0
        self.stats.merge(self.pending)
        self.pending.reset()
        n = self.n_buf
        name = f"{self.state['shards']:05d}.safetensors"
        self.state.update(docs=docs_done, tokens=self.state["tokens"] + n, shards=self.state["shards"] + 1)
        self.queue.put((self.buf, name, self.stats.state(), dict(self.state)))
        self.buf = {k: [] for k in self.buf}
        self.n_buf = 0
        return n

    def save_state(self, final: bool = False):
        self._save(self.stats.state(final), self.state)

    def _save(self, stats_state: dict[str, torch.Tensor], state: dict):
        save_file(stats_state, self.run / "stats.safetensors.tmp")
        (self.run / "stats.safetensors.tmp").rename(self.run / "stats.safetensors")
        (self.run / "state.json.tmp").write_text(json.dumps(state, indent=2))
        (self.run / "state.json.tmp").rename(self.run / "state.json")


# ---------------------------------------------------------------- main


def setup_run(cfg: Config) -> tuple[Config, Path]:
    """(config, run dir). A new run dir under data_root, symlinked from runs/lfm_sae/extract/, or the resumed one."""
    if cfg.resume is not None:
        run = cfg.resume.resolve()
        saved = config_from_json(json.loads((run / "config.json").read_text()))
        # runtime settings, not part of what the data depends on: taken from this command line
        saved.device, saved.compile = cfg.device, cfg.compile
        print(f"resuming {run} with its saved config (device {cfg.device}, compile {cfg.compile})")
        return saved, run
    for repo, attr in ((cfg.web_dataset, "web_revision"), (cfg.chat_dataset, "chat_revision")):
        if getattr(cfg, attr) is None:
            setattr(cfg, attr, hub_sha(repo))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = cfg.data_root / "extract" / stamp
    run.mkdir(parents=True)
    link = RUNS_ROOT / "extract" / stamp
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(run, target_is_directory=True)
    names = [hook_name(s, i) for i in sorted(set(cfg.layers)) for s in cfg.sites]
    device = torch.cuda.get_device_name() if cfg.device.startswith("cuda") else cfg.device
    meta = {"hooks": names, "git_commit": git_commit(), "device_name": device, "run_dir": str(run), "link": str(link)}
    (run / "config.json").write_text(json.dumps({**asdict(cfg), **meta}, indent=2, default=str))
    return cfg, run


def main(cfg: Config):
    cfg, run = setup_run(cfg)
    names = [hook_name(s, i) for i in sorted(set(cfg.layers)) for s in cfg.sites]
    stats = RunningStats(names, cfg.device)
    writer = ShardWriter(run, names, cfg.shard_tokens, stats)
    if cfg.resume is not None or (run / "state.json").exists():
        writer.restore()
    if writer.state["finished"]:
        raise SystemExit(f"{run} is already finished")

    stored = writer.state["tokens"]
    need = (cfg.num_tokens - stored) * len(names) * D_MODEL * 2
    free = shutil.disk_usage(run).free
    print(f"hooks: {names}\n{run}\nstoring {cfg.num_tokens - stored:,} more tokens x {len(names)} hooks "
          f"= {need / 1e9:.1f} GB ({free / 1e9:.1f} GB free)")
    if need > 0.95 * free:
        raise SystemExit("not enough disk space: lower --num-tokens or hook fewer points")

    sources = load_sources(cfg)  # downloads on first use, before the model loads
    lm, tok = load_language_model(cfg)
    rec = Recorder(lm, cfg.layers, cfg.sites)
    assert lm.config.hidden_size == D_MODEL, lm.config.hidden_size
    if cfg.compile:
        lm.compile(mode="reduce-overhead", dynamic=False)

    new_metrics = not (run / "metrics.csv").exists()
    metrics = open(run / "metrics.csv", "a", newline="")
    log = csv.writer(metrics)
    if new_metrics:
        log.writerow(["shard", "tokens", "docs", "seconds", "data_wait_s", "write_wait_s", "tokens_per_s"])

    t0 = t_shard = time.time()
    tokens_at_start, docs_at_shard = stored, writer.state["docs"]
    docs_done = writer.state["docs"]
    pending = 0  # rows buffered since the last shard
    wait = wwait = 0.0  # seconds blocked on the data worker / the shard writer since the last shard
    chunks = iter(chunk_loader(cfg, tok, sources, writer.state["docs"]))
    try:
        with torch.inference_mode():
            while True:
                t_wait = time.time()
                chunk = next(chunks, None)
                wait += time.time() - t_wait
                if chunk is None:  # a source ran out
                    stored += writer.commit(docs_done, force=True)
                    break
                for idx in chunk.batches:
                    ids, mask = collate([chunk.ids[i] for i in idx], tok.pad_token_id, 128 if cfg.compile else 8)
                    ids, mask = ids.to(cfg.device, non_blocking=True), mask.to(cfg.device, non_blocking=True)
                    lm(input_ids=ids, attention_mask=mask, use_cache=False)

                    keep = mask.bool()
                    if cfg.skip_bos:
                        keep[:, 0] = False
                    budget = cfg.num_tokens - stored - pending
                    per_doc = lambda v: torch.tensor(v, device=ids.device)[:, None].expand_as(ids)  # noqa: E731
                    rows = {k: rec.out[k][keep][:budget] for k in names}
                    rows["token_ids"] = ids[keep][:budget].to(torch.int32)
                    rows["positions"] = torch.arange(ids.shape[1], device=ids.device, dtype=torch.int32).expand_as(ids)[keep][:budget]
                    rows["doc_ids"] = per_doc([chunk.first_doc + i for i in idx])[keep][:budget]
                    rows["sources"] = per_doc([chunk.sources[i] for i in idx]).to(torch.uint8)[keep][:budget]
                    writer.add(rows)
                    pending += len(rows["token_ids"])
                    if stored + pending >= cfg.num_tokens:
                        break
                docs_done = chunk.first_doc + len(chunk.ids)
                last = stored + pending >= cfg.num_tokens
                t_write = time.time()
                n = writer.commit(docs_done, force=last)
                wwait += time.time() - t_write
                if n:
                    stored += n
                    pending = 0
                    dt = time.time() - t_shard
                    shard = writer.state["shards"] - 1
                    log.writerow([shard, n, docs_done - docs_at_shard, f"{dt:.1f}", f"{wait:.1f}", f"{wwait:.1f}", f"{n / dt:.0f}"])
                    metrics.flush()
                    rate = (stored - tokens_at_start) / (time.time() - t0)
                    eta = (cfg.num_tokens - stored) / rate / 60
                    print(f"shard {shard:05d}: {n:,} tokens, {stored:,}/{cfg.num_tokens:,} total, {n / dt:,.0f} tok/s "
                          f"(of {dt:.0f}s: {wait:.0f}s waiting on data, {wwait:.0f}s on disk), ETA {eta:.0f} min", flush=True)
                    t_shard, docs_at_shard, wait, wwait = time.time(), docs_done, 0.0, 0.0
                if last:
                    break
    finally:
        del chunks  # stop the data worker
        writer.close()  # finish the queued shards (their state.json is the resume point)
        metrics.close()

    writer.state["finished"] = True
    writer.save_state(final=True)
    elapsed = time.time() - t0
    results = {
        "tokens": writer.state["tokens"],
        "docs": writer.state["docs"],
        "shards": writer.state["shards"],
        "seconds_this_session": round(elapsed, 1),
        "tokens_per_s_this_session": round((stored - tokens_at_start) / elapsed),
        "bytes": sum(p.stat().st_size for p in (run / "shards").iterdir()),
        "hooks": stats.summary(),
    }
    if cfg.device.startswith("cuda"):
        results["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (run / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
