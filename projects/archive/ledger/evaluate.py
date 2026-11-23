"""Evaluate extraction on held-out test data, with Cloudflare's Clairic or our own (fine-tuned) model.

  uv run projects/ledger/evaluate.py --backend clairic          # needs CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_AUTH_TOKEN
  uv run projects/ledger/evaluate.py --backend local            # base model (common.MODEL)
  uv run projects/ledger/evaluate.py --backend local --adapter runs/ledger/train/<run>/adapter

Both backends get the same seeded examples (the ids are saved), the same inputs (OCR text and the same JPEG bytes
for images) and the same scoring, so their results.json files compare directly.

Each run writes to `<repo>/runs/ledger/eval/<timestamp>/`:
  config.json        config, git commit, package versions, dataset revisions, protocol, tesseract version
  sample_ids.json    the example ids per dataset
  predictions.jsonl  one row per example: gold, prediction, per-field scores, errors, latency, usage, raw output
  results.json       per dataset: mean per-field score, exact-example rate, null behaviour, failures, latency, usage
  scores.png         mean per-field score per dataset

Clairic responses are cached in `<repo>/data/cache/ledger/clairic/`, keyed by the payload hash, dataset revision
and protocol version. Failed requests count as 0 and stay in the denominator.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from tqdm import tqdm

from common import CACHE_ROOT, CONFIGS, build_messages, PROCEDURAL, SYNTHETIC, convert_row, hub_sha, new_ledger_run, parse_json_object, sample_indices
from scoring import anls, gold_number, is_null, relaxed_accuracy, score_extraction

PROTOCOL = "ledger-eval-v1"  # bump when inputs (prompts, OCR, image encoding) change, to invalidate the cache
DOCVQA = "lmms-lab-encoder/DocVQA"
CHARTQA = "HuggingFaceM4/ChartQA"
EVAL_SETS = ("synthetic", *(f"procedural/{c}" for c in CONFIGS), "docvqa", "chartqa_numeric")
MAX_IMAGE_BYTES = 4 * 1024 * 1024
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


@dataclass
class EvalConfig:
    backend: Literal["clairic", "openai", "local"] = "clairic"  # Clairic API; an OpenAI-compatible server (vLLM); or in-process HF generate
    adapter: str | None = None  # local: LoRA adapter dir from train.py (None = base model)
    datasets: tuple[str, ...] = EVAL_SETS  # which eval sets to run
    count: int = 100  # examples per dataset/config
    seed: int = 42  # sampling seed (fixes the example ids)
    quantize: bool | None = None  # local: 4-bit NF4 base weights (default: as the adapter was trained; bf16 without one)
    max_new_tokens: int = 512  # local: generation cap
    batch_size: int = 16  # local: requests generated together (left-padded, sorted by prompt length)
    timeout_s: float = 120.0  # clairic: request timeout
    max_retries: int = 6  # clairic: retries for 429 / 5xx / timeouts
    use_cache: bool = True  # clairic: reuse cached responses
    base_url: str = "http://localhost:8000/v1"  # openai: server URL (the served model is read from /models)
    concurrency: int = 256  # openai: requests in flight (the server batches them continuously)
    compile: bool = False  # local: model.compile(mode="reduce-overhead", dynamic=False)


# ---------------------------------------------------------------- inputs


def encode_jpeg(image, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Re-encode as a real JPEG under `max_bytes`: lower quality to 60, then shrink by 0.8x steps."""
    from PIL import Image

    img, quality = image.convert("RGB"), 90
    while True:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
        if quality > 60:
            quality -= 10
        else:
            img = img.resize((int(img.width * 0.8), int(img.height * 0.8)), Image.Resampling.LANCZOS)


def ocr(image_bytes: bytes) -> str:
    """Tesseract text of the image alone (never the annotations), cached on disk by image hash."""
    import pytesseract

    path = CACHE_ROOT / "ocr" / f"{hashlib.sha256(image_bytes).hexdigest()}.txt"
    if path.exists():
        return path.read_text()
    text = pytesseract.image_to_string(_decode(image_bytes))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def load_examples(name: str, count: int, seed: int) -> tuple[list[dict], str, str]:
    """Seeded examples for one eval set, as dicts {id, state, fields, gold, image_bytes, level, kind}.
    Also returns the repo and the pinned revision."""
    from datasets import load_dataset

    if name == "synthetic" or name.startswith("procedural/"):
        repo, config = (SYNTHETIC, None) if name == "synthetic" else (PROCEDURAL, name.split("/", 1)[1])
        rev = hub_sha(repo)
        ds = load_dataset(repo, name=config, split="test", revision=rev)
        out = []
        for row in ds.select(sample_indices(len(ds), count, seed)):
            ex = convert_row(row)  # same conversion as training
            out.append({"id": row["id"], "state": ex["state"], "fields": ex["fields"], "gold": ex["target"],
                        "image_bytes": None, "level": row.get("level"), "kind": "extraction"})
        return out, repo, rev

    if name == "docvqa":
        if shutil.which("tesseract") is None:
            raise SystemExit("docvqa needs the tesseract binary (e.g. `sudo apt install tesseract-ocr`), or drop it from --datasets.")
        rev = hub_sha(DOCVQA)
        ds = load_dataset(DOCVQA, name="DocVQA", split="validation", revision=rev)
        rows = list(ds.select(sample_indices(len(ds), count, seed)))
        images = [encode_jpeg(row["image"]) for row in rows]
        # OCR the same JPEG pixels the model sees; tesseract runs as a subprocess, so threads parallelize it.
        # One OpenMP thread per tesseract: otherwise cpu_count processes x cpu_count threads oversubscribe the CPU
        # (measured: 15-65 s per page instead of ~1.4 s).
        os.environ["OMP_THREAD_LIMIT"] = "1"
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            texts = list(tqdm(pool.map(ocr, images), total=len(images), desc="docvqa OCR"))
        out = []
        for row, image_bytes, text in zip(rows, images, texts):
            out.append({
                "id": f"docvqa:{row['questionId']}",
                "state": text,  # OCR supplies candidate text; the image supplies visual context
                "fields": {"answer": {"type": "text", "instructions": row["question"]}},
                "gold": list(row["answers"]),  # alternative acceptable answers
                "image_bytes": image_bytes, "level": None, "kind": "docvqa",
            })
        return out, DOCVQA, rev

    if name == "chartqa_numeric":
        rev = hub_sha(CHARTQA)
        ds = load_dataset(CHARTQA, split="test", revision=rev)
        # Gold labels only define the declared numeric subset; they never reach the model's input.
        numeric = [i for i, labels in enumerate(ds["label"]) if gold_number(labels) is not None]
        chosen = [numeric[j] for j in sample_indices(len(numeric), count, seed)]
        out = []
        for i, row in zip(chosen, ds.select(chosen)):
            out.append({
                "id": f"chartqa:test:{i}",
                "state": "Answer the question using the chart.",
                "fields": {"answer": {"type": "number", "instructions": row["query"]}},
                "gold": gold_number(row["label"]),
                "image_bytes": encode_jpeg(row["image"]), "level": None, "kind": "chartqa",
            })
        return out, CHARTQA, rev

    raise ValueError(f"unknown eval set {name!r}; choose from {EVAL_SETS}")


def _decode(image_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ---------------------------------------------------------------- backends


class Clairic:
    def __init__(self, cfg: EvalConfig):
        try:
            account, self.token = os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_AUTH_TOKEN"]
        except KeyError as e:
            raise SystemExit(f"--backend clairic needs {e.args[0]} in the environment") from None
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/clairic"
        self.cfg = cfg
        self.cache_dir = CACHE_ROOT / "clairic"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def payload(ex: dict) -> dict:
        p = {"model": "clairic", "state": ex["state"], "fields": ex["fields"], "options": {"return_state_text": True}}
        if ex["image_bytes"] is not None:
            p["images"] = [{"content_type": "image/jpeg", "base64": base64.b64encode(ex["image_bytes"]).decode("ascii")}]
        return p

    def _post(self, payload: dict) -> tuple[dict, int]:
        """POST with exponential backoff on transient failures only. Returns (result, attempts)."""
        import requests

        for attempt in range(self.cfg.max_retries + 1):
            wait = min(60.0, 2.0**attempt) * (0.5 + random.random())
            try:
                resp = requests.post(self.url, headers={"Authorization": f"Bearer {self.token}"}, json=payload, timeout=self.cfg.timeout_s)
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == self.cfg.max_retries:
                    raise RuntimeError(f"{type(e).__name__} after {attempt + 1} attempts: {e}") from e
            else:
                if resp.status_code in TRANSIENT_STATUS and attempt < self.cfg.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.replace(".", "", 1).isdigit():
                        wait = float(retry_after)
                else:
                    if resp.status_code >= 400:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                    envelope = resp.json()
                    if envelope.get("success") is False:
                        raise RuntimeError(f"API error: {envelope.get('errors')}")
                    return envelope["result"], attempt + 1
            time.sleep(wait)
        raise AssertionError("unreachable")

    def run(self, examples: list[dict]) -> list[dict]:
        return [self(ex, ex["revision"]) for ex in tqdm(examples, desc="requests")]

    def __call__(self, ex: dict, revision: str) -> dict:
        payload = self.payload(ex)
        key_src = json.dumps({"payload": payload, "revision": revision, "protocol": PROTOCOL, "url_model": "@cf/cloudflare/clairic"}, sort_keys=True)
        path = self.cache_dir / f"{hashlib.sha256(key_src.encode()).hexdigest()}.json"
        if self.cfg.use_cache and path.exists():
            return {**json.loads(path.read_text()), "cached": True}

        t0 = time.perf_counter()
        try:
            result, attempts = self._post(payload)
        except RuntimeError as e:  # recorded, scored 0, never cached
            return {"error": str(e), "latency_s": time.perf_counter() - t0, "cached": False}
        answers = result.get("answers") or {}
        rec = {
            "preds": {name: (answers.get(name) or {}).get("value") for name in ex["fields"]},
            "confidence": {name: (answers.get(name) or {}).get("confidence") for name in ex["fields"]},
            "p_null": {name: (answers.get(name) or {}).get("p_null") for name in ex["fields"]},
            "usage": result.get("usage"),
            "truncated": result.get("truncated"),
            "state_text": result.get("state_text"),
            "raw": result,
            "latency_s": time.perf_counter() - t0,
            "attempts": attempts,
        }
        path.write_text(json.dumps(rec, ensure_ascii=False))
        return {**rec, "cached": False}


class OpenAIServer:
    """Chat completions against an OpenAI-compatible server such as `vllm serve`. Requests go out concurrently, so
    the server's continuous batching (paged attention) keeps the GPU full. Prompts are the training ones: the same
    system and user JSON, the image (as a data URL) before the text, and the non-thinking chat template."""

    def __init__(self, cfg: EvalConfig):
        import requests

        self.cfg = cfg
        models = requests.get(f"{cfg.base_url}/models", timeout=30).json()["data"]
        self.model = models[0]["id"]
        self.meta = {"base_url": cfg.base_url, "served_model": self.model, "root": models[0].get("root")}
        print(f"Serving model: {self.model}")

    def messages(self, ex: dict) -> list[dict]:
        msgs = build_messages(ex["state"], ex["fields"], has_image=ex["image_bytes"] is not None)
        for m in msgs:
            for part in m["content"]:
                if part["type"] == "image":
                    part.clear()
                    part.update({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(ex["image_bytes"]).decode("ascii")}})
        return msgs

    def _call(self, ex: dict) -> dict:
        import requests

        body = {
            "model": self.model,
            "messages": self.messages(ex),
            "temperature": 0.0,
            "max_tokens": self.cfg.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(f"{self.cfg.base_url}/chat/completions", json=body, timeout=self.cfg.timeout_s)
                if resp.status_code in TRANSIENT_STATUS and attempt < self.cfg.max_retries:
                    time.sleep(min(60.0, 2.0**attempt))
                    continue
                if resp.status_code >= 400:
                    return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}", "latency_s": time.perf_counter() - t0}
                break
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == self.cfg.max_retries:
                    return {"error": f"{type(e).__name__}: {e}", "latency_s": time.perf_counter() - t0}
                time.sleep(min(60.0, 2.0**attempt))
        out = resp.json()
        choice = out["choices"][0]
        text = choice["message"]["content"] or ""
        parsed = parse_json_object(text)
        rec = {
            "preds": parsed or {},
            "raw": text,
            "usage": out.get("usage"),
            "truncated": choice.get("finish_reason") == "length",  # hit max_new_tokens
            "latency_s": time.perf_counter() - t0,
        }
        if parsed is None:
            rec["parse_error"] = True
        return rec

    def run(self, examples: list[dict]) -> list[dict]:
        with ThreadPoolExecutor(self.cfg.concurrency) as pool:
            return list(tqdm(pool.map(self._call, examples), total=len(examples), desc="requests"))


class Local:
    def __init__(self, cfg: EvalConfig):
        from model import load_for_inference

        self.cfg = cfg
        self.model, self.processor, self.meta = load_for_inference(cfg.adapter, cfg.quantize)
        if cfg.compile:
            self.model.compile(mode="reduce-overhead", dynamic=False)

    def run(self, examples: list[dict]) -> list[dict]:
        """Generate in batches of similar prompt length; latency_s is the batch time split evenly."""
        from model import generate_batch

        order = sorted(range(len(examples)), key=lambda i: len(examples[i]["state"]))
        recs: list[dict | None] = [None] * len(examples)
        bs = self.cfg.batch_size
        for b in tqdm(range(0, len(order), bs), desc="batches"):
            idx = order[b : b + bs]
            reqs = [(examples[i]["state"], examples[i]["fields"], _decode(examples[i]["image_bytes"]) if examples[i]["image_bytes"] else None) for i in idx]
            t0 = time.perf_counter()
            outs = generate_batch(self.model, self.processor, reqs, self.cfg.max_new_tokens)
            dt = (time.perf_counter() - t0) / len(idx)
            for i, (text, n_prompt) in zip(idx, outs):
                parsed = parse_json_object(text)
                recs[i] = {"preds": parsed or {}, "raw": text, "prompt_tokens": n_prompt, "latency_s": dt}
                if parsed is None:
                    recs[i]["parse_error"] = True
        return recs


# ---------------------------------------------------------------- scoring / summary


def score(ex: dict, rec: dict) -> list[dict]:
    if "error" in rec:  # failed request: every field scores 0
        return [{"field": f, "type": s["type"], "score": 0.0, "gold_null": None, "pred_null": None} for f, s in ex["fields"].items()]
    preds = rec["preds"]
    if ex["kind"] == "extraction":
        return score_extraction(ex["fields"], ex["gold"], preds)
    pred = preds.get("answer")
    s = anls(pred, ex["gold"]) if ex["kind"] == "docvqa" else relaxed_accuracy(pred, ex["gold"])
    return [{"field": "answer", "type": ex["fields"]["answer"]["type"], "score": s, "gold_null": False, "pred_null": is_null(pred)}]


def summarize(rows: list[dict]) -> dict:
    fields = [f for r in rows for f in r["fields"]]
    scalar = [f for f in fields if f["type"] != "list" and f["gold_null"] is not None]
    gold_null = [f for f in scalar if f["gold_null"]]
    gold_set = [f for f in scalar if not f["gold_null"]]
    out = {
        "n_examples": len(rows),
        "n_fields": len(fields),
        "field_score": sum(f["score"] for f in fields) / max(len(fields), 1),  # mean per-field score (dataset cards' metric)
        "example_exact": sum(all(f["score"] == 1 for f in r["fields"]) for r in rows) / max(len(rows), 1),
        "n_failed": sum("error" in r for r in rows),
        "n_parse_errors": sum(bool(r.get("parse_error")) for r in rows),
        "null_recall": sum(f["pred_null"] for f in gold_null) / len(gold_null) if gold_null else None,  # gold null -> predicted null
        "false_null_rate": sum(f["pred_null"] for f in gold_set) / len(gold_set) if gold_set else None,  # gold set -> predicted null
        "truncated_rate": sum(bool(r.get("truncated")) for r in rows) / max(len(rows), 1),
        "mean_latency_s": sum(r["latency_s"] for r in rows) / max(len(rows), 1),
    }
    usage = defaultdict(int)
    for r in rows:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] += v
    if usage:
        out["usage_total"] = dict(usage)
    levels = sorted({r["level"] for r in rows if r.get("level") is not None})
    if levels:
        out["field_score_by_level"] = {}
        for lv in levels:
            fs = [f for r in rows if r.get("level") == lv for f in r["fields"]]
            out["field_score_by_level"][str(lv)] = sum(f["score"] for f in fs) / len(fs)
    return out


def plot_scores(results: dict, path: Path, title: str) -> None:
    from zoo.plot import plt

    names = list(results)
    vals = [results[n]["field_score"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.2))
    ax.barh(names, vals, color="#4c72b0")
    for y, v in enumerate(vals):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.set_xlabel("mean per-field score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(cfg: EvalConfig):
    backend = {"clairic": Clairic, "openai": OpenAIServer, "local": Local}[cfg.backend](cfg)

    loaded = {name: load_examples(name, cfg.count, cfg.seed) for name in cfg.datasets}
    revisions = {name: {"repo": repo, "revision": rev} for name, (_, repo, rev) in loaded.items()}
    extra = {"revisions": revisions, "protocol": PROTOCOL}
    if "docvqa" in cfg.datasets:
        import pytesseract

        extra["tesseract"] = str(pytesseract.get_tesseract_version())
    if cfg.backend in ("local", "openai"):
        extra["model"] = backend.meta
    run_dir = new_ledger_run("eval", cfg, {"local": "cuda:0", "openai": cfg.base_url}.get(cfg.backend, "remote"), **extra)
    (run_dir / "sample_ids.json").write_text(json.dumps({n: [e["id"] for e in exs] for n, (exs, _, _) in loaded.items()}, indent=1))

    # Every request from every set goes to the backend at once, so a server never drains between sets.
    flat = [{**ex, "dataset": name, "revision": rev} for name, (exs, _, rev) in loaded.items() for ex in exs]
    t0 = time.perf_counter()
    recs = backend.run(flat)
    print(f"\n{len(flat)} requests in {time.perf_counter() - t0:.1f}s")

    rows_by_set = defaultdict(list)
    with (run_dir / "predictions.jsonl").open("w") as f:
        for ex, rec in zip(flat, recs):
            row = {"dataset": ex["dataset"], "id": ex["id"], "level": ex["level"], "gold": ex["gold"], "fields": score(ex, rec), **rec}
            rows_by_set[ex["dataset"]].append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    results = {}
    for name in loaded:
        r = results[name] = summarize(rows_by_set[name])
        print(f"{name:45s} field_score={r['field_score']:.3f} exact={r['example_exact']:.3f} failed={r['n_failed']} parse_errors={r['n_parse_errors']}")

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    label = {"clairic": "clairic", "openai": f"openai: {getattr(backend, 'model', '')}"}.get(cfg.backend, f"local ({'adapter' if cfg.adapter else 'base'})")
    plot_scores(results, run_dir / "scores.png", f"ledger eval: {label}")
    print(f"\nWrote {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
