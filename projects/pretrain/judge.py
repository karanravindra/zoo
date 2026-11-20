"""Grade the stories that `sample.py --prompts` wrote (`<run>/eval_stories.jsonl`) with Claude, an OpenAI model, the Codex CLI signed in with a ChatGPT plan, or a local Ollama model (a general one, or the d3 decision model).

`uv run --group pretrain projects/pretrain/judge.py [--judge claude-haiku-5-5] [--mode both] [--subset 1.0]`

Two modes, both over the shared prompts from `make_prompts.py`:
- `abs`: each story gets 1-10 scores for grammar, coherence, plot and creativity (structured JSON output).
- `pairwise`: each run's story is compared with the same prompt's story from the `--baseline` run (the top TinyStories
  model), in both orders so the judge's habit of preferring A or B cancels. A pair scores 1 / 0.5 / 0 for the run
  (the mean of the two orders), and the win rate is the mean over prompts.

Results are cached per run (`<run>/judge_<judge>_<mode>.jsonl`), so a rerun only grades what is missing and a batch
that was submitted but not collected (state in `--out-dir`/judge_<judge>_<mode>_batch.json) is resumed instead of
billed twice. Changing `RUBRIC_VERSION`, the baseline or a run's stories (each row keeps a hash of what was graded) ignores the old rows. Every run also gets `ended_rate`, the
share of stories that reached end-of-text within `--max-tokens` (measured, not judged).

It then writes `story_eval_<judge>.csv` and `.png` into `--out-dir` (one pair per judge), in-domain (TinyStories openings) and
out-of-domain (the other sources) reported separately. `--compare-judge OTHER` also prints how well a second judge's
cached grades agree with this one's. `--dry-run` prints what would be sent and calls nothing.

Needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile). With `--provider openai --judge <model id>` it needs an
`OPENAI_API_KEY` for the API instead (a ChatGPT subscription does not include API access). Batch requests cost half of
the standard price on both.
`--provider codex` (after `codex login`) instead runs one `codex exec` per judgement, `--workers` at a time, with Codex's
tools off and an empty working directory: no batch discount, and about 8k tokens of Codex overhead per call count
against the plan's usage limits. `--judge default` uses Codex's own model. Each grade is cached as it arrives, so a limit
that stops the run loses nothing: rerun later and it continues.
`--provider ollama --judge <tag>` (e.g. `lfm3:latest`) does the same against a local Ollama server (`--base-url`), asking for
schema-constrained JSON at temperature 0 with thinking off; nothing is billed. Grades are cached under the judge's name.
`--provider d3 --judge d3:9b` asks a decision model (Ollama's `/v1/systemone`): it does not write text, it scores the options
A / B / tie and returns their probabilities, so it only does `pairwise` (the probabilities are kept in the cache row's
`reason`). Its context is 4096 tokens.
"""

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from zoo.paths import runs_root

RUBRIC_VERSION = 1  # bump when a prompt or schema changes, so cached grades from the old rubric are ignored
CRITERIA = ("grammar", "coherence", "plot", "creativity")
IN_DOMAIN = "tinystories"  # the `source` of the prompts taken from the held-out TinyStories split

ABS_SYSTEM = """You grade short stories written by very small language models (2M to 100M parameters). Each story continues a given opening; the opening is the first sentence of the text you are shown.

Score four criteria from 1 to 10, as a strict grader who uses the whole range:
- grammar: spelling, grammar, punctuation and word choice.
- coherence: sentences follow from each other, names, facts and who-did-what stay consistent, and the text continues the opening sensibly in the same register.
- plot: something happens, with a beginning, a development and an ending, rather than repetition or wandering.
- creativity: specific, varied, surprising details rather than generic or repeated phrasing.

Anchors: 10 = a skilled human writer of this kind of text could have written it; 7 = fluent with minor slips; 4 = noticeably broken or repetitive; 1 = gibberish.
The text may be cut off at a length limit; do not penalise the cut itself. Write the comment as one short sentence naming the main flaw."""

PAIR_SYSTEM = """You compare two continuations of the same opening. Both were written by very small language models.

Pick the better one overall: grammar, coherence with the opening and within itself, plot, and creativity. Ignore the order they are shown in and their length. Either may be cut off at a length limit; do not penalise the cut itself. Answer "tie" only if you truly cannot separate them. Give the reason as one short sentence."""

ABS_SCHEMA = {
    "type": "object",
    "properties": {**{c: {"type": "integer"} for c in CRITERIA}, "comment": {"type": "string"}},
    "required": [*CRITERIA, "comment"],
    "additionalProperties": False,
}
PAIR_SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}, "reason": {"type": "string"}},
    "required": ["winner", "reason"],
    "additionalProperties": False,
}


@dataclass
class Config:
    runs: list[Path] | None = None  # run dirs holding eval_stories.jsonl (default: every such dir under runs/pretrain)
    baseline: Path = runs_root("tinystories") / "d512-l6-ctx2048-1e17-win512g5-mask"  # the model to beat; also graded in abs mode
    provider: Literal["anthropic", "openai", "codex", "ollama", "d3"] = "anthropic"  # who grades: Claude batches, OpenAI batches, the Codex CLI (a ChatGPT plan), a local Ollama model, or the d3 decision model on Ollama (comparisons only); all but anthropic need --judge too
    judge: str = "claude-haiku-5-5"  # model id of the grader (for --provider openai, an OpenAI model id: there is no default; for codex, a model id Codex accepts or "default" for its own)
    mode: Literal["abs", "pairwise", "both"] = "both"
    subset: float = 1.0  # fraction of the prompts to grade, the same prompts for every run, stratified by source (e.g. 0.2 for a cross-check judge)
    effort: Literal["low", "medium", "high"] = "low"  # anthropic: judge thinking effort; thinking tokens count toward --max-tokens
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None  # openai / codex: only sent when set, because models without reasoning reject it
    workers: int = 4  # codex / ollama / d3: parallel calls
    timeout_s: int = 300  # codex / ollama / d3: give up on one call after this long
    base_url: str = "http://localhost:11434"  # ollama / d3: server address
    num_ctx: int = 8192  # ollama: context window requested for every call (keep it fixed, a change reloads the model)
    max_tokens: int = 4096  # per judgement; the JSON answer is short, the rest is headroom for thinking
    poll_s: int = 30  # seconds between batch status checks
    compare_judge: str | None = None  # also print the agreement between this judge's cached grades and --judge's
    seed: int = 0  # picks which prompts --subset keeps
    out_dir: Path = runs_root("pretrain")  # where story_eval_<judge>.csv / .png and the pending-batch state files go
    dry_run: bool = False  # print the number of requests and one example, call nothing


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_stories(run: Path) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(run / "eval_stories.jsonl")}


def pick_ids(rows: dict[str, dict], frac: float, seed: int) -> list[str]:
    """A stable stratified subset: per source, the `frac` of ids that come first in a seeded hash order."""
    by_source: dict[str, list[str]] = {}
    for i, r in rows.items():
        by_source.setdefault(r["source"], []).append(i)
    keep = []
    for ids in by_source.values():
        ids = sorted(ids, key=lambda i: hashlib.sha1(f"{seed}:{i}".encode()).hexdigest())
        keep += ids[: max(1, math.ceil(frac * len(ids)))]
    return sorted(keep)


# ---- cache ----------------------------------------------------------------------------------------------------


def safe(judge: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", judge)  # a tag like lfm3:latest or org/model is not a file name


def cache_path(run: Path, judge: str, mode: str) -> Path:
    return run / f"judge_{safe(judge)}_{mode}.jsonl"


def digest(user: str) -> str:
    return hashlib.sha1(user.encode()).hexdigest()[:12]


def user_message(mode: str, key: str, cand: dict[str, dict], base: dict[str, dict]) -> str | None:
    """The judge's input for a cache key (`id` in abs mode, `id|ab` / `id|ba` in pairwise), or None if a story is missing."""
    i, _, order = key.partition("|")
    if i not in cand or i not in base:
        return None
    if mode == "abs":
        return abs_user(cand[i])
    a, b = (cand[i], base[i]) if order == "ab" else (base[i], cand[i])
    return pair_user(cand[i], a, b)


def load_cache(cfg: Config, run: Path, judge: str, mode: str, stories: dict[Path, dict[str, dict]]) -> dict[str, dict]:
    """The cached grades that are still valid: same rubric and baseline, and graded on the stories as they are now (each row
    carries a hash of the judge's input, so resampling a run or the baseline drops its old grades)."""
    p = cache_path(run, judge, mode)
    if not p.exists():
        return {}
    rows = {}
    for r in read_jsonl(p):
        if r.get("v") != RUBRIC_VERSION or (mode == "pairwise" and r.get("baseline") != cfg.baseline.name):
            continue
        msg = user_message(mode, r["key"], stories[run], stories[cfg.baseline])
        if msg is not None and r.get("h") == digest(msg):
            rows[r["key"]] = r
    return rows


def append_cache(run: Path, judge: str, mode: str, baseline: str, rows: list[dict]) -> None:
    with cache_path(run, judge, mode).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"v": RUBRIC_VERSION, "baseline": baseline, **r}) + "\n")


# ---- requests -------------------------------------------------------------------------------------------------


def abs_user(row: dict) -> str:
    return f"Opening:\n{row['prompt']}\n\nStory:\n{row['text']}"


def pair_user(row: dict, a: dict, b: dict) -> str:
    return f"Opening:\n{row['prompt']}\n\n=== Story A ===\n{a['text']}\n\n=== Story B ===\n{b['text']}"


def build_jobs(cfg: Config, mode: str, runs: list[Path], stories: dict[Path, dict[str, dict]], ids: list[str]) -> list[tuple[Path, str, str]]:
    """(run, cache key, user message) for every judgement the caches don't have yet."""
    jobs = []
    for run in runs:
        if mode == "pairwise" and run == cfg.baseline:
            continue
        done = load_cache(cfg, run, cfg.judge, mode, stories)
        for i in ids:
            for key in (i,) if mode == "abs" else (f"{i}|ab", f"{i}|ba"):
                user = user_message(mode, key, stories[run], stories[cfg.baseline])
                if user is not None and key not in done:
                    jobs.append((run, key, user))
    return jobs


def parse(mode: str, text: str | None) -> dict | None:
    """The judge's JSON answer, or None if there was none (refused, cut off) or it did not validate."""
    if text is None:
        return None
    try:
        d = json.loads(text)
        if mode == "abs":
            if not all(isinstance(d[c], (int, float)) and not isinstance(d[c], bool) and 1 <= d[c] <= 10 for c in CRITERIA):
                return None
            return {**{c: d[c] for c in CRITERIA}, "comment": str(d["comment"])}
        return {"winner": d["winner"], "reason": str(d["reason"])} if d["winner"] in ("A", "B", "tie") else None
    except (ValueError, KeyError, TypeError):
        return None


class AnthropicBatches:
    """Claude through the Message Batches API, answers constrained by a JSON schema."""

    def __init__(self, cfg: Config) -> None:
        import anthropic  # lazy: the other providers need no anthropic install

        self.cfg, self.client = cfg, anthropic.Anthropic()  # only built when something is graded: a cached rerun needs no credentials

    def submit(self, system: str, schema: dict, users: dict[str, str]) -> str:
        """Send one request per `custom_id: user message`; returns the batch id."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        cfg = self.cfg
        requests = [
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                model=cfg.judge, max_tokens=cfg.max_tokens, system=system, messages=[{"role": "user", "content": user}],
                output_config={"effort": cfg.effort, "format": {"type": "json_schema", "schema": schema}},
            ))
            for cid, user in users.items()
        ]
        return self.client.messages.batches.create(requests=requests).id

    def status(self, batch_id: str) -> tuple[bool, str]:
        b = self.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        return b.processing_status == "ended", f"{b.processing_status} (processing {c.processing}, succeeded {c.succeeded}, errored {c.errored})"

    def results(self, batch_id: str) -> Iterator[tuple[str, str | None, str]]:
        """(custom_id, the answer text or None, why there is none)."""
        for res in self.client.messages.batches.results(batch_id):
            if res.result.type != "succeeded":
                yield res.custom_id, None, res.result.type
                continue
            msg = res.result.message
            if msg.stop_reason in ("refusal", "max_tokens"):
                yield res.custom_id, None, msg.stop_reason
            else:
                yield res.custom_id, next((b.text for b in msg.content if b.type == "text"), None), ""


class OpenAIBatches:
    """An OpenAI model through the Batch API (Chat Completions endpoint, strict JSON-schema output)."""

    def __init__(self, cfg: Config) -> None:
        import openai

        self.cfg, self.client = cfg, openai.OpenAI()

    def submit(self, system: str, schema: dict, users: dict[str, str]) -> str:
        cfg = self.cfg
        lines = []
        for cid, user in users.items():
            body = {
                "model": cfg.judge, "max_completion_tokens": cfg.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "judgement", "strict": True, "schema": schema}},
            }
            if cfg.reasoning_effort:
                body["reasoning_effort"] = cfg.reasoning_effort
            lines.append(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": body}))
        f = self.client.files.create(file=("judge_requests.jsonl", "\n".join(lines).encode()), purpose="batch")
        return self.client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h").id

    def status(self, batch_id: str) -> tuple[bool, str]:
        b = self.client.batches.retrieve(batch_id)
        c = b.request_counts
        text = f"{b.status} (completed {c.completed} of {c.total}, failed {c.failed})" if c else b.status
        if b.errors and b.errors.data:  # e.g. a rejected request body fails the whole batch
            text += ": " + "; ".join(str(e.message) for e in b.errors.data[:3])
        return b.status in ("completed", "failed", "expired", "cancelled"), text

    def results(self, batch_id: str) -> Iterator[tuple[str, str | None, str]]:
        b = self.client.batches.retrieve(batch_id)
        for file_id in (b.output_file_id, b.error_file_id):
            if not file_id:
                continue
            for line in self.client.files.content(file_id).text.splitlines():
                r = json.loads(line)
                resp = r.get("response") or {}
                if r.get("error") or resp.get("status_code") != 200:
                    yield r["custom_id"], None, str(r.get("error") or f"status {resp.get('status_code')}")
                    continue
                choice = resp["body"]["choices"][0]
                if choice["message"].get("refusal") or choice["finish_reason"] == "length":
                    yield r["custom_id"], None, "refusal" if choice["message"].get("refusal") else "max_tokens"
                else:
                    yield r["custom_id"], choice["message"].get("content"), ""


CODEX_OFF = ("shell_tool", "unified_exec", "apps", "browser_use", "computer_use", "image_generation", "multi_agent", "plugins", "tool_suggest", "goals", "hooks", "shell_snapshot")
LIMIT = re.compile(r"usage limit|rate limit|too many requests|\b429\b|quota", re.I)


def codex_call(cfg: Config, schema_file: Path, cwd: Path, prompt: str) -> tuple[str | None, str]:
    """One `codex exec` judgement: (the answer text, or None and why not)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "last_message.txt"
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "-s", "read-only",
               "-C", str(cwd), "--output-schema", str(schema_file), "-o", str(out), "-c", 'web_search="disabled"']
        for feature in CODEX_OFF:
            cmd += ["--disable", feature]
        if cfg.judge != "default":
            cmd += ["-m", cfg.judge]
        if cfg.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{cfg.reasoning_effort}"']
        try:
            r = subprocess.run([*cmd, "-"], input=prompt, capture_output=True, text=True, timeout=cfg.timeout_s)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        text = out.read_text().strip() if out.exists() else ""
        if r.returncode == 0 and text:
            return text, ""
        err = (r.stderr + r.stdout).strip()[-400:]
        return None, ("limit: " if LIMIT.search(err) else "") + (err or f"exit {r.returncode}")


def run_pool(cfg: Config, mode: str, jobs: list[tuple[Path, str, str]], call) -> None:
    """Judge every job with `call(user message) -> (answer text or None, why not)` on `cfg.workers` threads, caching each
    grade the moment it arrives; stops early when a call reports a usage limit."""
    jobs = sorted(jobs, key=lambda j: (j[1].split("|")[0], str(j[0])))  # prompt by prompt across runs, so a run cut short still leaves comparable coverage
    lock, stop = threading.Lock(), threading.Event()
    graded, failed, skipped = 0, [], 0

    def work(job):
        if stop.is_set():
            return job, None, "skipped"
        try:
            return (job, *call(job[2]))
        except Exception as e:  # noqa: BLE001 - an odd response from one call must not abort the run
            return job, None, f"{type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(cfg.workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(work, j) for j in jobs]), 1):
            (run, key, user), text, why = fut.result()
            out = parse(mode, text)
            if why == "skipped":
                skipped += 1
            elif out is None:
                failed.append(f"{run.name} {key}: {why or 'invalid answer'}")
                if why.startswith("limit:") and not stop.is_set():
                    stop.set()
                    print(f"[{mode}] usage limit hit: {why[7:200]}\n[{mode}] stopping; rerun later to continue")
            else:
                with lock:
                    append_cache(run, cfg.judge, mode, cfg.baseline.name, [{"key": key, "h": digest(user), **out}])
                graded += 1
            if n % 100 == 0 or n == len(jobs):
                print(f"[{mode}] {n}/{len(jobs)} done (graded {graded}, failed {len(failed)})", flush=True)
    print(f"[{mode}] graded {graded}, failed {len(failed)}, skipped {skipped}" + "".join(f"\n  {f}" for f in failed[:5]))


def run_codex(cfg: Config, mode: str, system: str, schema: dict, jobs: list[tuple[Path, str, str]]) -> None:
    if not shutil.which("codex"):
        raise SystemExit("--provider codex needs the codex CLI on PATH (and `codex login`)")
    with tempfile.TemporaryDirectory() as tmp:
        cwd, schema_file = Path(tmp) / "cwd", Path(tmp) / "schema.json"
        cwd.mkdir()  # an empty working root, so the agent has no repository to wander into
        schema_file.write_text(json.dumps(schema))
        suffix = "\n\nAnswer directly with only the JSON object. Do not run any commands."
        run_pool(cfg, mode, jobs, lambda user: codex_call(cfg, schema_file, cwd, f"{system}\n\n{user}{suffix}"))


def ollama_call(cfg: Config, system: str, schema: dict, user: str) -> tuple[str | None, str]:
    """One chat call to a local Ollama server: schema-constrained JSON, temperature 0, thinking off."""
    import urllib.error
    import urllib.request

    body = {"model": cfg.judge, "stream": False, "think": False, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_ctx": cfg.num_ctx, "num_predict": cfg.max_tokens}}
    req = urllib.request.Request(f"{cfg.base_url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            d = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    if d.get("done_reason") == "length":
        return None, "max_tokens"
    return d["message"]["content"], ""


def run_ollama(cfg: Config, mode: str, system: str, schema: dict, jobs: list[tuple[Path, str, str]]) -> None:
    run_pool(cfg, mode, jobs, lambda user: ollama_call(cfg, system, schema, user))


D3_ASK = "Which continuation of the opening is better overall: grammar, coherence with the opening and within itself, plot, and creativity? Ignore the order they are shown in and their length. Either may be cut off at a length limit; do not penalise the cut itself."
D3_CHOICES = {"A": "Story A is the better continuation", "B": "Story B is the better continuation", "tie": "They are equally good, or cannot be separated"}


D3_LEVELS = ["Very poor", "Poor", "Fair", "Good", "Excellent"]  # ordered, lowest first; d3 scores 2-10 levels and 5 keeps them distinct
D3_CRITERIA = {
    "grammar": "How grammatical and fluent is the story's English (spelling, punctuation, sentence structure)? Ignore a cut-off ending.",
    "coherence": "How consistent is the story with its opening and with itself (characters, setting and events that fit together, no contradictions)?",
    "plot": "How well-formed is the plot: a clear sequence of events that builds toward a resolution? Ignore a cut-off ending.",
    "creativity": "How original and interesting is the story compared with a generic, repetitive one?",
}


def d3_abs_call(cfg: Config, state: str) -> tuple[str | None, str]:
    """One d3 request with a `score` question per criterion. d3 returns the expected level (0 to len-1, a probability-weighted
    average, so it is continuous); that is mapped linearly onto 1-10 so it reads like the other judges' scores."""
    import urllib.error
    import urllib.request

    qs = {c: {"type": "score", "instructions": q, "criteria": D3_LEVELS} for c, q in D3_CRITERIA.items()}
    req = urllib.request.Request(f"{cfg.base_url}/v1/systemone", json.dumps({"model": cfg.judge, "state": state, "questions": qs}).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            ans = json.loads(r.read())["answers"]
        out = {c: round(1 + ans[c]["score"] * 9 / (len(D3_LEVELS) - 1), 3) for c in CRITERIA}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    return json.dumps({**out, "comment": "d3 expected level: " + json.dumps({c: round(ans[c]["score"], 3) for c in CRITERIA})}), ""


def d3_call(cfg: Config, state: str) -> tuple[str | None, str]:
    """One d3 decision: the comparison as the usual JSON answer (the option probabilities go into `reason`)."""
    import urllib.error
    import urllib.request

    body = {"model": cfg.judge, "state": state, "questions": {"winner": {"type": "choice", "instructions": D3_ASK, "criteria": D3_CHOICES}}}
    req = urllib.request.Request(f"{cfg.base_url}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            ans = json.loads(r.read())["answers"]["winner"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    probs = {k: round(v, 4) for k, v in ans["probabilities"].items()}
    return json.dumps({"winner": ans["choice"], "reason": json.dumps(probs)}), ""


def run_d3(cfg: Config, mode: str, jobs: list[tuple[Path, str, str]]) -> None:
    call = d3_call if mode == "pairwise" else d3_abs_call
    run_pool(cfg, mode, jobs, lambda user: call(cfg, user))


def make_provider(cfg: Config) -> AnthropicBatches | OpenAIBatches:
    return OpenAIBatches(cfg) if cfg.provider == "openai" else AnthropicBatches(cfg)


def collect(cfg: Config, provider: AnthropicBatches | OpenAIBatches, mode: str, state: Path) -> None:
    """Wait for the batch in `state`, write its answers into the caches, and remove the state file."""
    st = json.loads(state.read_text())
    while True:
        done, line = provider.status(st["batch_id"])
        print(f"[{mode}] batch {st['batch_id']}: {line}")
        if done:
            break
        time.sleep(cfg.poll_s)
    per_run: dict[str, list[dict]] = {}
    failed: list[str] = []
    for cid, text, why in provider.results(st["batch_id"]):
        run, key, h = st["jobs"][cid]
        out = parse(mode, text)
        if out is None:
            failed.append(f"{Path(run).name} {key}: {why or 'invalid answer'}")
        else:
            per_run.setdefault(run, []).append({"key": key, "h": h, **out})
    for run, rows in per_run.items():
        append_cache(Path(run), cfg.judge, mode, st["baseline"], rows)  # the baseline the batch was sent with, not the current --baseline
    state.unlink()
    n_ok = sum(map(len, per_run.values()))
    lost = len(st["jobs"]) - n_ok - len(failed)  # sent but no result came back at all (a batch that failed or expired)
    print(f"[{mode}] graded {n_ok}, failed {len(failed)}, no result {lost}" + "".join(f"\n  {f}" for f in failed[:5]))


def grade(cfg: Config, mode: str, runs: list[Path], stories: dict[Path, dict[str, dict]], ids: list[str]) -> None:
    system, schema = (ABS_SYSTEM, ABS_SCHEMA) if mode == "abs" else (PAIR_SYSTEM, PAIR_SCHEMA)
    state = cfg.out_dir / f"judge_{safe(cfg.judge)}_{mode}_batch.json"
    if state.exists() and not cfg.dry_run:  # a batch was submitted earlier and never collected
        print(f"[{mode}] resuming {state}")
        collect(cfg, make_provider(cfg), mode, state)
    jobs = build_jobs(cfg, mode, runs, stories, ids)
    if not jobs:
        print(f"[{mode}] nothing left to grade")
        return
    if cfg.dry_run:
        print(f"[{mode}] would send {len(jobs)} requests to {cfg.judge} ({cfg.provider}). Example user message:\n{jobs[0][2]}\n")
        return
    if cfg.provider == "codex":
        run_codex(cfg, mode, system, schema, jobs)
        return
    if cfg.provider == "ollama":
        run_ollama(cfg, mode, system, schema, jobs)
        return
    if cfg.provider == "d3":
        run_d3(cfg, mode, jobs)
        return
    provider = make_provider(cfg)
    batch_id = provider.submit(system, schema, {f"q{n}": user for n, (_, _, user) in enumerate(jobs)})
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"batch_id": batch_id, "baseline": cfg.baseline.name, "jobs": {f"q{n}": [str(run), key, digest(user)] for n, (run, key, user) in enumerate(jobs)}}))
    print(f"[{mode}] submitted {len(jobs)} requests as batch {batch_id}")
    collect(cfg, provider, mode, state)


# ---- report ---------------------------------------------------------------------------------------------------


def mean_se(x) -> tuple[float, float]:
    x = np.asarray(list(x), dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), (float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"))


def candidate_score(key: str, winner: str) -> float:
    """1 / 0.5 / 0 for the run's story in one comparison (in order "ab" the run's story is A, in "ba" it is B)."""
    if winner == "tie":
        return 0.5
    return 1.0 if (winner == "A") == key.endswith("|ab") else 0.0


def summarize(cfg: Config, run: Path, stories: dict[Path, dict[str, dict]], ids: set[str]) -> dict:
    mine = stories[run]
    row = {"run": run.name, "flops": float("nan"), "ended_rate": float(np.mean([s["ended"] for s in mine.values()]))}
    if (run / "results.json").exists():
        row["flops"] = json.loads((run / "results.json").read_text()).get("total_flops", float("nan"))
    abs_rows = {k: v for k, v in load_cache(cfg, run, cfg.judge, "abs", stories).items() if k in ids}
    overall = {k: np.mean([v[c] for c in CRITERIA]) for k, v in abs_rows.items()}
    splits = {"in": lambda k: mine[k]["source"] == IN_DOMAIN, "ood": lambda k: mine[k]["source"] != IN_DOMAIN}
    row["n_abs"] = len(abs_rows)
    for name, sel in splits.items():
        row[f"score_{name}"], row[f"score_{name}_se"] = mean_se(overall[k] for k in overall if sel(k))
    for c in CRITERIA:
        row[c] = mean_se(v[c] for v in abs_rows.values())[0]
    pair_rows = {k: v for k, v in load_cache(cfg, run, cfg.judge, "pairwise", stories).items() if k.split("|")[0] in ids}
    by_id: dict[str, list[float]] = {}
    for k, v in pair_rows.items():
        by_id.setdefault(k.split("|")[0], []).append(candidate_score(k, v["winner"]))
    pairs = {i: float(np.mean(s)) for i, s in by_id.items() if len(s) == 2}  # only prompts judged in both orders
    row["n_pairs"] = len(pairs)
    for name, sel in splits.items():
        row[f"win_{name}"], row[f"win_{name}_se"] = mean_se(s for i, s in pairs.items() if sel(i))
    row["pick_a_rate"] = float(np.mean([v["winner"] == "A" for v in pair_rows.values()])) if pair_rows else float("nan")
    return row


def agreement(cfg: Config, runs: list[Path], stories: dict[Path, dict[str, dict]], ids: set[str]) -> None:
    """Pooled agreement of `cfg.compare_judge`'s cached grades with `cfg.judge`'s, on the judgements both made."""
    from scipy.stats import spearmanr

    a_scores, b_scores, same, n_pair = [], [], 0, 0
    for run in runs:
        for mode in ("abs", "pairwise"):
            ca = load_cache(cfg, run, cfg.judge, mode, stories)
            cb = load_cache(cfg, run, cfg.compare_judge, mode, stories)
            for k in ca.keys() & cb.keys():
                if k.split("|")[0] not in ids:
                    continue
                if mode == "abs":
                    a_scores.append(np.mean([ca[k][c] for c in CRITERIA]))
                    b_scores.append(np.mean([cb[k][c] for c in CRITERIA]))
                else:
                    n_pair += 1
                    same += candidate_score(k, ca[k]["winner"]) == candidate_score(k, cb[k]["winner"])
    if len(a_scores) > 2:
        print(f"abs agreement with {cfg.compare_judge}: Spearman {spearmanr(a_scores, b_scores).statistic:.3f} on {len(a_scores)} stories")
    if n_pair:
        print(f"pairwise agreement with {cfg.compare_judge}: {same / n_pair:.1%} of {n_pair} comparisons give the same verdict")
    if len(a_scores) <= 2 and not n_pair:
        print(f"no judgements in common with {cfg.compare_judge} yet")


def report(cfg: Config, runs: list[Path], stories: dict[Path, dict[str, dict]], ids: list[str]) -> None:
    rows = sorted((summarize(cfg, r, stories, set(ids)) for r in runs), key=lambda r: (r["flops"], r["run"]))
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"story_eval_{safe(cfg.judge)}"
    with (out / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'run':<44}{'FLOPs':>9}{'score in':>10}{'score ood':>11}{'win in':>9}{'win ood':>9}{'ended':>8}{'pickA':>7}")
    f2 = lambda v: "-" if v != v else f"{v:.2f}"  # noqa: E731
    for r in rows:
        mark = " (baseline)" if r["run"] == cfg.baseline.name else ""
        print(f"{r['run'] + mark:<44}{r['flops']:>9.0e}{f2(r['score_in']):>10}{f2(r['score_ood']):>11}{f2(r['win_in']):>9}{f2(r['win_ood']):>9}{r['ended_rate']:>8.0%}{f2(r['pick_a_rate']):>7}")
    base = next((r for r in rows if r["run"] == cfg.baseline.name), None)
    cand = [r for r in rows if r is not base and r["flops"] == r["flops"]]
    from zoo.plot import plt  # lazy: importing matplotlib is slow and grading needs none of it

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color, label in (("in", "tab:blue", "TinyStories openings"), ("ood", "tab:orange", "other sources")):
        pts = [(r["flops"], r[f"score_{name}"], r[f"score_{name}_se"]) for r in cand if r[f"score_{name}"] == r[f"score_{name}"]]
        if pts:
            x, y, e = zip(*pts)
            ax0.errorbar(x, y, yerr=e, marker="o", color=color, label=f"pretrain, {label}")
        if base and base[f"score_{name}"] == base[f"score_{name}"]:
            ax0.axhline(base[f"score_{name}"], color=color, ls="--", alpha=0.7, label=f"{base['run']}, {label}")
        pts = [(r["flops"], r[f"win_{name}"], r[f"win_{name}_se"]) for r in cand if r[f"win_{name}"] == r[f"win_{name}"]]
        if pts:
            x, y, e = zip(*pts)
            ax1.errorbar(x, y, yerr=e, marker="o", color=color, label=label)
    ax0.set(xscale="log", xlabel="training FLOPs", ylabel="mean judge score (1-10)", title=f"story quality ({cfg.judge})")
    ax0.legend(fontsize=7)
    ax1.axhline(0.5, color="gray", ls="--")
    ax1.set(xscale="log", xlabel="training FLOPs", ylabel="win rate vs baseline", ylim=(0, 1), title=f"head-to-head vs {cfg.baseline.name}")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {out / (stem + '.csv')} and {out / (stem + '.png')}")


def main(cfg: Config) -> None:
    if cfg.provider != "anthropic" and cfg.judge.startswith("claude"):
        raise SystemExit(f"--provider {cfg.provider} needs --judge <a model id it accepts>" + (" or --judge default" if cfg.provider == "codex" else ""))
    if cfg.provider == "anthropic" and not cfg.judge.startswith("claude"):
        raise SystemExit(f"--judge {cfg.judge} is not a Claude model; pass --provider openai")
    cfg.baseline = cfg.baseline.resolve()  # one spelling per run, so the baseline is never listed twice or judged against itself
    runs = [Path(r).resolve() for r in cfg.runs] if cfg.runs else sorted(p.parent.resolve() for p in runs_root("pretrain").glob("*/eval_stories.jsonl"))
    if cfg.baseline not in runs:
        runs.append(cfg.baseline)
    missing = [r for r in runs if not (r / "eval_stories.jsonl").exists()]
    if missing:
        raise SystemExit("No eval_stories.jsonl in: " + ", ".join(str(m) for m in missing) + " (run sample.py --prompts first)")
    stories = {r: load_stories(r) for r in runs}
    ids = pick_ids(stories[cfg.baseline], cfg.subset, cfg.seed)
    print(f"{len(runs)} runs, {len(ids)} prompts each, judge {cfg.judge}")
    for mode in ("abs", "pairwise") if cfg.mode == "both" else (cfg.mode,):
        grade(cfg, mode, runs, stories, ids)
    if cfg.dry_run:
        return
    report(cfg, runs, stories, ids)
    if cfg.compare_judge:
        agreement(cfg, runs, stories, set(ids))


if __name__ == "__main__":
    main(tyro.cli(Config, description=__doc__))
