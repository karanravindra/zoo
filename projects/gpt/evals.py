"""The evaluation suite: loss sets in bits per byte, and zero-shot multiple choice scored by log-likelihood.

Loss sets (windows of `block_size` tokens, exactly as `val/loss` is measured, documents joined by end-of-text):
  val        the corpus's own validation split, in full; for FineWeb-Edu also sliced by URL domain, edu score and
             document length (needs `prepare-evals`: the shards hold no metadata), and by position in the window
  wikitext   WikiText-2/103 test (the same text), detokenized
  pg19       the middle of each of the 100 PG-19 test books: book-length text
Likelihood sets (a prompt, then each answer as its continuation, all behind an end-of-text token, as a document starts):
  blimp      grammar: is the good sentence more likely than the bad one? overall, per phenomenon and per paradigm
  arc_easy, arc_challenge, mmlu (as cloze: the answer text, not its letter), hellaswag, piqa, sciq
  lambada    the last word of a passage: greedy accuracy, and perplexity of the word

Every set reports what it can of `loss` (nats per token), `bpb` (bits per UTF-8 byte: comparable across tokenizers),
`acc`. For a multiple-choice set `acc` picks the answer with the highest total log-probability and `acc_norm` the
highest per byte of answer (the lm-evaluation-harness conventions); `bpb` is that of the correct answer (it moves
smoothly while accuracy is still at chance); `chance` is the accuracy of guessing. A standard error `<acc>_se` (the
binomial one) comes with every accuracy: a model's accuracy on n items is only known to that much.

`prepare_evals` downloads what the Hub has and the HF cache lacks; `run_eval` then needs no network.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from config import ALL_TASKS, TASKS_BY_CORPUS, EvalConfig, PrepareEvalsConfig
from data import SPECIALS, corpus_dir, load_meta, load_tokenizer, val_window_ids
from pretrain import LitGPT
from tqdm import tqdm

from zoo.data.text import Shards
from zoo.training import load_module
from zoo.utils.device import get_device

LN2 = math.log(2)
PROJECT = "gpt"

# Evaluation sets that are files of a Hub dataset repo (read with pyarrow; hf_hub_download caches them in HF_HOME).
HUB_FILES = {
    "wikitext": ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1/wikitext-2-raw-v1-test.parquet"),  # 60 articles; WikiText-103's test is the same text
    "pg19": ("emozilla/pg19-test", "data/test-00000-of-00001-29a571947c0b5ccc.parquet"),  # the 100 test books
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet"),  # 14,042 questions in 57 subjects
}
# Evaluation sets that load through `datasets`: (repo, config, split).
HF_DATASETS = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "piqa": ("baber/piqa", None, "validation"),
    "sciq": ("allenai/sciq", None, "test"),
    "lambada": ("EleutherAI/lambada_openai", "en", "test"),
}
BLIMP = """adjunct_island anaphor_gender_agreement anaphor_number_agreement animate_subject_passive animate_subject_trans causative
complex_NP_island coordinate_structure_constraint_complex_left_branch coordinate_structure_constraint_object_extraction
determiner_noun_agreement_1 determiner_noun_agreement_2 determiner_noun_agreement_irregular_1 determiner_noun_agreement_irregular_2
determiner_noun_agreement_with_adj_2 determiner_noun_agreement_with_adjective_1 determiner_noun_agreement_with_adj_irregular_1
determiner_noun_agreement_with_adj_irregular_2 distractor_agreement_relational_noun distractor_agreement_relative_clause drop_argument
ellipsis_n_bar_1 ellipsis_n_bar_2 existential_there_object_raising existential_there_quantifiers_1 existential_there_quantifiers_2
existential_there_subject_raising expletive_it_object_raising inchoative intransitive irregular_past_participle_adjectives
irregular_past_participle_verbs irregular_plural_subject_verb_agreement_1 irregular_plural_subject_verb_agreement_2
left_branch_island_echo_question left_branch_island_simple_question matrix_question_npi_licensor_present npi_present_1 npi_present_2
only_npi_licensor_present only_npi_scope passive_1 passive_2 principle_A_case_1 principle_A_case_2 principle_A_c_command
principle_A_domain_1 principle_A_domain_2 principle_A_domain_3 principle_A_reconstruction regular_plural_subject_verb_agreement_1
regular_plural_subject_verb_agreement_2 sentential_negation_npi_licensor_present sentential_negation_npi_scope sentential_subject_island
superlative_quantifiers_1 superlative_quantifiers_2 tough_vs_raising_1 tough_vs_raising_2 transitive wh_island wh_questions_object_gap
wh_questions_subject_gap wh_questions_subject_gap_long_distance wh_vs_that_no_gap wh_vs_that_no_gap_long_distance wh_vs_that_with_gap
wh_vs_that_with_gap_long_distance""".split()  # the 67 paradigms, 1,000 pairs each, in 12 phenomena

LENGTH_EDGES = (256, 512, 1024, 2048)  # document lengths (tokens) that the validation slices split at
LENGTH_NAMES = ("lt256", "256-512", "512-1k", "1k-2k", "ge2k")
DOMAIN_NAMES = ("wikipedia", "edu", "gov", "org", "com", "net", "other")


# --- fetching ---------------------------------------------------------------------------------------------------------


def hub_file(task: str, *, download: bool = False) -> Path:
    """The local path of a Hub file of `HUB_FILES` (downloaded to the HF cache first when `download`)."""
    from huggingface_hub import hf_hub_download

    repo, name = HUB_FILES[task]
    try:
        return Path(hf_hub_download(repo, name, repo_type="dataset", local_files_only=not download))
    except Exception as e:  # not cached: LocalEntryNotFoundError, whose module has moved between versions
        if download:
            raise
        raise SystemExit(f"{task} ({repo}) is not in the HF cache: run `uv run projects/gpt/cli.py prepare-evals` first ({type(e).__name__})") from None


def read_parquet(path: Path) -> dict[str, list]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pydict()


def hf_dataset(task: str, config: str | None = None):
    from datasets import load_dataset

    repo, cfg, split = HF_DATASETS[task] if task in HF_DATASETS else ("nyu-mll/blimp", config, "train")
    return load_dataset(repo, cfg, split=split)


def domain_group(netloc: str) -> str:
    """A coarse source of a web document, from its host."""
    host = netloc.lower().split(":")[0].removeprefix("www.")
    if host.endswith("wikipedia.org"):
        return "wikipedia"
    if host.endswith(".edu") or ".edu." in host or ".ac." in host:
        return "edu"
    if host.endswith(".gov") or ".gov." in host:
        return "gov"
    return next((t for t in ("org", "com", "net") if host.endswith("." + t)), "other")


def prepare_evals(cfg: PrepareEvalsConfig) -> None:
    from datasets import load_dataset
    from prepare import SOURCES

    for task in HUB_FILES:
        print(f"{task}: {hub_file(task, download=True)}")
    for task in HF_DATASETS:
        ds = hf_dataset(task)
        print(f"{task}: {len(ds):,} items")
    n = sum(len(hf_dataset("blimp", name)) for name in BLIMP)
    print(f"blimp: {n:,} pairs in {len(BLIMP)} paradigms")
    if not cfg.dataset:
        return

    # The validation documents of a corpus whose val split was held out of the training stream: the first `val_docs`
    # documents of it, filtered as `prepare` filters them. Their order is the order of the end-of-text delimiters in
    # validation/shard_*.npy, and their number of documents and bytes must equal what `prepare` recorded.
    source, meta = SOURCES[cfg.dataset], load_meta(cfg.dataset, "validation")
    if source.val_split is not None:
        print(f"{cfg.dataset}: its validation split is a split of the source, which has no URLs to slice by")
        return
    docs: dict[str, list] = {"domain": [], "score": [], "int_score": [], "bytes": []}
    stream = load_dataset(source.hub, source.name, split="train", streaming=True)
    for row in tqdm(stream, total=source.val_docs, desc="validation documents", disable=None):
        text = row[source.text]
        for s in SPECIALS:
            text = text.replace(s, "")
        if not text.strip():
            continue
        docs["domain"].append(urlparse(row.get("url") or "").netloc.lower().removeprefix("www."))
        docs["score"].append(float(row.get("score", 0.0)))
        docs["int_score"].append(int(row.get("int_score", 0)))
        docs["bytes"].append(len(text.encode("utf-8")))
        if len(docs["bytes"]) == source.val_docs:
            break
    if len(docs["bytes"]) != meta["num_docs"] or sum(docs["bytes"]) != meta["num_bytes"]:
        raise SystemExit(f"the first {source.val_docs} documents of {source.hub} are not the corpus's validation split: {len(docs['bytes'])} documents, {sum(docs['bytes']):,} bytes against {meta['num_docs']} and {meta['num_bytes']:,}")
    path = corpus_dir(cfg.dataset) / "validation" / "docs.json"
    path.write_text(json.dumps(docs))
    groups = defaultdict(int)
    for d in docs["domain"]:
        groups[domain_group(d)] += 1
    print(f"wrote {path}: {len(docs['bytes']):,} documents; by domain {dict(groups)}; by edu score {dict(sorted(defaultdict(int, {s: docs['int_score'].count(s) for s in set(docs['int_score'])}).items()))}")


# --- scoring ----------------------------------------------------------------------------------------------------------


class Scorer:
    """Runs a pretrained `LitGPT` over token streams and (context, continuation) pairs."""

    def __init__(self, module, batch_tokens: int) -> None:
        self.net, self.cfg = module.net.eval(), module.cfg
        self.buckets = module.position_buckets()
        self.tok = load_tokenizer(self.cfg.dataset)
        self.device = next(module.net.parameters()).device
        self.bf16 = self.device.type == "cuda" and "bf16" in self.cfg.precision
        self.eot, self.bs, self.batch_tokens = self.cfg.eot_id, self.cfg.block_size, batch_tokens
        self.rows = max(256, (1 << 26) // self.cfg.vocab_size)  # rows of logits (float32) held at a time: 256 MB

    def encode(self, text: str) -> list[int]:
        for s in SPECIALS:  # as `prepare` does, so that the text of a corpus never holds a special token
            text = text.replace(s, "")
        return np.asarray(self.tok.encode(text)).tolist()

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.bf16):
            return self.net(ids).float()

    def nll_hits(self, h: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy (nats) and top-1 correctness of the next-token predictions of hidden states `h` (N, dim) for targets `y` (N,)."""
        w = self.net.classifier_weight
        nll = torch.empty(len(y), device=h.device)
        hit = torch.empty(len(y), dtype=torch.bool, device=h.device)
        for a in range(0, len(y), self.rows):
            logits = F.linear(h[a : a + self.rows], w).float()
            t = y[a : a + self.rows]
            nll[a : a + self.rows] = torch.logsumexp(logits, -1) - logits.gather(1, t[:, None])[:, 0]
            hit[a : a + self.rows] = logits.argmax(-1) == t
        return nll, hit

    @torch.inference_mode()
    def stream(self, tokens: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-token cross-entropy and top-1 correctness over consecutive windows of block_size + 1 tokens (neighbours share
        one token, so every token but the first is a target once, as in training): (n * block_size,) arrays, target i is token i + 1."""
        bs = self.bs
        n = (len(tokens) - 1) // bs
        if n < 1:
            raise SystemExit(f"{label}: {len(tokens)} tokens are fewer than one window of {bs + 1}")
        per = max(1, self.batch_tokens // bs)
        nll, hit = np.empty(n * bs, np.float32), np.empty(n * bs, bool)
        for a in tqdm(range(0, n, per), desc=label, leave=False, disable=None):
            b = min(a + per, n)
            w = torch.from_numpy(np.stack([tokens[i * bs : i * bs + bs + 1] for i in range(a, b)]).astype(np.int64)).to(self.device)
            h = self.hidden(w[:, :-1])
            l, k = self.nll_hits(h.flatten(0, 1), w[:, 1:].flatten())
            nll[a * bs : b * bs], hit[a * bs : b * bs] = l.cpu().numpy(), k.cpu().numpy()
        return nll, hit

    @torch.inference_mode()
    def score(self, ctxs: list[list[int]], conts: list[list[int]], label: str) -> tuple[np.ndarray, np.ndarray]:
        """`log p(continuation | context)` summed over the continuation's tokens, and whether greedy decoding from the
        context reproduces the continuation exactly, for every pair. A pair longer than block_size loses its context's start."""
        n = len(conts)
        seqs = [(c + k)[-self.bs :] for c, k in zip(ctxs, conts)]
        ncont = np.array([len(k) for k in conts])
        assert (ncont < np.array([len(s) for s in seqs])).all(), "a continuation fills the whole window"
        order = np.argsort([-len(s) for s in seqs], kind="stable")  # longest first: batches of similar length, so little padding
        lp, ok = np.zeros(n), np.zeros(n, bool)
        i = 0
        with tqdm(total=n, desc=label, leave=False, disable=None) as bar:
            while i < n:
                t = len(seqs[order[i]])
                idx = order[i : i + max(1, self.batch_tokens // t)]
                ids = torch.zeros(len(idx), t, dtype=torch.long)
                for r, j in enumerate(idx):
                    ids[r, : len(seqs[j])] = torch.tensor(seqs[j])
                ids = ids.to(self.device)
                h = self.hidden(ids)
                rows, cols = [], []
                for r, j in enumerate(idx):
                    length, c = len(seqs[j]), ncont[j]
                    rows += [r] * c  # the continuation's tokens are at length - c .. length - 1, predicted from the position before
                    cols += range(length - c - 1, length - 1)
                rows, cols = torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device)
                nll, hit = self.nll_hits(h[rows, cols], ids[rows, cols + 1])
                seg = np.repeat(np.arange(len(idx)), ncont[idx])
                lp[idx] = -np.bincount(seg, weights=nll.cpu().numpy(), minlength=len(idx))
                ok[idx] = np.bincount(seg, weights=(~hit).cpu().numpy().astype(float), minlength=len(idx)) == 0
                i += len(idx)
                bar.update(len(idx))
        return lp, ok


def se(acc: float, n: int) -> float:
    """The binomial standard error of an accuracy measured on n items."""
    return math.sqrt(acc * (1 - acc) / n)


# --- loss sets --------------------------------------------------------------------------------------------------------


def stream_report(sc: Scorer, nll: np.ndarray, hit: np.ndarray, tokens_per_byte: float) -> dict:
    """loss, bpb and top-1 accuracy over the whole stream, and by position in the window."""
    bs, n = sc.bs, len(nll) // sc.bs
    loss = float(nll.mean())
    report = {"tokens": len(nll), "loss": loss, "bpb": loss / LN2 * tokens_per_byte, "acc": float(hit.mean()), "pos": {}}
    m, k = nll.reshape(n, bs), hit.reshape(n, bs)
    for a, b in sc.buckets:
        pos_loss = float(m[:, a:b].mean())
        report["pos"][f"{a}-{b}"] = {"loss": pos_loss, "bpb": pos_loss / LN2 * tokens_per_byte, "acc": float(k[:, a:b].mean())}
    return report


def slice_report(nll: np.ndarray, hit: np.ndarray, tokens: np.ndarray, eot: int, docs: dict | None) -> dict:
    """The loss by slices of the validation documents: their length, and with `docs` (validation/docs.json) their domain and edu score.

    A document's tokens are its text and the end-of-text after it (the target the loss is also charged for), so the
    bytes of its text carry the delimiter's cost, as in `val/bpb`. The first document (whose first token is no target)
    and the last, if the last window cuts it, are left out."""
    n = len(nll)
    ends = np.flatnonzero(tokens[: n + 1] == eot)  # position of each document's end-of-text
    last = len(ends) - 1  # documents 1..last are whole
    pos = np.arange(1, n + 1)
    doc_of = np.searchsorted(ends, pos, side="left")  # the number of end-of-texts before a target: its document
    keep = (doc_of >= 1) & (doc_of <= last)
    length = np.diff(np.concatenate([[-1], ends]))  # tokens of each document, its end-of-text included
    by: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {"length": (np.digitize(length, LENGTH_EDGES), LENGTH_NAMES)}
    nbytes = None
    if docs is not None:
        if len(docs["bytes"]) <= last:
            raise SystemExit(f"validation/docs.json has {len(docs['bytes'])} documents but the validation tokens have {last + 1}: run `prepare-evals` again")
        nbytes = np.array(docs["bytes"][: last + 1], dtype=float)
        by["domain"] = (np.array([DOMAIN_NAMES.index(domain_group(d)) for d in docs["domain"][: last + 1]]), DOMAIN_NAMES)
        scores = sorted(set(docs["int_score"]))
        by["edu"] = (np.array([scores.index(s) for s in docs["int_score"][: last + 1]]), tuple(f"edu{s}" for s in scores))
    out: dict[str, dict] = {}
    for key, (cat_doc, names) in by.items():
        cat_tok = cat_doc[doc_of[keep]]
        k = len(names)
        toks = np.bincount(cat_tok, minlength=k)
        nll_sum = np.bincount(cat_tok, weights=nll[keep], minlength=k)
        hits = np.bincount(cat_tok, weights=hit[keep].astype(float), minlength=k)
        doc_n = np.bincount(cat_doc[1 : last + 1], minlength=k)
        doc_bytes = np.bincount(cat_doc[1 : last + 1], weights=nbytes[1:] if nbytes is not None else None, minlength=k) if nbytes is not None else None
        out[key] = {}
        for i, name in enumerate(names):
            if toks[i] == 0:
                continue
            row = {"docs": int(doc_n[i]), "tokens": int(toks[i]), "loss": float(nll_sum[i] / toks[i]), "acc": float(hits[i] / toks[i])}
            if doc_bytes is not None:
                row["bpb"] = float(nll_sum[i] / LN2 / doc_bytes[i])
            out[key][name] = row
    return out


def task_val(sc: Scorer, cfg: EvalConfig) -> dict:
    dataset = sc.cfg.dataset
    meta = load_meta(dataset, "validation")
    tokens = np.asarray(Shards(corpus_dir(dataset) / "validation")[0 : meta["length"]])
    if cfg.limit_tokens:
        tokens = tokens[: cfg.limit_tokens + 1]
    nll, hit = sc.stream(tokens, "val")
    report = stream_report(sc, nll, hit, meta["length"] / meta["num_bytes"])
    if not cfg.limit_tokens:  # the windows that training's val/loss averages: the check that this is the same measurement
        report["loss_2m"] = float(nll.reshape(-1, sc.bs)[val_window_ids(len(nll) // sc.bs, sc.bs, sc.cfg.val_tokens)].mean())
    docs_path = corpus_dir(dataset) / "validation" / "docs.json"
    report["slices"] = slice_report(nll, hit, tokens, sc.eot, json.loads(docs_path.read_text()) if docs_path.exists() else None)
    return report


def text_stream(sc: Scorer, docs: list[str], limit_tokens: int) -> tuple[np.ndarray, float, float]:
    """The documents as one token stream (an end-of-text after each, as in the corpus shards), its tokens per byte, and
    the share of the documents' bytes it covers (1 unless `limit_tokens` cuts it)."""
    ids: list[int] = []
    for d in docs:
        ids += sc.encode(d) + [sc.eot]
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    tokens = np.array(ids, dtype=np.int64)
    share = 1.0
    if limit_tokens:
        tokens = tokens[: limit_tokens + 1]
        share = len(tokens) / len(ids)
    return tokens, len(ids) / n_bytes, share


def wikitext_detokenizer(s: str) -> str:
    """Undoes WikiText's tokenization (spaces around punctuation, " @-@ ") as GPT-2 and lm-evaluation-harness do, so the text is prose again."""
    s = s.replace("s '", "s'")
    s = re.sub(r"/' [0-9]/", r"/'[0-9]/", s)
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for p in (":", ";", ".", "!", "?", ","):
        s = s.replace(f" {p} ", f"{p} ")
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", s)
    s = s.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    s = s.replace(" " + chr(176) + " ", chr(176)).replace(" \n", "\n").replace("\n ", "\n").replace(" N ", " 1 ").replace(" 's", "'s")
    return s


def task_wikitext(sc: Scorer, cfg: EvalConfig) -> dict:
    docs = [wikitext_detokenizer(p) for p in read_parquet(hub_file("wikitext"))["page"]]
    tokens, tpb, share = text_stream(sc, docs, cfg.limit_tokens)
    nll, hit = sc.stream(tokens, "wikitext")
    report = stream_report(sc, nll, hit, tpb)
    n_words = sum(len(d.split()) for d in docs) * share
    report["word_ppl"] = math.exp(report["loss"] * len(tokens) / n_words)  # per whitespace-separated word, the number papers quote for WikiText
    return report


def task_pg19(sc: Scorer, cfg: EvalConfig) -> dict:
    docs = []
    for text in read_parquet(hub_file("pg19"))["text"]:
        if len(text) > cfg.pg19_chars:
            a = (len(text) - cfg.pg19_chars) // 2
            a = text.find(" ", a) + 1 or a  # start at a word
            text = text[a : a + cfg.pg19_chars]
        docs.append(text)
    tokens, tpb, _ = text_stream(sc, docs, cfg.limit_tokens)
    nll, hit = sc.stream(tokens, "pg19")
    return stream_report(sc, nll, hit, tpb)


# --- likelihood sets --------------------------------------------------------------------------------------------------


def hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def load_mc(task: str, limit: int) -> list[tuple[str, list[str], int, str]]:
    """(prompt, answers, index of the correct answer, group) per item, in the lm-evaluation-harness formats."""
    if task == "mmlu":
        d = read_parquet(hub_file("mmlu"))
        items = [(f"Question: {q}\nAnswer:", list(c), a, s) for q, c, a, s in zip(d["question"], d["choices"], d["answer"], d["subject"])]
    else:
        ds = hf_dataset(task)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        if task in ("arc_easy", "arc_challenge"):
            items = [(f"Question: {r['question']}\nAnswer:", r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]), "") for r in ds]
        elif task == "hellaswag":
            items = [(hellaswag_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()), [hellaswag_clean(e) for e in r["endings"]], int(r["label"]), "") for r in ds]
        elif task == "piqa":
            items = [(f"Question: {r['goal']}\nAnswer:", [r["sol1"], r["sol2"]], int(r["label"]), "") for r in ds]
        elif task == "sciq":
            items = [(f"{r['support'].lstrip()}\nQuestion: {r['question']}\nAnswer:".strip(), [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]], 3, "") for r in ds]
        else:
            raise ValueError(task)
    return items[:limit] if limit else items


def task_mc(sc: Scorer, task: str, cfg: EvalConfig) -> dict:
    items = load_mc(task, cfg.limit)
    ctxs, conts, nbytes = [], [], []
    for prompt, answers, _, _ in items:
        c = [sc.eot] + sc.encode(prompt)
        for a in answers:
            ctxs.append(c)
            conts.append(sc.encode(" " + a))
            nbytes.append(len((" " + a).encode("utf-8")))
    lp, _ = sc.score(ctxs, conts, task)
    nbytes = np.array(nbytes, dtype=float)
    hits, hits_norm, gold_nll, gold_bytes, chance = [], [], 0.0, 0.0, 0.0
    by_group: dict[str, list[bool]] = defaultdict(list)
    start = 0
    for _, answers, gold, group in items:
        k = len(answers)
        s, b = lp[start : start + k], nbytes[start : start + k]
        hits.append(int(np.argmax(s)) == gold)
        hits_norm.append(int(np.argmax(s / b)) == gold)
        gold_nll, gold_bytes, chance = gold_nll - s[gold], gold_bytes + b[gold], chance + 1 / k
        by_group[group].append(hits[-1])
        start += k
    n = len(items)
    acc, acc_norm = float(np.mean(hits)), float(np.mean(hits_norm))
    report = {"n": n, "chance": chance / n, "acc": acc, "acc_se": se(acc, n), "acc_norm": acc_norm, "acc_norm_se": se(acc_norm, n), "bpb": float(gold_nll / LN2 / gold_bytes)}
    if len(by_group) > 1:  # mmlu: the 57 subjects
        report["acc_macro"] = float(np.mean([np.mean(v) for v in by_group.values()]))
        report["subjects"] = {g: {"n": len(v), "acc": float(np.mean(v))} for g, v in sorted(by_group.items())}
    return report


def task_blimp(sc: Scorer, cfg: EvalConfig) -> dict:
    pairs: list[tuple[str, str, str, str]] = []
    for name in tqdm(BLIMP, desc="blimp: load", leave=False, disable=None):
        ds = hf_dataset("blimp", name)
        ds = ds.select(range(min(cfg.limit, len(ds)))) if cfg.limit else ds
        pairs += [(r["sentence_good"], r["sentence_bad"], r["linguistics_term"], r["UID"]) for r in ds]
    ctxs, conts = [[sc.eot]] * (2 * len(pairs)), []
    for good, bad, _, _ in pairs:
        conts += [sc.encode(good), sc.encode(bad)]
    lp, _ = sc.score(ctxs, conts, "blimp")
    ok = lp[0::2] > lp[1::2]  # a tie is a miss
    groups: dict[str, dict[str, list[bool]]] = {"terms": defaultdict(list), "paradigms": defaultdict(list)}
    for (_, _, term, uid), o in zip(pairs, ok):
        groups["terms"][term].append(bool(o))
        groups["paradigms"][uid].append(bool(o))
    acc = float(ok.mean())
    report = {"n": len(pairs), "chance": 0.5, "acc": acc, "acc_se": se(acc, len(pairs))}
    for key, g in groups.items():
        report[key] = {name: {"n": len(v), "acc": float(np.mean(v))} for name, v in sorted(g.items())}
    return report


def task_lambada(sc: Scorer, cfg: EvalConfig) -> dict:
    ds = hf_dataset("lambada")
    texts = ds["text"][: cfg.limit] if cfg.limit else ds["text"]
    ctxs, conts = [], []
    for t in texts:
        head, _, word = t.rpartition(" ")
        ctxs.append([sc.eot] + sc.encode(head))
        conts.append(sc.encode(" " + word))
    lp, ok = sc.score(ctxs, conts, "lambada")
    acc = float(ok.mean())
    return {"n": len(texts), "acc": acc, "acc_se": se(acc, len(texts)), "ppl": float(np.exp(-lp.mean())), "loss": float(-lp.mean())}


# --- running ----------------------------------------------------------------------------------------------------------


def run_task(sc: Scorer, task: str, cfg: EvalConfig) -> dict:
    match task:
        case "val":
            return task_val(sc, cfg)
        case "wikitext":
            return task_wikitext(sc, cfg)
        case "pg19":
            return task_pg19(sc, cfg)
        case "blimp":
            return task_blimp(sc, cfg)
        case "lambada":
            return task_lambada(sc, cfg)
        case _:
            return task_mc(sc, task, cfg)


def flatten(prefix: str, value, out: dict[str, float]) -> dict[str, float]:
    """eval/<task>/<metric>/... for the numbers of a nested result (the 67 blimp paradigms and 57 mmlu subjects stay in eval.json)."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in ("paradigms", "subjects"):
                flatten(f"{prefix}/{k}", v, out)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)
    return out


def summary_line(task: str, r: dict) -> str:
    parts = []
    for key, fmt in (("bpb", "{:.4f}"), ("loss", "{:.4f}"), ("word_ppl", "{:.1f}"), ("ppl", "{:.1f}")):
        if key in r:
            parts.append(f"{key} {fmt.format(r[key])}")
    for key in ("acc", "acc_norm"):
        if key in r:
            parts.append(f"{key} {r[key]:.4f}" + (f" ± {r[key + '_se']:.4f}" if key + "_se" in r else ""))
    if "chance" in r:
        parts.append(f"(chance {r['chance']:.2f}, n {r['n']:,})")
    elif "tokens" in r:
        parts.append(f"({r['tokens']:,} tokens)")
    return f"  {task:<14}" + "  ".join(parts)


def resolve_ckpt(path: Path) -> Path:
    return path / "checkpoints" / "last.ckpt" if path.is_dir() else path


def run_eval(cfg: EvalConfig) -> None:
    if not cfg.ckpts:
        raise SystemExit("give at least one run directory or checkpoint: `eval runs/gpt/<run id>`")
    device = get_device()
    for path in map(resolve_ckpt, cfg.ckpts):
        module = load_module(LitGPT, path, map_location=device).to(device)
        mc = module.cfg
        tasks = cfg.tasks or TASKS_BY_CORPUS.get(mc.dataset, ALL_TASKS)
        run_dir = path.parent.parent
        print(f"{run_dir.name}: {mc.n_layers} layers x {mc.dim}, {mc.n_params / 1e6:.2f}M parameters ({mc.n_params_body / 1e6:.2f}M outside the embedding), {module.tokens_seen:,} tokens, {mc.dataset}; tasks {', '.join(tasks)}")
        sc = Scorer(module, cfg.batch_tokens)
        results = {}
        for task in tasks:
            results[task] = run_task(sc, task, cfg)
            print(summary_line(task, results[task]), flush=True)
        # eval.json keeps the tasks of earlier calls, so the suite can be filled in a task at a time
        out = run_dir / "eval.json"
        old = json.loads(out.read_text()) if out.exists() else {}
        info = {
            "ckpt": str(path), "dataset": mc.dataset, "n_params": mc.n_params, "n_params_body": mc.n_params_body,
            "tokens_seen": module.tokens_seen, "flops": module.tokens_seen * mc.train_flops_per_token,
        }  # fmt: skip
        out.write_text(json.dumps({**info, "tasks": {**old.get("tasks", {}), **results}}, indent=1))
        print(f"  wrote {out}")
        if cfg.wandb:
            log_wandb(run_dir, flatten("eval", results, {}))
        del module, sc
        torch.cuda.empty_cache()


def wandb_run_id(run_dir: Path) -> str:
    """The W&B id of a run: the one in its `wandb/run-<time>-<id>` folder. A sweep agent chooses the id itself, so it is not always the run directory's name."""
    folders = sorted((run_dir / "wandb").glob("run-*-*"))
    return folders[-1].name.rsplit("-", 1)[1] if folders else run_dir.name


def log_wandb(run_dir: Path, values: dict[str, float]) -> None:
    """Adds the numbers to the summary of the run's W&B run. A failure here loses nothing: eval.json is already written."""
    try:
        import wandb

        run_id = wandb_run_id(run_dir)
        run = wandb.init(project=PROJECT, id=run_id, resume="allow", dir=str(run_dir), settings=wandb.Settings(silent=True))
        run.summary.update(values)
        run.finish()
        print(f"  logged {len(values)} values to W&B run {run_id}")
    except Exception as e:
        print(f"  W&B logging failed ({type(e).__name__}: {e}); the numbers are in eval.json")
