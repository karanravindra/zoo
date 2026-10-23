"""Evaluate a pretrained chat-base GPT.

Sections: per-source validation loss, zero-shot multiple choice, BLiMP grammar,
LAMBADA, and repetition/degeneration of sampled text. The default is a short run
for quick checks; --full runs every example.
"""

import argparse
import hashlib
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

import gigatoken as gt
import numpy as np
import torch
from data import DATA_DIR, ROLES
from datasets import get_dataset_config_names, load_dataset
from datasets.utils.logging import set_verbosity_error
from model import GPT, GPTConfig
from prepare import SOURCES, source_documents
from tokenizers import Tokenizer
from train import OUTPUT_DIR, choose_device, precision_context

# Hide "using the latest cached version" notices when datasets are loaded offline.
set_verbosity_error()

# Examples or documents per section: 0 means all.
PRESETS = {
    "short": {
        "val_docs": 50,
        "mix_windows": 50,
        "task_examples": 200,
        "blimp_per_config": 10,
        "lambada_examples": 500,
        "repetition_prompts": 2,
        "repetition_tokens": 100,
    },
    "full": {
        "val_docs": 2000,
        "mix_windows": 0,
        "task_examples": 0,
        "blimp_per_config": 1000,
        "lambada_examples": 0,
        "repetition_prompts": 20,
        "repetition_tokens": 200,
    },
}
CACHE_DOCS = 2000
TOKEN_BUDGET = 16384
SAMPLE_PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "def fibonacci(n):\n",
    "The derivative of x^2 is",
    f"{ROLES['user']}\nWhat is the boiling point of water?<|end|>\n{ROLES['assistant']}\n",
]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = gt.Tokenizer(Tokenizer.from_str(checkpoint["tokenizer"]))
    config = checkpoint["config"]
    if "n_kv_heads" not in config:
        config = {**config, "n_kv_heads": config["n_heads"]}
    model = GPT(GPTConfig(**config)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), tokenizer, checkpoint["data"]["eos_id"], checkpoint["tokenizer"]


def encode(tokenizer, text):
    return tokenizer.encode(text).tolist()


def batches(lengths, max_rows=64):
    """Group indices of similar length, capped at TOKEN_BUDGET padded tokens per batch."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    chunk = []
    for i in order:
        if chunk and (len(chunk) >= max_rows or (len(chunk) + 1) * lengths[i] > TOKEN_BUDGET):
            yield chunk
            chunk = []
        chunk.append(i)
    if chunk:
        yield chunk


def pad(sequences, device):
    batch = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids)
    return batch.to(device)


@torch.inference_mode()
def window_nll(model, windows, device):
    """Summed next-token NLL and target count over id lists (inputs are all but the last id)."""
    total, count = 0.0, 0
    for chunk in batches([len(w) for w in windows]):
        batch = pad([windows[i] for i in chunk], device)
        lengths = torch.tensor([len(windows[i]) for i in chunk], device=device)
        with precision_context(device):
            logits, _ = model(batch[:, :-1])
        logprobs = logits.float().log_softmax(-1)
        token_lp = logprobs.gather(-1, batch[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = torch.arange(batch.size(1) - 1, device=device)[None] < (lengths - 1)[:, None]
        total -= (token_lp * mask).sum().item()
        count += int(mask.sum().item())
    return total, count


def windows_of(ids, context_length):
    return [
        w
        for w in (ids[i : i + context_length + 1] for i in range(0, len(ids) - 1, context_length))
        if len(w) > 1
    ]


def source_validation_docs(data_dir, tokenizer, eos_id, tokenizer_json):
    """Token ids and byte counts of each source's held-out documents, cached on disk."""
    path = data_dir / "validation_by_source.npz"
    digest = hashlib.sha256(tokenizer_json.encode()).hexdigest()
    if path.exists():
        cache = np.load(path)
        if str(cache["tokenizer"]) == digest:
            return {
                s["name"]: (cache[f"{s['name']}.tokens"], cache[f"{s['name']}.bytes"])
                for s in SOURCES
            }
    arrays = {"tokenizer": np.array(digest)}
    docs = {}
    for source in SOURCES:
        tokens, sizes = [], []
        for text in source_documents(source, "validation", seed=0):
            tokens.extend(encode(tokenizer, text) + [eos_id])
            sizes.append(len(text.encode()))
            if len(sizes) == CACHE_DOCS:
                break
        arrays[f"{source['name']}.tokens"] = np.array(tokens, dtype=np.uint16)
        arrays[f"{source['name']}.bytes"] = np.array(sizes, dtype=np.int64)
        docs[source["name"]] = (arrays[f"{source['name']}.tokens"], arrays[f"{source['name']}.bytes"])
    np.savez(path, **arrays)
    return docs


def validation(model, tokenizer, eos_id, tokenizer_json, data_dir, preset, device):
    """Loss per source (bits per byte makes sources and tokenizers comparable) plus the packed mix."""
    context_length = model.config.context_length
    result = {"sources": {}}
    docs = source_validation_docs(data_dir, tokenizer, eos_id, tokenizer_json)
    for name, (tokens, sizes) in docs.items():
        ends = np.flatnonzero(tokens == eos_id) + 1
        starts = np.concatenate(([0], ends[:-1]))
        n_docs = min(len(sizes), preset["val_docs"] or len(sizes))
        windows = []
        for start, end in zip(starts[:n_docs], ends[:n_docs]):
            # Each document starts after an EOS, as in the packed training stream.
            windows += windows_of([eos_id, *tokens[start:end].astype(np.int64).tolist()], context_length)
        nll, count = window_nll(model, windows, device)
        result["sources"][name] = {
            "loss": nll / count,
            "perplexity": math.exp(nll / count),
            "bits_per_byte": nll / math.log(2) / int(sizes[:n_docs].sum()),
            "docs": n_docs,
            "tokens": count,
        }
    losses = [v["loss"] for v in result["sources"].values()]
    result["mean_source_loss"] = sum(losses) / len(losses)
    packed = np.memmap(data_dir / "validation.bin", dtype=np.uint16, mode="r")
    n_windows = (len(packed) - 1) // context_length
    n_windows = min(n_windows, preset["mix_windows"] or n_windows)
    windows = [
        packed[i * context_length : (i + 1) * context_length + 1].astype(np.int64).tolist()
        for i in range(n_windows)
    ]
    nll, count = window_nll(model, windows, device)
    result["mix"] = {"loss": nll / count, "perplexity": math.exp(nll / count), "tokens": count}
    return result


@torch.inference_mode()
def score_pairs(model, tokenizer, eos_id, pairs, device):
    """For (context, continuation) text pairs return (log-prob, greedy match) of the continuation."""
    limit = model.config.context_length
    encoded = []
    for context, continuation in pairs:
        # Start from EOS so the first token is predicted from a document boundary.
        context_ids = [eos_id] + (encode(tokenizer, context) if context else [])
        ending = encode(tokenizer, continuation)
        encoded.append(((context_ids + ending)[-limit:], len(ending)))
    results = [None] * len(pairs)
    for chunk in batches([len(ids) for ids, _ in encoded]):
        batch = pad([encoded[i][0] for i in chunk], device)
        lengths = torch.tensor([len(encoded[i][0]) for i in chunk], device=device)
        spans = torch.tensor([encoded[i][1] for i in chunk], device=device)
        with precision_context(device):
            logits, _ = model(batch[:, :-1])
        logprobs = logits.float().log_softmax(-1)
        targets = batch[:, 1:]
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        hit = logprobs.argmax(-1) == targets
        positions = torch.arange(targets.size(1), device=device)[None]
        mask = (positions >= (lengths - spans - 1)[:, None]) & (positions < (lengths - 1)[:, None])
        total = (token_lp * mask).sum(1).tolist()
        greedy = (hit | ~mask).all(1).tolist()
        for row, i in enumerate(chunk):
            results[i] = (total[row], greedy[row])
    return results


def choose(contexts, endings, label):
    return [(c, " " + e) for c, e in zip(contexts, endings)], label


def hellaswag(row):
    return choose([row["ctx"]] * 4, row["endings"], int(row["label"]))


def arc(row):
    texts = row["choices"]["text"]
    prompt = f"Question: {row['question']}\nAnswer:"
    return choose([prompt] * len(texts), texts, row["choices"]["label"].index(row["answerKey"]))


def piqa(row):
    prompt = f"Question: {row['goal']}\nAnswer:"
    return choose([prompt] * 2, [row["sol1"], row["sol2"]], int(row["label"]))


def openbookqa(row):
    texts = row["choices"]["text"]
    return choose([row["question_stem"]] * len(texts), texts, row["choices"]["label"].index(row["answerKey"]))


def sciq(row):
    endings = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
    prompt = f"{row['support'].strip()}\nQuestion: {row['question']}\nAnswer:"
    return choose([prompt] * 4, endings, 3)


def winogrande(row):
    # Score the text after the blank given the sentence up to it with each option filled in.
    head, tail = row["sentence"].split("_")
    pairs = [(head + row[key].strip(), " " + tail.strip()) for key in ("option1", "option2")]
    return pairs, int(row["answer"]) - 1


def copa(row):
    connector = {"cause": " because", "effect": " therefore"}[row["question"]]
    prompt = row["premise"].strip().rstrip(".") + connector
    endings = [c[0].lower() + c[1:] for c in (row["choice1"], row["choice2"])]
    return choose([prompt] * 2, endings, int(row["label"]))


TASKS = {
    "hellaswag": (lambda: load_dataset("Rowan/hellaswag", split="validation"), hellaswag),
    "arc_easy": (lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"), arc),
    "arc_challenge": (lambda: load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"), arc),
    "piqa": (lambda: load_dataset("ybisk/piqa", split="validation"), piqa),
    "openbookqa": (lambda: load_dataset("allenai/openbookqa", "main", split="test"), openbookqa),
    "sciq": (lambda: load_dataset("allenai/sciq", split="test"), sciq),
    "winogrande": (lambda: load_dataset("allenai/winogrande", "winogrande_xl", split="validation"), winogrande),
    "copa": (lambda: load_dataset("aps/super_glue", "copa", split="validation"), copa),
}
SECTIONS = ["validation", *TASKS, "blimp", "lambada", "repetition", "samples"]


def multiple_choice(model, tokenizer, eos_id, name, limit, device):
    """Zero-shot accuracy by summed (acc) and per-character (acc_norm) continuation log-prob."""
    load, build = TASKS[name]
    dataset = load()
    dataset = dataset.select(range(min(limit or len(dataset), len(dataset))))
    items = [build(row) for row in dataset]
    scores = score_pairs(model, tokenizer, eos_id, [p for pairs, _ in items for p in pairs], device)
    correct = correct_norm = offset = 0
    for pairs, label in items:
        logprobs = [s for s, _ in scores[offset : offset + len(pairs)]]
        offset += len(pairs)
        normalised = [s / max(1, len(c)) for s, (_, c) in zip(logprobs, pairs)]
        correct += int(np.argmax(logprobs) == label)
        correct_norm += int(np.argmax(normalised) == label)
    n = len(items)
    return {"acc": correct / n, "acc_norm": correct_norm / n, "n": n, "chance": 1 / len(items[0][0])}


def blimp(model, tokenizer, eos_id, per_config, device):
    """Share of minimal pairs where the grammatical sentence gets the higher log-prob."""
    pairs, terms = [], []
    for config in get_dataset_config_names("nyu-mll/blimp"):
        dataset = load_dataset("nyu-mll/blimp", config, split="train")
        for row in dataset.select(range(min(per_config, len(dataset)))):
            pairs += [("", row["sentence_good"]), ("", row["sentence_bad"])]
            terms.append(row["linguistics_term"])
    scores = [s for s, _ in score_pairs(model, tokenizer, eos_id, pairs, device)]
    wins = [scores[i] > scores[i + 1] for i in range(0, len(scores), 2)]
    by_term = defaultdict(list)
    for term, win in zip(terms, wins):
        by_term[term].append(win)
    return {
        "acc": sum(wins) / len(wins),
        "n": len(wins),
        "chance": 0.5,
        "by_term": {t: sum(w) / len(w) for t, w in sorted(by_term.items())},
    }


def lambada(model, tokenizer, eos_id, limit, device):
    """Predict the final word of a passage: greedy accuracy and per-word perplexity."""
    dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    dataset = dataset.select(range(min(limit or len(dataset), len(dataset))))
    pairs = []
    for row in dataset:
        context, word = row["text"].rsplit(" ", 1)
        pairs.append((context, " " + word))
    scores = score_pairs(model, tokenizer, eos_id, pairs, device)
    return {
        "acc": sum(g for _, g in scores) / len(scores),
        "perplexity": math.exp(-sum(s for s, _ in scores) / len(scores)),
        "n": len(scores),
    }


def distinct(ids, n):
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def repetition(model, tokenizer, eos_id, tokenizer_json, data_dir, preset, args, device):
    """Sample continuations of held-out document openings and measure repeats and stopping."""
    torch.manual_seed(args.seed)
    docs = source_validation_docs(data_dir, tokenizer, eos_id, tokenizer_json)
    prompts = []
    for name, (tokens, _) in docs.items():
        ends = np.flatnonzero(tokens == eos_id) + 1
        starts = np.concatenate(([0], ends[:-1]))
        step = max(1, len(starts) // preset["repetition_prompts"])
        for start in starts[::step][: preset["repetition_prompts"]]:
            prompts.append((name, tokens[start : start + 16].astype(np.int64).tolist()))
    rows = []
    for name, prompt in prompts:
        ids = torch.tensor([prompt], device=device)
        result = model.generate(ids, preset["repetition_tokens"], args.temperature, args.top_k, eos_id)
        generated = result[0, len(prompt) :].tolist()
        rows.append(
            {
                "source": name,
                "length": len(generated),
                "ended": generated[-1] == eos_id if generated else False,
                "distinct": {n: distinct(generated, n) for n in (1, 2, 4)},
            }
        )
    summary = {"samples": len(rows), "temperature": args.temperature, "top_k": args.top_k}
    long_rows = [r for r in rows if r["length"] >= 8]
    for n in (1, 2, 4):
        summary[f"distinct_{n}"] = sum(r["distinct"][n] for r in long_rows) / max(1, len(long_rows))
    # A sample is degenerate when over half of its 4-grams are repeats.
    summary["degenerate_rate"] = sum(r["distinct"][4] < 0.5 for r in long_rows) / max(1, len(long_rows))
    summary["eos_rate"] = sum(r["ended"] for r in rows) / len(rows)
    summary["mean_length"] = sum(r["length"] for r in rows) / len(rows)
    summary["by_source"] = {
        name: {
            "distinct_4": sum(r["distinct"][4] for r in group) / len(group),
            "eos_rate": sum(r["ended"] for r in group) / len(group),
        }
        for name in docs
        if (group := [r for r in long_rows if r["source"] == name])
    }
    return summary


def samples(model, tokenizer, eos_id, args, device):
    torch.manual_seed(args.seed)
    outputs = []
    for prompt in SAMPLE_PROMPTS:
        ids = torch.tensor([encode(tokenizer, prompt)], device=device)
        result = model.generate(ids, 150, args.temperature, args.top_k, eos_id)
        outputs.append({"prompt": prompt, "completion": tokenizer.decode(result[0].tolist()).decode()})
    return outputs


def preview(text, max_lines=8, max_chars=400):
    """Indented, trimmed view of a completion that survives runaway newlines."""
    lines = text[:max_chars].rstrip().splitlines()
    cut = len(text) > max_chars or len(lines) > max_lines
    return textwrap.indent("\n".join(lines[:max_lines]) + (" ..." if cut else ""), "    ")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained chat-base GPT")
    parser.add_argument("--full", action="store_true", help="run every example (default: short run)")
    parser.add_argument("--sections", nargs="*", default=SECTIONS, choices=SECTIONS)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, help="default: outputs/evaluation_<mode>.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mode = "full" if args.full else "short"
    preset = PRESETS[mode]
    output = args.output or OUTPUT_DIR / f"evaluation_{mode}.json"
    device = choose_device(args.device)
    model, tokenizer, eos_id, tokenizer_json = load_model(args.checkpoint, device)
    wanted = set(args.sections)
    results = {"checkpoint": str(args.checkpoint), "mode": mode}

    if "validation" in wanted:
        v = results["validation"] = validation(
            model, tokenizer, eos_id, tokenizer_json, args.data_dir, preset, device
        )
        print(f"{'validation':<24}{'loss':>7}{'ppl':>8}{'bits/byte':>11}{'docs':>7}")
        for name, s in v["sources"].items():
            print(f"{name:<24}{s['loss']:>7.3f}{s['perplexity']:>8.2f}{s['bits_per_byte']:>11.3f}{s['docs']:>7}")
        print(f"{'mean of sources':<24}{v['mean_source_loss']:>7.3f}")
        print(f"{'packed mix':<24}{v['mix']['loss']:>7.3f}{v['mix']['perplexity']:>8.2f}")

    results["tasks"] = {}
    print(f"\n{'task':<14}{'acc':>7}{'acc_norm':>10}{'chance':>8}{'n':>7}")
    for name in TASKS:
        if name in wanted:
            r = results["tasks"][name] = multiple_choice(
                model, tokenizer, eos_id, name, preset["task_examples"], device
            )
            print(f"{name:<14}{r['acc']:>7.3f}{r['acc_norm']:>10.3f}{r['chance']:>8.2f}{r['n']:>7}")

    if "blimp" in wanted:
        r = results["blimp"] = blimp(model, tokenizer, eos_id, preset["blimp_per_config"], device)
        print(f"{'blimp':<14}{r['acc']:>7.3f}{'':>10}{r['chance']:>8.2f}{r['n']:>7}")
        terms = ", ".join(f"{t} {a:.2f}" for t, a in r["by_term"].items())
        print(textwrap.fill(terms, 100, initial_indent="  by term: ", subsequent_indent="    "))
    if "lambada" in wanted:
        r = results["lambada"] = lambada(model, tokenizer, eos_id, preset["lambada_examples"], device)
        print(f"{'lambada':<14}{r['acc']:>7.3f}  word ppl {r['perplexity']:.1f}  n={r['n']}")

    if "repetition" in wanted:
        r = results["repetition"] = repetition(
            model, tokenizer, eos_id, tokenizer_json, args.data_dir, preset, args, device
        )
        print(
            f"\nrepetition over {r['samples']} samples: distinct-1 {r['distinct_1']:.2f}, "
            f"distinct-2 {r['distinct_2']:.2f}, distinct-4 {r['distinct_4']:.2f}, "
            f"degenerate {r['degenerate_rate']:.0%}, ended with EOS {r['eos_rate']:.0%}, "
            f"mean length {r['mean_length']:.0f}"
        )
    if "samples" in wanted:
        results["samples"] = samples(model, tokenizer, eos_id, args, device)
        for s in results["samples"]:
            print(f"\n--- {s['prompt']!r}\n{preview(s['completion'])}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
