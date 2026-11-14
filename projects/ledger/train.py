"""LoRA (optionally QLoRA) fine-tune of Qwen3.5-0.8B on tasksource extraction data (JSON answers).

First run: `uv run projects/ledger/train.py --max-steps 20`, then check peak memory and preview.jsonl.
Data check only (tokenizer, no model): `uv run projects/ledger/train.py --data-only`

Each run writes to `<repo>/runs/ledger/train/<timestamp>/`:
  config.json      config, git commit, device, package versions, model/dataset revisions
  sample_ids.json  train/validation example ids used, and the ids rejected as over-length
  metrics.csv      one row per Trainer log (train loss, lr, grad norm, eval loss)
  checkpoints/     Trainer checkpoints (adapter + optimizer state), last `save_total_limit` kept
  adapter/         final LoRA adapter + processor + ledger.json (base model id and revision)
  preview.jsonl    greedy predictions and scores on the first `n_preview` validation examples
  results.json     final validation loss, peak GPU memory, rejection counts, preview score
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from transformers import Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONFIGS,
    MODEL,
    PROCEDURAL,
    SYNTHETIC,
    build_messages,
    encode_example,
    hub_sha,
    load_extraction,
    new_run_dir,
    parse_json_object,
)
from model import ADAPTER_META, generate, load_model, load_processor  # noqa: E402
from scoring import score_extraction  # noqa: E402

from zoo.utils.device import seed_everything  # noqa: E402

# Full-attention layers: q/k/v/o_proj. Gated-DeltaNet (linear attention) layers: in_proj_qkv, in_proj_z, out_proj
# (the tiny per-head in_proj_a/b gates stay frozen). Every layer: gate/up/down_proj.
LORA_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"}
METRIC_FIELDS = ["step", "epoch", "loss", "grad_norm", "learning_rate", "eval_loss", "eval_runtime"]


@dataclass
class TrainConfig:
    model: str = MODEL  # base model on the Hub
    configs: tuple[str, ...] = CONFIGS  # procedural-extraction configs to train on
    train_per_config: int = 1000  # train examples sampled per procedural config
    val_per_config: int = 100  # validation examples sampled per procedural config
    train_synthetic: int = 2000  # train examples sampled from synthetic-extraction
    val_synthetic: int = 200  # validation examples sampled from synthetic-extraction
    max_len: int = 2048  # token cap; longer examples are rejected (and counted), never truncated
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 8  # per-device train batch size
    eval_batch_size: int = 8
    grad_accum: int = 2  # gradient accumulation steps (effective batch = batch_size * grad_accum)
    quantize: bool = False  # QLoRA: 4-bit NF4 base weights (saves little on a 0.8B model and slows every matmul)
    gradient_checkpointing: bool = True  # recompute activations in backward to save memory
    group_by_length: bool = True  # batch examples of similar length together (less padding)
    num_workers: int = 4  # dataloader workers running the collator (chat template + tokenization)
    lr: float = 1e-4
    epochs: float = 1.0
    max_steps: int = -1  # optimizer steps; overrides epochs when > 0 (use 20 for a first run)
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    logging_steps: int = 10
    seed: int = 42
    n_preview: int = 8  # validation examples to generate answers for after training
    max_new_tokens: int = 512  # generation cap for the preview
    data_only: bool = False  # only build the datasets and report token lengths / rejections, no model
    compile: bool = False  # model.compile(mode="reduce-overhead", dynamic=False)


def build_split(split: str, cfg: TrainConfig, revisions: dict[str, str]):
    """Equal seeded samples per procedural config plus synthetic, within official split boundaries."""
    from datasets import concatenate_datasets

    train = split == "train"
    parts = [
        load_extraction(PROCEDURAL, c, split, cfg.train_per_config if train else cfg.val_per_config, cfg.seed, revisions[PROCEDURAL])
        for c in cfg.configs
    ]
    parts.append(load_extraction(SYNTHETIC, None, split, cfg.train_synthetic if train else cfg.val_synthetic, cfg.seed, revisions[SYNTHETIC]))
    return concatenate_datasets(parts).shuffle(seed=cfg.seed)


def row_messages(row: dict) -> list[dict]:
    return build_messages(row["state"], json.loads(row["fields_json"]), json.loads(row["target_json"]), has_image=row.get("image") is not None)


def filter_length(ds, processor, max_len: int, split: str):
    """Drop examples over `max_len` tokens. Returns (kept dataset, rejection report, rejected ids)."""
    ds = ds.map(lambda r: {"n_tokens": len(encode_example(processor, row_messages(r))["input_ids"])})
    lengths, sources, ids = ds["n_tokens"], ds["source"], ds["id"]
    total, rejected = Counter(sources), Counter(s for s, n in zip(sources, lengths) if n > max_len)
    report = {
        s: {"n": total[s], "rejected": rejected[s], "rate": rejected[s] / total[s], "max_tokens": max(n for s2, n in zip(sources, lengths) if s2 == s)}
        for s in sorted(total)
    }
    print(f"\n{split}: token lengths and rejections (> {max_len})")
    for s, r in report.items():
        print(f"  {s:45s} n={r['n']:5d} rejected={r['rejected']:4d} ({r['rate']:6.1%}) max={r['max_tokens']}")
    rejected_ids = [i for i, n in zip(ids, lengths) if n > max_len]
    return ds.filter(lambda r: r["n_tokens"] <= max_len), report, rejected_ids


class Collator:
    """Encodes rows with the chat template and left-pads; labels keep only the assistant answer and <|im_end|>.

    Left padding puts every answer at the end of its row, so the batch also carries `logits_to_keep`: the model
    computes logits only for that tail instead of the whole sequence (the 248k-token vocab makes full logits the
    biggest cost of a step for a 0.8B model)."""

    def __init__(self, processor, max_len: int):
        self.processor, self.max_len = processor, max_len
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        encs = []
        for row in rows:
            image = row.get("image")
            enc = encode_example(self.processor, row_messages(row), [image] if image is not None else None)
            if len(enc["input_ids"]) > self.max_len:  # filter_length should have removed these
                raise ValueError(f"{row['id']}: {len(enc['input_ids'])} tokens > max_len {self.max_len}")
            encs.append(enc)
        n = max(len(e["input_ids"]) for e in encs)
        batch = {
            "input_ids": torch.full((len(encs), n), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(encs), n), dtype=torch.long),
            "labels": torch.full((len(encs), n), -100, dtype=torch.long),
        }
        for i, e in enumerate(encs):  # left padding
            k = len(e["input_ids"])
            batch["input_ids"][i, n - k :] = e["input_ids"]
            batch["attention_mask"][i, n - k :] = e["attention_mask"]
            batch["labels"][i, n - k :] = e["labels"]
        # Logits for the last `keep` positions cover every supervised token's prediction (made one position earlier).
        first = (batch["labels"] != -100).int().argmax(1)
        batch["logits_to_keep"] = int((n - first).max()) + 1
        with_images = [e for e in encs if "pixel_values" in e]
        if with_images:
            batch["pixel_values"] = torch.cat([e["pixel_values"] for e in with_images])
            batch["image_grid_thw"] = torch.cat([e["image_grid_thw"] for e in with_images])
        return batch


def tail_loss(outputs, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
    """Next-token cross-entropy on the supervised tail, with logits for only the last `keep` positions.

    Summed and divided by `num_items_in_batch` (supervised tokens in the whole accumulated batch) in training, so
    gradient accumulation weighs every token equally; a per-batch token mean in evaluation."""
    logits = outputs.logits[:, :-1].float()
    targets = labels[:, labels.shape[1] - logits.shape[1] :].to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction="sum")
    if num_items_in_batch is None:
        return loss / (targets != -100).sum()
    return loss / num_items_in_batch


class CsvLogger(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        with path.open("w", newline="") as f:
            csv.DictWriter(f, METRIC_FIELDS).writeheader()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, METRIC_FIELDS, extrasaction="ignore").writerow({"step": state.global_step, **logs})


def show_masking(processor, row: dict) -> None:
    """Print one example's supervised span, to eyeball the label mask."""
    enc = encode_example(processor, row_messages(row))
    kept = enc["input_ids"][enc["labels"] != -100]
    print(f"\nexample {row['id']}: {len(enc['input_ids'])} tokens, {len(kept)} supervised")
    print("supervised text:", repr(processor.tokenizer.decode(kept)))


@torch.no_grad()
def preview(model, processor, ds, n: int, max_new_tokens: int, path: Path) -> float | None:
    """Greedy answers for the first `n` validation rows -> preview.jsonl; returns the mean per-field score."""
    if n <= 0:
        return None
    model.eval()
    model.config.use_cache = True
    scores = []
    with path.open("w") as f:
        for row in ds.select(range(min(n, len(ds)))):
            fields, gold = json.loads(row["fields_json"]), json.loads(row["target_json"])
            text, _ = generate(model, processor, row["state"], fields, max_new_tokens=max_new_tokens)
            pred = parse_json_object(text)
            fs = score_extraction(fields, gold, pred or {})
            scores += [r["score"] for r in fs]
            f.write(json.dumps({"id": row["id"], "source": row["source"], "gold": gold, "raw": text, "parsed": pred is not None, "fields": fs}, ensure_ascii=False) + "\n")
            print(f"  {row['id']}: {sum(r['score'] for r in fs) / len(fs):.2f}  {text[:120]!r}")
    return sum(scores) / len(scores)


def main(cfg: TrainConfig):
    seed_everything(cfg.seed)
    revisions = {PROCEDURAL: hub_sha(PROCEDURAL), SYNTHETIC: hub_sha(SYNTHETIC), cfg.model: hub_sha(cfg.model, "model")}
    processor = load_processor(cfg.model, revisions[cfg.model])

    train_ds, train_report, train_rejected = filter_length(build_split("train", cfg, revisions), processor, cfg.max_len, "train")
    val_ds, val_report, val_rejected = filter_length(build_split("validation", cfg, revisions), processor, cfg.max_len, "validation")
    print(f"\ntrain {len(train_ds)}, validation {len(val_ds)}")
    show_masking(processor, train_ds[0])
    if cfg.data_only:
        return

    run_dir = new_run_dir("train", cfg, "cuda:0", revisions=revisions)
    (run_dir / "sample_ids.json").write_text(json.dumps(
        {"train": list(train_ds["id"]), "validation": list(val_ds["id"]), "rejected": {"train": train_rejected, "validation": val_rejected}}, indent=1
    ))

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = load_model(cfg.model, revisions[cfg.model], quantize=cfg.quantize)
    model.config.use_cache = False
    gc_kwargs = {"use_reentrant": False}
    if cfg.quantize:
        model = prepare_model_for_kbit_training(model, cfg.gradient_checkpointing, gc_kwargs)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()
    # Exact decoder projection names; the vision tower has its own gate/up/down_proj, which stay frozen.
    targets = [
        name for name, _ in model.named_modules()
        if name.split(".")[-1] in LORA_PROJECTIONS and "visual" not in name.split(".")
    ]
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=targets, bias="none",
    ))
    print(f"LoRA on {len(targets)} modules")
    model.print_trainable_parameters()
    if cfg.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so adapter keys are unchanged

    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(
        model=model,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(processor, cfg.max_len),
        compute_loss_func=tail_loss,
        callbacks=[CsvLogger(run_dir / "metrics.csv")],
        args=TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs=gc_kwargs,
            train_sampling_strategy="group_by_length" if cfg.group_by_length else "random",
            length_column_name="n_tokens",
            dataloader_num_workers=cfg.num_workers,
            learning_rate=cfg.lr,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            bf16=True,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            prediction_loss_only=True,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            logging_steps=cfg.logging_steps,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="none",
            seed=cfg.seed,
        ),
    )
    train_out = trainer.train()
    final_eval = trainer.evaluate()  # also covers short runs that never reach eval_steps
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nPeak GPU memory allocated: {peak_gb:.2f} GiB (reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB)")

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    (adapter_dir / ADAPTER_META).write_text(json.dumps({"model": cfg.model, "model_revision": revisions[cfg.model], "quantize": cfg.quantize}, indent=2))

    print(f"\nPreview on {cfg.n_preview} validation examples:")
    preview_score = preview(model, processor, val_ds, cfg.n_preview, cfg.max_new_tokens, run_dir / "preview.jsonl")

    results = {
        "train_loss": train_out.training_loss,
        "global_step": train_out.global_step,
        "eval_loss": final_eval.get("eval_loss"),  # diagnostic only; accuracy comes from evaluate.py
        "peak_gpu_allocated_gib": peak_gb,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "n_train": len(train_ds),
        "n_validation": len(val_ds),
        "rejections": {"train": train_report, "validation": val_report},
        "preview_field_score": preview_score,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "rejections"}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
