"""GPT: a Llama-style language model, pretrained on a token corpus.

  uv run projects/gpt/cli.py prepare [--dataset tinystories]   download, train the tokenizer, write token shards
  uv run projects/gpt/cli.py pretrain [flags]                  pretrain on the prepared corpus
  uv run projects/gpt/cli.py sample <checkpoint> [flags]       continue a prompt
  uv run projects/gpt/cli.py prepare-evals [--dataset ...]     download the evaluation sets, save the validation documents' metadata
  uv run projects/gpt/cli.py eval <run dir or checkpoint>...   score checkpoints: loss sets, grammar, zero-shot multiple choice; writes eval.json
  uv run projects/gpt/cli.py fit-lr [--group ...]              scaling study: the best learning rate per width, and a rule
  uv run projects/gpt/cli.py fit-scaling [--group ...]         scaling study: IsoFLOP profiles and the L(N, D) law of a ladder sweep
  uv run projects/gpt/cli.py check-anneal [--group ...]        scaling study: constant LR with EMA against an annealed LR
  uv run projects/gpt/cli.py fit-shape [--group ...]           scaling study: depth x width, MLP size, head size and context length at equal compute
  uv run projects/gpt/cli.py fit-optim [--group ...]           scaling study: weight decay, AdamW beta2, warmup and batch size at equal compute

`<command> --help` lists a command's flags. Later stages (sft, dpo, grpo) will be one command each, taking the
previous stage's checkpoint (`--base-ckpt`, `--sft-ckpt`). A run is a W&B run in the `gpt` project (group `pretrain`),
with local files in `<runs>/gpt/<run id>/`: `checkpoints/` (the best by val/loss, plus last.ckpt), `config.json`,
`csv/metrics.csv`.
"""

from dataclasses import asdict
from typing import Annotated

import torch
import tyro
from config import AnnealConfig, Config, EvalConfig, LRFitConfig, OptimConfig, PrepareConfig, PrepareEvalsConfig, SampleConfig, ScalingConfig, ShapeConfig
from data import GPTData, checkpoint_corpus, corpus_dir, corpus_root, load_meta, load_tokenizer
from prepare import prepare_corpus
from pretrain import LitGPT
from torchinfo import summary

from zoo.training import load_module, make_trainer
from zoo.utils.device import get_device, seed_everything

PROJECT = "gpt"


def pretrain(cfg: Config) -> None:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    root = corpus_dir(cfg.dataset)
    cfg.corpus_build = str(root.resolve().relative_to(corpus_root(cfg.dataset).resolve()))
    train_meta = load_meta(cfg.dataset, "train", root=root)
    val_meta = load_meta(cfg.dataset, "validation", root=root)
    # The corpus decides the vocabulary, so the config that is logged and saved in checkpoints is the one that ran.
    cfg.vocab_size, cfg.eot_id = train_meta["vocab_size"], train_meta["eot_id"]
    cfg.tokens_per_byte = val_meta["length"] / val_meta["num_bytes"]
    if cfg.max_flops:  # needs the vocabulary above: the output projection is part of the FLOPs per token
        cfg.max_steps = cfg.budget_steps(train_meta["length"])
    tokens = cfg.max_steps * cfg.global_batch
    print(f"{cfg.dataset}: {train_meta['length']:,} train tokens; {cfg.n_layers} layers x {cfg.dim} wide, {cfg.n_params / 1e6:.2f}M parameters ({cfg.n_params_body / 1e6:.2f}M outside the embedding), peak lr {cfg.peak_lr:.2e}")
    print(f"{cfg.global_batch:,} tokens per step ({cfg.grad_accum} micro-batches of {cfg.batch_size} x {cfg.block_size}); {cfg.max_steps:,} steps = {tokens:,} tokens = {tokens * cfg.train_flops_per_token:.2e} FLOPs")
    module = LitGPT(asdict(cfg))
    summary(module.net, input_data=torch.zeros(1, cfg.block_size, dtype=torch.long), device="cpu", depth=2)

    trainer = make_trainer(
        cfg, PROJECT, stage="pretrain", monitor="val/loss", mode="min",
        val_check_interval=cfg.val_every * cfg.grad_accum, check_val_every_n_epoch=None,  # in micro-batches, across epochs
        accumulate_grad_batches=cfg.grad_accum,
        devices=cfg.devices, strategy="ddp" if cfg.devices > 1 else "auto",
    )  # fmt: skip
    trainer.fit(module, GPTData(cfg, root))
    # The corpus has no test split: the validation windows chose the checkpoint and are the final score.


def sample(cfg: SampleConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device()
    module = load_module(LitGPT, cfg.ckpt, map_location=device).to(device)
    model_cfg = module.cfg
    tok = load_tokenizer(model_cfg.dataset, root=checkpoint_corpus(model_cfg))
    ids = [int(t) for t in tok.encode(cfg.prompt)] if cfg.prompt else [model_cfg.eot_id]
    prompt = torch.tensor([ids] * cfg.num_samples, device=device)
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        out = module.net.generate(prompt, cfg.max_tokens, cfg.temperature, cfg.top_k, eos_id=model_cfg.eot_id, generator=g)
    for i, row in enumerate(out.tolist()):
        n = len(ids)
        if model_cfg.eot_id in row[n:]:  # drop everything from the first generated end-of-text on
            row = row[: n + row[n:].index(model_cfg.eot_id)]
        print(f"--- sample {i} ---\n{tok.decode(row).decode('utf-8', errors='replace')}\n")


Command = (
    Annotated[PrepareConfig, tyro.conf.subcommand("prepare")]
    | Annotated[Config, tyro.conf.subcommand("pretrain")]
    | Annotated[SampleConfig, tyro.conf.subcommand("sample")]
    | Annotated[PrepareEvalsConfig, tyro.conf.subcommand("prepare-evals")]
    | Annotated[EvalConfig, tyro.conf.subcommand("eval")]
    | Annotated[LRFitConfig, tyro.conf.subcommand("fit-lr")]
    | Annotated[ScalingConfig, tyro.conf.subcommand("fit-scaling")]
    | Annotated[AnnealConfig, tyro.conf.subcommand("check-anneal")]
    | Annotated[ShapeConfig, tyro.conf.subcommand("fit-shape")]
    | Annotated[OptimConfig, tyro.conf.subcommand("fit-optim")]
)


def main(cmd: Command) -> None:
    match cmd:
        case PrepareConfig():
            prepare_corpus(cmd)
        case Config():
            pretrain(cmd)
        case SampleConfig():
            sample(cmd)
        case PrepareEvalsConfig():
            from evals import prepare_evals

            prepare_evals(cmd)
        case EvalConfig():
            from evals import run_eval

            run_eval(cmd)
        case LRFitConfig():
            from scaling import run_fit_lr  # numpy and matplotlib only: nothing to load for the other commands

            run_fit_lr(cmd)
        case ScalingConfig():
            from scaling import run_fit_scaling

            run_fit_scaling(cmd)
        case AnnealConfig():
            from scaling import run_check_anneal

            run_check_anneal(cmd)
        case ShapeConfig():
            from scaling import run_fit_shape

            run_fit_shape(cmd)
        case OptimConfig():
            from scaling import run_fit_optim

            run_fit_optim(cmd)


if __name__ == "__main__":
    main(tyro.cli(Command, description=__doc__))
