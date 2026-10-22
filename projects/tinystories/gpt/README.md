# TinyStories GPT

Train a decoder-only GPT from scratch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
The default model has about 6.7 million trainable parameters, including embeddings.

## Setup

Run these commands from the repository root:

```bash
uv sync --extra tinystories
```

## Prepare the data

```bash
uv run --extra tinystories python projects/tinystories/gpt/prepare.py
```

Preparation trains a 4,096-token byte-level BPE (byte pair encoding) tokenizer on the first 100,000 training stories. It uses Gigatoken by default to encode the corpus.
It then streams the official training and validation splits into packed token files.
Every story ends with `<|endoftext|>`. The tokenizer never trains on validation stories.
Byte-level encoding can represent any UTF-8 text without an unknown-token fallback.

Files go into `data/tinystories/`:

- `tokenizer.json`: tokenizer vocabulary and encoding rules.
- `train.bin` and `validation.bin`: token IDs stored as unsigned 16-bit integers.
- `metadata.json`: vocabulary size, end-of-story ID, and story and token counts.

Preparation needs network access and disk space for the corpus and Hugging Face cache.
Streaming avoids loading the whole corpus into memory. Training reads token files through memory mapping.
Rerunning preparation replaces the prepared data and retrains the tokenizer. Use a separate `--data-dir` for experiments.

For a smaller first run:

```bash
uv run --extra tinystories python projects/tinystories/gpt/prepare.py \
  --data-dir data/tinystories-small --train-stories 10000 --val-stories 1000
```

Small corpora may produce fewer than 4,096 vocabulary entries, which slightly reduces the parameter count.

For local data, pass both `--train-file train.txt` and `--val-file validation.txt`.
These files must contain one story per nonempty line. This also supports offline smoke tests.

## Train

```bash
uv run --extra tinystories python projects/tinystories/gpt/train.py
```

The device defaults to CUDA, then Apple MPS, then CPU, depending on availability.
CUDA devices that support bfloat16 use mixed precision. Other devices use float32.
Use `--device cpu`, `--device mps`, or `--device cuda` to choose explicitly.
Training and validation use `torch.compile(model, mode="reduce-overhead")`.
The first iterations include compilation time. Checkpoints store the original model weights.

| Setting | Default |
| --- | --- |
| Trainable parameters | 6,693,376 with a 4,096-token vocabulary |
| Transformer layers | 8 |
| Hidden width | 256 |
| Attention heads | 4 |
| Key/value heads | 1 |
| Feed-forward width | 1,024 |
| Context length | 2,048 tokens |
| Tokenizer vocabulary | 4,096 tokens |
| Microbatch size | 16 sequences |
| Gradient accumulation | 4 microbatches |
| Tokens per optimizer step | 32,768 tokens |
| Training steps | 10,000 optimizer steps |
| Total sampled training tokens | 327,680,000 tokens |
| Peak learning rate | 0.0003 |
| Warmup | 200 optimizer steps |

The model uses learned position embeddings, normalization before each sublayer, GELU activations, and causal grouped-query attention. Four query heads share one key/value head.
Its output projection shares the token embedding weights.
Training uses Muon for attention and feed-forward matrix weights.
AdamW updates token embeddings, position embeddings, and normalization parameters, with no weight decay on normalization parameters.
Muon uses momentum 0.95 and PyTorch's `match_rms_adamw` update scaling, allowing both optimizers to share `--lr`.
Both use cosine learning-rate decay to 10% of the peak rate, with gradient clipping at a norm of 1 before either optimizer steps.
Loss is next-token cross-entropy. Training windows can cross story boundaries, with end-of-story tokens marking each boundary.
Windows are sampled randomly with replacement; a step count does not guarantee coverage of every story.

For a short run on the smaller dataset:

```bash
uv run --extra tinystories python projects/tinystories/gpt/train.py \
  --data-dir data/tinystories-small --output-dir projects/tinystories/gpt/outputs/small \
  --steps 100 --warmup-steps 10 --batch-size 4 --grad-accum 1 \
  --eval-every 50 --eval-batches 5
```

Reduce `--batch-size` if memory is limited. Increase `--grad-accum` to preserve the effective batch size.
Reducing `--context-length` also reduces memory use, but changes the model's position embeddings.
A short run checks the pipeline; coherent stories require substantially more training.

## Checkpoints and metrics

The default output directory is `projects/tinystories/gpt/outputs/`.
Each evaluation writes:

- `checkpoint.pt`: latest model, both optimizer states, random states, tokenizer, and training configuration.
- `best.pt`: checkpoint with the lowest measured validation loss.
- `metrics.csv`: training loss, validation loss, validation perplexity, learning rate, and sampled token count.
- `curves.png`: training loss, validation perplexity, and learning-rate curves.
- `summary.json`: latest metrics and model configuration.

Perplexity is the exponential of the average next-token loss. Lower values indicate better next-token predictions.
Validation uses a fixed sample of windows for comparable measurements, rather than evaluating the entire split.

Resume the default run after an interruption:

```bash
uv run --extra tinystories python projects/tinystories/gpt/train.py \
  --resume projects/tinystories/gpt/outputs/checkpoint.pt
```

`--steps` is the total target step count, including completed steps.
For customized runs, pass the same data, output, context, and training settings again.
The checkpoint restores model, both optimizers, and random states. The supplied learning-rate settings determine the remaining schedule.
Changing the total step count changes the cosine schedule.
Resuming requires a Muon + AdamW checkpoint. Older AdamW-only checkpoints still work for generation.

## Generate stories

```bash
uv run --extra tinystories python projects/tinystories/gpt/generate.py \
  --prompt "Once upon a time, a little fox found a key."
```

Use `--checkpoint path/to/checkpoint.pt` for another run.
Generation loads the tokenizer from the checkpoint, so the prepared dataset is not needed.
`--temperature 0.8`, `--top-k 40`, and `--max-new-tokens 200` control sampling.
Use `--top-k 0` to sample from the full vocabulary.
Generation stops at the end-of-story token or the token limit.
When text exceeds the context length, the model uses the most recent context window.

## Verify

```bash
uv run --extra tinystories python -m unittest discover -s tests -p 'test_tinystories.py' -v
```

The tests check the parameter budget, causal attention, learning on a repeated batch, and the local preparation/training/resume/generation pipeline.
