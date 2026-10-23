import argparse
import time

import torch
from model import GPT, GPTConfig
from train import initialize_gradients, make_optimizers, precision_context

VOCAB_SIZE = 8192


def benchmark(config, args, compile_mode):
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = GPT(config).to(device)
    optimizers = make_optimizers(model, 1e-3, 0.1)
    run = model if compile_mode == "eager" else torch.compile(model, mode=compile_mode)
    initialize_gradients(model)
    x = torch.randint(VOCAB_SIZE, (args.batch_size, config.context_length), device=device)

    def step():
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=False)
        for _ in range(args.grad_accum):
            with precision_context(device):
                _, loss = run(x, x)
            (loss / args.grad_accum).backward()
        for optimizer in optimizers.values():
            optimizer.step()
        if config.moe:
            model.update_router_bias()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / args.steps
    tokens = args.batch_size * args.grad_accum * config.context_length
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "ms_per_step": seconds * 1e3,
        "tokens_per_second": tokens / seconds,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark dense vs MoE training speed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--modes", nargs="+", default=["eager", "reduce-overhead"], help="compile modes"
    )
    args = parser.parse_args()
    variants = {
        "dense": GPTConfig(vocab_size=VOCAB_SIZE, context_length=args.context_length),
        "moe": GPTConfig(
            vocab_size=VOCAB_SIZE, context_length=args.context_length, moe=True
        ),
    }
    print(f"{'mode':<16}{'model':<7}{'params':>12}{'ms/step':>10}{'tok/s':>11}{'GB':>7}")
    for mode in args.modes:
        results = {}
        for name, config in variants.items():
            try:
                results[name] = benchmark(config, args, mode)
            except RuntimeError as error:
                print(f"{mode:<16}{name:<7}failed: {str(error).splitlines()[0]}")
                break
            r = results[name]
            print(
                f"{mode:<16}{name:<7}{r['params']:>12,}{r['ms_per_step']:>10.1f}"
                f"{r['tokens_per_second']:>11,.0f}{r['peak_gb']:>7.2f}"
            )
        if len(results) < len(variants):
            continue
        ratio = results["moe"]["tokens_per_second"] / results["dense"]["tokens_per_second"]
        print(f"{mode:<16}moe throughput = {ratio:.2f}x dense")


if __name__ == "__main__":
    main()
