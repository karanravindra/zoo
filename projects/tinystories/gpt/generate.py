import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from train import OUTPUT_DIR, choose_device


def main():
    parser = argparse.ArgumentParser(description="Generate a story with a trained TinyStories GPT")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "best.pt")
    parser.add_argument("--prompt", default="Once upon a time, there was a little")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.temperature <= 0 or args.top_k < 0:
        parser.error("Require max-new-tokens >= 0, temperature > 0, top-k >= 0")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    ids = tokenizer.encode(args.prompt).ids
    if not ids:
        ids = [checkpoint["data"]["eos_id"]]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    result = model.generate(tokens, args.max_new_tokens, args.temperature, args.top_k,
                            checkpoint["data"]["eos_id"])
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
