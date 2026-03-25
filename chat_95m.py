#!/usr/bin/env python3
"""Chat with the 95M SimPO model."""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, "/home/m/simpo-training")

from train_simpo_95m import GPT95MConfig, GPT95M, SimpleTokenizer
from datasets import load_dataset
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def load_model():
    ckpt = torch.load(
        "/home/m/simpo-95m/final.pt", map_location="cpu", weights_only=False
    )
    config = ckpt["config"]
    model = GPT95M(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt["step"]
    loss = ckpt["loss"]
    print(f"Loaded: {step} steps, loss {loss:.4f}")
    return model, config


def build_tokenizer():
    print("Building tokenizer...")
    tok = SimpleTokenizer(16384)
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")
    texts = []
    for i, item in enumerate(ds):
        if i >= 5000:
            break
        texts.append(
            str(item.get("instruction", ""))
            + " "
            + str(item.get("chosen_response", ""))
        )
    tok.build(texts)
    return tok


def generate(model, tokenizer, prompt, max_new=50, temperature=0.8, top_k=40):
    tokens = tokenizer.encode(prompt, max_len=200)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    for _ in range(max_new):
        with torch.no_grad():
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1))
                )
                next_logits[next_logits < top_k_vals[-1]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        if next_token == 0:  # pad
            break

        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

        if input_ids.shape[1] > 256:
            break

    return tokenizer.decode(input_ids[0].tolist())


def main():
    print("=" * 50)
    print("95M SimPO Chat")
    print("=" * 50)

    model, config = load_model()
    tokenizer = build_tokenizer()

    print(f"\nModel ready. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        response = generate(model, tokenizer, prompt, max_new=60, temperature=0.9)
        # Extract just the generated part
        if prompt.lower() in response.lower():
            response = response[len(prompt) :].strip()
        print(f"Model: {response}\n")


if __name__ == "__main__":
    main()
