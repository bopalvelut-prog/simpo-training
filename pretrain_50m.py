#!/usr/bin/env python3
"""
Pretrain 50M GPT on text data (next-token prediction).
Then fine-tune with SimPO for preference alignment.

Usage:
  python pretrain_50m.py                    # 5h pretraining
  python pretrain_50m.py --hours 1          # 1h pretraining (test)
  python train_simpo_50m.py --pretrained ./pretrain-50m/best.pt  # SimPO from pretrained
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from train_simpo_50m import GPT50MConfig, GPT50M, SimpleTokenizer


# ==================== Text Dataset ====================


class TextDataset(Dataset):
    """Tokenized text for next-token prediction."""

    def __init__(self, tokenizer, max_len=256, max_samples=50000):
        print("Loading text for pretraining...")
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")

        self.sequences = []
        count = 0
        for item in ds:
            # Combine all text fields
            parts = []
            for key in ["instruction", "chosen_response", "rejected_response"]:
                text = str(item.get(key, ""))
                if len(text) > 10:
                    parts.append(text)

            full_text = " ".join(parts)
            tokens = tokenizer.encode(full_text, max_len + 1)

            if len(tokens) >= 8:  # Need at least 8 tokens
                self.sequences.append(tokens[: max_len + 1])
                count += 1

            if count >= max_samples:
                break

        print(f"Loaded {len(self.sequences)} text sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def collate_text(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    def pad(seqs):
        return torch.stack(
            [F.pad(s, (0, max_len - len(s)), value=pad_id) for s in seqs]
        )

    return pad(xs), pad(ys)


# ==================== Pretraining ====================


def pretrain_50m():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pretraining 50M GPT (Next-Token Prediction)")
    print("=" * 60)
    print(f"  Time: {args.hours}h | LR: {args.lr} | Batch: {args.batch_size}")

    config = GPT50MConfig(sequence_len=args.max_len)

    # Tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = SimpleTokenizer(config.vocab_size)
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
    tokenizer.build(texts)
    print(f"  Vocab: {len(tokenizer.word2id)} words")

    # Dataset
    dataset = TextDataset(tokenizer, max_len=args.max_len, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_text,
        num_workers=0,
    )

    # Model
    print("\nBuilding 50M model...")
    model = GPT50M(config)

    start_step = 0
    best_loss = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        best_loss = ckpt.get("loss", float("inf"))
        print(f"  Resumed from step {start_step}, best_loss={best_loss:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training
    max_seconds = args.hours * 3600
    start_time = time.time()
    step = start_step
    total_loss = 0
    os.makedirs("./pretrain-50m", exist_ok=True)

    print(f"\nStarting pretraining for {args.hours}h...\n")

    epoch = 0
    while True:
        epoch += 1
        for x, y in loader:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                break

            # Forward
            model.train()
            logits = model(x)  # [B, T, V]
            logits = logits[:, :-1, :]  # align with next token

            # Trim y to match logits length
            seq_len = logits.shape[1]
            y_trimmed = y[:, :seq_len]

            # Cross-entropy loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_trimmed.reshape(-1),
                ignore_index=0,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            total_loss += loss.item()

            # Log
            if step % 10 == 0:
                avg = (
                    total_loss / (step - start_step)
                    if step > start_step
                    else loss.item()
                )
                rem = (max_seconds - elapsed) / 3600
                tps = (step - start_step) * args.batch_size / max(elapsed, 1)
                print(
                    f"  Step {step} | Epoch {epoch} | Loss: {loss.item():.4f} | "
                    f"Avg: {avg:.4f} | Elapsed: {elapsed / 3600:.1f}h | "
                    f"Rem: {rem:.1f}h | {tps:.1f} samples/s"
                )

            # Save best
            if step % 50 == 0:
                avg = total_loss / (step - start_step)
                if avg < best_loss:
                    best_loss = avg
                    path = "./pretrain-50m/best.pt"
                    torch.save(
                        {
                            "step": step,
                            "model": model.state_dict(),
                            "config": config,
                            "loss": best_loss,
                            "tokenizer_word2id": tokenizer.word2id,
                        },
                        path,
                    )
                    print(f"  New best: {best_loss:.4f} -> {path}")

        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            break

    # Final save
    final_loss = total_loss / max(step - start_step, 1)
    path = "./pretrain-50m/final.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "config": config,
            "loss": final_loss,
            "tokenizer_word2id": tokenizer.word2id,
        },
        path,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Pretraining Complete!")
    print(f"  Steps: {step - start_step}")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Time: {elapsed / 3600:.2f}h")
    print(f"  Saved: {path}")
    print(f"\nNext: python train_simpo_50m.py --pretrained {path}")
    print("=" * 60)


if __name__ == "__main__":
    pretrain_50m()
