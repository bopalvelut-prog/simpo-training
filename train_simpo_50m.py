#!/usr/bin/env python3
"""
SimPO Training with 50M Parameter Model
Runs on CPU — trains for 5 hours by default.

Uses the autoresearch GPT architecture (50M params):
  Vocab: 16384, Layers: 11, Heads: 16, Embed: 512

Trains with SimPO (reference-free preference optimization) on UltraFeedback data.
"""

import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from datasets import load_dataset


# ==================== 50M GPT Model ====================


@dataclass
class GPT50MConfig:
    vocab_size: int = 16384
    n_layer: int = 11
    n_head: int = 16
    n_embd: int = 512
    sequence_len: int = 512


def rms_norm(x, eps=1e-6):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], -1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT50M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        cos, sin = self._rope(config.sequence_len, head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self._init_weights()
        nparams = sum(p.numel() for p in self.parameters())
        print(f"Model: {nparams:,} params ({nparams / 1e6:.1f}M)")

    def _rope(self, seq_len, head_dim, base=10000):
        d = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (d / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

    @torch.no_grad()
    def _init_weights(self):
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.c_q.weight)
            nn.init.xavier_uniform_(block.attn.c_k.weight)
            nn.init.xavier_uniform_(block.attn.c_v.weight)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.xavier_uniform_(block.mlp.c_fc.weight)
            nn.init.zeros_(block.mlp.c_proj.weight)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.wte(input_ids)
        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(rms_norm(x))

    def log_probs(self, input_ids):
        """Get log probs for next-token prediction."""
        logits = self.forward(input_ids)
        return F.log_softmax(logits[:, :-1, :], dim=-1)


# ==================== SimPO Dataset ====================


class SimPODataset(Dataset):
    def __init__(self, tokenizer_encode, max_len=512, max_samples=10000):
        print("Loading UltraFeedback dataset...")
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")
        self.pairs = []
        self.max_len = max_len

        for item in ds:
            prompt = str(item.get("instruction", ""))
            chosen = str(item.get("chosen_response", ""))
            rejected = str(item.get("rejected_response", ""))

            if len(prompt) < 5 or len(chosen) < 10 or len(rejected) < 10:
                continue

            prompt_ids = tokenizer_encode(prompt, max_len // 2)
            chosen_ids = tokenizer_encode(chosen, max_len // 2)
            rejected_ids = tokenizer_encode(rejected, max_len // 2)

            if len(prompt_ids) < 3 or len(chosen_ids) < 3 or len(rejected_ids) < 3:
                continue

            self.pairs.append(
                {
                    "prompt_len": len(prompt_ids),
                    "chosen": prompt_ids + chosen_ids,
                    "rejected": prompt_ids + rejected_ids,
                }
            )

            if len(self.pairs) >= max_samples:
                break

        print(f"Loaded {len(self.pairs)} preference pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return {
            "prompt_len": p["prompt_len"],
            "chosen": torch.tensor(p["chosen"], dtype=torch.long),
            "rejected": torch.tensor(p["rejected"], dtype=torch.long),
        }


def collate_fn(batch, pad_id=0):
    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        return torch.stack(
            [F.pad(s, (0, max_len - len(s)), value=pad_id) for s in seqs]
        )

    return {
        "prompt_len": torch.tensor([b["prompt_len"] for b in batch]),
        "chosen": pad([b["chosen"] for b in batch]),
        "rejected": pad([b["rejected"] for b in batch]),
    }


# ==================== Simple Tokenizer ====================


class SimpleTokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.word2id = {}
        self.id2word = {}

    def build(self, texts):
        from collections import Counter

        freq = Counter()
        for text in texts:
            for word in text.lower().split():
                freq[word] += 1
        # Reserve 0=pad, 1=unk
        self.word2id = {"<pad>": 0, "<unk>": 1}
        for word, _ in freq.most_common(self.vocab_size - 2):
            self.word2id[word] = len(self.word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def encode(self, text, max_len=256):
        ids = [self.word2id.get(w, 1) for w in text.lower().split()]
        return ids[:max_len]

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "?") for i in ids if i > 1)


# ==================== SimPO Training ====================


def simpo_loss(log_probs_chosen, log_probs_rejected, prompt_lens, gamma=1.0, beta=0.5):
    """Compute SimPO loss with sequence-level avg log-prob as reward."""
    B = log_probs_chosen.shape[0]
    losses = []
    for i in range(B):
        p_len = prompt_lens[i].item()
        # Average log-prob of response tokens only
        c_resp = log_probs_chosen[i, p_len - 1 :]
        r_resp = log_probs_rejected[i, p_len - 1 :]

        # Mask padding (non-zero positions)
        c_mask = (c_resp != 0).float()
        r_mask = (r_resp != 0).float()

        c_avg = c_resp.sum() / c_mask.sum().clamp(min=1)
        r_avg = r_resp.sum() / r_mask.sum().clamp(min=1)

        # SimPO: -log(sigmoid((c_avg - r_avg - gamma) / beta))
        diff = (c_avg - r_avg - gamma) / beta
        loss = -F.logsigmoid(diff)
        losses.append(loss)

    return torch.stack(losses).mean()


def train_simpo_50m():
    """Train 50M model with SimPO for 5 hours."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Load pretrained checkpoint"
    )
    parser.add_argument("--hours", type=float, default=5.0)
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("SimPO Training — 50M Parameter Model")
    print("=" * 60)

    # Config
    training_hours = args.hours
    gamma = 1.0
    beta = 0.5
    lr = 3e-4
    batch_size = 4
    max_len = 256
    max_samples = 10000
    grad_clip = 1.0
    save_every = 600  # Save every 10 min

    config = GPT50MConfig(sequence_len=max_len)
    print(f"  Training time: {training_hours}h")
    print(f"  gamma={gamma}, beta={beta}, lr={lr}")
    print(f"  batch_size={batch_size}, max_len={max_len}")
    if args.pretrained:
        print(f"  Pretrained: {args.pretrained}")

    # Build tokenizer
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
    dataset = SimPODataset(tokenizer.encode, max_len=max_len, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=0),
        num_workers=0,
    )

    # Model
    print("\nBuilding 50M model...")
    model = GPT50M(config)

    # Load pretrained weights if provided
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        pretrained_loss = ckpt.get("loss", "?")
        print(f"  Loaded (pretrain loss: {pretrained_loss})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    print(f"\nStarting SimPO training for {training_hours}h...")
    max_seconds = training_hours * 3600
    start_time = time.time()
    step = 0
    total_loss = 0
    os.makedirs("./simpo-50m", exist_ok=True)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\nTime limit reached: {elapsed / 3600:.2f}h")
            break

        for batch in loader:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                break

            chosen = batch["chosen"]
            rejected = batch["rejected"]
            prompt_lens = batch["prompt_len"]

            # Forward pass
            model.train()
            log_probs_chosen = model.log_probs(chosen)
            log_probs_rejected = model.log_probs(rejected)

            # SimPO loss
            loss = simpo_loss(
                log_probs_chosen, log_probs_rejected, prompt_lens, gamma, beta
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            total_loss += loss.item()

            # Log
            if step % 10 == 0:
                avg_loss = total_loss / step
                rem_hours = (max_seconds - elapsed) / 3600
                tps = step * batch_size / elapsed  # samples/sec estimate
                print(
                    f"  Step {step} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | "
                    f"Elapsed: {elapsed / 3600:.1f}h | Remaining: {rem_hours:.1f}h | "
                    f"{tps:.1f} samples/s"
                )

            # Save checkpoint
            if step % (save_every // 2) == 0:  # roughly every 10 min
                path = f"./simpo-50m/step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "loss": total_loss / step,
                    },
                    path,
                )
                print(f"  Saved checkpoint: {path}")

    # Final save
    path = "./simpo-50m/final.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "config": config,
            "loss": total_loss / max(step, 1),
        },
        path,
    )

    final_loss = total_loss / max(step, 1)
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"SimPO Training Complete!")
    print(f"  Steps: {step}")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Time: {elapsed / 3600:.2f}h")
    print(f"  Saved: {path}")
    print("=" * 60)


if __name__ == "__main__":
    train_simpo_50m()
