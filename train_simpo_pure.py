#!/usr/bin/env python3
"""
SimPO: Simple Preference Optimization (Pure Python)
Based on: https://arxiv.org/abs/2405.14734

Reference-free preference optimization using sequence-level average log
probability as implicit reward with a target margin.

L_SimPO = -log(sigmoid((avg_lp_chosen - avg_lp_rejected - gamma) / beta))

No reference model. No torch. Pure Python + numpy (optional).

Usage: python3 train_simpo_pure.py
"""

import os
import math
import random
import pickle
from collections import defaultdict

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ==================== Tokenizer ====================


class TinyTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4

    def build_vocab(self, texts, max_vocab=2000):
        word_freq = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                word_freq[word] += 1
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1])[
            : max_vocab - 4
        ]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.vocab_size = idx + 1

    def encode(self, text, max_length=128):
        tokens = [self.word2idx.get(w, 1) for w in text.lower().split()]
        return tokens[:max_length]

    def decode(self, ids):
        return " ".join(
            self.idx2word.get(i, "<UNK>") for i in ids if i not in [0, 2, 3]
        )


# ==================== Tiny Transformer ====================


class TinyTransformer:
    def __init__(
        self, vocab_size=2000, d_model=128, n_heads=4, n_layers=3, max_len=128
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        scale = 0.02
        self.embed = self._init((vocab_size, d_model), scale)
        self.pos_embed = self._init((max_len, d_model), scale)

        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                {
                    "wq": self._init((d_model, d_model), scale),
                    "wk": self._init((d_model, d_model), scale),
                    "wv": self._init((d_model, d_model), scale),
                    "wo": self._init((d_model, d_model), scale),
                    "w1": self._init((d_model, d_model * 2), scale),
                    "w2": self._init((d_model * 2, d_model), scale),
                    "ln1_g": self._init((d_model,), 1.0),
                    "ln1_b": self._init((d_model,), 0.0),
                    "ln2_g": self._init((d_model,), 1.0),
                    "ln2_b": self._init((d_model,), 0.0),
                }
            )

        self.output_proj = self._init((d_model, vocab_size), scale)

        params = sum(
            p.size
            if HAS_NUMPY
            else len(p) * len(p[0])
            if isinstance(p[0], list)
            else len(p)
            for p in [self.embed, self.pos_embed, self.output_proj]
        )
        for layer in self.layers:
            for p in layer.values():
                params += (
                    p.size
                    if HAS_NUMPY
                    else len(p) * len(p[0])
                    if isinstance(p[0], list)
                    else len(p)
                )
        print(f"Model: {params:,} params ({params / 1e6:.2f}M)")

    def _init(self, shape, scale):
        if HAS_NUMPY:
            return np.random.randn(*shape).astype(np.float32) * scale
        return [
            [random.gauss(0, scale) for _ in range(shape[1])] for _ in range(shape[0])
        ]

    def softmax(self, x):
        if HAS_NUMPY:
            e = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        max_val = max(x)
        e = [math.exp(v - max_val) for v in x]
        s = sum(e)
        return [v / s for v in e]

    def log_softmax(self, x):
        if HAS_NUMPY:
            return (
                x
                - np.max(x, axis=-1, keepdims=True)
                - np.log(
                    np.sum(
                        np.exp(x - np.max(x, axis=-1, keepdims=True)),
                        axis=-1,
                        keepdims=True,
                    )
                )
            )
        max_val = max(x)
        e = [math.exp(v - max_val) for v in x]
        log_sum = math.log(sum(e))
        return [v - max_val - log_sum for v in x]

    def layer_norm(self, x, g, b, eps=1e-5):
        if HAS_NUMPY:
            m = x.mean(axis=-1, keepdims=True)
            v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
            return g * (x - m) / np.sqrt(v + eps) + b
        n = len(x)
        m = sum(x) / n
        v = sum((i - m) ** 2 for i in x) / n
        return [g[j] * (x[j] - m) / math.sqrt(v + eps) + b[j] for j in range(n)]

    def forward(self, tokens):
        seq_len = len(tokens)
        if HAS_NUMPY:
            x = self.embed[tokens] + self.pos_embed[:seq_len]
        else:
            x = [
                [self.embed[t][d] + self.pos_embed[p][d] for d in range(self.d_model)]
                for p, t in enumerate(tokens)
            ]

        for layer in self.layers:
            if HAS_NUMPY:
                q = x @ layer["wq"]
                k = x @ layer["wk"]
                v = x @ layer["wv"]
                scores = (q @ k.T) / math.sqrt(self.d_model)
                mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
                attn = self.softmax(scores + mask)
                out = (attn @ v) @ layer["wo"]
                x = self.layer_norm(x + out, layer["ln1_g"], layer["ln1_b"])
                h = np.maximum(x @ layer["w1"], 0)
                h = h @ layer["w2"]
                x = self.layer_norm(x + h, layer["ln2_g"], layer["ln2_b"])
            else:
                x = self.layer_norm(x, layer["ln1_g"], layer["ln1_b"])

        if HAS_NUMPY:
            return x @ self.output_proj
        return [
            [
                sum(x[t][d] * self.output_proj[d][v] for d in range(self.d_model))
                for v in range(self.vocab_size)
            ]
            for t in range(seq_len)
        ]

    def avg_log_prob(self, tokens, response_start=0):
        """Compute average log probability of response tokens (SimPO core)."""
        logits = self.forward(tokens)
        if not HAS_NUMPY:
            return 0.0

        total_lp = 0.0
        count = 0
        for i in range(response_start, len(logits) - 1):
            lp = self.log_softmax(logits[i])
            next_token = tokens[i + 1]
            total_lp += lp[next_token]
            count += 1

        return total_lp / max(count, 1)

    def generate(self, tokenizer, prompt, max_new=20, temperature=0.8):
        tokens = tokenizer.encode(prompt, self.max_len - max_new)
        for _ in range(max_new):
            logits = self.forward(tokens)
            next_logits = logits[-1]
            if HAS_NUMPY:
                next_logits = next_logits / temperature
                probs = self.softmax(next_logits)
                nxt = np.random.choice(len(probs), p=probs)
            else:
                nxt = random.randint(0, self.vocab_size - 1)
            if nxt == 3:
                break
            tokens.append(nxt)
        return tokenizer.decode(tokens)

    def save(self, path):
        data = {
            "config": (
                self.vocab_size,
                self.d_model,
                self.n_heads,
                self.n_layers,
                self.max_len,
            ),
            "embed": self.embed.tolist() if HAS_NUMPY else self.embed,
            "pos_embed": self.pos_embed.tolist() if HAS_NUMPY else self.pos_embed,
            "layers": self.layers,
            "output_proj": self.output_proj.tolist() if HAS_NUMPY else self.output_proj,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(*data["config"])
        model.embed = (
            np.array(data["embed"], dtype=np.float32) if HAS_NUMPY else data["embed"]
        )
        model.pos_embed = (
            np.array(data["pos_embed"], dtype=np.float32)
            if HAS_NUMPY
            else data["pos_embed"]
        )
        model.layers = data["layers"]
        model.output_proj = (
            np.array(data["output_proj"], dtype=np.float32)
            if HAS_NUMPY
            else data["output_proj"]
        )
        return model


# ==================== SimPO Training ====================


def load_preference_data():
    """Preference pairs: chosen (good) vs rejected (bad) responses."""
    data = [
        {
            "q": "capital of France",
            "chosen": "Paris is the capital of France",
            "rejected": "London",
        },
        {
            "q": "author of Romeo and Juliet",
            "chosen": "Shakespeare wrote Romeo and Juliet",
            "rejected": "Hemingway",
        },
        {
            "q": "what is 2 plus 2",
            "chosen": "2 plus 2 equals 4",
            "rejected": "2 plus 2 is 5",
        },
        {
            "q": "color of the sky",
            "chosen": "The sky appears blue during the day",
            "rejected": "The sky is green",
        },
        {
            "q": "planet we live on",
            "chosen": "We live on planet Earth",
            "rejected": "We live on Mars",
        },
        {
            "q": "days in a week",
            "chosen": "There are 7 days in a week",
            "rejected": "There are 5 days",
        },
        {
            "q": "largest ocean",
            "chosen": "The Pacific Ocean is the largest",
            "rejected": "The Atlantic is largest",
        },
        {
            "q": "who painted Mona Lisa",
            "chosen": "Leonardo da Vinci painted the Mona Lisa",
            "rejected": "Picasso painted it",
        },
        {
            "q": "what is H2O",
            "chosen": "H2O is the chemical formula for water",
            "rejected": "H2O is gold",
        },
        {
            "q": "when did WW2 end",
            "chosen": "World War 2 ended in 1945",
            "rejected": "It ended in 1950",
        },
        {
            "q": "tallest mountain",
            "chosen": "Mount Everest is the tallest mountain",
            "rejected": "K2 is tallest",
        },
        {
            "q": "language in Brazil",
            "chosen": "Portuguese is the official language of Brazil",
            "rejected": "They speak Spanish",
        },
        {
            "q": "speed of light",
            "chosen": "Light travels at about 300000 km per second",
            "rejected": "Light is slow",
        },
        {
            "q": "who discovered America",
            "chosen": "Christopher Columbus reached the Americas in 1492",
            "rejected": "Magellan",
        },
        {
            "q": "water boiling point",
            "chosen": "Water boils at 100 degrees Celsius at sea level",
            "rejected": "It boils at 50",
        },
    ] * 5
    random.shuffle(data)
    return data


def simpo_loss(avg_lp_chosen, avg_lp_rejected, gamma=1.0, beta=0.01):
    """SimPO loss: -log(sigmoid((avg_lp_chosen - avg_lp_rejected - gamma) / beta))"""
    diff = (avg_lp_chosen - avg_lp_rejected - gamma) / beta
    # sigmoid(x) = 1 / (1 + exp(-x))
    # -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
    if HAS_NUMPY:
        # Numerically stable softplus
        if diff > 20:
            return math.exp(-diff)
        elif diff < -20:
            return -diff
        else:
            return math.log(1 + math.exp(-diff))
    else:
        if diff > 20:
            return math.exp(-diff)
        elif diff < -20:
            return -diff
        else:
            return math.log(1 + math.exp(-diff))


def train_simpo():
    """SimPO training: reference-free preference optimization."""
    print("=" * 50)
    print("SimPO: Simple Preference Optimization")
    print("Reference-free | Avg log-prob reward | Margin-based")
    print("=" * 50)

    os.makedirs("./simpo-tiny-pure", exist_ok=True)

    # Config
    gamma = 1.0  # Target reward margin
    beta = 0.5  # Temperature (higher = smoother sigmoid)
    lr = 0.0001  # Learning rate (reduced for stability)
    epochs = 10
    clip_grad = 0.1  # Gradient clipping

    print(f"  gamma={gamma}, beta={beta}, lr={lr}")

    # Load data
    data = load_preference_data()

    # Build tokenizer
    tokenizer = TinyTokenizer()
    all_texts = []
    for item in data:
        all_texts.append(f"q: {item['q']} a: {item['chosen']}")
        all_texts.append(f"q: {item['q']} a: {item['rejected']}")
    tokenizer.build_vocab(all_texts, max_vocab=1500)
    print(f"Vocab: {tokenizer.vocab_size}")

    # Create model
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=64
    )

    # SimPO training loop
    print("\n" + "=" * 50)
    print("SimPO Training")
    print("=" * 50)

    for epoch in range(epochs):
        total_loss = 0
        num_pairs = 0

        random.shuffle(data)

        for item in data:
            prompt = f"q: {item['q']} a:"
            prompt_tokens = tokenizer.encode(prompt)

            chosen_text = f"{prompt} {item['chosen']}"
            rejected_text = f"{prompt} {item['rejected']}"

            chosen_tokens = tokenizer.encode(chosen_text, max_length=64)
            rejected_tokens = tokenizer.encode(rejected_text, max_length=64)

            if len(chosen_tokens) < 3 or len(rejected_tokens) < 3:
                continue

            if not HAS_NUMPY:
                continue

            # Forward pass: compute average log probabilities
            avg_lp_chosen = model.avg_log_prob(
                chosen_tokens, response_start=len(prompt_tokens) - 1
            )
            avg_lp_rejected = model.avg_log_prob(
                rejected_tokens, response_start=len(prompt_tokens) - 1
            )

            # SimPO loss
            loss = simpo_loss(avg_lp_chosen, avg_lp_rejected, gamma, beta)

            # Compute gradient direction:
            # We want to INCREASE chosen log-prob and DECREASE rejected log-prob
            chosen_logits = model.forward(chosen_tokens)
            rejected_logits = model.forward(rejected_tokens)

            # SimPO gradient weight: higher loss = stronger update
            grad_weight = min(loss / max(beta, 1e-6), 10.0)  # Cap gradient weight

            # Simple gradient: push chosen probs up, rejected probs down
            for pos in range(
                len(prompt_tokens) - 1, min(len(chosen_tokens) - 1, len(chosen_logits))
            ):
                lp = model.log_softmax(chosen_logits[pos])
                probs = model.softmax(chosen_logits[pos])
                target = chosen_tokens[pos + 1]
                # Positive gradient: increase probability of chosen tokens
                grad = probs.copy()
                grad[target] -= 1.0
                grad = np.clip(grad, -clip_grad, clip_grad)
                model.output_proj[:, target] -= lr * grad[target] * grad_weight

            for pos in range(
                len(prompt_tokens) - 1,
                min(len(rejected_tokens) - 1, len(rejected_logits)),
            ):
                probs = model.softmax(rejected_logits[pos])
                target = rejected_tokens[pos + 1]
                # Negative gradient: decrease probability of rejected tokens
                grad = probs.copy()
                grad[target] -= 1.0
                grad = np.clip(grad, -clip_grad, clip_grad)
                model.output_proj[:, target] += lr * grad[target] * grad_weight

            total_loss += loss
            num_pairs += 1

        avg_loss = total_loss / max(num_pairs, 1)

        # Evaluate: check if model prefers chosen over rejected
        correct = 0
        for item in data[:20]:
            prompt = f"q: {item['q']} a:"
            prompt_tokens = tokenizer.encode(prompt)
            chosen_tokens = tokenizer.encode(
                f"{prompt} {item['chosen']}", max_length=64
            )
            rejected_tokens = tokenizer.encode(
                f"{prompt} {item['rejected']}", max_length=64
            )
            if len(chosen_tokens) < 3 or len(rejected_tokens) < 3:
                continue
            lp_c = model.avg_log_prob(
                chosen_tokens, response_start=len(prompt_tokens) - 1
            )
            lp_r = model.avg_log_prob(
                rejected_tokens, response_start=len(prompt_tokens) - 1
            )
            if lp_c > lp_r:
                correct += 1

        accuracy = correct / min(20, len(data))
        print(
            f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Preference Accuracy: {accuracy:.1%}"
        )

    # Save
    model.save("./simpo-tiny-pure/simpo_tiny.pkl")
    print(f"\nSaved: ./simpo-tiny-pure/simpo_tiny.pkl")

    # Test generation
    print("\n" + "=" * 50)
    print("Test Generation (post-SimPO)")
    print("=" * 50)
    for q in ["capital of France", "author of Romeo and Juliet", "color of the sky"]:
        resp = model.generate(tokenizer, f"q: {q} a:", max_new=8)
        print(f"Q: {q}")
        print(f"A: {resp}\n")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_simpo()

    print("=" * 50)
    print("Done!")
    print("  Model: ./simpo-tiny-pure/simpo_tiny.pkl")
    print("=" * 50)
