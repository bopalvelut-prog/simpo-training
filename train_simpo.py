#!/usr/bin/env python3
"""
SimPO: Simple Preference Optimization with a Reference-Free Reward
Based on: https://arxiv.org/abs/2405.14734

Key idea: DPO uses a reference model to prevent reward hacking.
SimPO removes the reference model entirely by using sequence-level
average log probability as the implicit reward, with a target
reward margin (gamma) to separate chosen from rejected.

Loss: L_SimPO = -log(σ((avg_lp_chosen - avg_lp_rejected - γ) / β))

Where:
  avg_lp_chosen   = log π(y_w|x) / |y_w|
  avg_lp_rejected = log π(y_l|x) / |y_l|
  γ (gamma)       = target reward margin (default 1.0)
  β (beta)        = temperature (default 0.01)

No reference model needed. No KL penalty. Simpler and often better than DPO.

Usage:
  python train_simpo.py --model Qwen/Qwen2.5-1.5B-Instruct
  python train_simpo.py --model Qwen/Qwen2.5-7B-Instruct --use_lora
"""

import os
import json
import math
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType

import warnings

warnings.filterwarnings("ignore")


# ==================== Config ====================


@dataclass
class SimPOConfig:
    model: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    output_dir: str = field(default="./simpo-qwen2.5-1.5b")
    model_size: str = field(default="1.5b")

    # SimPO specific
    gamma: float = field(
        default=1.0,
        metadata={"help": "Target reward margin between chosen and rejected"},
    )
    beta: float = field(default=0.01, metadata={"help": "Temperature for sigmoid"})

    # Training
    training_hours: float = field(
        default=5.0, metadata={"help": "Max training time in hours (overrides epochs)"}
    )
    epochs: int = field(default=3)
    batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-7)
    warmup_ratio: float = field(default=0.1)
    max_prompt_length: int = field(default=1024)
    max_response_length: int = field(default=1024)
    max_seq_length: int = field(default=2048)

    # LoRA
    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)

    # Datasets
    dataset_name: str = field(default="argilla/ultrafeedback-binarized-preferences")
    max_train_samples: int = field(default=50000)
    wandb_project: str = field(default="simpo")

    # System
    use_flash_attention: bool = field(default=True)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)


# ==================== SimPO Dataset ====================


def load_simpo_dataset(config: SimPOConfig, tokenizer) -> Dataset:
    """Load preference dataset for SimPO training.

    Expects dataset with 'chosen' and 'rejected' conversation columns.
    Supports: argilla/ultrafeedback-binarized-preferences, Anthropic/hh-rlhf, etc.
    """
    print(f"Loading dataset: {config.dataset_name}")

    try:
        ds = load_dataset(config.dataset_name, split="train")
    except Exception:
        # Fallback: try without config
        ds = load_dataset(config.dataset_name, split="train")

    def format_conversation(messages):
        """Convert message list to formatted text."""
        if isinstance(messages, list):
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"Human: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            return "\n\n".join(parts) + "\n\nAssistant: "
        return str(messages)

    def extract_prompt_and_responses(example):
        """Extract prompt, chosen, rejected from various dataset formats."""
        # UltraFeedback format
        if "chosen" in example and "rejected" in example:
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Messages format (list of dicts)
            if isinstance(chosen, list):
                # Find the common prompt (everything before the assistant turn)
                chosen_text = format_conversation(chosen)
                rejected_text = format_conversation(rejected)

                # Extract prompt: everything before the last Assistant: turn
                if "Human:" in chosen_text:
                    prompt = chosen_text.rsplit("Assistant:", 1)[0] + "Assistant: "
                    chosen_resp = chosen_text.rsplit("Assistant:", 1)[1].strip()
                    rejected_resp = rejected_text.rsplit("Assistant:", 1)[1].strip()
                else:
                    prompt = ""
                    chosen_resp = chosen_text
                    rejected_resp = rejected_text

                return {
                    "prompt": prompt.strip(),
                    "chosen": chosen_resp.strip(),
                    "rejected": rejected_resp.strip(),
                }

            # String format
            elif isinstance(chosen, str):
                return {
                    "prompt": example.get("prompt", ""),
                    "chosen": chosen,
                    "rejected": rejected,
                }

        # Fallback
        return {
            "prompt": example.get("prompt", example.get("question", "")),
            "chosen": example.get("chosen", example.get("preferred", "")),
            "rejected": example.get("rejected", example.get("dispreferred", "")),
        }

    processed = ds.map(extract_prompt_and_responses, remove_columns=ds.column_names)

    # Filter out empty examples
    processed = processed.filter(
        lambda x: (
            x["prompt"]
            and x["chosen"]
            and x["rejected"]
            and len(x["chosen"]) > 10
            and len(x["rejected"]) > 10
        )
    )

    if config.max_train_samples and len(processed) > config.max_train_samples:
        processed = processed.select(range(config.max_train_samples))

    print(f"Loaded {len(processed)} preference pairs")
    return processed


def tokenize_simpo(examples, tokenizer, max_length):
    """Tokenize preference pairs for SimPO."""
    chosen_texts = [p + c for p, c in zip(examples["prompt"], examples["chosen"])]
    rejected_texts = [p + r for p, r in zip(examples["prompt"], examples["rejected"])]
    prompt_texts = examples["prompt"]

    chosen_encodings = tokenizer(
        chosen_texts, truncation=True, max_length=max_length, padding=False
    )
    rejected_encodings = tokenizer(
        rejected_texts, truncation=True, max_length=max_length, padding=False
    )
    prompt_encodings = tokenizer(
        prompt_texts, truncation=True, max_length=max_length, padding=False
    )

    return {
        "chosen_input_ids": chosen_encodings["input_ids"],
        "chosen_attention_mask": chosen_encodings["attention_mask"],
        "rejected_input_ids": rejected_encodings["input_ids"],
        "rejected_attention_mask": rejected_encodings["attention_mask"],
        "prompt_input_ids": prompt_encodings["input_ids"],
    }


# ==================== SimPO Trainer ====================


class SimPOTrainer(Trainer):
    """Custom trainer for SimPO (Simple Preference Optimization).

    SimPO uses sequence-level average log probability as the implicit reward,
    with a target reward margin (gamma) to separate chosen from rejected.

    L_SimPO = -log(σ((avg_lp_chosen - avg_lp_rejected - γ) / β))
    """

    def __init__(self, gamma=1.0, beta=0.01, **kwargs):
        self.gamma = gamma
        self.beta = beta
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute SimPO loss."""
        # Forward pass for chosen
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_input_ids"]
        rejected_mask = inputs["rejected_attention_mask"]

        # Get logits for chosen
        chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        chosen_logits = chosen_outputs.logits

        # Get logits for rejected
        rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)
        rejected_logits = rejected_outputs.logits

        # Compute average log probability for chosen responses
        # We only count the response tokens (not the prompt)
        chosen_log_probs = self._get_avg_log_prob(
            chosen_logits, chosen_ids, chosen_mask, inputs.get("prompt_input_ids")
        )

        # Compute average log probability for rejected responses
        rejected_log_probs = self._get_avg_log_prob(
            rejected_logits, rejected_ids, rejected_mask, inputs.get("prompt_input_ids")
        )

        # SimPO loss: -log(sigmoid((chosen_avg_lp - rejected_avg_lp - gamma) / beta))
        logits_diff = (chosen_log_probs - rejected_log_probs - self.gamma) / self.beta
        loss = -F.logsigmoid(logits_diff).mean()

        if return_outputs:
            return loss, {
                "chosen_log_probs": chosen_log_probs.mean().item(),
                "rejected_log_probs": rejected_log_probs.mean().item(),
                "reward_margin": (chosen_log_probs - rejected_log_probs).mean().item(),
            }
        return loss

    def _get_avg_log_prob(self, logits, input_ids, attention_mask, prompt_ids=None):
        """Compute average log probability of the response tokens.

        Args:
            logits: Model logits [batch, seq, vocab]
            input_ids: Full sequence token ids [batch, seq]
            attention_mask: Attention mask [batch, seq]
            prompt_ids: Prompt token ids (to mask out prompt from loss)

        Returns:
            Average log probability per sequence [batch]
        """
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out prompt tokens if provided
        if prompt_ids is not None:
            # Create a mask that's 0 for prompt tokens and 1 for response tokens
            prompt_lengths = prompt_ids.ne(0).sum(dim=1)
            seq_len = shift_labels.shape[1]
            position_ids = torch.arange(seq_len, device=shift_labels.device).unsqueeze(
                0
            )
            response_mask = (position_ids >= (prompt_lengths.unsqueeze(1) - 1)).float()
            shift_mask = shift_mask * response_mask

        # Average log prob over response tokens
        masked_log_probs = token_log_probs * shift_mask
        seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
        avg_log_probs = masked_log_probs.sum(dim=1) / seq_lengths

        return avg_log_probs


# ==================== Data Collator ====================


class SimPODataCollator:
    """Data collator for SimPO that pads chosen/rejected pairs."""

    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        chosen_ids = [f["chosen_input_ids"] for f in features]
        chosen_masks = [f["chosen_attention_mask"] for f in features]
        rejected_ids = [f["rejected_input_ids"] for f in features]
        rejected_masks = [f["rejected_attention_mask"] for f in features]
        prompt_ids = [f["prompt_input_ids"] for f in features]

        # Pad all sequences
        chosen_ids = self._pad(chosen_ids)
        chosen_masks = self._pad(chosen_masks)
        rejected_ids = self._pad(rejected_ids)
        rejected_masks = self._pad(rejected_masks)
        prompt_ids = self._pad(prompt_ids)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_masks,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_masks,
            "prompt_input_ids": prompt_ids,
        }

    def _pad(self, sequences):
        max_len = max(len(s) for s in sequences)
        max_len = min(max_len, self.max_length)
        padded = []
        for s in sequences:
            s = s[:max_len]
            pad_len = max_len - len(s)
            padded.append(s + [self.tokenizer.pad_token_id or 0] * pad_len)
        return torch.tensor(padded, dtype=torch.long)


# ==================== Time Stopping Callback ====================


class TimeStoppingCallback(TrainerCallback):
    """Stop training after a specified duration."""

    def __init__(self, max_hours: float = 5.0):
        self.max_seconds = max_hours * 3600
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        hours = self.max_seconds / 3600
        print(f"Time limit: {hours:.1f} hours ({self.max_seconds:.0f}s)")

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        remaining = self.max_seconds - elapsed

        if remaining <= 0:
            control.should_training_stop = True
            hours = elapsed / 3600
            print(f"\nTime limit reached: {hours:.2f}h elapsed. Stopping.")
        elif state.global_step % 50 == 0:
            rem_hours = remaining / 3600
            elapsed_hours = elapsed / 3600
            print(f"  Time: {elapsed_hours:.1f}h elapsed, {rem_hours:.1f}h remaining")

        return control


# ==================== Main ====================


def main():
    parser = HfArgumentParser(SimPOConfig)
    config = parser.parse_args_into_dataclasses()[0]

    print("=" * 60)
    print("SimPO: Simple Preference Optimization")
    print(f"Model: {config.model}")
    print(f"Gamma (margin): {config.gamma}")
    print(f"Beta (temperature): {config.beta}")
    print("=" * 60)

    # Initialize wandb
    if config.wandb_project:
        import wandb

        wandb.init(project=config.wandb_project, config=vars(config))

    # Create output dir
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {config.model}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float32,
        "device_map": "auto",
    }
    if config.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

    # Apply LoRA
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset
    raw_dataset = load_simpo_dataset(config, tokenizer)

    # Tokenize
    tokenized = raw_dataset.map(
        lambda x: tokenize_simpo(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    # Filter out sequences that are too short
    tokenized = tokenized.filter(
        lambda x: len(x["chosen_input_ids"]) > 2 and len(x["rejected_input_ids"]) > 2
    )

    print(f"Training on {len(tokenized)} preference pairs")

    # Training arguments — use max_steps=1M (we stop by time, not steps)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=1_000_000,  # We stop via TimeStoppingCallback
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        bf16=config.bf16,
        report_to="wandb" if config.wandb_project else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # Data collator
    data_collator = SimPODataCollator(tokenizer, config.max_seq_length)

    # Time-based stopping callback
    time_callback = TimeStoppingCallback(max_hours=config.training_hours)

    # Create trainer
    trainer = SimPOTrainer(
        gamma=config.gamma,
        beta=config.beta,
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[time_callback],
    )

    # Train
    print(f"Starting SimPO training for {config.training_hours}h...")
    trainer.train()

    # Save
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("=" * 60)
    print(f"SimPO training complete! Model saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
