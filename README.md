# SimPO Training

**SimPO: Simple Preference Optimization with a Reference-Free Reward**

Based on [SimPO (arxiv 2405.14734)](https://arxiv.org/abs/2405.14734)

## Key Idea

DPO uses a reference model to prevent reward hacking. **SimPO removes the reference model entirely** by using sequence-level average log probability as the implicit reward, with a target reward margin (gamma) to separate chosen from rejected responses.

**Loss function:**

```
L_SimPO = -log(σ((avg_lp_chosen - avg_lp_rejected - γ) / β))
```

Where:
- `avg_lp_chosen = log π(y_w|x) / |y_w|` — avg log-prob of chosen response
- `avg_lp_rejected = log π(y_l|x) / |y_l|` — avg log-prob of rejected response
- `γ` (gamma) — target reward margin (default: 1.0)
- `β` (beta) — temperature for sigmoid (default: 0.01)

## Advantages over DPO

1. **No reference model** — saves ~50% memory during training
2. **No KL penalty** — simpler objective
3. **Length normalization** — sequence-level avg log-prob prevents length bias
4. **Often better** — margin-based reward is more stable

## Files

| File | Description |
|------|-------------|
| `train_simpo.py` | Full SimPO trainer (PyTorch + transformers + PEFT) |
| `train_simpo_pure.py` | Pure Python SimPO (no torch, numpy optional) |
| `train_all_sizes.sh` | Train across Qwen2.5 sizes (1.5B–14B) |
| `requirements.txt` | Python dependencies |

## Quick Start

### Pure Python (tiny model, CPU)
```bash
python3 train_simpo_pure.py
```

### Full GPU training
```bash
pip install -r requirements.txt

# Single model
python train_simpo.py --model Qwen/Qwen2.5-1.5B-Instruct

# With LoRA (for larger models)
python train_simpo.py --model Qwen/Qwen2.5-7B-Instruct --use_lora

# All sizes
bash train_all_sizes.sh
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gamma` | 1.0 | Reward margin between chosen and rejected |
| `--beta` | 0.01 | Temperature for sigmoid |
| `--epochs` | 3 | Training epochs |
| `--batch_size` | 4 | Per-device batch size |
| `--learning_rate` | 5e-7 | Learning rate |
| `--use_lora` | False | Enable LoRA for large models |

## Supported Datasets

- `argilla/ultrafeedback-binarized-preferences` (default)
- Any dataset with `chosen`/`rejected` columns
- Custom: provide prompt + chosen + rejected format
