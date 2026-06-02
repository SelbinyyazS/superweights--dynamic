# SAGE: SuperActivation-Guided Edge Growth

This is a minimal PyTorch prototype for dynamic sparse training with dense weights and binary masks. It does not use PyTorch sparse tensors yet.

Included:

- `MaskedLinear`: dense `weight` parameter plus a binary edge mask
- `SparseMLP`: `784 -> hidden_dim -> hidden_dim -> 10`
- random, gradient, and SAGE growth
- pruning by lowest active weight magnitude
- MNIST and Fashion-MNIST training
- CSV logging for epoch, loss, accuracy, active parameter count, and superweight concentration

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --dataset mnist --growth_mode sage --sparsity 0.95 --epochs 20
```

The first run downloads datasets into `data/`. Logs are written to `logs/train.csv` by default.

## Smoke Run

```bash
python train.py \
  --dataset mnist \
  --growth_mode sage \
  --sparsity 0.95 \
  --epochs 1 \
  --hidden_dim 64 \
  --growth_interval 10 \
  --train_batches 5 \
  --eval_batches 5 \
  --log_path logs/smoke.csv
```

## CLI Arguments

- `--dataset`: `mnist`, `fashion-mnist`, or `fashion_mnist`
- `--sparsity`: fraction of masked-out weights at initialization
- `--growth_mode`: `random`, `gradient`, or `sage`
- `--epochs`: number of training epochs
- `--hidden_dim`: hidden layer width
- `--growth_interval`: optimizer steps between prune-grow updates; set to `0` to disable growth
- `--prune_fraction`: fraction of active weights to prune and regrow at each update

Additional convenience arguments include `--batch_size`, `--lr`, `--data_dir`, `--log_path`, `--seed`, `--num_workers`, `--train_batches`, and `--eval_batches`.

## Notes

`MaskedLinear` stores `score_ema[j, i]` as an EMA of:

```text
mean_batch(|grad_output_j|) * mean_batch(|activation_i|)
```

Inactive edges are kept inactive by masking gradients before `optimizer.step()` and masking weights immediately after the step. Superweight concentration is logged as the share of active weight magnitude held by the top 1% of active weights.
