# SAGE: SuperActivation-Guided Focus

This is a minimal PyTorch prototype for studying superactivation-guided sparse structure. It supports both sparse edge growth and dense-to-focused pruning with dense weights plus binary masks. It does not use PyTorch sparse tensors yet.

Included:

- `MaskedLinear`: dense `weight` parameter plus a binary edge mask
- `SparseMLP`: `784 -> hidden_dim -> hidden_dim -> 10`
- random, gradient, and SAGE growth
- pruning by lowest active weight magnitude
- MNIST and Fashion-MNIST training
- CSV logging for epoch, loss, accuracy, active parameter count, and superweight concentration
- layer-wise SAGE score and rewiring diagnostics
- dense-start focused pruning, masked neuron pruning, and physical hidden-layer compaction

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dense-To-Focused Training

This is the main path for testing whether superweights/superactivations can form first, then guide pruning:

```bash
python train.py \
  --dataset mnist \
  --dense_start \
  --growth_interval 0 \
  --epochs 20 \
  --hidden_dim 256 \
  --neuron_prune_start_epoch 5 \
  --neuron_prune_interval 1 \
  --neuron_prune_fraction 0.05 \
  --neuron_protect_fraction 0.05 \
  --compact_epoch -1 \
  --log_path logs/sage_focus.csv
```

The phases are:

1. Dense warmup: all edges are active while SAGE tracks activation and gradient-output signals.
2. Stage 1 masked neuron pruning: weak hidden neurons are killed by zeroing their incoming row and outgoing column masks.
3. Focused masked training: later epochs train only the surviving masked subnetwork.
4. Stage 2 final physical compaction: after the last training epoch, surviving hidden neurons are copied into a smaller `SparseMLP`, reducing actual dense layer dimensions.

Use a positive value like `--compact_epoch 12` to compact during training. Use `--compact_epoch -1` for the cleaner first experiment: prune with masks throughout training, then compact once at the end and log the final smaller physical model.

## Sparse-Growth Training

```bash
python train.py --dataset mnist --growth_mode sage --sparsity 0.95 --epochs 20
```

The first run downloads datasets into `data/`. Logs are written to `logs/train.csv` by default.

## Synthetic Smoke Test

This checks the sparse mechanics without downloading a dataset:

```bash
python smoke_test.py
```

It verifies that SAGE scores update during backward, inactive gradients are zeroed before the optimizer step, inactive weights are zero after the step, the active edge count stays constant after prune-grow, hidden neurons can be mask-pruned, and the compacted model matches the masked model's output.

## Dataset Smoke Run

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
- `--dense_start`: start with all edges active
- `--neuron_prune_start_epoch`: first epoch for masked neuron pruning; `0` disables it
- `--neuron_prune_interval`: epochs between neuron-pruning events
- `--neuron_prune_fraction`: fraction of currently active hidden neurons to prune per event
- `--neuron_protect_fraction`: fraction of highest-score hidden neurons protected from pruning
- `--min_hidden_neurons`: minimum surviving neurons per hidden layer
- `--compact_epoch`: epoch when surviving hidden neurons are copied into a physically smaller model; `-1` compacts after the final training epoch, `0` disables compaction

Additional convenience arguments include `--batch_size`, `--lr`, `--data_dir`, `--log_path`, `--seed`, `--num_workers`, `--train_batches`, and `--eval_batches`.

## Comparing Growth Modes

Run the same settings while changing only `--growth_mode`:

```bash
python train.py --dataset mnist --growth_mode random --sparsity 0.95 --epochs 20 --log_path logs/random.csv
python train.py --dataset mnist --growth_mode gradient --sparsity 0.95 --epochs 20 --log_path logs/gradient.csv
python train.py --dataset mnist --growth_mode sage --sparsity 0.95 --epochs 20 --log_path logs/sage.csv
```

The CSV includes the required training fields plus extra diagnostics:

- `growth_events`, `pruned_edges`, `grown_edges`
- `mean_pruned_weight_magnitude`, `mean_grown_score`
- `neuron_prune_events`, `pruned_neurons`, `pruned_hidden1`, `pruned_hidden2`, `mean_pruned_neuron_score`
- `physical_parameter_count`, `hidden1_dim`, `hidden2_dim`, `active_hidden1`, `active_hidden2`
- per-layer active parameter counts, densities, active/inactive SAGE score means, max SAGE score, and last prune-grow counts

## Notes

`MaskedLinear` stores `score_ema[j, i]` as an EMA of:

```text
mean_batch(|grad_output_j|) * mean_batch(|activation_i|)
```

Inactive edges are kept inactive by masking gradients before `optimizer.step()` and masking weights immediately after the step. Superweight concentration is logged as the share of active weight magnitude held by the top 1% of active weights.

For hidden neuron pruning, SAGE scores each hidden neuron using activation EMA, gradient-output EMA, incoming/outgoing active weight flow, and incoming/outgoing SAGE edge score flow. Stage 1 reduces logical active structure with masks. Stage 2 reduces actual dense layer dimensions by copying only surviving hidden neurons.
