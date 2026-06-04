# SAGE: SuperActivation-Guided Focus

This is a minimal PyTorch prototype for studying superactivation-guided sparse structure. It supports both sparse edge growth and dense-to-focused pruning with dense weights plus binary masks. It does not use PyTorch sparse tensors yet.

Included:

- `MaskedLinear`: dense `weight` parameter plus a binary edge mask
- `MaskedConv2d`: dense convolution weights plus a binary kernel mask
- `SparseMLP`: `784 -> hidden_dim -> hidden_dim -> 10`
- `SageCifarCNN`: compactable CIFAR-10 CNN with structured channel pruning
- random, gradient, and SAGE growth
- pruning by lowest active weight magnitude
- MNIST, Fashion-MNIST, and CIFAR-10 training
- CSV logging for epoch, loss, accuracy, active parameter count, and superweight concentration
- layer-wise SAGE score and rewiring diagnostics
- dense-start focused pruning, masked neuron pruning, and physical hidden-layer compaction
- dense-start focused channel pruning and physical convolutional compaction for CIFAR-10

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running On Colab

Use Colab for CIFAR-10 or larger runs if your laptop is slow. The notebook is:

```text
notebooks/sage_cifar10_colab.ipynb
```

Colab clones from GitHub, so push your latest local `main` first:

```bash
git add README.md run_multiseed.py smoke_test.py summarize_logs.py train.py src notebooks
git commit -m "Add CIFAR-10 SAGE channel pruning"
git push origin main
```

Then open `notebooks/sage_cifar10_colab.ipynb` in Colab, choose `Runtime -> Change runtime type -> T4 GPU` or better, and run the cells in order. Start with the short CIFAR sanity run before launching the full multiseed sweep.

If the GitHub repo is private, create a GitHub token with read access to this repo, then add it in Colab under `Secrets` with the name `GITHUB_TOKEN` before running the clone cell. The notebook will use that secret automatically.

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
  --neuron_prune_mode sage \
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

## CIFAR-10 Channel Pruning

MNIST and Fashion-MNIST are sanity checks. The next meaningful benchmark is CIFAR-10 with a convolutional model:

```bash
python train.py \
  --dataset cifar10 \
  --model cifar_cnn \
  --dense_start \
  --growth_interval 0 \
  --epochs 80 \
  --post_compact_epochs 20 \
  --base_channels 64 \
  --batch_size 128 \
  --lr 0.001 \
  --neuron_prune_start_epoch 20 \
  --neuron_prune_end_epoch 80 \
  --neuron_prune_interval 2 \
  --neuron_prune_fraction 0.03 \
  --neuron_prune_mode sage \
  --neuron_protect_fraction 0.05 \
  --min_hidden_neurons 8 \
  --compact_epoch -1 \
  --sage_focus_start_epoch 20 \
  --sage_grad_boost 1.15 \
  --sage_boost_fraction 0.03 \
  --weak_grad_decay 0.95 \
  --log_path logs/cifar10_sage_channels.csv
```

For CIFAR-10, `prune_weak_neurons` means "prune weak convolution channels." A pruned channel zeros the producer conv output, the corresponding downstream conv input, and finally the classifier input for the last conv layer. At compaction, surviving channels are copied into smaller dense convolution tensors, so `physical_parameter_count` drops without using sparse tensors.

Run the CIFAR multiseed comparison:

```bash
python run_multiseed.py \
  --dataset cifar10 \
  --model cifar_cnn \
  --seeds 1 2 3 \
  --epochs 80 \
  --post_compact_epochs 20 \
  --base_channels 64 \
  --output_dir logs/cifar10_stage3 \
  --neuron_prune_start_epoch 20 \
  --neuron_prune_end_epoch 80 \
  --neuron_prune_interval 2 \
  --neuron_prune_fraction 0.03 \
  --sage_focus_start_epoch 20 \
  --sage_grad_boost 1.15 \
  --sage_boost_fraction 0.03 \
  --weak_grad_decay 0.95
```

Then summarize:

```bash
python summarize_logs.py --aggregate logs/cifar10_stage3/*.csv \
  --output results/cifar10_stage3_aggregate.csv
```

## SAGE Strengthening

SAGE-Focus v2 can actively strengthen high-SAGE paths during training. After backward, before the optimizer step, the top active edges by SAGE score can receive a gradient boost while the weaker active edges can be damped:

```bash
python train.py \
  --dataset fashion-mnist \
  --dense_start \
  --growth_interval 0 \
  --epochs 20 \
  --post_compact_epochs 10 \
  --hidden_dim 256 \
  --neuron_prune_start_epoch 5 \
  --neuron_prune_end_epoch 20 \
  --neuron_prune_interval 1 \
  --neuron_prune_fraction 0.05 \
  --neuron_prune_mode sage \
  --neuron_protect_fraction 0.05 \
  --compact_epoch -1 \
  --sage_focus_start_epoch 5 \
  --sage_grad_boost 1.25 \
  --sage_boost_fraction 0.05 \
  --weak_grad_decay 0.90 \
  --log_path logs/sage_strengthened.csv
```

This uses 20 prune/discovery epochs, compacts at epoch 20, then fine-tunes the compact model for 10 more epochs. For a fair dense baseline with the same total training budget, run dense for 30 epochs.

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
It also runs a synthetic CIFAR-shaped CNN check for channel pruning and convolutional compaction.

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

CIFAR-10 quick debug run:

```bash
python train.py \
  --dataset cifar10 \
  --model cifar_cnn \
  --dense_start \
  --epochs 1 \
  --base_channels 16 \
  --train_batches 2 \
  --eval_batches 2 \
  --num_workers 0 \
  --log_path logs/cifar10_smoke.csv
```

## CLI Arguments

- `--dataset`: `mnist`, `fashion-mnist`, `fashion_mnist`, `cifar10`, or `cifar-10`
- `--model`: `auto`, `mlp`, or `cifar_cnn`; `auto` uses `cifar_cnn` for CIFAR-10
- `--sparsity`: fraction of masked-out weights at initialization
- `--growth_mode`: `random`, `gradient`, or `sage`
- `--epochs`: number of training epochs
- `--hidden_dim`: hidden layer width
- `--base_channels`: starting width for the CIFAR CNN; the conv stack is `C, C, 2C, 2C, 4C`
- `--growth_interval`: optimizer steps between prune-grow updates; set to `0` to disable growth
- `--prune_fraction`: fraction of active weights to prune and regrow at each update
- `--dense_start`: start with all edges active
- `--neuron_prune_start_epoch`: first epoch for masked neuron pruning; `0` disables it
- `--neuron_prune_end_epoch`: last epoch for masked neuron pruning; `0` means the main `--epochs` value
- `--neuron_prune_interval`: epochs between neuron-pruning events
- `--neuron_prune_fraction`: fraction of currently active hidden neurons to prune per event
- `--neuron_prune_mode`: `sage`, `magnitude`, or `random`
- `--neuron_protect_fraction`: fraction of highest-score hidden neurons protected from pruning
- `--min_hidden_neurons`: minimum surviving neurons per hidden layer
- `--compact_epoch`: epoch when surviving hidden neurons are copied into a physically smaller model; `-1` compacts after the final training epoch, `0` disables compaction
- `--post_compact_epochs`: extra epochs after the main phase, useful for fine-tuning the compacted model
- `--sage_focus_start_epoch`: first epoch for SAGE-focused gradient scaling; `0` follows `--neuron_prune_start_epoch`
- `--sage_focus_end_epoch`: final epoch for SAGE-focused gradient scaling; `0` means no end
- `--sage_grad_boost`: gradient multiplier for top-SAGE active edges
- `--sage_boost_fraction`: fraction of active edges to boost by SAGE score
- `--weak_grad_decay`: gradient multiplier for non-boosted active edges

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

## Comparing Focused Pruning Modes

After running a dense baseline and SAGE-Focus, run these pruning baselines:

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
  --neuron_prune_mode magnitude \
  --neuron_protect_fraction 0.05 \
  --compact_epoch -1 \
  --log_path logs/magnitude_focus.csv

python train.py \
  --dataset mnist \
  --dense_start \
  --growth_interval 0 \
  --epochs 20 \
  --hidden_dim 256 \
  --neuron_prune_start_epoch 5 \
  --neuron_prune_interval 1 \
  --neuron_prune_fraction 0.05 \
  --neuron_prune_mode random \
  --neuron_protect_fraction 0.05 \
  --compact_epoch -1 \
  --log_path logs/random_focus.csv
```

Summarize the comparison:

```bash
python summarize_logs.py \
  logs/dense_baseline.csv \
  logs/sage_focus.csv \
  logs/magnitude_focus.csv \
  logs/random_focus.csv
```

## Multi-Seed Experiments

Run the full dense/SAGE/magnitude/random comparison for seeds 1, 2, and 3:

```bash
python run_multiseed.py --seeds 1 2 3
```

Logs are written to `logs/multiseed/` with names like `sage_seed2.csv`.
The runner uses `.venv/bin/python` automatically when that file exists. You can override it with `--python /path/to/python`.

Summarize each run:

```bash
python summarize_logs.py logs/multiseed/*.csv
```

Summarize mean and standard deviation by pruning mode:

```bash
python summarize_logs.py --aggregate logs/multiseed/*.csv
```

Useful runner options:

- `--modes dense sage magnitude random`: choose which experiment modes to run
- `--dataset mnist` or `--dataset fashion-mnist`
- `--skip_existing`: resume without rerunning completed logs
- `--dry_run`: print commands without running them
- `--train_batches` and `--eval_batches`: quick debugging runs
- `--python`: choose the Python executable used for `train.py`
- SAGE strengthening runner options include `--post_compact_epochs`, `--sage_grad_boost`, `--sage_boost_fraction`, and `--weak_grad_decay`

## Notes

`MaskedLinear` stores `score_ema[j, i]` as an EMA of:

```text
mean_batch(|grad_output_j|) * mean_batch(|activation_i|)
```

Inactive edges are kept inactive by masking gradients before `optimizer.step()` and masking weights immediately after the step. Superweight concentration is logged as the share of active weight magnitude held by the top 1% of active weights.

For hidden neuron pruning, SAGE scores each hidden neuron using activation EMA, gradient-output EMA, incoming/outgoing active weight flow, and incoming/outgoing SAGE edge score flow. Stage 1 reduces logical active structure with masks. Stage 2 reduces actual dense layer dimensions by copying only surviving hidden neurons.
