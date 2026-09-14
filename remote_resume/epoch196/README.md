# Remote continuation: Epoch 196 to 290

This bundle contains the exact code and state needed to continue the current
joint self-improving experiment for 95 more epochs.

## Contents

- `checkpoints/checkpoint-190.pt`: baseline state before the five-epoch trial.
- `checkpoints/checkpoint-191.pt` through `checkpoint-195.pt`: every state from
  the exact-RL + replay + diverse-Top-K trial.
- `checkpoint-193.pt` is the best fixed-validation checkpoint in that trial
  (`18.297120`).
- `checkpoint-195.pt` is the continuation state. Use it so that the next epoch
  is exactly Epoch 196 and the replay pool, reference policy, optimizer state,
  edge predictor, and RNG state are restored.
- `metadata/`: parameters, metrics, validation manifest, and final summary.
- `ENVIRONMENT.md`: source environment versions and setup guidance.
- `../../validation/macsim_fixed/`: fixed validation tensors used by training.

The six checkpoint files are committed directly to this branch, so a normal
`git clone` downloads them. Their SHA256 hashes are provided below for transfer
verification.

## Verify checkpoint transfer

From this directory:

```bash
sha256sum -c SHA256SUMS
```

All six lines must report `OK`.

## Start the 95-epoch continuation

From the repository root, pass the physical GPU index and worker count:

```bash
bash scripts/resume_epoch196_to290_tmux.sh 4 96
```

The script masks the selected physical GPU with `CUDA_VISIBLE_DEVICES`, so the
Python process correctly uses logical CUDA device 0. Override the Python binary
or output location when needed:

```bash
PYTHON_BIN=/path/to/env/bin/python \
OUTPUT_DIR=/path/to/output \
bash scripts/resume_epoch196_to290_tmux.sh 4 96
```

Monitor with:

```bash
tmux attach -t cvtsp_e196_290
```

or:

```bash
tail -f outputs/train__resume_epoch196_to290_exactrl_replay30_diverse10/live.log
```

The remote host must provide a CUDA-compatible PyTorch installation, a working
Gurobi installation/license, `git-lfs`, and the remaining packages in
`requirements.txt`. Do not copy a Gurobi license into Git.
