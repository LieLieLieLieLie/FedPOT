# FedPOT

Core implementation of FedPOT: privacy-preserving federated transfer learning via differentially private prototype release, partial optimal transport alignment, and transport-conditioned generation.

## Repository Layout

```text
core/              # DP prototype release, OT alignment, generation, filtering
data/              # Dataset placement guide only; raw data are not included
evaluation/        # Downstream evaluation utilities
experiments/       # Baselines, ablation, diagnostics, visualization scripts
feddata/           # CWRU and Office-Home data loaders
models/            # Neural network modules
config.py          # Hyperparameters and experiment configuration
main.py            # Main experiment entry
prepare_data.py    # Dataset preparation helper
trainer.py         # FedPOT training and evaluation pipeline
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data

Put datasets under `data/`. See `data/README.md` for the expected layout.
Raw datasets are intentionally ignored by Git because CWRU and Office-Home are large local experiment assets.

You can check or prepare the expected datasets with:

```bash
python prepare_data.py --dataset cwru
python prepare_data.py --dataset office_home
```

## Usage

Run from the repository root:

```bash
python main.py --dataset cwru --baselines
python main.py --dataset cwru --ablation
python main.py --dataset office_home --baselines
python main.py --dataset office_home --ablation
```

Additional experiment switches include:

```bash
python main.py --privacy
python main.py --hyperparam
python main.py --viz
python main.py --comm
```

By default, experiment outputs are written to `results/`, logs to `logs/`, and model artifacts to `checkpoints/`.
These generated directories are ignored by Git.

## Author

Zilong Yin  
College of Electronic and Information Engineering, Tongji University  
Email: zilong_yin@163.com
