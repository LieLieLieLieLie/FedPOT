# FedPOT

Core implementation of FedPOT: privacy-preserving federated transfer learning via differentially private prototype release, partial optimal transport alignment, and transport-conditioned generation.

## Repository Layout

```text
code/
  core/              # DP prototype release, OT alignment, generation, filtering
  data/              # Dataset placement guide only; raw data are not included
  evaluation/        # Downstream evaluation utilities
  experiments/       # Baselines, ablation, diagnostics, visualization scripts
  feddata/           # CWRU and Office-Caltech-10 data loaders
  models/            # Neural network modules
  config.py          # Hyperparameters and experiment configuration
  main.py            # Main experiment entry
  prepare_data.py    # Dataset preparation helper
  trainer.py         # FedPOT training and evaluation pipeline
```

## Installation

Install the required Python packages:

```bash
pip install torch torchvision numpy scipy scikit-learn POT pandas matplotlib tqdm Pillow
```

## Data

Put datasets under `code/data/`. See `code/data/README.md` for the expected layout.

## Usage

Run from the `code/` directory:

```bash
python main.py --dataset cwru --baselines
python main.py --dataset cwru --ablation
python main.py --dataset office_caltech --baselines
python main.py --dataset office_caltech --ablation
```

Additional experiment switches include:

```bash
python main.py --privacy
python main.py --hyperparam
python main.py --viz
python main.py --comm
```

## Author

Zilong Yin  
College of Electronic and Information Engineering, Tongji University  
Email: zilong_yin@163.com
