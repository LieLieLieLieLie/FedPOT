# FedPOT

FedPOT is a privacy-preserving federated transfer learning framework based on differentially private prototype release, relational partial optimal transport, and transport-conditioned feature generation. The project includes the experimental code and paper assets for evaluating FedPOT on CWRU bearing fault diagnosis and Office-Caltech-10 cross-domain transfer.

## Overview

Federated transfer learning is useful when two parties own complementary feature views but cannot exchange raw data. FedPOT addresses three coupled challenges in this setting:

- privacy-preserving source-side knowledge release through DP-sanitized prototypes;
- semantic alignment between heterogeneous source and target feature spaces through relational partial optimal transport;
- complementary feature generation through a transport-conditioned CVAE and late-fusion downstream classifier.

The implementation supports baseline comparison, ablation studies, privacy-budget analysis, hyperparameter sensitivity, representation visualization, OT heatmaps, calibration plots, and communication-cost analysis.

## Repository Structure

```text
FedPOT/
  code/                       # Main implementation
    core/                     # DP prototype, OT alignment, CVAE generation, filtering
    evaluation/               # Downstream metrics and classifiers
    experiments/              # Baselines, ablation, visualization, diagnostics
    feddata/                  # CWRU and Office-Caltech-10 data loaders
    models/                   # Neural network modules
    main.py                   # Main experiment entry
    prepare_data.py           # Dataset preparation helper
    requirements.txt          # Python dependencies
  docs/                       # Notes and experiment guides
  drawio/                     # Source files for method diagrams
  paper/                      # LaTeX paper source and figures
```

Generated files such as checkpoints, logs, caches, local datasets, Python bytecode, and LaTeX auxiliary files are ignored by `.gitignore`.

## Installation

Create a Python environment and install the dependencies:

```bash
cd code
pip install -r requirements.txt
```

The experiments were developed with Python 3.9/3.10 and PyTorch 2.x. GPU acceleration is recommended for Office-Caltech-10 feature extraction and CVAE training.

## Data

Place datasets under `code/data/`:

```text
code/data/
  cwru/
    97.mat
    98.mat
    ...
    133.mat
  office_caltech_10/
    amazon/
    caltech/
    dslr/
    webcam/
```

CWRU uses four load domains and four fault classes. Office-Caltech-10 uses four visual domains and ten object classes. The Office-Caltech loader extracts ResNet-50 features and caches them in `code/data/office_caltech_10/.cache/`.

If data are not present, use the dataset preparation helper where applicable:

```bash
cd code
python prepare_data.py --dataset cwru
```

## Running Experiments

Run the complete pipeline:

```bash
cd code
python main.py --full
```

Run the main baseline and ablation experiments:

```bash
python main.py --dataset cwru --baselines
python main.py --dataset cwru --ablation
python main.py --dataset office_caltech --baselines
python main.py --dataset office_caltech --ablation
```

Run additional diagnostics:

```bash
python main.py --privacy
python main.py --hyperparam
python main.py --viz
python main.py --comm
```

Outputs are written to:

```text
code/results/figures/       # PDF figures
code/results/tables/        # XLSX result tables
code/checkpoints/           # Model states and intermediate artifacts
code/logs/                  # Experiment logs
```

## Paper

The manuscript source is in `paper/`:

```bash
cd paper
pdflatex cas-sc-template.tex
pdflatex cas-sc-template.tex
```

Experiment figures used by the paper are stored in `paper/figures/experiment/`.

## Reproducibility Notes

- The default random seed is `42`.
- Default privacy parameters are `epsilon=2.0`, `delta=1e-5`, and `max_norm=1.0`.
- CWRU defaults to source load `0` and target load `2`.
- Office-Caltech-10 defaults to source domain `amazon` and target domain `dslr`.
- Some results depend on stochastic neural training and may vary slightly across hardware, PyTorch versions, and CUDA/cuDNN settings.

## Citation

If this repository helps your research, please cite the associated paper when it becomes available:

```bibtex
@article{yin2026fedpot,
  title   = {FedPOT: Privacy-Preserving Federated Transfer Learning via Partial Optimal Transport and Transport-Conditioned Generation},
  author  = {Yin, Zilong},
  journal = {Preprint},
  year    = {2026}
}
```

## Author

Zilong Yin  
College of Electronic and Information Engineering, Tongji University  
Shanghai 201804, China  
ORCID: 0009-0002-7994-2772  
Email: zilong_yin@163.con

## License

The open-source license has not been specified yet. Please add a `LICENSE` file before public release if you want others to reuse, modify, or redistribute the code under explicit terms.
