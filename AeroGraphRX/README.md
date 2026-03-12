# AeroGraphRX: Graph Signal Processing for Cooperative Multi-Station Signal Detection and Tracking

**AeroGraphRX** is a comprehensive framework for simulating and evaluating cooperative signal detection, classification, and tracking using Graph Signal Processing (GSP) techniques.

## Overview

AeroGraphRX implements a multi-station GSP framework combining CNN-GCN networks, VAE-based novelty detection, JPDA tracking, and stealth detection algorithms. The system is designed for cooperative detection of aircraft signals (ADS-B), unknown modulations, and low-RCS (stealth) targets.

## Installation

### Requirements
- Python 3.8+
- pip

### Quick Start

```bash
# Clone or download the repository
cd AeroGraphRX

# Install dependencies
pip install -r requirements.txt

# (Optional) Install the package in development mode
pip install -e .
```

## Quick Start Guide

### 1. Generate Synthetic Dataset

```bash
python scripts/generate_dataset.py
```

This generates synthetic signal datasets for all modulation types with SNR variations:
- Output: `data/synthetic/train.npz`, `data/synthetic/val.npz`, `data/synthetic/test.npz`
- Includes seed log for reproducibility

### 2. Run Monte Carlo Simulation

```bash
python scripts/run_simulation.py
```

Executes 10,000 Monte Carlo trials comparing our methods against baselines:
- Outputs: `data/results.npz` with AUC, confidence intervals, and statistical test results
- Performs DeLong tests, McNemar tests, and paired t-tests with Bonferroni correction

### 3. Run Ablation Studies

```bash
python scripts/run_ablation.py
```

Tests system sensitivity to key hyperparameters:
- Spectral cutoff K0 (optimal ~10)
- Graph sparsity threshold epsilon (optimal ~50% edge density)
- Adjacency weight components (spatial, spectral, temporal)
- GCN layer count (optimal = 2)
- Graph smoothness mu (optimal = 0.01)

### 4. Generate Paper Figures

```bash
python scripts/generate_figures.py
```

Produces all 7 publication-quality figures:
- fig1_roc_curves.png - Detection ROC with bootstrap CI
- fig2_graph_analysis.png - GSP ablation studies
- fig3_classification.png - Modulation classification with calibration
- fig4_tracking.png - JPDA tracking vs baselines
- fig5_stealth.png - Stealth detection performance
- fig6_architecture.png - System architecture diagram
- fig7_vae_detection.png - VAE-based novelty detection

### Run Complete Pipeline

```bash
bash scripts/run_all.sh
```

Executes all steps (dataset generation → simulation → ablation → figures) in sequence.

## Repository Structure

```
AeroGraphRX/
├── configs/
│   └── default.yaml              # Complete hyperparameter configuration
├── scripts/
│   ├── generate_dataset.py        # Synthetic signal generation
│   ├── run_simulation.py          # Monte Carlo simulation
│   ├── run_ablation.py            # Ablation studies
│   ├── generate_figures.py        # Publication figures
│   └── run_all.sh                 # Complete pipeline
├── tests/
│   ├── test_core.py               # Core module tests
│   └── test_models.py             # Model tests
├── data/
│   ├── synthetic/                 # Generated datasets
│   ├── results.npz                # Simulation results
│   └── ablation_results.npz       # Ablation results
├── figures/                       # Output figures
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Reproducing Paper Results

### Full Reproducibility

To fully reproduce all paper results:

```bash
bash scripts/run_all.sh
```

This will:
1. Generate 10 million synthetic signal samples across all modulation types
2. Run 10,000 Monte Carlo trials
3. Execute 5 ablation studies
4. Generate all 7 paper figures

**Runtime**: ~2-4 hours on a modern CPU

### Expected Results

Key results should match paper claims:

| Metric | Expected | Figure |
|--------|----------|--------|
| ADS-B Detection AUC | 0.97 +/- 0.01 | Fig 1a |
| Stealth Detection AUC | 0.89 +/- 0.03 | Fig 1b |
| Novelty Detection AUC | 0.93 +/- 0.02 | Fig 1c |
| Classification Accuracy @ SNR=10dB | 0.90+ | Fig 3b |
| TDoA CEP50 | 420 m | Fig 4a |
| JPDA Track Continuity | 96.3% | Fig 4b |
| Stealth Pd @ RCS=-20dBsm | 0.70+ | Fig 5b |
| VAE Silhouette Score | 0.72 | Fig 7a |

## Configuration

All hyperparameters are in `configs/default.yaml`:

### Key Parameters

**Simulation:**
- n_mc_trials: 10000 (Monte Carlo trials)
- n_signal_events: 500 (events per trial)
- snr_range_db: [-10, 20] dB

**Graph Construction:**
- alpha_s, alpha_f, alpha_t: Spatial/spectral/temporal weights (0.4, 0.4, 0.2)
- spectral_cutoff_K0: 50 (spectral bands)
- epsilon: 0.5 (sparsity threshold)

**CNN-GCN:**
- cnn_filters: [32, 64, 128]
- gcn_layers: 2
- gcn_hidden_dim: 128

**VAE:**
- latent_dim: 32
- beta: 0.1 (KL weight)

**Evaluation:**
- n_bootstrap: 1000 (bootstrap samples)
- bonferroni_pairs: 6 (number of pairwise comparisons)

Modify `configs/default.yaml` to experiment with different settings.

## Testing

Run pytest to verify core functionality:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run specific test class
pytest tests/test_core.py::TestGraphLaplacian -v
```

### Test Coverage

**Core tests** (test_core.py):
- Graph Laplacian properties
- Graph Fourier Transform
- Chebyshev polynomial filters
- TDoA measurement geometry
- CRLB bounds
- Signal generation
- ROC/AUC computation
- Bootstrap confidence intervals
- Statistical tests (McNemar, DeLong)

**Model tests** (test_models.py):
- CNN-GCN forward pass shapes
- VAE reconstruction quality
- VAE novelty detection
- JPDA track continuity
- Stealth detector PFA calibration

## Citation

If you use AeroGraphRX in your research, please cite:

```
@article{aerographrx2024,
  title={AeroGraphRX: Simulation of a Graph Signal Processing Framework
         for Cooperative Multi-Station Detection and Tracking},
  author={Usha, A. and George, Noel},
  journal={IEEE Transactions on Signal Processing},
  year={2025},
  institution={Alliance University}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2025 Usha A, Noel George, Alliance University.

## Authors

- Usha A - Alliance University
- Noel George - Alliance University

## Contact

For questions or issues:
- Email: usha@allianceuniversity.edu.in
- Email: noel@allianceuniversity.edu.in

## Acknowledgments

- Graph Signal Processing fundamentals based on Shuman et al., 2013
- JPDA tracking implementation inspired by Bar-Shalom & Li, 1995
- Statistical testing following DeLong et al., 1988 and McNemar, 1947

## References

1. Shuman, D. I., et al. (2013). The emerging field of signal processing on graphs. IEEE Signal Processing Magazine, 30(3), 83-98.

2. Bar-Shalom, Y., & Li, X. R. (1995). Multitarget-multisensor tracking. Artech House.

3. DeLong, E. R., et al. (1988). Comparing areas under ROC curves. Biometrics, 44(3), 837-845.

4. McNemar, Q. (1947). Note on sampling error of difference between correlated proportions. Psychometrika, 12(2), 153-157.

---

**Last Updated**: March 2025
**Status**: Production Ready
**Tested On**: Python 3.8, 3.9, 3.10, 3.11
