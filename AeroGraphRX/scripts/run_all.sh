#!/bin/bash
# AeroGraphRX Complete Reproducibility Pipeline
# Runs the complete pipeline from data generation to figure generation

set -e

echo ""
echo "========================================================================"
echo "AeroGraphRX Reproducibility Pipeline"
echo "========================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Check for required packages
echo ""
echo "Checking dependencies..."
python3 -c "import numpy, scipy, sklearn, torch, matplotlib, yaml; print('All dependencies available.')" || {
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
}

# Step 1: Generate synthetic dataset
echo ""
echo "========================================================================"
echo "Step 1: Generating synthetic dataset..."
echo "========================================================================"
python3 scripts/generate_dataset.py

# Step 2: Run Monte Carlo simulation
echo ""
echo "========================================================================"
echo "Step 2: Running Monte Carlo simulation..."
echo "========================================================================"
python3 scripts/run_simulation.py

# Step 3: Run ablation studies
echo ""
echo "========================================================================"
echo "Step 3: Running ablation studies..."
echo "========================================================================"
python3 scripts/run_ablation.py

# Step 4: Generate paper figures
echo ""
echo "========================================================================"
echo "Step 4: Generating paper figures..."
echo "========================================================================"
python3 scripts/generate_figures.py

echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Generated outputs:"
echo "  - Dataset: data/synthetic/{train,val,test}.npz"
echo "  - Simulation results: data/results.npz"
echo "  - Ablation results: data/ablation_results.npz"
echo "  - Figures: figures/fig*.png"
echo ""
