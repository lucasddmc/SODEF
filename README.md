# SODEF: Stable Neural ODE with Lyapunov-Stable Equilibrium Points

This repository contains a clean implementation of SODEF (Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks) optimized for the APUANA cluster at CIn-UFPE.

## Overview

SODEF is a defense mechanism that enhances the adversarial robustness of neural networks by:
1. Using a pretrained backbone (ResNet) for feature extraction
2. Adding a Neural ODE component with Lyapunov stability constraints
3. Training the ODE component to create stable equilibrium points that resist adversarial perturbations

## Project Structure

```
lucas/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── slurm/                   # SLURM job scripts for APUANA cluster
│   ├── setup_env.sh         # Environment setup
│   ├── train_baseline.sh    # Baseline model training
│   └── train_sodef.sh       # SODEF defense training
├── src/                     # Source code
│   ├── models/              # Model implementations
│   │   ├── resnet.py        # ResNet backbone for CIFAR-10
│   │   ├── sodef.py         # SODEF Neural ODE components
│   │   └── combined.py      # Combined backbone + SODEF models
│   ├── training/            # Training modules
│   │   ├── baseline_trainer.py  # Baseline training logic
│   │   └── sodef_trainer.py     # SODEF training logic
│   ├── utils/               # Utility functions
│   │   └── data_loaders.py  # CIFAR-10 data loading
│   ├── train_baseline.py    # Baseline training script
│   ├── train_sodef.py       # SODEF training script
│   └── evaluate.py          # Adversarial evaluation script
├── config/                  # Configuration files
│   ├── baseline_config.yaml # Baseline training config
│   ├── sodef_config.yaml    # SODEF training config
│   └── eval_config.yaml     # Evaluation config
├── data/                    # Data directory (auto-created)
└── results/                 # Results directory (auto-created)
```

## Setup Instructions

### 1. Environment Setup on APUANA Cluster

```bash
# Submit environment setup job
sbatch slurm/setup_env.sh

# Or manually setup:
module load Python/3.10
python -m venv $HOME/sodef_env
source $HOME/sodef_env/bin/activate
pip install -r requirements.txt
```

### 2. Local Development Setup

```bash
# Create virtual environment
python -m venv sodef_env
source sodef_env/bin/activate  # Linux/Mac
# or
sodef_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Stage 1: Train Baseline ResNet

**On APUANA Cluster:**
```bash
sbatch slurm/train_baseline.sh
```

**Locally:**
```bash
cd src
python train_baseline.py --config ../config/baseline_config.yaml
```

### Stage 2: Train SODEF Defense

**On APUANA Cluster:**
```bash
sbatch slurm/train_sodef.sh
```

**Locally:**
```bash
cd src
python train_sodef.py --config ../config/sodef_config.yaml
```

### Stage 3: Evaluate Adversarial Robustness

**Locally (recommended):**
```bash
cd src
python evaluate.py --config ../config/eval_config.yaml --model both
```

## Configuration

### Baseline Training (`config/baseline_config.yaml`)
- Model architecture (ResNet-18/34/50)
- Training hyperparameters (learning rate, epochs, etc.)
- Data loading settings

### SODEF Training (`config/sodef_config.yaml`)
- ODE dimension and integration settings
- Lyapunov stability regularization weights
- Two-phase training configuration

### Evaluation (`config/eval_config.yaml`)
- Attack configurations (AutoAttack, APGD, FAB, Square)
- Model checkpoint paths
- Evaluation settings

## Key Features

### SODEF Components
- **Neural ODE Function**: Implements the differential equation with stability constraints
- **Lyapunov Regularizers**: Diagonal and off-diagonal Jacobian regularization
- **Two-Phase Training**: Stability training followed by classification fine-tuning
- **Adaptive Integration**: Temporary ODE for training, full integration for inference

### Training Pipeline
1. **Baseline Training**: Standard ResNet training on CIFAR-10
2. **Stability Phase**: Train ODE with only stability regularizers (backbone frozen)
3. **Classification Phase**: Fine-tune with classification + stability losses
4. **Full Integration**: Switch to full ODE integration for final evaluation

### Evaluation
- **AutoAttack**: Standard adversarial robustness evaluation
- **Multiple Attacks**: L∞ and L2 norm attacks
- **Comparison**: Baseline vs SODEF-enhanced models

## APUANA Cluster Specifics

### Resource Requirements
- **Baseline Training**: 1 GPU, 32GB RAM, 6 hours
- **SODEF Training**: 1 GPU, 32GB RAM, 8 hours (due to Jacobian computation)
- **Evaluation**: Can run locally (CPU sufficient for CIFAR-10)

### Job Submission
```bash
# Check job status
squeue -u $USER

# Monitor job output
tail -f logs/baseline_*.out
tail -f logs/sodef_*.out

# Cancel job if needed
scancel <job_id>
```

## Expected Results

Based on the original SODEF paper, you should expect:
- **Baseline ResNet**: ~85% clean accuracy, ~53% robust accuracy (AutoAttack L∞)
- **SODEF-Enhanced**: Similar clean accuracy, ~58-70% robust accuracy improvement

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size in config files
2. **Slow Jacobian computation**: Reduce `num_samples` in SODEF config
3. **Checkpoint not found**: Ensure baseline training completed successfully

### Monitoring
- Check SLURM logs in `logs/` directory
- Monitor GPU usage with `nvidia-smi`
- Tensorboard logs saved in checkpoint directories

## References

- [Original SODEF Paper](https://openreview.net/forum?id=9CPc4EIr2t1)
- [APUANA Cluster Documentation](https://apuana.cin.ufpe.br/)
- [AutoAttack Library](https://github.com/fra31/auto-attack)

## Contact

For questions about this implementation or cluster usage, contact the APUANA cluster support team at cluster.apuana-l@cin.ufpe.br.