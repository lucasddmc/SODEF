#!/bin/bash
#SBATCH --job-name=train_baseline_cifar10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=long-simple
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# Train baseline CIFAR-10 classifier
echo "Starting baseline CIFAR-10 training..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load Python module
module load Python/3.10

# Activate virtual environment
source $HOME/sodef_env/bin/activate

# Create logs directory
mkdir -p logs

# Print system information
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run baseline training
echo "Training baseline ResNet on CIFAR-10..."
python src/train_baseline.py --config config/baseline_config.yaml

echo "Baseline training completed!"