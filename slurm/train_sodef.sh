#!/bin/bash
#SBATCH --job-name=train_sodef_defense
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=logs/sodef_%j.out
#SBATCH --error=logs/sodef_%j.err

# SODEF Defense Training Script
echo "Starting SODEF defense training..."
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

# Train SODEF defense component (requires pretrained baseline)
echo "Training SODEF defense component..."
python src/train_sodef.py --config config/sodef_config.yaml

echo "SODEF defense training completed!"