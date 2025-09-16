#!/bin/bash
#SBATCH --job-name=eval_fgsm_pgd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Minimal evaluation: Clean + FGSM/PGD on Baseline and SODEF

echo "Starting minimal evaluation (FGSM/PGD)..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

module load Python/3.10
source $HOME/sodef_env/bin/activate
mkdir -p logs

# System info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run evaluation with FGSM/PGD only (ensure eval_config.yaml lists only these attacks or ignores others)
echo "Evaluating Baseline and SODEF under FGSM/PGD..."
python src/evaluate.py --config config/eval_config.yaml --model both --device cuda

echo "Evaluation completed! Results at results/evaluation_results.json"
