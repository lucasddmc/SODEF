#!/bin/bash
#SBATCH --job-name=setup_sodef_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# Script to set up SODEF environment on APUANA cluster
echo "Setting up SODEF environment..."

# Load Python module
module load Python/3.10

# Create logs directory if it doesn't exist
mkdir -p logs

# Create virtual environment
if [ ! -d "$HOME/sodef_env" ]; then
    echo "Creating virtual environment..."
    python -m venv $HOME/sodef_env
fi

# Activate virtual environment
source $HOME/sodef_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchdiffeq; print('torchdiffeq installed successfully')"

echo "Environment setup completed!"