#!/bin/bash
#SBATCH --job-name=Edu_LLM_Finetune
#SBATCH --account=project_2015971      # <--- Ensure this is your correct Project ID
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --mem=64G
#SBATCH --output=logs/train_%j.out

# ---------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ---------------------------------------------------------

# Load system drivers and basic tools (Required for AMD GPUs)
module load pytorch

# --- CRITICAL CHANGE HERE ---
# activate your custom environment
# Replace the path below with the REAL path to your environment's 'activate' file
source /projappl/project_2015971/my_env/bin/activate

# (Optional) Verify it worked
echo "Using Python from: $(which python)"

# ---------------------------------------------------------
# 2. RUN TRAINING
# ---------------------------------------------------------

echo "Job started on node: $SLURMD_NODENAME"
echo "GPU info:"
nvidia-smi

mkdir -p logs

# Run the script
srun $(which python) scripts/train.py --config config/training_config.yaml