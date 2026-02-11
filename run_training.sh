#!/bin/bash
#SBATCH --job-name=Edu_LLM_4GPU
#SBATCH --account=project_2015971
#SBATCH --partition=gpumedium      # <--- UPGRADE: Medium partition
#SBATCH --time=02:00:00
#SBATCH --nodes=1                  # <--- KEEP 1 NODE (Simpler)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # <--- MAX POWER (Use all CPUs)
#SBATCH --gres=gpu:a100:4,nvme:3800 # <--- MAX POWER (4 GPUs + 3.8TB Disk)
#SBATCH --mem=0                    # <--- MAX RAM (All available memory)
#SBATCH --output=logs/train_%j.out

# 1. Environment
module load pytorch
source /projappl/project_2015971/my_env/bin/activate

# 2. Run with Data Parallel
# We use 'torchrun' to utilize all 4 GPUs automatically
# --nproc_per_node=4 tells it to use 4 GPUs
srun torchrun --nproc_per_node=4 scripts/train.py --config config/training_config.yaml