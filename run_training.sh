#!/bin/bash
#SBATCH --job-name=Edu_LLM_4GPU
#SBATCH --account=project_2015971
#SBATCH --partition=gpumedium      # Medium partition (up to 24 GPUs, 36h)
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # 1 task — torchrun handles GPU workers
#SBATCH --cpus-per-task=128        # All 128 cores on the node
#SBATCH --gres=gpu:a100:4,nvme:3500  # 4 full A100 GPUs + 3.5TB NVMe (max)
#SBATCH --mem=0                    # All available memory (~490 GiB)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# 1. Environment
module load pytorch
source /projappl/project_2015971/my_env/bin/activate

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# 2. Run with Distributed Data Parallel (4 GPUs)
# torchrun launches 4 processes, one per GPU
# HuggingFace Trainer auto-detects the distributed environment
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    scripts/train.py --config config/training_config.yaml

echo "End: $(date)"