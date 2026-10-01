#!/bin/bash
#SBATCH --job-name=0ZnO_mace_r2scan_330
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-5-447
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=7-00:00:00
#SBATCH --output=md_%j.out
#SBATCH --error=md_%j.err

############################
# EXACT Nextflow environment
############################
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Java/17.0.13

source /mimer/NOBACKUP/groups/snic2022-5-322/sudhama/project/pinn-env-2/bin/activate

############################
# Sanity check (optional)
############################
echo "SLURM job running on: $(hostname)"
nvidia-smi

python - << EOF
import torch
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

############################
# Run MD
############################
python continue_NVT_mace_md.py
