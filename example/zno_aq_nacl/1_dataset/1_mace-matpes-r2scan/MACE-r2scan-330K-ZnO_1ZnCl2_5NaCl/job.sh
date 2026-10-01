#!/bin/bash
#SBATCH -A naiss2025-5-447-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:10:00
#SBATCH -J 4mace_1ZnCl2_5NaCl

# Activate the native ARM virtual environment
source "$HOME/.venvs/aarch64/mace-env/bin/activate"

# Run your Python script
python nvt_mace_md.py
