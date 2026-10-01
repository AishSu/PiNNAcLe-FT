#!/bin/bash

IMAGE=/nobackup/proj/disk/snic2022-5-322/shared/zhanyun/PiNNAcLe-FT/docker/PiNN/pinn-gpu-acle-plumed-cp2k2026.sif
PROJECT=/nobackup/proj/disk/snic2022-5-322
SCRIPT="$PROJECT/personal/aishu/PiNNAcLe-FT/example/zno_aq_nacl/2_pinet2_pr/1_mace-matpes-r2scan/compute_error.py"
for seed in 0 1 2; do
    echo "Evaluating seed $seed"

    apptainer exec --nv \
        --bind "$PROJECT:$PROJECT" \
        "$IMAGE" \
        python -u "$SCRIPT" --seed "$seed" \
        > "metrics_seed${seed}.log" 2>&1

    status=$?
    echo "Seed $seed finished with exit code $status"
done

grep -h '^Seed [0-9].*emax=' \
    metrics_seed0.log metrics_seed1.log metrics_seed2.log > metrics.log
