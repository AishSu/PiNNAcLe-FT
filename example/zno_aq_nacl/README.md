# ZnO/aqueous NaCl example

This example adapts the PiNNAcLe-FT workflow to the ZnO/aqueous NaCl interface.

## Status

The workflow is under development.

## System

The system contains Zn, O, H, Na, and Cl.

## Workflow

1. Prepare the DFT reference dataset.
2. Train and evaluate the PiNN models.
3. Run molecular dynamics and collect new configurations.
# Example: ZnO/aqueous NaCl interface.
This example demonstrates how to build PiNet2-P3 models for a ZnO/aqueous NaCl interface, with either r2scan or pbe level. There is two kinds of sampling strategies for finetuning, unbiased md and steered md. 

# Construct the dataset
  + Run MD simulations using the mace-matpes-r2scan-0 (or mace-omat-0-m-pbe) foundation model for two different systems (ZnO/aq. NaCl and ZnO/aq. NaCl + ZnCl2 solution).

  + Move the trajectory files to the corresponding subfolder, and build the dataset

# Pre-train the PiNet2-P3 models
  + Run the training script
```
python build_pinet2.py  # run by slurm, and change the random seed manully
```
  + After the training finished, estimate the performance of PiNet2-P3 models on energy and force
```
python compute_error.py # run by slurm, and change the random seed manully
```
# Fine-tune the PiNet2-P3 models at the CP2K r2SCAN/TZV2P level

 + Modify the hyperparameters in nextflow/acle-cp2k-from-user-model.nf, especially the four tolerances (frmsetol, ermsetol, fmaxtol, emaxtol) for convergence check.
 + If disk space is limited, using the release_space channel to delete some intermediate files is a good option.
 + Then, we can start the fine-tuning
```
NXF_VER=23.10.1 nextflow run main.nf -profile arrhen -bg > log.out
```
# Plot the fine-tuning curve

