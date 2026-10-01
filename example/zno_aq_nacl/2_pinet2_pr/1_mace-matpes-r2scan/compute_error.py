import math
import os
import numpy as np
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# pinn log . essentially does the same thing. Since it only prints two parameters E_RMSE and F_RMSE, we want to print the maximum error as well. So we write our own code to compute the error metrics. /AS01102026

WORK_DIR = Path("/nobackup/proj/disk/snic2022-5-322/personal/aishu/PiNNAcLe-FT/example/zno_aq_nacl/2_pinet2_pr/1_mace-matpes-r2scan")
DATASET_DIR = WORK_DIR / "split_ds_seed"
MODEL_DIR = WORK_DIR / "PiNet2_models"

# get the inital energy and force metricx for selecting the hyper-parameters in PiNNAncL package
def get_ener_force_metrics(nSeed):

    from pinn import get_calc
    from ase import Atoms
    from pinn.io import load_tfrecord

    dataset = load_tfrecord(str(DATASET_DIR / f"eval_seed{nSeed}.yml"))
    fields = ['elems', 'coord', 'cell', 'e_data', 'f_data']
    refData = {k: [] for k in fields}
    for example in dataset:
        for k in fields:
            refData[k].append(example[k].numpy())

    nNumFrame = len(refData['e_data'])
    print("Number of frames = %d"%(nNumFrame))

    # get predictions 

    strModelPath = str(MODEL_DIR / f"ZnO_aqNaCl_ZnCl2-330K-macer2scan-5854-pinet2-B1-5E6-{nSeed}")
    print(f"\nEvaluating seed {nSeed}", flush=True)
    print(f"Dataset: {DATASET_DIR / f'eval_seed{nSeed}.yml'}", flush=True)
    print(f"Model: {strModelPath}", flush=True)

    calc = get_calc(strModelPath)
    calc.properties = ['energy', 'force']

    # based on the code in tips.nf of PiNNAcLe, the energy is per atom value
    e_label = []
    f_label = []
    e_pred = []
    f_pred = []
    for nFramIndex in range(0, nNumFrame):

        e_label.append(refData['e_data'][nFramIndex] / len(refData['elems'][nFramIndex]))
        f_label.append(refData['f_data'][nFramIndex])

        atoms = Atoms(numbers=refData['elems'][nFramIndex],
                      positions=refData['coord'][nFramIndex],
                      cell=refData['cell'][nFramIndex],
                      pbc=True)

        atoms.set_calculator(calc)
        
        dPredPotEner = atoms.get_potential_energy()
        dPredForce = atoms.get_forces()

        e_pred.append(dPredPotEner / len(refData['elems'][nFramIndex]))
        f_pred.append(dPredForce)

    # e_label = np.array(e_label)
    # f_label = np.array(f_label)
    # e_pred = np.array(e_pred)
    # f_pred = np.array(f_pred)

    # assert e_pred.shape == e_label.shape
    # assert f_pred.shape == f_label.shape

    # emax = np.max(np.abs(e_pred-e_label))
    # fmax = np.max(np.abs(f_pred-f_label))
    # ermse = np.sqrt(np.mean((e_pred-e_label)**2))
    # frmse = np.sqrt(np.mean((f_pred-f_label)**2))

    # print(
    #     f"Seed {nSeed}: "
    #     f"emax={1000 * emax:.6f} meV/atom, "
    #     f"fmax={1000 * fmax:.6f} meV/Å, "
    #     f"ermse={1000 * ermse:.6f} meV/atom, "
    #     f"frmse={1000 * frmse:.6f} meV/Å",
    #     flush=True,
    # )

    # One energy value per frame.
    e_label = np.asarray(e_label, dtype=float).reshape(-1)
    e_pred = np.asarray(e_pred, dtype=float).reshape(-1)

    assert e_pred.shape == e_label.shape, (
        f"Energy shape mismatch: {e_pred.shape} vs {e_label.shape}"
    )

    # Compare forces within each frame, allowing different atom counts.
    force_errors = []

    for frame_index, (predicted, reference) in enumerate(zip(f_pred, f_label)):
        predicted = np.asarray(predicted, dtype=float)
        reference = np.asarray(reference, dtype=float)

        if predicted.shape != reference.shape:
            raise ValueError(
                f"Frame {frame_index}: force shape mismatch: "
                f"prediction {predicted.shape}, reference {reference.shape}"
            )

        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                f"Frame {frame_index}: expected forces with shape (N, 3), "
                f"got {reference.shape}"
            )

        force_errors.append((predicted - reference).ravel())

    energy_errors = e_pred - e_label
    force_errors = np.concatenate(force_errors)

    emax = np.max(np.abs(energy_errors))
    ermse = np.sqrt(np.mean(energy_errors**2))

    fmax = np.max(np.abs(force_errors))
    frmse = np.sqrt(np.mean(force_errors**2))

    print(
        f"Seed {nSeed}: "
        f"emax={1000 * emax:.6f} meV/atom, "
        f"fmax={1000 * fmax:.6f} meV/Å, "
        f"ermse={1000 * ermse:.6f} meV/atom, "
        f"frmse={1000 * frmse:.6f} meV/Å",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    get_ener_force_metrics(args.seed)
