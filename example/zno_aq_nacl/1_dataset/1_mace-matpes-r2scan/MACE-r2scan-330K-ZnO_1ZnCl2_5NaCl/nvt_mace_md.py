import os
import math
import numpy as np


def startModelMD():
    from mace.calculators import mace_mp
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.bussi import Bussi
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    # 1. Setup Parameters
    setup = {
        'ensemble': 'NVT',
        't': 500,          # Total target time in ps
        'dt': 0.5,          # Timestep in fs
        'temp': 330,        # Temperature in K
        'log-every': 500,   # Logging interval in steps
        'init_structure': 'ZnO_1ZnCl2_5NaCl.vasp', # Input geometry (vasp)
    }

    ensemble = setup['ensemble']
    t = float(setup['t'])
    dt = float(setup['dt'])
    temp = float(setup['temp'])
    every = int(setup['log-every'])

    strWorkDir = os.getcwd() + "/"
    strSystem = f"{ensemble}-{int(temp)}K-ZnO_2ZnCl2_1NaCl"
    strLogFile = strWorkDir + f"{strSystem}.log"
    strTrajFile = strWorkDir + f"{strSystem}.traj"

    # Ensure output files do not overwrite existing runs inadvertently
    if os.path.exists(strLogFile) or os.path.exists(strTrajFile):
        print(f"Warning: Existing simulation files detected for {strSystem}.")
        print("Use continueModelMD() to resume, or clear existing files to restart.")
        return

    # 2. Read Initial Structure & Set Calculator
    print(f"Loading initial structure from: {setup['init_structure']}")
    atoms = read(setup['init_structure'],format='vasp')

    macemp = mace_mp(model="mace-matpes-r2scan-0", device="cuda")
    atoms.set_calculator(macemp)

    # 3. Initialize Velocities at Target Temperature
    # Draws initial momenta from Maxwell-Boltzmann distribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    # Remove center of mass translation
    Stationary(atoms)

    # 4. Instantiate Bussi Thermostat Dynamics
    dyn = Bussi(
        atoms, 
        timestep=dt * units.fs, 
        temperature_K=temp, 
        taut=dt * units.fs * 400
    )

    # 5. Setup Native Logging and Trajectory Output
    # header=True writes column titles in mode="w"
    native_logger = MDLogger(dyn, atoms, strLogFile, header=True, stress=True, mode="w")
    native_traj = Trajectory(strTrajFile, mode='w', atoms=atoms)

    # Attach callbacks for periodic logging
    dyn.attach(native_logger, interval=every)
    dyn.attach(native_traj, interval=every)

    # Write initial frame (step 0) to both trajectory and log
    native_logger()
    native_traj.write()

    # 6. Execute Run
    total_steps = int((t * 1000) / dt)
    print(f"Starting initial MD run for system: {strSystem}")
    print(f"Total duration: {t:.2f} ps ({total_steps} steps at dt={dt} fs)")
    print(f"Logging every {every} steps ({every * dt / 1000:.2f} ps)")

    dyn.run(total_steps)
    print("Initial MD simulation completed successfully.")

def continueModelMD():
    from mace.calculators import mace_mp
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.bussi import Bussi

    setup = {
        'ensemble': 'NVT',
        't': 2000,          # Total target time in ps
        'dt': 0.5,         # timestep in fs
        'temp': 330,
        'log-every': 500,
    }

    ensemble = setup['ensemble']
    t = float(setup['t'])
    dt = float(setup['dt'])
    temp = float(setup['temp'])
    every = int(setup['log-every'])

    strWorkDir = os.getcwd() + "/"
    strSystem = "NVT-330K-ZnO_2ZnCl2_1NaCl"
    strLogFile = strWorkDir + f"{strSystem}.log"
    strTrajFile = strWorkDir + f"{strSystem}.traj"

    assert os.path.exists(strLogFile) and os.path.exists(strTrajFile), "Log or Trajectory file missing!"
    assert (ensemble in strSystem) and (str(setup['temp']) in strSystem)

    # 2. Parse previous time steps from Log Safely
    with open(strLogFile, "r") as f:
        listLines = f.readlines()

    arrTime = []
    for l in listLines:
        parts = l.split()
        if not parts:
            continue
        try:
            arrTime.append(float(parts[0]))
        except ValueError:
            continue

    assert len(arrTime) >= 2, "Not enough history data found in the log file!"

    dSaveTimeSpan = arrTime[1] - arrTime[0]
    dSaveTimeSpan_fs = dSaveTimeSpan * 1000

    assert math.fabs(dSaveTimeSpan_fs/dt - every) < 1e-6, f"Interval mismatch! Log has {dSaveTimeSpan_fs/dt} steps, config expected {every}."
    dTotPrevTime = arrTime[-1]

    remaining_t = t - dTotPrevTime
    print(f"Detected previous simulation time: {dTotPrevTime:.2f} ps")
    print(f"Remaining time to reach target {t} ps: {remaining_t:.2f} ps")

    if remaining_t <= 0:
        print("Simulation target already reached or exceeded.")
        return

    # 3. Read structure and velocities safely
    macemp = mace_mp(model="mace-matpes-r2scan-0", device="cuda")
    # Explicitly state format='traj' to ensure velocities transfer properly
    atoms = read(strTrajFile, index=-1, format='traj')
    atoms.set_calculator(macemp)

    assert atoms.get_velocities() is not None, "Velocities/Momenta missing from trajectory file!"

    dyn = Bussi(atoms, timestep=dt*units.fs, temperature_K=temp, taut=dt * units.fs * 400)

    # 4. FIXED: Chronological time-patching injection using internal ASE definitions
    steps_previously_done = int((dTotPrevTime * 1000) / dt)

    def corrected_get_time():
        return (steps_previously_done + dyn.nsteps) * dyn.dt

    dyn.get_time = corrected_get_time
    print(f"Adjusted starting time hook: {dyn.get_time() / units.fs / 1000:.2f} ps")

    # 5. Native Logging definitions (with custom execution wrappers)
    native_logger = MDLogger(dyn, atoms, strLogFile, header=False, stress=True, mode="a")
    def log_wrapper():
        current_steps = dyn.get_number_of_steps()
        if current_steps != 0 and current_steps % every == 0:
            native_logger()

    native_traj = Trajectory(strTrajFile, 'a', atoms)
    def traj_wrapper():
        current_steps = dyn.get_number_of_steps()
        if current_steps != 0 and current_steps % every == 0:
            native_traj.write()

    dyn.attach(log_wrapper, interval=1)
    dyn.attach(traj_wrapper, interval=1)

    # 6. Execute remaining steps
    total_steps_to_run = int((remaining_t * 1000) / dt)
    print(f"Running for {total_steps_to_run} steps...")
    dyn.run(total_steps_to_run)
    print("Simulation complete.")

if __name__ == '__main__':
    continueModelMD()
#    startModelMD()
