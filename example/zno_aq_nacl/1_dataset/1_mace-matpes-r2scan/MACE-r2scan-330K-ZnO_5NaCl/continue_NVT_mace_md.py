import os
import math
import numpy as np

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
    strSystem = "NVT-330K-ZnO_aqNaCl-rep1"
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
