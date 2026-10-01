#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Input settings
params.md_init = "./inputs/geo/*.vasp"
params.md_ps = 500
params.replicates = 1  // Run each file 3 times with different seeds

// 1. Get the files
// 2. Combine with temperature (330K)
// 3. Combine with a range 1..N for replicates
model_config = Channel.fromPath(params.md_init)
                .combine(Channel.of(330,473,573))
                .combine(Channel.of(1..params.replicates))

workflow {
    pinn_nvt(model_config)
}

process pinn_nvt {
    publishDir "mds/${md_geo.simpleName}", mode: 'link'
    label 'pinn'

    input:
    tuple (file(md_geo), val(temp), val(rep_index))

    output:
    file "*.log"
    file "*.traj"

    script:
    // Generate a seed that is unique to the file AND the replicate number
    def combined_seed = (md_geo.simpleName.hashCode() + rep_index).abs() % 10000
    
    """
    #!/usr/bin/env python3
    import numpy as np
    import random
    from mace.calculators import mace_mp
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.bussi import Bussi

    seed_val = ${combined_seed}
    np.random.seed(seed_val)
    random.seed(seed_val)

    atoms = read("${md_geo}")
    atoms.set_calculator(mace_mp(model="mace-matpes-r2scan-0", device="cuda"))

    # Assign initial velocities based on the unique seed
    MaxwellBoltzmannDistribution(atoms, temperature_K=${temp}, rng=np.random.default_rng(seed_val))
    
    dt = 0.5 * units.fs
    dyn = Bussi(atoms, timestep=dt, temperature_K=${temp}, taut=dt*400)

    # Filename now includes the replicate index (rep1, rep2, etc.)
    file_base = "NVT-${temp}K-${md_geo.simpleName}-rep${rep_index}"

    dyn.attach(MDLogger(dyn, atoms, f"{file_base}.log", stress=True, mode="w"), interval=500)
    dyn.attach(Trajectory(f"{file_base}.traj", 'w', atoms).write, interval=500)

    # Run for requested picoseconds
    steps = int((${params.md_ps} * 1000) / 0.5) 
    dyn.run(steps)
    """
}
