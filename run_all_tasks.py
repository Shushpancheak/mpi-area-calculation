import subprocess

filename = "tmp_run_batch.sh"

for cores_n in range(1, 9):
    for N in [3, 6, 8]:
        with open(filename, "w") as f:
            f.write(
"""#!/bin/bash
#SBATCH --ntasks={tasks}
#SBATCH --cpus-per-task=4
#SBATCH --partition=RT
#SBATCH --job-name=task-1-mumladze
#SBATCH --comment="Task 1. Mumladze Maximelian."
#SBATCH --output=out-{n}-{cores}.txt
#SBATCH --error=error.txt
mpiexec -np {cores} ./program {parts}""".format(tasks=(8 if cores_n < 4 else 16), cores=cores_n, parts=10 ** N, n=N))
        
        program = subprocess.run(['sbatch', filename])
