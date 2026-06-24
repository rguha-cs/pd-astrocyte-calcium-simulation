import os
import numpy as np

sigma_values = [round(x, 2) for x in np.arange(45000, 50000, 1000)]

num_trials = 2

os.makedirs("jobs", exist_ok=True)

for sigma in sigma_values:
    for trial in range(num_trials):
        #job_name = f"jobs/sigma_{int(sigma):2f}_trial_{trial}.sh"
        job_name = f"jobs/sigma_{sigma:.2f}_trial_{trial}.sh"
        with open(job_name, "w") as f:
            f.write(f'''#!/bin/bash
echo "Running sigma={sigma} trial={trial}"
python run_single_sim.py --sigma {sigma} --trial {trial} --output_dir batch_results
''')
        os.chmod(job_name, 0o755)
print("Job scripts generated.")
