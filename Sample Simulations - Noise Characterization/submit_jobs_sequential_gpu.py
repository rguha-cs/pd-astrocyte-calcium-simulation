import os
import subprocess
import time

job_dir = "jobs"
job_files = sorted([f for f in os.listdir(job_dir) if f.endswith(".sh")])
job_paths = [os.path.join(job_dir, f) for f in job_files]

print(f"Launching {len(job_paths)} jobs sequentially on GPU...")

for job_script in job_paths:
    print(f"\n Running job: {job_script}")
    start = time.time()
    result = subprocess.run(["bash", job_script], capture_output=True, text=True)
    end = time.time()
    
    print(f"Duration: {end - start:.2f} seconds")

    if result.returncode == 0:
        print("Job completed successfully.")
    else:
        print("Job failed.")
        print("Stderr:", result.stderr.strip())

print("\nAll jobs completed.")
