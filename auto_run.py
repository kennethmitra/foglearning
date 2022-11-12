"""
Run experiments automatically
"""
import os.path
import subprocess
import sys
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

print(f"Using {sys.executable}")

# Store/resume completed tasks if interrupted
PROGRESS_DIR = "./progress"
Path(PROGRESS_DIR).mkdir(exist_ok=True, parents=True)
PROGRESS_FILE = f"{PROGRESS_DIR}/share_devices_exp.json"
OVERRIDE_PROGRESS_SAVE = True

run_params = []
for s in [2, 3]:
    for val in [0, 1, 2, 5, 14]:
        run_params.append(f"--num_share_devices {val} --run_name share_{val}_of_30 --seed {s}")

run_params.append(f"--num_share_devices 29 --run_name share_29_of_30 --seed 1")
run_params.append(f"--num_share_devices 29 --run_name share_29_of_30 --seed 2")
run_params.append(f"--num_share_devices 29 --run_name share_29_of_30 --seed 3")

for param in run_params:
    print(param)

if os.path.isfile(PROGRESS_FILE):
    print("Resuming from previous run...")
    with open(PROGRESS_FILE, 'r') as f:
        completed = json.load(f)
else:
    completed = []

for param in tqdm(run_params):
    if not OVERRIDE_PROGRESS_SAVE and param in completed:
        continue

    print(os.path.realpath(__file__))
    subprocess.call(f"venv\\Scripts\\python.exe decentralized_fl.py {param}", cwd=os.getcwd())

    # Save progress
    completed.append(param)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(completed, ensure_ascii=False, indent=4, fp=f)