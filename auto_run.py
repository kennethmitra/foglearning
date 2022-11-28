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
OVERRIDE_PROGRESS_SAVE = False

run_params = []
for s in [1, 2, 3, 4]:
    for val in [1.0, 0.9, 0.8, 0.7, 0.5]:
        run_params.append(f"--num_share_devices 2 --comm_reliability {val} --run_name cifar_comm_{int(val*100)}_share_2 --seed {s}")
        run_params.append(f"--num_share_devices 5 --comm_reliability {val} --run_name cifar_comm_{int(val * 100)}_share_5 --seed {s}")

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
        print(f"Skipping {param}")
        continue

    print(os.path.realpath(__file__))
    subprocess.call(f"venv\\Scripts\\python.exe decentralized_fl.py {param}", cwd=os.getcwd())

    # Save progress
    completed.append(param)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(completed, ensure_ascii=False, indent=4, fp=f)