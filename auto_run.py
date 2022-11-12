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

# Sweep through num_share_devices
values = [0, 1, 2, 5, 9]
print(values)

if os.path.isfile(PROGRESS_FILE):
    print("Resuming from previous run...")
    with open(PROGRESS_FILE, 'r') as f:
        completed = json.load(f)
else:
    completed = []

for val in tqdm(values):
    if val in completed:
        continue

    print(os.path.realpath(__file__))
    # subprocess.call(f".\\venv\\Scripts\\activate.bat && python decentralized_fl.py --num_devices {NUM_DEVICES} --num_share_devices {val} --run_name share_{val}_of_{NUM_DEVICES}", cwd=os.getcwd())
    subprocess.call(f"venv\\Scripts\\python.exe decentralized_fl.py --num_share_devices {val} --run_name cifar_numshare_{val}", cwd=os.getcwd())

    # Save progress
    completed.append(val)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(completed, ensure_ascii=False, indent=4, fp=f)