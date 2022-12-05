import os
import re
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

"""
Used to download a large number of csv's from Tensorboard quickly
"""

SAVE_PATH = "results/cifar_strat_compare"
Path(SAVE_PATH).mkdir(exist_ok=True, parents=True)

matching_runs = [f for f in os.listdir("runs/") if re.search(r"cifar_strat_", f)]

for run in tqdm(matching_runs):
    URL = f"http://localhost:6006/data/plugin/scalars/scalars?tag=Accuracy%2Favg_acc&run={run}&format=csv"
    urlretrieve(URL, f"{SAVE_PATH}/{run}-tag-Accuracy_avg_acc.csv")
