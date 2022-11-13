from pathlib import Path

import pandas as pd
import seaborn as sns
from glob import glob
import re
import matplotlib.pyplot as plt

EXPERIMENT_TITLE = "MNIST Varying num_share_devices"
CSVS_DIR = "results/mnist_num_share_exp/acc"
FIG_OUT_DIR = "results/figures"

data = pd.DataFrame()
for fname in glob(f"{CSVS_DIR}/*.csv"):
    print(fname)
    # Extract parameter value
    m = re.search(r"share_(\d+)_of_30-tag-Accuracy_avg_acc.csv", fname)
    if m:
        # Parse experiment
        parameter_val = int(m.group(1))
        if parameter_val in (0, 1, 14, 29):
            # Parse csv to get desired timeseries
            df = pd.read_csv(fname, index_col=False)
            df.drop(columns='Wall time', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['num_share_devices'] = parameter_val
            df['type'] = 'decentralized'
            data = pd.concat([data, df])
        else:
            continue
    else:
        # Parse control
        df = pd.read_csv(fname, index_col=False)
        df.drop(columns='Wall time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['num_share_devices'] = 'fedavg'
        df['type'] = 'fedavg'
        data = pd.concat([data, df])

sns.set(rc={'figure.figsize':(6, 5)})
g = sns.lineplot(data=data, x="Step", y="Value", hue="num_share_devices", errorbar=("se", 2), hue_order=(0, 1, 14, 29, 'fedavg'))
# g.set(xscale='log', yscale='log')
plt.title(EXPERIMENT_TITLE)
plt.ylabel("Test Accuracy")

Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}")
plt.show()


# EXPERIMENT_TITLE = "CIFAR10 Varying num_share_devices"
# CSVS_DIR = "results/cifar_num_share_exp"
# FIG_OUT_DIR = "results/figures"
#
# data = pd.DataFrame()
# for fname in glob(f"{CSVS_DIR}/*.csv"):
#     print(fname)
#     # Extract parameter value
#     m = re.search(r"cifar_numshare_(\d+)-tag-Accuracy_avg_acc.csv", fname)
#     if m:
#         # Parse experiment
#         parameter_val = int(m.group(1))
#         # Parse csv to get desired timeseries
#         df = pd.read_csv(fname, index_col=False)
#         df.drop(columns='Wall time', inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df['num_share_devices'] = parameter_val
#         df['type'] = 'decentralized'
#         data = pd.concat([data, df])
#     else:
#         # Parse control
#         df = pd.read_csv(fname, index_col=False)
#         df.drop(columns='Wall time', inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df['num_share_devices'] = 'fedavg'
#         df['type'] = 'fedavg'
#         data = pd.concat([data, df])
#
# sns.set(rc={'figure.figsize':(6, 5)})
# sns.lineplot(data=data, x="Step", y="Value", hue="num_share_devices", errorbar=("se", 2))
# plt.title(EXPERIMENT_TITLE)
# plt.ylabel("Test Accuracy")
#
# Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
# plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}")
# plt.show()
