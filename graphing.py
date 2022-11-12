import pandas as pd
import seaborn as sns
from glob import glob
import re
import matplotlib.pyplot as plt

EXPERIMENT_TITLE = "MNIST Varying num_share_devices"
CSVS_DIR = "results/mnist_num_share_exp/acc"

data = pd.DataFrame()
for fname in glob(f"{CSVS_DIR}/*.csv"):
    print(fname)
    # Extract parameter value
    m = re.search(r"share_(\d+)_of_30-tag-Accuracy_avg_acc.csv", fname)
    if m:
        # Parse experiment
        parameter_val = int(m.group(1))
        # Parse csv to get desired timeseries
        df = pd.read_csv(fname, index_col=False)
        df.drop(columns='Wall time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['num_share_devices'] = parameter_val
        df['type'] = 'decentralized'
        data = pd.concat([data, df])
    else:
        # Parse control
        df = pd.read_csv(fname, index_col=False)
        df.drop(columns='Wall time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['num_share_devices'] = 'fedavg'
        df['type'] = 'fedavg'
        data = pd.concat([data, df])

sns.lineplot(data=data, x="Step", y="Value", hue="num_share_devices")
plt.show()
