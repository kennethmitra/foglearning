import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

####################
# MNIST Line graph
####################
# EXPERIMENT_TITLE = "MNIST Varying num_share_devices"
# CSVS_DIR = "results/mnist_num_share_exp/acc"
# FIG_OUT_DIR = "results/figures"
#
# data = pd.DataFrame()
# for fname in glob(f"{CSVS_DIR}/*.csv"):
#     print(fname)
#     # Extract parameter value
#     m = re.search(r"share_(\d+)_of_30-tag-Accuracy_avg_acc.csv", fname)
#     if m:
#         # Parse experiment
#         parameter_val = int(m.group(1))
#         if parameter_val in (0, 1, 14, 29):  # SELECT certain param values cause they overlap too much
#             # Parse csv to get desired timeseries
#             df = pd.read_csv(fname, index_col=False)
#             df.drop(columns='Wall time', inplace=True)
#             df.reset_index(drop=True, inplace=True)
#             df['num_share_devices'] = parameter_val
#             df['type'] = 'decentralized'
#             data = pd.concat([data, df])
#         else:
#             continue
#     else:
#         # Parse control
#         df = pd.read_csv(fname, index_col=False)
#         df.drop(columns='Wall time', inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df['num_share_devices'] = 'fedavg'
#         df['type'] = 'fedavg'
#         data = pd.concat([data, df])
#
# sns.set(rc={'figure.figsize':(11.5, 8)})
# # Put the params here too
# g = sns.lineplot(data=data, x="Step", y="Value", hue="num_share_devices", errorbar=("se", 2), hue_order=(0, 1, 14, 29, 'fedavg'))
# # g.set(xscale='log', yscale='log')
# plt.title(EXPERIMENT_TITLE)
# plt.ylabel("Test Accuracy")
#
# Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
# plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}_large")
# plt.show()

####################
# MNIST Bar graph
####################
# max_step = 10
# EXPERIMENT_TITLE = f"MNIST Avg accuracy over steps 0 through {max_step}"
# CSVS_DIR = "results/mnist_num_share_exp/acc"
# FIG_OUT_DIR = "results/figures"
#
#
#
# data = pd.DataFrame()
# for fname in glob(f"{CSVS_DIR}/*.csv"):
#     print(fname)
#     # Extract parameter value
#     m = re.search(r"share_(\d+)_of_30-tag-Accuracy_avg_acc.csv", fname)
#     if m:
#         # Parse experiment
#         parameter_val = int(m.group(1))
#         if parameter_val in (0, 1, 2, 5, 14, 29):  # SELECT certain param values cause they overlap too much
#             # Parse csv to get desired timeseries
#             log = pd.read_csv(fname, index_col=False)
#             log.drop(columns='Wall time', inplace=True)
#             log.reset_index(drop=True, inplace=True)
#             df = pd.DataFrame(log.query(f"Step < {max_step}").mean()).transpose()
#             df['num_share_devices'] = parameter_val
#             df['type'] = 'decentralized'
#             data = pd.concat([data, df])
#         else:
#             continue
#     else:
#         # Parse control
#         log = pd.read_csv(fname, index_col=False)
#         log.drop(columns='Wall time', inplace=True)
#         log.reset_index(drop=True, inplace=True)
#         df = pd.DataFrame(log.query(f"Step < {max_step}").mean()).transpose()
#         df['num_share_devices'] = 'fedavg'
#         df['type'] = 'fedavg'
#         data = pd.concat([data, df])
#
# data.reset_index(drop=True, inplace=True)
# sns.set(rc={'figure.figsize':(11.5,8)})
# # Put the params here too
# g = sns.barplot(data=data, x="num_share_devices", y="Value", errorbar=("se", 2), order=(0, 1, 2, 5, 14, 29, 'fedavg'))
# plt.title(EXPERIMENT_TITLE)
# plt.ylabel("Test Accuracy")
# plt.ylim(0, 0.86)
#
# Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
# plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}_large")
# plt.show()


####################
# CIFAR10 Line graphs
####################
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
# sns.set(rc={'figure.figsize':(11.5,8)})
# sns.lineplot(data=data, x="Step", y="Value", hue="num_share_devices", errorbar=("se", 2))
# plt.title(EXPERIMENT_TITLE)
# plt.ylabel("Test Accuracy")
#
# Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
# plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}_large")
# plt.show()

####################
# Comm Reliability Bar graphs
####################
EXPERIMENT_TITLE = "Impact of Communication Reliability (CIFAR10)"
CSVS_DIR = "results/cifar_comm_reliability_exp"
FIG_OUT_DIR = "results/figures"
min_step = 140

data = pd.DataFrame()
for fname in glob(f"{CSVS_DIR}/*.csv"):
    print(fname)
    # Extract parameter value
    m = re.search(r"cifar_comm_(\d+)_share_(\d+)-tag-Accuracy_avg_acc.csv$", fname)
    if m:
        # Parse experiment
        comm_reliab = float(m.group(1))
        share_devices = int(m.group(2))
        if comm_reliab in (50, 70, 80, 90, 100):
            # Parse csv to get desired timeseries
            # df = pd.read_csv(fname, index_col=False)
            # df.drop(columns='Wall time', inplace=True)
            # df.reset_index(drop=True, inplace=True)
            # df['num_share_devices'] = share_devices
            # df['comm_reliability'] = comm_reliab
            # df['type'] = 'decentralized'
            # data = pd.concat([data, df])
            log = pd.read_csv(fname, index_col=False)
            log.drop(columns='Wall time', inplace=True)
            log.reset_index(drop=True, inplace=True)
            df = pd.DataFrame(log.query(f"Step > {min_step}").mean()).transpose()
            df['num_share_devices'] = share_devices
            df['comm_reliability'] = comm_reliab
            df['type'] = 'decentralized'
            data = pd.concat([data, df])
        else:
            continue
    else:
        m2 = re.search(r"cifar_comm_(\d+)_fedavg.csv$", fname)
        comm_reliab = float(m2.group(1))
        # Parse control
        # df = pd.read_csv(fname, index_col=False)
        # df.drop(columns='Wall time', inplace=True)
        # df.reset_index(drop=True, inplace=True)
        # df['num_share_devices'] = 'fedavg'
        # df['comm_reliability'] = comm_reliab
        # df['type'] = 'fedavg'
        # data = pd.concat([data, df])
        log = pd.read_csv(fname, index_col=False)
        log.drop(columns='Wall time', inplace=True)
        log.reset_index(drop=True, inplace=True)
        df = pd.DataFrame(log.query(f"Step > {min_step}").mean()).transpose()
        df['num_share_devices'] = 'fedavg'
        df['comm_reliability'] = comm_reliab
        df['type'] = 'fedavg'
        data = pd.concat([data, df])

# sns.set(rc={'figure.figsize':(11.5,8)})
sns.set(rc={'figure.figsize':(9, 6)})
# sns.lineplot(data=data, x="Step", y="Value", hue="comm_reliability", style="num_share_devices", markers=False, errorbar=("se", 0), palette='flare')  #errorbar=("se", 2)
g = sns.barplot(data=data, x="num_share_devices", y="Value", hue="comm_reliability", errorbar=("se", 2), order=(2, 5, 'fedavg'))
plt.legend(loc='lower right')
plt.title(EXPERIMENT_TITLE)
plt.ylabel(f"Avg Test Accuracy (Steps {min_step} to 150)")
plt.xlabel("Number of share devices")

Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}")
plt.show()

####################
# Strategy line graphs
####################
# EXPERIMENT_TITLE = "Impact of Communication Strategies"
# CSVS_DIR = "results/cifar_strat_compare"
# FIG_OUT_DIR = "results/figures"
#
# data = pd.DataFrame()
# for fname in glob(f"{CSVS_DIR}/*.csv"):
#     print(fname)
#     # Extract parameter value
#     m = re.search(r"cifar_strat_(dist|rand)_share_(\d+)-tag-Accuracy_avg_acc.csv", fname)
#     if m:
#         # Parse experiment
#         strategy = (m.group(1))
#         num_share_devices = int(m.group(2))
#         # Parse csv to get desired timeseries
#         df = pd.read_csv(fname, index_col=False)
#         df.drop(columns='Wall time', inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df['num_share_devices'] = num_share_devices
#         df['strategy'] = strategy
#         data = pd.concat([data, df])
#     else:
#         raise NotImplemented("Error")
#
# sns.set(rc={'figure.figsize':(9, 6)})
# sns.lineplot(data=data, x="Step", y="Value", hue="strategy", errorbar=("se", 2))
# plt.title(EXPERIMENT_TITLE)
# plt.ylabel("Test Accuracy")
#
# Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
# plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}")
# plt.show()

####################
# NonIID line graphs
####################
# EXPERIMENT_TITLE = "NonIID Results"
# CSVS_DIR = "results/cifar_noniid_exp"
# FIG_OUT_DIR = "results/figures"
#
# data = pd.DataFrame()
# for fname in glob(f"{CSVS_DIR}/*.csv"):
#     print(fname)
#     # Extract parameter value
#     m = re.search(r"cifar_noniid_(\w+).csv$", fname)
#     if m:
#         # Parse experiment
#         type = (m.group(1))
#         # Parse csv to get desired timeseries
#         df = pd.read_csv(fname, index_col=False)
#         df.drop(columns='Wall time', inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df['type'] = type
#         data = pd.concat([data, df])
#     else:
#         raise NotImplemented("Error")
#
# sns.set(rc={'figure.figsize':(9, 6)})
# sns.lineplot(data=data, x="Step", y="Value", hue="type", errorbar=("se", 2))
# plt.title(EXPERIMENT_TITLE)
# plt.ylabel("Test Accuracy")
#
# Path(f"{FIG_OUT_DIR}").mkdir(parents=True, exist_ok=True)
# plt.savefig(f"{FIG_OUT_DIR}/{EXPERIMENT_TITLE}")
# plt.show()