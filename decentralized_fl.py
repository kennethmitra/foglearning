"""
Main script for decentralized federated learning

Decentralized learning steps:
1. Data Collection
2. Local Training
3. Weight sharing
"""
import os

import numpy as np
import torch
import torchvision.datasets as datasets
from joblib import Parallel, delayed
from tensorboardX import SummaryWriter
from torchvision import transforms

import average
from device import Device
from options import args_parser


##############################
# Initial Setup and Logging
##############################

print(f"PyTorch version: {torch.__version__}")

# Parse args
args = args_parser()

# Logging
writer = SummaryWriter(comment=args.run_name)

# Print args
print("========== ARGS ===========")
argstring = []
argtable_head = []
argtable_mid = []
argtable_vals = []
for k, v in vars(args).items():
    print(f"{k}: {v}")
    argstring.append(f"{k}: {v}")
    argtable_head.append(str(k))
    argtable_mid.append("---")
    argtable_vals.append(str(v))
print("===========================")
writer.add_text("hparams/dump", " \n ".join(argstring))
writer.add_text("hparams/table",
                f"| {' | '.join(argtable_head)} |  \n| {' | '.join(argtable_mid)} |  \n| {' | '.join(argtable_vals)} |  ")

# Set seed from provided args
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.use_gpu:
    torch.cuda.manual_seed(args.seed)
compute_device = torch.device('cuda' if args.use_gpu else "cpu")
print("Compute device:", compute_device)

##############################
# Define Dataset
##############################

# For MNIST
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# train_ds = datasets.MNIST('./data', download=True, train=True, transform=transform)
# test_ds = datasets.MNIST('./data', download=True, train=False, transform=transform)

# For CIFAR10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_ds = datasets.CIFAR10('./data', download=True, train=True, transform=transform_train)
test_ds = datasets.CIFAR10('./data', download=True, train=False, transform=transform_test)
train_size = len(train_ds)

data_indices = []
for label in range(10):
    data_indices.append(np.nonzero(np.array(train_ds.targets) == label)[0])


####################################
# Functions for Dataset processing
####################################

def get_device_data(class_dist, total_data_count):
    return_indices = np.array([])
    label_count = []
    for labels in range(10):
        class_data_count = int(class_dist[labels] * total_data_count)
        label_count.append(class_data_count)
        return_indices = np.concatenate(
            [return_indices, np.random.choice(data_indices[labels].flatten(), class_data_count)])
    train_subset = torch.utils.data.Subset(train_ds, return_indices)
    print("data distribution:", label_count)
    return train_subset.dataset

def get_device_distribution(device_id):
    if args.iid:
        return [0.1] * 10
    else:
        class_dist = 0.0*np.ones(10)
        major_labels = (device_id%10, (device_id%10+1)%10)
        for i in major_labels:
            class_dist[i] = 0.5
        return class_dist

##############################
# Create/Initialize devices
##############################

devices = [Device(id=devid, train_ds=get_device_data(get_device_distribution(devid), train_size//args.num_devices), test_ds=test_ds, device=compute_device, args=args, x_pos=float(devid), y_pos=0, radio_range=4.5) for devid in range(args.num_devices)]


#######################################
# Define parallelization functions
#######################################

def call_train_local(dev):
    return dev.train_local(num_iter=args.num_local_update, device=compute_device)


def call_send_target_devices(dev):
    dev.send_target_devices(device_list=devices, sample_list=[args.num_local_update for el in devices])


def call_aggregate_weights(dev):
    dev.aggregate_weights()


def call_test_local(dev):
    return dev.test_local(device=compute_device)


##############################
# Run experiment
##############################

print(f"Running on {os.cpu_count()} processes")
with Parallel(n_jobs=os.cpu_count() - 2, backend="threading") as parallel:
    for round in range(args.num_total_rounds):
        losses = parallel(delayed(call_train_local)(dev) for dev in devices)
        avg_loss = np.mean(losses)
        loss_var = np.var(losses)
        writer.add_scalar("Loss/avg_loss", avg_loss, round)
        writer.add_scalar("Loss/loss_var", loss_var, round)

        if args.fed_avg:
            # Federated averaging
            weights = []
            for dev in devices:
                weights.append(dev.send_to_cloud())

            all_weights = [el[0] for el in weights if el is not None]
            all_samples = [el[1] for el in weights if el is not None]

            new_weights = average.average_weights(all_weights, all_samples)
            for dev in devices:
                dev.receive_from_cloud(new_weights)
        else:
            parallel(delayed(call_send_target_devices)(dev) for dev in devices)
            parallel(delayed(call_aggregate_weights)(dev) for dev in devices)

        # Compute test accuracy
        results = parallel(delayed(call_test_local)(dev) for dev in devices)
        avg_acc = np.sum([el[0] for el in results]) / np.sum([el[1] for el in results])
        acc_var = np.var([el[0] / el[1] for el in results])
        print(
            f"Round: {round}, Avg Loss = {avg_loss}, var = {loss_var} \t | \t Average acc = {avg_acc}, variance = {acc_var}")
        writer.add_scalar("Accuracy/avg_acc", avg_acc, round)
        writer.add_scalar("Accuracy/acc_var", acc_var, round)

        # Individual Accuracies
        for i in range(len(results)):
            writer.add_scalar(f"Individual_Accuracy/device_{results[i][2]}", results[i][0] / results[i][1], round)

    for device in devices:
        writer.add_text(f"stats/tx_dist_avg_{device.id}", str(np.mean(device.transmission_dist_hist)), round)
        writer.add_text(f"stats/tx_dist_var_{device.id}", str(np.var(device.transmission_dist_hist)), round)