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
from joblib import Parallel, delayed
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from torchvision import transforms
from options import args_parser
from device import Device

# Logging
writer = SummaryWriter()

# Parse args
args = args_parser()
print("========== ARGS ===========")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("===========================")
# writer.add_hparams(vars(args), name='Args')

# Set seed from provided args
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.use_gpu:
    torch.cuda.manual_seed(args.seed)
compute_device = torch.device('cuda' if args.use_gpu else "cpu")
print("Compute device:", compute_device)

# Create Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST('./data', download=True, train=True, transform=transform)
test_ds = datasets.MNIST('./data', download=True, train=False, transform=transform)

# Create/Initialize devices
devices = [Device(id=devid, train_ds=train_ds, test_ds=test_ds, device=compute_device, args=args) for
           devid in range(args.num_devices)]


def call_train_local(dev):
    return dev.train_local(num_iter=args.num_local_update, device=compute_device)


def call_send_target_devices(dev):
    dev.send_target_devices(device_list=devices, sample_list=[args.num_local_update for el in devices])


def call_aggregate_weights(dev):
    dev.aggregate_weights()


def call_test_local(dev):
    return dev.test_local(device=compute_device)


print(f"Running on {os.cpu_count()} processes")
with Parallel(n_jobs=os.cpu_count(),backend="threading") as parallel:
    for round in range(10):
        losses = parallel(delayed(call_train_local)(dev) for dev in devices)
        avg_loss = np.mean(losses)
        loss_var = np.var(losses)
        writer.add_scalar("Loss/avg_loss", avg_loss, round)
        writer.add_scalar("Loss/loss_var", loss_var, round)


        # parallel(delayed(call_send_target_devices)(dev) for dev in devices)
        # parallel(delayed(call_aggregate_weights)(dev) for dev in devices)

        # Compute test accuracy
        results = parallel(delayed(call_test_local)(dev) for dev in devices)
        avg_acc = np.sum([el[0] for el in results]) / np.sum([el[1] for el in results])
        acc_var = np.var([el[0] / el[1] for el in results])
        print(f"Round: {round}, Avg Loss = {avg_loss}, var = {loss_var} \t | \t Average acc = {avg_acc}, variance = {acc_var}")
        writer.add_scalar("Accuracy/avg_acc", avg_acc, round)
        writer.add_scalar("Accuracy/acc_var", acc_var, round)

