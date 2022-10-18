"""
Main script for decentralized federated learning

Decentralized learning steps:
1. Data Collection
2. Local Training
3. Weight sharing
"""
import numpy as np
import torch
from joblib import Parallel, delayed
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from torchvision import transforms
from options import args_parser
from device import Device

# Parse args
args = args_parser()

print("========== ARGS ===========")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("===========================")

# Logging
writer = SummaryWriter()

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
devices = [Device(id=devid, train_ds=train_ds, test_ds=test_ds, device=compute_device, args=args) for devid in range(args.num_devices)]


def call_train_local(dev):
    dev.train_local(num_iter=args.num_local_update, device=compute_device)


with Parallel(n_jobs=10) as parallel:
    parallel(delayed(call_train_local)(dev) for dev in devices)

# devices[0].train_local(100, compute_device)