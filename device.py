"""
Represents an edge device for decentralized federated learning

Decentralized learning steps:
1. Data Collection
2. Local Training
3. Weight sharing
"""
import math

import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from models.initialize_model import initialize_model
import torch
from copy import deepcopy
from average import average_weights
from torchsummary import summary
import random

class Device:
    def __init__(self, id, train_ds, test_ds, device, args, x_pos, y_pos, radio_range):
        self.id = id
        self.bs = args.batch_size
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.shuffle_ds = args.shuffle_dataset
        self.model = initialize_model(args, device)
        #print(f"Device {self.id} model summary: ")
        #summary(self.model.shared_layers, (3, 32, 32), device='cuda' if args.use_gpu else "cpu")

        self.args = args

        # Device location
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radio_range = radio_range
        self.transmission_dist_hist = []

        # Training
        self.train_dl = DataLoader(self.train_ds, batch_size=self.bs, shuffle=self.shuffle_ds, num_workers=0)
        self.epoch = 0  # Number of times iterated through train dataset

        # Aggregation
        self.received_weights = []
        self.target_share_devs = []
        self.share_k_devices = args.num_share_devices

        # Testing
        self.test_dl = DataLoader(self.test_ds, batch_size=self.bs, shuffle=False, num_workers=0)  # Using same ds for train/test

    def train_local(self, num_iter, device):
        print(f"Training device {self.id} for {num_iter} iterations using {device}")
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_dl:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.model.optimize_model(input_batch=inputs, label_batch=labels)
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end: break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch=self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        return loss

    def test_local(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_dl:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total, self.id

    def _is_within_range(self, other):
        return math.sqrt((self.x_pos - other.x_pos)**2 + (self.y_pos - other.y_pos)**2) <= self.radio_range

    def record_transmission(self, other):
        self.transmission_dist_hist.append(math.sqrt((self.x_pos - other.x_pos)**2 + (self.y_pos - other.y_pos)**2))

    def _choose_target_devices(self, device_list):
        if self.args.model_share_strategy == 'random':
            idxs = np.random.choice(list(range(len(device_list))), self.share_k_devices, replace=False)
            self.target_share_devs = [device_list[i] for i in idxs]
        elif self.args.model_share_strategy == 'distance':
            devices_in_range = list(filter(self._is_within_range, device_list))
            idxs = np.random.choice(list(range(len(devices_in_range))), self.share_k_devices, replace=False)
            self.target_share_devs = [devices_in_range[i] for i in idxs]

    def send_target_devices(self, device_list, sample_list):
        self._choose_target_devices([d for d in device_list if d.id != self.id])

        for dev, sample in zip(self.target_share_devs, sample_list):
            if random.random() < self.args.comm_reliability:
                print(f"device {self.id} sending to {dev.id}")
                dev._receive_from_node(deepcopy(self.model.shared_layers.state_dict()), sample)
                self.record_transmission(dev)
            else:
                print(f"comm failed from device {self.id} to {dev.id}")

    def send_to_cloud(self):
        # TODO Add in call to record_transmission_distance()
        # TODO add transmission failures
        return deepcopy((self.model.shared_layers.state_dict(), self.args.num_local_update))

    def _receive_from_node(self, weights, n_samples):
        # TODO add transmission failures
        self.received_weights.append((weights, n_samples))

    def receive_from_cloud(self, new_weights):
        # TODO add transmission failures
        self.model.update_model(new_weights)

    def aggregate_weights(self):
        # Add current device's weights to list
        self.received_weights.append((deepcopy(self.model.shared_layers.state_dict()), self.args.num_local_update))

        all_weights = [el[0] for el in self.received_weights]
        all_samples = [el[1] for el in self.received_weights]

        new_weights = average_weights(all_weights, all_samples)

        self.model.update_model(new_weights)

        self.received_weights.clear()
        self.target_share_devs.clear()
