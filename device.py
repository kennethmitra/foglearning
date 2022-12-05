"""
Represents an edge device for decentralized federated learning

Decentralized learning steps:
1. Data Collection
2. Local Training
3. Weight sharing
"""
import math
import random
from copy import deepcopy

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from average import average_weights
from models.initialize_model import initialize_model


class Device:
    """
    This class simulates an edge device. Each edge device has a (x,y) coordinate, a train and test dataset, id, and radio range (applicable to distance-based communication strategies)
    Other parameters are passed with the args object
    """
    def __init__(self, id, train_ds, test_ds, device, args, x_pos, y_pos, radio_range):
        self.id = id
        self.bs = args.batch_size
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.shuffle_ds = args.shuffle_dataset
        self.model = initialize_model(args, device)
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
        
        # communication failure
        self.comm_fail = False

        # Testing
        self.test_dl = DataLoader(self.test_ds, batch_size=self.bs, shuffle=False, num_workers=0)  # Using same ds for train/test

    def train_local(self, num_iter, device):
        """
        Performs local training steps
        :param num_iter: Number of gradient update steps to perform
        :param device: Use GPU or CPU
        :return: Return the training loss
        """
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
        """
        Test device model on local test dataset
        :param device: PyTorch device (Use GPU or CPU)
        :return: Tuple of (number of correct test samples, total test samples, device id)
        """
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
        """
        Internal function. Returns true if other device is within radio range
        :param other: Other device
        :return: Boolean
        """
        return math.sqrt((self.x_pos - other.x_pos)**2 + (self.y_pos - other.y_pos)**2) <= self.radio_range

    def record_transmission(self, other):
        """
        Saves the communication distance
        :param other: Other device in communication
        :return: None
        """
        self.transmission_dist_hist.append(math.sqrt((self.x_pos - other.x_pos)**2 + (self.y_pos - other.y_pos)**2))
        
    def record_cloud_transmission(self):
        """
        Saves the communication distance to cloud
        :return: None
        """
        # Assumes cloud distance is the furthest distance between the devices
        self.transmission_dist_hist.append(self.args.num_devices)

    def _choose_target_devices(self, device_list):
        """
        Choose target devices for sharing model weights according to different strategies
        :param device_list: List of devices to consider as candidates to select from
        :return: None
        """
        if self.args.model_share_strategy == 'random':
            idxs = np.random.choice(list(range(len(device_list))), self.share_k_devices, replace=False)
            self.target_share_devs = [device_list[i] for i in idxs]

        elif self.args.model_share_strategy == 'distance':
            devices_in_range = list(filter(self._is_within_range, device_list))
            idxs = np.random.choice(list(range(len(devices_in_range))), min(self.share_k_devices, len(devices_in_range)), replace=False)
            self.target_share_devs = [devices_in_range[i] for i in idxs]

        elif self.args.model_share_strategy == 'ring':
            # communicate with neighbors only
            idxs = [(self.id)%(self.args.num_devices-1), (self.id-1)%(self.args.num_devices-1)]
            print(idxs)
            self.target_share_devs = [device_list[i] for i in idxs]

    def send_target_devices(self, device_list, sample_list):
        """
        Sends this device's model's weights to selected target devices
        :param device_list: Candidate target devices
        :param sample_list: Number of samples trained on for each device in device_list
        :return: None
        """
        self._choose_target_devices([d for d in device_list if d.id != self.id])

        for dev, sample in zip(self.target_share_devs, sample_list):
            if random.random() < self.args.comm_reliability:
                print(f"device {self.id} sending to {dev.id}")
                dev._receive_from_node(deepcopy(self.model.shared_layers.state_dict()), sample)
                self.record_transmission(dev)
            else:
                print(f"comm failed from device {self.id} to {dev.id}")
                self.comm_fail = True

    def send_to_cloud(self):
        """
        Send current device's model weights to the cloud server
        :return:
        """
        if np.random.rand() < self.args.comm_reliability:
            self.record_cloud_transmission()
            print(f"device {self.id} sending to cloud")
            return deepcopy((self.model.shared_layers.state_dict(), self.args.num_local_update))
        else:
            print(f"device {self.id} failed to send to cloud")
            self.comm_fail = True
            return None

    def _receive_from_node(self, weights, n_samples):
        """
        Receive model weights from another device
        :param weights: The weights to receive
        :param n_samples: The number of samples used to train the weights
        :return: None
        """
        self.received_weights.append((weights, n_samples))

    def receive_from_cloud(self, new_weights):
        """
        Receive new weight from the cloud server
        :param new_weights: The weights to receive
        :return: None
        """
        if np.random.rand() < self.args.comm_reliability:
            self.model.update_model(new_weights)
        else:
            print(f"device {self.id} failed to receive from cloud")

    def aggregate_weights(self):
        """
        Update this device's model's weights with the weighted average of the received weights
        :return: None
        """
        # Add current device's weights to list
        self.received_weights.append((deepcopy(self.model.shared_layers.state_dict()), self.args.num_local_update))

        all_weights = [el[0] for el in self.received_weights]
        all_samples = [el[1] for el in self.received_weights]

        new_weights = average_weights(all_weights, all_samples)

        self.model.update_model(new_weights)

        self.received_weights.clear()
        self.target_share_devs.clear()
