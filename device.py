"""
Represents an edge device for decentralized federated learning

Decentralized learning steps:
1. Data Collection
2. Local Training
3. Weight sharing
"""
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from models.initialize_model import initialize_model
import torch
from copy import deepcopy
from average import average_weights

class Device:
    def __init__(self, id, train_ds, test_ds, device, args):
        self.id = id
        self.bs = args.batch_size
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.shuffle_ds = args.shuffle_dataset
        self.model = initialize_model(args, device)
        self.args = args

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
        return correct, total

    def _choose_target_devices(self, device_list):
        idxs = np.random.choice(list(range(len(device_list))), self.share_k_devices)
        self.target_share_devs = [device_list[i] for i in idxs]

    def send_target_devices(self, device_list, sample_list):
        self._choose_target_devices(device_list)

        for dev, sample in zip(self.target_share_devs, sample_list):
            dev._receive_from_node(deepcopy(self.model.shared_layers.state_dict()), sample)

    def _receive_from_node(self, weights, n_samples):
        self.received_weights.append((weights, n_samples))

    def aggregate_weights(self):
        # Add current device's weights to list
        self.received_weights.append((deepcopy(self.model.shared_layers.state_dict()), self.args.num_local_update))

        all_weights = [el[0] for el in self.received_weights]
        all_samples = [el[1] for el in self.received_weights]

        new_weights = average_weights(all_weights, all_samples)

        self.model.update_model(new_weights)

        self.received_weights.clear()
        self.target_share_devs.clear()
