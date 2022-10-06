from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import gc
import typing

from tqdm import tqdm


class Node:
    def __init__(self, id, local_dataset, batch_size, local_epochs, criterion):
        self.id = id
        self.local_dataset = local_dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.children = []
        self.parents = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self.model = None

        self.dataloader = DataLoader(self.local_dataset, batch_size=self.batch_size, shuffle=True)

    def addChild(self, child):
        self.children.append(child)

    def addParent(self, parent):
        self.parents.append(parent)

    def receiveModel(self, model):
        self.model = model

    def transmitModel(self):
        return self.model

    def train_local(self):
        if self.model is None:
            raise Exception(f"Model not initialized for node {self.id}")

        self.model.train()
        self.model.to(self.device)

        optim = torch.optim.SGD(params=self.model.params(), lr=1e-2, momentum=0.9)

        for e in range(self.local_epoch):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optim.zero_grad()
                outputs = self.model(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optim.step()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

    def aggregate_children(self, include_self):
        averaged_weights = OrderedDict()
        model_list = [child.transmitModel() for child in self.children] + [self.model]
        for m_idx, model in enumerate(model_list):
            child_weights = model.transmitModel().state_dict()
            for key in self.model.state_dict().keys():
                if m_idx == 0 and not include_self:
                    averaged_weights[key] = 1/len(model_list) * child_weights[key]
                else:
                    averaged_weights[key] += 1/len(model_list) * child_weights[key]
        self.model.load_state_dict(averaged_weights)
        gc.collect()