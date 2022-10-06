import torchvision
from node import Node


mnist = torchvision.datasets.MNIST('./data2', download=True)
server = Node(0, local_dataset=mnist, )