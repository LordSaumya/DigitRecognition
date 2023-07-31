# Imports 
import pytorch_lightning as pl
import math as maths
import torch as pt
from torch import nn
from torch.nn import functional as F
import torchmetrics as tm
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

# Data
from dataLoader import MNISTDataLoader

# Number and nature of convolutional layers
conv_layers = 8
growing = True

# Model
class recognitionModel(pl.LightningModule):
    def __init__(self, lr=0.001, batch_size=32):
        super().__init__()

        # Define and save hpams
        self.lr = lr
        self.batch_size = batch_size

        # Model Architecture
        self.linear = nn.Linear(28*28, 10)

        # Function definitions
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        if (conv_layers > 0):
            if (not growing):
                # Convolutional layers (constant number of channels in each layer [32])
                self.conv = nn.ModuleList([nn.Conv2d(32, 32, 3, 1) for i in range(conv_layers)])
                # First layer has 1 input channel
                self.conv[0] = nn.Conv2d(1, 32, 3, padding=1)
                # Last layer has 784 output channels (same as input channels of linear layer)
                self.conv[conv_layers-1] = nn.Conv2d(32, 784, 3, 1)
            else:
                # Convolutional layers (doubles number of channels in each layer starting from 32)
                self.conv = nn.ModuleList([nn.Conv2d(int(maths.pow(2, i + 4)), int(maths.pow(2, i + 5)), 3, 1) for i in range(conv_layers)])
                # First layer has 1 input channel
                self.conv[0] = nn.Conv2d(1, 32, 3, 1)
                # Last layer has 784 output channels (same as input channels of linear layer)
                self.conv[conv_layers-1] = nn.Conv2d(int(maths.pow(2, conv_layers - 1 + 4)), 784, 3, 1)

        elif (conv_layers == 1):
            self.conv = nn.ModuleList([nn.Conv2d(1, 784, 3, 1)])

        # Metrics
        self.train_acc = tm.Accuracy(task = "multiclass", num_classes = 10)
        self.train_acc = tm.Accuracy(task = "multiclass" , num_classes = 10)
        self.train_acc = tm.Accuracy(task = "multiclass" , num_classes = 10)

    def forward(self, x):

        if (conv_layers > 0):
            # Convolutional layer(s)
            for i in range(conv_layers):
                x = self.conv[i](x)
                x = self.relu(x)

        # Flatten
        x = nn.Flatten()(x)

        # linear layer
        x = self.linear(x)

        return x