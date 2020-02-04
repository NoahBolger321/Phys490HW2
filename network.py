import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    """
    Class to define one hidden layer NN.
    Consists of two fully connected layers and Relu activation function
    """
    def __init__(self, n_pixels):
        super(Net, self).__init__()
        self.n_pixels = n_pixels
        self.fc1 = nn.Linear(self.n_pixels, 250)
        self.fc2 = nn.Linear(250, 9)

    # Feedforward function
    def forward(self, x):
        x_1 = self.fc1(x)
        x_2 = func.relu(x_1)
        return self.fc2(x_2)
