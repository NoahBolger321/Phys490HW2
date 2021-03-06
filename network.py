import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    """
    Class to define two hidden layer NN.
    Consists of 3 fully connected layers and Relu activation functions
    """
    def __init__(self, n_pixels):
        super(Net, self).__init__()
        self.n_pixels = n_pixels
        self.fc1 = nn.Linear(self.n_pixels, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 9)

    def forward(self, x):
        x_1 = self.fc1(x)
        x_2 = func.relu(x_1)
        x_3 = self.fc2(x_2)
        x_4 = func.relu(x_3)
        return self.fc3(x_4)

    def test(self, X_test, Y_test, loss):
        self.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test).float()
            targets = torch.from_numpy(Y_test).long()
            output = self.forward(inputs)
            cross_val = loss(output, targets.reshape(-1))
        return cross_val.item(), output, targets
