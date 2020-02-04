from data_handling import MnistData
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import Net
import argparse


if __name__ == '__main__':

    data = MnistData(test_size=0.1017)

    X_train = np.matrix(data.X_train.tolist())
    Y_train = np.matrix(data.Y_train.tolist())
    X_test = np.matrix(data.X_test.tolist())
    Y_test = np.matrix(data.Y_test.tolist())

    # Command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Tutorial')
    parser.add_argument('-p', dest='param', metavar='data/parameters.json.json',
                        help='parameter file name')
    args = parser.parse_args()

    with open(args.param) as paramfile:
        param = json.load(paramfile)

    model = Net(14*14)
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1, int(param['num_epochs']) + 1):

            inputs = torch.from_numpy(X_train).float()
            targets = torch.from_numpy(Y_train).long()

            output = model.forward(inputs)

            loss = loss_func(output, targets.reshape(-1))
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:

                estimates = output.argmax(1)
                actual = targets[0]
                differences = np.array(np.subtract(estimates, actual))
                num_correct = len(differences[np.where(differences == 0)])
                accuracy = 100*(num_correct / len(actual))
                print('Epoch [{}/{}]'.format(epoch+1, param['num_epochs'])+\
                      '\tTraining Loss: {:.4f}'.format(loss.item())+\
                      '\tAccuracy: {:.4f}'.format(accuracy))
