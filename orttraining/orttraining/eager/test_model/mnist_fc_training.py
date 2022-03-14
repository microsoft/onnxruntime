## This code is from https://github.com/pytorch/examples/blob/master/mnist/main.py
## with modification to do training using onnxruntime as backend on cuda device.
## A private PyTorch build from https://aiinfra.visualstudio.com/Lotus/_git/pytorch (ORTTraining branch) is needed to run the demo.

## Model testing is not complete.

from __future__ import print_function
import argparse
import torch
from onnxruntime.training import ORTModule
from onnxruntime.capi import _pybind_state as torch_ort_eager
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def my_loss(x, target):
    return F.nll_loss(F.log_softmax(x, dim=1), target)

def train_with_eager(args, model, optimizer, device, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data_cpu = data.reshape(data.shape[0], -1)
        data = data_cpu.to(device)

        x = model(data)
        loss = my_loss(x.cpu(), target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        # Since the output corresponds to [loss_desc, probability_desc], the first value is taken as loss.
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_cpu), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

def main():
#Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # set device
    torch_ort_eager.set_device(0, 'CPUExecutionProvider', {})

    device = torch.device('ort', index=0)
    input_size = 784
    hidden_size = 500
    num_classes = 10
    model = NeuralNet(input_size, hidden_size, num_classes)
    model.to(device)
    model = ORTModule(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print('\nStart Training.')

    for epoch in range(1, args.epochs + 1):
        train_with_eager(args, model, optimizer, device, train_loader, epoch)


if __name__ == '__main__':
    main()
