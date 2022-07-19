# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring
# pylint: disable=C0103
# pylint: disable=R0903

# The following is a simple neural network trained and tested using FashinMINST data.
# It is using eager mode targeting the ort device. After building eager mode run
# PYTHONPATH=~/{repo root}/build/Linux/Debug python ~/{repo root}/orttraining/orttraining/eager/test/training_test.py

import os

import onnxruntime_pybind11_state as torch_ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# we copy traing data to build folder as it is gitignored
dataset_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "build/data")
training_data = datasets.FashionMNIST(root=dataset_root_dir, train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root=dataset_root_dir, train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = torch_ort.device()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, sample):
        sample = self.flatten(sample)
        logits = self.linear_relu_stack(sample)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        x_ort = X.to(device)
        y_ort = y.to(device)
        pred = model(x_ort)
        loss = loss_fn(pred, y_ort)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            x_ort = X.to(device)
            y_ort = y.to(device)
            pred = model(x_ort)
            test_loss += loss_fn(pred, y_ort).item()
            correct += (pred.argmax(1) == y_ort).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model_nn = NeuralNetwork().to(device)
learning_rate = 1e-3

loss_fn_nn = nn.CrossEntropyLoss().to(device)
optimizer_nn = torch.optim.SGD(model_nn.parameters(), lr=learning_rate)

batch_size = 64
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model_nn, loss_fn_nn, optimizer_nn)
    test_loop(test_dataloader, model_nn, loss_fn_nn)
print("Done!")
