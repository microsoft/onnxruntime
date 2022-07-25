## This code is from https://github.com/pytorch/examples/blob/master/mnist/main.py
## with modification to do training using onnxruntime as backend.

# pylint: disable=missing-docstring
# pylint: disable=C0103

from __future__ import print_function

import argparse
import os

import onnxruntime_pybind11_state as torch_ort
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

# we use the build directory so gitignore applies.
dataset_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "build/data")


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
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
        target_ort = target.to(device)

        x = model(data)
        loss = my_loss(x, target_ort)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Since the output corresponds to [loss_desc, probability_desc], the first value is taken as loss.
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data_cpu),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {"num_workers": 0, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            dataset_root_dir,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            dataset_root_dir,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    device_ort = torch_ort.device()
    input_size = 784
    hidden_size = 500
    num_classes = 10
    model_nn = NeuralNet(input_size, hidden_size, num_classes)
    model_nn.to(device_ort)
    optimizer = optim.SGD(model_nn.parameters(), lr=0.01)

    print("\nStart Training.")

    for epoch in range(1, args.epochs + 1):
        train_with_eager(args, model_nn, optimizer, device_ort, train_loader, epoch)


if __name__ == "__main__":
    main()
