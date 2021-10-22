## This code is from https://github.com/pytorch/examples/blob/master/mnist/main.py
## with modification to do training using onnxruntime as backend on cuda device.
## A private PyTorch build from https://aiinfra.visualstudio.com/Lotus/_git/pytorch (ORTTraining branch) is needed to run the demo.

## Model testing is not complete.

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

from onnxruntime.capi.ort_trainer import IODescription, ModelDescription, ORTTrainer
from mpi4py import MPI
try:
    from onnxruntime.capi._pybind_state import set_cuda_device_id
except ImportError:
    pass

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

def train_with_trainer(args, trainer, device, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        learning_rate = torch.tensor([args.lr])
        loss = trainer.train_step(data, target, learning_rate)

        # Since the output corresponds to [loss_desc, probability_desc], the first value is taken as loss.
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss[0]))

# TODO: comple this once ORT training can do evaluation.
def test_with_trainer(args, trainer, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)
            output = F.log_softmax(trainer.eval_step(data, fetches=['probability']), dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()     # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)   # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def mnist_model_description():
    input_desc = IODescription('input1', ['batch', 784], torch.float32)
    label_desc = IODescription('label', ['batch', ], torch.int64, num_classes=10)
    loss_desc = IODescription('loss', [], torch.float32)
    probability_desc = IODescription('probability', ['batch', 10], torch.float32)
    return ModelDescription([input_desc, label_desc], [loss_desc, probability_desc])

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
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    comm = MPI.COMM_WORLD
    args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) if ('OMPI_COMM_WORLD_LOCAL_RANK' in os.environ) else 0
    args.world_rank = int(os.environ['OMPI_COMM_WORLD_RANK']) if ('OMPI_COMM_WORLD_RANK' in os.environ) else 0
    args.world_size=comm.Get_size()
    if use_cuda:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        set_cuda_device_id(args.local_rank)
    else:
        device = torch.device("cpu")

    input_size = 784
    hidden_size = 500
    num_classes = 10
    model = NeuralNet(input_size, hidden_size, num_classes)

    model_desc = mnist_model_description()
    # use log_interval as gradient accumulate steps
    trainer = ORTTrainer(model,
                         my_loss,
                         model_desc,
                         "SGDOptimizer",
                         None,
                         IODescription('Learning_Rate', [1,], torch.float32),
                         device,
                         1,
                         args.world_rank,
                         args.world_size,
                         use_mixed_precision=False,
                         allreduce_post_accumulation=True)
    print('\nBuild ort model done.')

    for epoch in range(1, args.epochs + 1):
        train_with_trainer(args, trainer, device, train_loader, epoch)
        import pdb
        test_with_trainer(args, trainer, device, test_loader)


if __name__ == '__main__':
    main()
