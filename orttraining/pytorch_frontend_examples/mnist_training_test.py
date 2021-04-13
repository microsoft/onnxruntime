## This code is from https://github.com/pytorch/examples/blob/master/mnist/main.py
## with modification to do training using onnxruntime as backend on cuda device.
## A private PyTorch build from https://aiinfra.visualstudio.com/Lotus/_git/pytorch (ORTTraining branch) is needed to run the demo.
## To run the demo with ORT backend:
##      python mnist_training.py --use-ort

## When "--use-ort" is not given, it will run training with PyTorch as backend.
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

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def my_loss(x, target):
    #return F.nll_loss(F.log_softmax(x, dim=1), target)
    #return F.nll_loss(F.softmax(x, dim=1), to_one_hot(target, 10))
    return nn.BCEWithLogitsLoss(x, to_one_hot(target, 10))

# simple cross entropy cost (might be numerically unstable if pred has 0)
def xentropy_cost(x_target, x_pred):
    import pdb
    pdb.set_trace()
    assert x_target.size() == x_pred.size(), "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
    logged_x_pred = F.log_softmax(x_pred, dim=1)
    cost_value = -torch.sum(x_target * logged_x_pred)
    return cost_value

def train_with_trainer(args, trainer, device, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        import pdb
        pdb.set_trace()
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        learning_rate = torch.tensor([args.lr])
        x = trainer(data)
        loss = xentropy_cost(x,  to_one_hot(target, 10))
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
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--use-ort', action='store_true', default=False,
                        help='to use onnxruntime as training backend')

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


    args.local_rank = 0
    args.world_rank = 0
    args.world_size=1
    if use_cuda:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cpu")
    args.n_gpu = 1
    

    input_size = 784
    hidden_size = 160
    num_classes = 10
    model = NeuralNet(input_size, hidden_size, num_classes)

    dummy_input = torch.randn(256, 784)
    input_names = ["X"]
    output_names = ["predictions"]
    import pdb
    pdb.set_trace()
    torch.onnx.export(model, dummy_input, "mnist_pt.onnx", verbose=True, input_names=input_names, output_names=output_names)
    
    for epoch in range(1, args.epochs + 1):
        train_with_trainer(args, model, device, train_loader, epoch)
        import pdb
        test_with_trainer(args, trainer, device, test_loader)


if __name__ == '__main__':
    main()
