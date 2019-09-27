## This code is from https://github.com/pytorch/examples/blob/master/mnist/main.py
## with modification to do training using onnxruntime as backend on cuda device.
## A private PyTorch build from https://aiinfra.visualstudio.com/Lotus/_git/pytorch is needed to run the demo.
## To run the demo with ORT backend:
##      python mnist_training.py --use_ort
## When "--use_ort" is not given, it will run training with PyTorch as backend.
## Model testing is not complete.

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from ort_trainer import ORTTrainer

## 
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

def parameter_count(parameters):
    count = 0
    for p in parameters:
        print(p.shape)

def my_loss(x, target):
    return F.nll_loss(F.log_softmax(x, dim=1), target)


def train(args, trainer, device, train_loader, epoch):    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss = trainer.train_step(data, target)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#TODO : comple this once ORT training can do evaluation.
def test(args, trainer, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)
            output = F.log_softmax(trainer.eval(data), dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

    parser.add_argument('--use_ort', action='store_true', default=False,
                        help='to use onnxruntime as training backend')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not use_cuda:
        print("Training with ORT only works with CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

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
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    input_size = 784
    hidden_size = 500
    num_classes = 10
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    optimizer_constructor_lambda = lambda model_parameters: optim.SGD(model_parameters, args.lr, args.momentum)

    trainer = ORTTrainer(model, my_loss, optimizer_constructor_lambda)

    if args.use_ort:
#call trainer.compile() to setup ORT backend for training.
#if trainer.compile() is not called, it will use PyTorch as training backend
        trainer.compile(batch_size = 64, feature_shape = 28 * 28, input_dtype = torch.float32, number_classes = 10, label_type=torch.int64)

    for epoch in range(1, args.epochs + 1):
        train(args, trainer, device, train_loader, epoch)
        test(args, trainer, device, test_loader)

#if (args.save_model) :
#torch.save(model.state_dict(), "mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
