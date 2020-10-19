import argparse
import torch
from torchvision import datasets, transforms

import onnxruntime
from onnxruntime.training import ORTModule


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train(args, model, device, optimizer, loss_fn, train_loader, epoch):
    model.train()
    for iteration, (data, target) in enumerate(train_loader):
        if iteration == args.train_steps:
            break
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        optimizer.zero_grad()
        if args.pytorch_only:
            probability = model(data)
        else:
            probability = model(data)

        if args.view_graphs:
            import torchviz
            pytorch_backward_graph = torchviz.make_dot(probability, params=dict(list(model.named_parameters())))
            pytorch_backward_graph.view()

        loss = loss_fn(probability, target)
        loss.backward()
        optimizer.step()

        # Stats
        if iteration % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, iteration * len(data), len(train_loader.dataset),
                100. * iteration / len(train_loader), loss))


def test(args, model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)
            output = model(data)

            # Stats
            test_loss += loss_fn(output, target, False).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def my_loss(x, target, is_train=True):
    if is_train:
        return torch.nn.CrossEntropyLoss()(x, target)
    else:
        return torch.nn.CrossEntropyLoss(reduction='sum')(x, target)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-steps', type=int, default=-1, metavar='N',
                        help='number of steps to train. Set -1 to run through whole dataset (default: -1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--view-graphs', action='store_true', default=False,
                        help='views forward and backward graphs')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()


    # Common setup
    torch.manual_seed(args.seed)
    onnxruntime.set_seed(args.seed)

    # TODO: CUDA support is broken due to copying from PyTorch into ORT
    # if not args.no_cuda and torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = 'cpu'

    ## Data loader
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),

                                                                            transforms.Normalize((0.1307,), (0.3081,))])),
                                            batch_size=args.batch_size,
                                            shuffle=True)
    if args.test_batch_size > 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.test_batch_size, shuffle=True)

    # Model architecture
    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10).to(device)
    if not args.pytorch_only:
        print('Training MNIST on ORTModule....')
        model = ORTModule(model)
    else:
        print('Training MNIST on vanilla PyTorch....')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Train loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, optimizer, my_loss, train_loader, epoch)
        if args.test_batch_size > 0:
            test(args, model, device, my_loss, test_loader)


if __name__ == '__main__':
    main()
