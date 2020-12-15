# This code is from https://github.com/pytorch/examples/blob/master/mnist/main.py
# with modification to do training using onnxruntime as backend on cuda device.

# To print nodes from ORT backend
# Add --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1 to build.sh
# export ORT_DEBUG_NODE_IO_NAME_FILTER="SoftmaxCrossEntropyLoss_3_Grad/SoftmaxCrossEntropyLossGrad_0"
# export ORT_DEBUG_NODE_IO_NAME_FILTER="SoftmaxCrossEntropyLoss_3"
# export ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=1
# export ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
# See https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/debug_node_inputs_outputs_utils.h

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import onnxruntime
from onnxruntime.training import ORTTrainer, ORTTrainerOptions, optim


# Pytorch model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ONNX Runtime training
def mnist_model_description():
    return {'inputs': [('input1', ['batch', 784]),
                       ('label', ['batch'])],
            'outputs': [('loss', [], True),
                        ('probability', ['batch', 10])]}

def my_loss(x, target):
    return F.nll_loss(F.log_softmax(x, dim=1), target)

# Helpers
def train(log_interval, trainer, device, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Fetch data
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        # Train step
        loss, prob = trainer.train_step(data, target)

        if batch_idx == 0:
            # trainer.save_as_onnx('/home/thiagofc/mnist_onnx/pytorch_as_onnx.onnx')
            # import pdb; pdb.set_trace()
            pass
        else:
            break

        # Stats
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(trainer, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)

            # Using fetches around without eval_step to not pass 'target' as input
            trainer._train_step_info.fetches = ['probability']
            output = F.log_softmax(trainer.eval_step(data), dim=1)
            trainer._train_step_info.fetches = []

            # Stats
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='ONNX Runtime MNIST Example')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # Basic setup
    args = parser.parse_args()
    if not args.no_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.manual_seed(args.seed)
    onnxruntime.set_seed(args.seed)

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.test_batch_size, shuffle=True)

    # Modeling
    model = NeuralNet(784, 500, 10)
    model_desc = mnist_model_description()
    optim_config = optim.SGDConfig(lr=args.lr)
    opts = ORTTrainerOptions({'device': {'id': device}})

    # import onnx
    # model = onnx.load('/home/thiagofc/mnist_onnx/mnist_with_training_probability_grad.onnx')
    # my_loss=None
    trainer = ORTTrainer(model,
                         model_desc,
                         optim_config,
                         loss_fn=my_loss,
                         options=opts)

    # Train loop
    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, trainer, device, train_loader, epoch)
        # test(trainer, device, test_loader)


if __name__ == '__main__':
    main()
