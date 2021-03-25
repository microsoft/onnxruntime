"""Test for a simple ORTModule using the high-level DeepSpeed API.

To run on the local GPU(s):

```
$ pip install deepspeed
$ deepspeed orttraining_test_ortmodule_deepspeed_zero_stage_1.py \
    --deepspeed_config=orttraining_test_ortmodule_deepspeed_zero_stage_1_config.json
```
"""
import argparse
import logging
import os
import torch
import time
from torchvision import datasets, transforms
import torch.distributed as dist

import onnxruntime
from onnxruntime.training import ORTModule

import deepspeed

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
    print('\n======== Epoch {:} / {:} with batch size {:} ========'.format(
        epoch+1, args.epochs, model.train_batch_size()))
    model.train()
    # Measure how long the training epoch takes.
    t0 = time.time()
    start_time = t0

    # Reset the total loss for this epoch.
    total_loss = 0

    for iteration, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1).half()

        optimizer.zero_grad()
        probability = model(data)

        if args.view_graphs:
            import torchviz
            pytorch_backward_graph = torchviz.make_dot(probability, params=dict(list(model.named_parameters())))
            pytorch_backward_graph.view()

        loss = loss_fn(probability, target)
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        model.backward(loss)
        model.step()

        # Stats
        if iteration % args.log_interval == 0:
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            print('[{:5}/{:5} ({:2.0f}%)]\tLoss: {:.6f}\tExecution time: {:.4f}'.format(
                iteration * len(data), len(train_loader.dataset),
                100. * iteration / len(train_loader), loss, elapsed_time))
            start_time = curr_time

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_loader)

    epoch_time = time.time() - t0
    print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:.4f}s".format(epoch_time))
    return epoch_time


def test(args, model, device, loss_fn, test_loader):
    model.eval()

    t0 = time.time()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1).half()
            output = model(data)

            # Stats
            test_loss += loss_fn(output, target, False).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Batch size: {:}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        args.test_batch_size, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Report the final accuracy for this validation run.
    epoch_time = time.time() - t0
    print("  Accuracy: {0:.2f}".format(float(correct)/len(test_loader.dataset)))
    print("  Validation took: {:.4f}s".format(epoch_time))
    return epoch_time

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
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime training')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status (default: 300)')
    parser.add_argument('--view-graphs', action='store_true', default=False,
                        help='views forward and backward graphs')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='WARNING',
                        help='Log level (default: WARNING)')
    parser.add_argument('--data-dir', type=str, default='./mnist',
                        help='Path to the mnist data directory')

    # DeepSpeed-related settings
    parser.add_argument('--local_rank',
                        type=int,
                        required=True,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()


    # Common setup
    torch.manual_seed(args.seed)
    onnxruntime.set_seed(args.seed)

    # TODO: CUDA support is broken due to copying from PyTorch into ORT
    if not args.no_cuda and torch.cuda.is_available():
        device = "cuda:" + str(args.local_rank)
    else:
        device = "cpu"

    ## Data loader

    dist.init_process_group(backend='nccl')
    if args.local_rank == 0:
        # download only once on rank 0
        datasets.MNIST(args.data_dir, download=True)
    dist.barrier()
    train_set = datasets.MNIST(args.data_dir, train=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))]))

    test_loader = None
    if args.test_batch_size > 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.test_batch_size, shuffle=True)

    # Model architecture
    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10).to(device)
    if not args.pytorch_only:
        print('Training MNIST on ORTModule....')
        model = ORTModule(model)

        # TODO: change it to False to stop saving ONNX models
        model._save_onnx = True
        model._save_onnx_prefix = 'MNIST'

        # Set log level
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % args.log_level)
        logging.basicConfig(level=numeric_level)
    else:
        print('Training MNIST on vanilla PyTorch....')

    model, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_set)
    
    # Train loop
    total_training_time, total_test_time, epoch_0_training = 0, 0, 0
    for epoch in range(0, args.epochs):
        total_training_time += train(args, model, device, optimizer, my_loss, train_loader, epoch)
        if not args.pytorch_only and epoch == 0:
            epoch_0_training = total_training_time
        if args.test_batch_size > 0:
            total_test_time += test(args, model, device, my_loss, test_loader)

    print('\n======== Global stats ========')
    if not args.pytorch_only:
        estimated_export = 0
        if args.epochs > 1:
            estimated_export = epoch_0_training - (total_training_time - epoch_0_training)/(args.epochs-1)
            print("  Estimated ONNX export took:               {:.4f}s".format(estimated_export))
        else:
            print("  Estimated ONNX export took:               Estimate available when epochs > 1 only")
        print("  Accumulated training without export took: {:.4f}s".format(total_training_time - estimated_export))
    print("  Accumulated training took:                {:.4f}s".format(total_training_time))
    print("  Accumulated validation took:              {:.4f}s".format(total_test_time))


if __name__ == '__main__':
    main()
