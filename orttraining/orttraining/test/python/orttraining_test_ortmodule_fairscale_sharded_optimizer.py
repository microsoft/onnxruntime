import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
import torchvision
from torchvision import datasets, transforms
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import onnxruntime
from onnxruntime.training import ORTModule
import numpy as np

# Usage : 
# pip install fairscale
# python3 orttraining_test_ortmodule_fairscale_sharded_optimizer.py --use_sharded_optimizer --use_ortmodule 


def dist_init(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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

def get_dataloader(args,rank,batch_size):

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    ) 
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
       train_dataset,
       num_replicas=args.world_size,
       rank=rank
    )
 
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)
   
    test_loader = None
    if args.test_batch_size > 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.test_batch_size, shuffle=True)

    return train_loader,test_loader

def my_loss(x, target, is_train=True):
    if is_train:
        return torch.nn.CrossEntropyLoss()(x, target)
    else:
        return torch.nn.CrossEntropyLoss(reduction='sum')(x, target)

def train_step(args, model, device, optimizer, loss_fn, train_loader, epoch):
    print('\n======== Epoch {:} / {:} with batch size {:} ========'.format(epoch+1, args.epochs, args.batch_size))
    model.train()
    # Measure how long the training epoch takes.
    t0 = time.time()
    start_time = t0

    # Reset the total loss for this epoch.
    total_loss = 0

    for iteration, (data, target) in enumerate(train_loader):
        if iteration == args.train_steps:
            break
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

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

        loss.backward()
        optimizer.step()

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
            data = data.reshape(data.shape[0], -1)
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
    accuracy = float(correct)/len(test_loader.dataset)
    print("  Accuracy: {0:.2f}".format(accuracy))
    print("  Validation took: {:.4f}s".format(epoch_time))
    return epoch_time, accuracy

def train(
    rank: int,
    args,
    world_size: int,
    epochs: int):

    # DDP init example
    dist_init(rank, world_size) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Setup
    if not args.cpu:
        torch.cuda.set_device(rank)
        torch.cuda.manual_seed(0)
    torch.manual_seed(0)  # also sets the cuda seed
    np.random.seed(0)

    # Problem statement
    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10).to(rank)

    if args.use_ortmodule:
        print("Converting to ORTModule....")
        model = ORTModule(model)

    train_dataloader, test_dataloader = get_dataloader(args,rank,args.batch_size)
    loss_fn = my_loss
    base_optimizer = torch.optim.SGD # pick any pytorch compliant optimizer here
    base_optimizer_arguments = {} # pass any optimizer specific arguments here, or directly below when instantiating OSS
    if args.use_sharded_optimizer:
        # Wrap the optimizer in its state sharding brethren
        optimizer = OSS(params=model.parameters(), optim=base_optimizer,lr = args.lr )

        # Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
        model = ShardedDDP(model, optimizer)
    else:
        device_ids = None if args.cpu else [rank]
        model = DDP(model, device_ids=device_ids, find_unused_parameters=False)  # type: ignore
        
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # Any relevant training loop, nothing specific to OSS. For example:
    model.train()
    total_training_time, total_test_time, epoch_0_training, validation_accuracy = 0, 0, 0, 0
    for epoch in range(epochs):
        total_training_time += train_step(args, model, rank, optimizer, loss_fn, train_dataloader, epoch)
        if epoch == 0:
            epoch_0_training = total_training_time
        if args.test_batch_size > 0:
            test_time, validation_accuracy = test(args, model, rank, loss_fn, test_dataloader)
            total_test_time += test_time

    print('\n======== Global stats ========')
    if args.use_ortmodule:
        estimated_export = 0
        if args.epochs > 1:
            estimated_export = epoch_0_training - (total_training_time - epoch_0_training)/(args.epochs-1)
            print("  Estimated ONNX export took:               {:.4f}s".format(estimated_export))
        else:
            print("  Estimated ONNX export took:               Estimate available when epochs > 1 only")
        print("  Accumulated training without export took: {:.4f}s".format(total_training_time - estimated_export))
    print("  Accumulated training took:                {:.4f}s".format(total_training_time))
    print("  Accumulated validation took:              {:.4f}s".format(total_test_time))


    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark the optimizer state sharding, on a typical computer vision workload"
    )
    parser.add_argument("--world_size", action="store", default=2, type=int)
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--batch_size", action="store", default=256, type=int)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--use_sharded_optimizer', action='store_true', default=False, 
                        help='use sharded optim')
    parser.add_argument('--train-steps', type=int, default=-1, metavar='N',
                        help='number of steps to train. Set -1 to run through whole dataset (default: -1)')
    parser.add_argument('--view-graphs', action='store_true', default=False,
                        help='views forward and backward graphs')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status (default: 300)')
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--use_ortmodule", action="store_true", default=False, help = "use ortmodule")
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./mnist',
                        help='Path to the mnist data directory') 
    args = parser.parse_args()

    # Supposing that WORLD_SIZE and EPOCHS are somehow defined somewhere
    mp.spawn(
        train,
        args=(
            args,
            args.world_size,
            args.epochs,
        ),
        nprocs=args.world_size,
        join=True,
    )
