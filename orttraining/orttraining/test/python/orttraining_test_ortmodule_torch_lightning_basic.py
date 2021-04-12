import argparse
from multiprocessing import cpu_count

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import onnxruntime
from onnxruntime.training import ORTModule


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, lr, use_ortmodule=True):
        super().__init__()
        self.lr = lr
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )
        if use_ortmodule:
            self.encoder = ORTModule(self.encoder)
            self.decoder = ORTModule(self.decoder)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)

        z = self.encoder(x)

        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-steps', type=int, default=-1, metavar='N',
                        help='number of steps to train. Set -1 to run through whole dataset (default: -1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime training')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--data-dir', type=str, default='./mnist',
                        help='Path to the mnist data directory')

    args = parser.parse_args()

    # Common setup
    torch.manual_seed(args.seed)
    onnxruntime.set_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Data loader
    dataset = MNIST(args.data_dir, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, num_workers=cpu_count(), batch_size=args.batch_size)

    # Model architecture
    autoencoder = LitAutoEncoder(lr=args.lr, use_ortmodule=not args.pytorch_only)

    # Train loop
    kwargs = {}
    if device == 'cuda':
        kwargs.update({'gpus': 1})
    if args.train_steps > 0:
        kwargs.update({'max_steps': args.train_steps})
    if args.epochs > 0:
        kwargs.update({'max_epochs': args.epochs})
    trainer = pl.Trainer(**kwargs)
    trainer.fit(autoencoder, train_loader)


if __name__ == '__main__':
    main()
