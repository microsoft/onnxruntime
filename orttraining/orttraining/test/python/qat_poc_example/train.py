import logging

import numpy as np
import torch
from torchvision import datasets, transforms

import onnxruntime.training.api as orttraining


def _get_dataloaders(data_dir, batch_size):
    """Preprocesses the data and returns dataloaders."""

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size), torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )


def _train_epoch(model, optimizer, train_loader):
    """Trains the model for one epoch."""

    model.train()
    cumulative_loss = 0
    for data, target in train_loader:
        train_loss = model(data.reshape(len(data), 784).numpy(), target.numpy().astype(np.int64))
        optimizer.step()
        model.lazy_reset_grad()
        cumulative_loss += train_loss

    return cumulative_loss / len(train_loader)


def _eval(model, test_loader):
    """Evaluates the model on the test set."""
    model.eval()
    cumulative_loss = 0
    for data, target in test_loader:
        test_loss = model(data.reshape(len(data), 784).numpy(), target.numpy().astype(np.int64))
        cumulative_loss += test_loss

    return cumulative_loss / len(test_loader)


def train_model(qat_train_model, qat_eval_model, qat_optimizer_model, qat_checkpoint):
    """Trains the given QAT models.

    This function uses onnxruntime training apis for training.
    It takes in the prepared (from onnxblock) QAT models and the checkpoint as input.
    """

    # Preprocess data and create dataloaders.
    batch_size = 64
    train_loader, test_loader = _get_dataloaders("data", batch_size)

    # Load the checkpoint state.
    state = orttraining.CheckpointState.load_checkpoint(qat_checkpoint)

    # Create the training module.
    model = orttraining.Module(qat_train_model, state, qat_eval_model)

    # Create the optimizer.
    optimizer = orttraining.Optimizer(qat_optimizer_model, model)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        logging.info(f"Starting epoch: {epoch+1}")
        training_loss = _train_epoch(model, optimizer, train_loader)
        eval_loss = _eval(model, test_loader)

        logging.info(f"End of epoch: {epoch+1}, training loss: {training_loss:.4f}, eval loss: {eval_loss:.4f}")
