import argparse

import torch
import torch.nn as nn
from pt_model import TransformerModel
from utils import get_batch, prepare_data


def train(model, data_source, device, epoch, args, bptt=35):
    total_loss = 0.0
    model.train()
    for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt)):
        data, targets = get_batch(data_source, i)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 28785), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            print(
                "epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f}".format(
                    epoch, batch, len(data_source) // bptt, cur_loss
                )
            )
            total_loss = 0


def evaluate(model, data_source, criterion, bptt=35):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output_flat = output.view(-1, 28785)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch TransformerModel example")
    parser.add_argument(
        "--batch-size", type=int, default=20, metavar="N", help="input batch size for training (default: 20)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=20, metavar="N", help="input batch size for testing (default: 20)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="how many batches to wait before logging training status (default: 200)",
    )

    # Basic setup
    args = parser.parse_args()
    if not args.no_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.manual_seed(args.seed)

    # Model
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    model = TransformerModel(28785, 200, 2, 200, 2, 0.2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Preparing data
    train_data, val_data, test_data = prepare_data(device, args.batch_size, args.test_batch_size)

    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, train_data, device, epoch, args)
        val_loss = evaluate(model, val_data, criterion)
        print("-" * 89)
        print(f"| end of epoch {epoch:3d} | valid loss {val_loss:5.2f} | ")
        print("-" * 89)

    # Evaluate
    test_loss = evaluate(model, test_data, criterion)
    print("=" * 89)
    print(f"| End of training | test loss {test_loss:5.2f}")
    print("=" * 89)
