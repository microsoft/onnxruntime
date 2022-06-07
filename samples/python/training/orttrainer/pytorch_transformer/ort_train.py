import argparse

import torch
from ort_utils import my_loss, transformer_model_description_dynamic_axes
from pt_model import TransformerModel
from utils import get_batch, prepare_data

import onnxruntime


def train(trainer, data_source, device, epoch, args, bptt=35):
    total_loss = 0.0
    for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt)):
        data, targets = get_batch(data_source, i)

        loss, pred = trainer.train_step(data, targets)
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            print(
                "epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f}".format(
                    epoch, batch, len(data_source) // bptt, cur_loss
                )
            )
            total_loss = 0


def evaluate(trainer, data_source, bptt=35):
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            loss, pred = trainer.eval_step(data, targets)
            total_loss += len(data) * loss.item()
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
    onnxruntime.set_seed(args.seed)

    # Model
    optim_config = onnxruntime.training.optim.SGDConfig(lr=args.lr)
    model_desc = transformer_model_description_dynamic_axes()
    model = TransformerModel(28785, 200, 2, 200, 2, 0.2).to(device)

    # Preparing data
    train_data, val_data, test_data = prepare_data(device, args.batch_size, args.test_batch_size)
    trainer = onnxruntime.training.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss)

    # Train
    for epoch in range(1, args.epochs + 1):
        train(trainer, train_data, device, epoch, args)
        val_loss = evaluate(trainer, val_data)
        print("-" * 89)
        print("| end of epoch {:3d} | valid loss {:5.2f} | ".format(epoch, val_loss))
        print("-" * 89)

    # Evaluate
    test_loss = evaluate(trainer, test_data)
    print("=" * 89)
    print("| End of training | test loss {:5.2f}".format(test_loss))
    print("=" * 89)
