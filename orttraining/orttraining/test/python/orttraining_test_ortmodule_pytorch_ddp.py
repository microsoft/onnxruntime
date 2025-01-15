# This test script is a modified version of Pytorch's tutorial.
# For details, see https://pytorch.org/tutorials/intermediate/ddp_tutorial.html.
import argparse
import os
import sys  # noqa: F401
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

import onnxruntime  # noqa: F401
from onnxruntime.training.ortmodule import ORTModule


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size, use_ort_module):
    torch.manual_seed(0)
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    if use_ort_module:
        model = ORTModule(model)
        print(f"  Rank {rank} uses ORTModule.")
    else:
        print(f"  Rank {rank} uses Pytorch's nn.Module.")

    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.Adagrad(ddp_model.parameters(), lr=0.01)

    x = torch.randn(20, 10).to(rank)
    y = torch.randn(20, 5).to(rank)

    loss_history = []

    for i in range(5):
        optimizer.zero_grad()
        p = ddp_model(x)
        loss = loss_fn(p, y)
        with torch.no_grad():
            print(f"  Rank {rank} at iteration {i} has loss {loss}.")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_history.append(torch.unsqueeze(loss, 0))

    loss_history = torch.cat(loss_history).cpu()
    expected_loss_history = torch.FloatTensor(
        [1.4909229278564453, 1.432194471359253, 1.39592707157135, 1.367714762687683, 1.3445055484771729]
    )

    assert torch.allclose(expected_loss_history, loss_history)

    cleanup()


def demo_checkpoint(rank, world_size, use_ort_module):
    torch.manual_seed(rank)
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    if use_ort_module:
        print(f"  Rank {rank} uses ORTModule.")
        model = ToyModel().to(rank)
        model = ORTModule(model)
    else:
        print(f"  Rank {rank} uses Pytorch's nn.Module.")
        model = ToyModel().to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), "model.checkpoint")  # noqa: N806
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {"cuda:0": f"cuda:{rank}"}
    ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Rank {rank} sees loss {loss}")

    if rank == 0:
        assert torch.allclose(loss.cpu(), torch.FloatTensor([1.4909229278564453]))
    elif rank == 1:
        assert torch.allclose(loss.cpu(), torch.FloatTensor([1.0177688598632812]))
    elif rank == 2:
        assert torch.allclose(loss.cpu(), torch.FloatTensor([1.290669322013855]))
    elif rank == 3:
        assert torch.allclose(loss.cpu(), torch.FloatTensor([0.825118362903595]))
    else:
        raise AssertionError()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


"""
CustomEmbedding is adapted from
https://github.com/huggingface/transformers/blob/312b104ff65514736c0475814fec19e47425b0b5/src/transformers/models/distilbert/modeling_distilbert.py#L91.
"""


class CustomEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 511
        dim = 10
        pad_token_id = 0
        max_position_embeddings = 16
        self.word_embeddings = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        seq_length = input_embeds.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


"""
This module calls `CustomEmbeddings`, which will generate a series of nodes (Shape->Gather->Unsqueeze...) whos
allocate output tensors on CPU memory. Then this test can converge failures when CUDA EP enabled, CPU device
allocation and usage are correct.
"""


class AnotherLayerOfToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CustomEmbeddings()
        self.t = ToyModel()

    def forward(self, x):
        embed_val = self.embedding(x)
        return self.t(embed_val)


"""
`Mixed device allocation` here means ORT backend allocates output tensors on CPU for some nodes and
on CUDA for other nodes. This test could help catch regression when ORT allocation planner logic got changed with bugs.
"""


def demo_mixed_device_allocation_training(rank, world_size, use_ort_module):
    torch.manual_seed(0)
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    device = "cuda:" + str(rank)

    # create a model and move it to GPU with id rank
    model = AnotherLayerOfToyModel().to(device)
    if use_ort_module:
        model = ORTModule(model)
        print(f"  Rank {rank} uses ORTModule.")
    else:
        print(f"  Rank {rank} uses Pytorch's nn.Module.")

    ddp_model = DDP(model, device_ids=[device])

    loss_fn = nn.MSELoss()
    optimizer = optim.Adagrad(ddp_model.parameters(), lr=0.01)

    batch = 2
    max_seq_length = 16
    x = torch.randint(1, 511, (batch, max_seq_length)).to(device)
    y = torch.randn(batch, max_seq_length, 5).to(device)

    loss_history = []

    for i in range(5):
        optimizer.zero_grad()
        p = ddp_model(x)
        loss = loss_fn(p, y)
        with torch.no_grad():
            print(f"  Rank {rank} at iteration {i} has loss {loss}.")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_history.append(torch.unsqueeze(loss, 0))

    loss_history = torch.cat(loss_history).cpu()
    expected_loss_history = torch.FloatTensor([1.1589857340, 1.0975260735, 1.0628030300, 1.0386666059, 1.0196533203])

    assert torch.allclose(expected_loss_history, loss_history)

    cleanup()


def run_demo(demo_fn, world_size, use_ort_module):
    mp.spawn(demo_fn, args=(world_size, use_ort_module), nprocs=world_size, join=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ort_module", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(demo_basic, 4, args.use_ort_module)
    run_demo(demo_checkpoint, 4, args.use_ort_module)
    run_demo(demo_mixed_device_allocation_training, 4, args.use_ort_module)
