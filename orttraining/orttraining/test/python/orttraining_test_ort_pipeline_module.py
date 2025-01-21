import argparse

import deepspeed
import torch
from deepspeed.pipe import LayerSpec
from torch import nn, utils

from onnxruntime.training.ortmodule.experimental.pipe import ORTPipelineModule

# This script demonstrates how to set up a pipeline parallel training session
# using DeepSpeed's ORTPipelineModule for a simple neural network model.


# USAGE:
# pip install deepspeed
# deepspeed orttraining_test_ort_pipeline_module.py --deepspeed_config=orttraining_test_ortmodule_deepspeed_pipeline_parallel_config.json --pipeline-parallel-size 2 --steps=100
def get_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from distributed launcher")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps to run")
    parser.add_argument("--pipeline-parallel-size", type=int, default=2, help="Number of pipeline stages")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--layer_spec", type=bool, default=False, help="Use LayerSpec for layer specification")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


class SampleData(utils.data.Dataset):
    """Custom dataset to facilitate loading and batching of data."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SimpleNetPipeInput(nn.Module):
    """First stage of the pipeline, responsible for initial processing."""

    def __init__(self, config: dict[str, int]):
        super().__init__()
        self.linear = nn.Linear(config["input_size"], config["hidden_size"])
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x


class SimpleNetPipeBlock(nn.Module):
    """Intermediate stage of the pipeline, can be duplicated to deepen the network."""

    def __init__(self, config: dict[str, int]):
        super().__init__()
        self.linear = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x


class SimpleNetPipeOutput(nn.Module):
    """Final stage of the pipeline, producing the output."""

    def __init__(self, config: dict[str, int]):
        super().__init__()
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


def build_model(config: dict[str, int], n: int, layer_spec: bool) -> nn.Module:
    """Constructs and returns the model either using LayerSpec or nn.Sequential."""
    if layer_spec:
        print("Wrapping layers with LayerSpec")
        model = (
            [LayerSpec(SimpleNetPipeInput, config)]
            + [LayerSpec(SimpleNetPipeBlock, config) for _ in range(n)]
            + [LayerSpec(SimpleNetPipeOutput, config)]
        )
    else:
        print("Wrapping layers with nn.Sequential")
        model = nn.Sequential(
            SimpleNetPipeInput(config),
            SimpleNetPipeBlock(config),
            SimpleNetPipeBlock(config),
            SimpleNetPipeOutput(config),
        )
    return model


args = get_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
deepspeed.init_distributed(dist_backend=args.backend)
torch.manual_seed(args.seed)

model = build_model({"input_size": 4, "hidden_size": 8, "output_size": 3}, n=10, layer_spec=args.layer_spec)

model = ORTPipelineModule(
    layers=model,
    loss_fn=torch.nn.CrossEntropyLoss(),
    num_stages=args.pipeline_parallel_size,
    partition_method="uniform",
    activation_checkpoint_interval=0,
)

# Setup input data
x = torch.rand((10, 4))
y = torch.randint(0, 3, (10,))
dataset = SampleData(x, y)

print("Initialize deepspeed")
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters(), training_data=dataset
)

for step in range(args.steps):
    loss = model_engine.train_batch()
    if step % 10 == 0:
        print(f"step = {step}, loss = {loss}")
