import argparse
import os
from pathlib import Path

import onnx
import torch

from onnxruntime.training.ortmodule import ORTModule

torch.distributed.init_process_group(backend="nccl")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ortmodule_cache_dir", required=True, help="directory for ortmodule to cache exported model")

    args = parser.parse_args()
    return args


args = get_args()

os.environ["ORTMODULE_CACHE_DIR"] = args.ortmodule_cache_dir


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc(x))
        return x


model = Net()
model = ORTModule(model)

data = torch.randn(1, 10)
_ = model(data)

root_dir = Path(__file__).resolve().parent
cache_dir = root_dir / os.environ["ORTMODULE_CACHE_DIR"]

cached_files = sorted(os.listdir(cache_dir))
assert len(cached_files) == 2

rank = torch.distributed.get_rank()

_ = onnx.load(str(cache_dir / os.listdir(cache_dir)[rank]))

del os.environ["ORTMODULE_CACHE_DIR"]
