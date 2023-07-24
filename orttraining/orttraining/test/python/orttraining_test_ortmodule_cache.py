import argparse
import os
import time
from pathlib import Path

import torch

from onnxruntime.training.ortmodule import DebugOptions, LogLevel, ORTModule

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


data = torch.randn(1, 10)

# first time seeing the model, architecture should be cached under ORTMODULE_CACHE_DIR
model_pre_cache = Net()
model_pre_cache = ORTModule(model_pre_cache, DebugOptions(log_level=LogLevel.INFO))

pre_cache_start = time.time()
_ = model_pre_cache(data)
pre_cache_duration = time.time() - pre_cache_start

root_dir = Path(__file__).resolve().parent
cache_dir = root_dir / os.environ["ORTMODULE_CACHE_DIR"]

cached_files = sorted(os.listdir(cache_dir))
assert len(cached_files) == 2

# second time seeing the model, architecture should be loaded from ORTMODULE_CACHE_DIR
model_post_cache = Net()
model_post_cache = ORTModule(model_post_cache, DebugOptions(log_level=LogLevel.INFO))

post_cache_start = time.time()
_ = model_post_cache(data)
post_cache_duration = time.time() - post_cache_start

assert post_cache_duration < pre_cache_duration

del os.environ["ORTMODULE_CACHE_DIR"]
