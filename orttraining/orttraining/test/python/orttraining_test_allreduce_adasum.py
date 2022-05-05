#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import subprocess
import os
import shutil
import sys
import torch
import argparse
from onnxruntime.capi._pybind_state import set_cuda_device_id, get_mpi_context_world_rank, get_mpi_context_world_size
from onnxruntime.training import optim, orttrainer
from _test_commons import _load_pytorch_transformer_model
from onnxruntime import set_seed


def _run_adasum_tests(opts):
    # Common setup
    seed = 42
    optim_config = optim.LambConfig()
    # Setup ORTTRainer
    torch.manual_seed(seed)
    set_seed(seed)
    model, model_desc, my_loss, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(opts.device.id)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)
    # Train once to see flag going through
    data, targets = batcher_fn(train_data, 0)
    result = trainer.train_step(data, targets)
    assert result is not None


def test_single_precision_adasum_on_gpu():
    # Common setup
    world_rank = get_mpi_context_world_rank()
    world_size = get_mpi_context_world_size()
    set_cuda_device_id(world_rank)
    device = "cuda:" + str(world_rank)
    opts = orttrainer.ORTTrainerOptions(
        {
            "debug": {"deterministic_compute": True},
            "device": {
                "id": device,
            },
            "distributed": {
                "world_rank": world_rank,
                "world_size": world_size,
                "enable_adasum": True,
            },
        }
    )
    _run_adasum_tests(opts)


def test_half_precision_adasum_on_gpu():
    # Common setup
    world_rank = get_mpi_context_world_rank()
    world_size = get_mpi_context_world_size()
    set_cuda_device_id(world_rank)
    device = "cuda:" + str(world_rank)
    opts = orttrainer.ORTTrainerOptions(
        {
            "debug": {"deterministic_compute": True},
            "device": {
                "id": device,
            },
            "mixed_precision": {"enabled": True},
            "distributed": {
                "world_rank": world_rank,
                "world_size": world_size,
                "enable_adasum": True,
            },
        }
    )
    _run_adasum_tests(opts)


function_map = {
    "test_single_precision_adasum_on_gpu": test_single_precision_adasum_on_gpu,
    "test_half_precision_adasum_on_gpu": test_half_precision_adasum_on_gpu,
}
parser = argparse.ArgumentParser(description="Test adasum allreduce")
parser.add_argument(
    "--scenario", choices=function_map.keys(), help="training scenario to test adasum allreduce", required=True
)
args = parser.parse_args()
function_map[args.scenario]()
