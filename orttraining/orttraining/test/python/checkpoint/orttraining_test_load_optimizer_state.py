#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

################################################################################
# Refer to orttraining_test_checkpoint.py for an overview about Checkpoint tests
################################################################################

import os
import pickle
from numpy.testing import assert_allclose
import argparse
import glob

import torch
import torch.distributed as dist

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime
from onnxruntime.training import checkpoint, optim
from _test_helpers import distributed_setup, load_model_optim_state_and_eval, aggregate_states
from _test_commons import assert_all_states_close_ort


def verify_optimizer_state_match(device, opts, checkpoint_dir, world_rank, use_lamb=False):
    expected_optim_state, trainer_optim_state = load_model_optim_state_and_eval(device, opts, use_lamb)

    # verify optimizer states are matching by:
    # - Saving the state dictionaries for each rank in the zero run in a pickle file.
    # - Loading them one by one and aggregating them into a single state dictionary
    # - Comparing this aggregated state dictionary with the full dummy optimizer dictionary (expected_optim_state)
    # created by load_model_optim_state_and_eval

    with open(os.path.join(checkpoint_dir, "distributed_state_" + str(world_rank) + ".pkl"), "wb") as f:
        pickle.dump(trainer_optim_state, f)
    dist.barrier()

    if world_rank == 0:
        # aggregate states and compare
        aggregated_state_dict = aggregate_states(
            checkpoint_dir, filename_prefix="distributed_state", state_dict_key_name=None
        )

        # compare all states
        assert_all_states_close_ort(aggregated_state_dict, expected_optim_state, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, "distributed_state_" + str(world_rank) + ".pkl"))


@distributed_setup
def test_optim_load_to_distributed_zero_full_precision_adam(
    world_rank, world_size, device, checkpoint_dir="checkpoint_dir/distributed_zero/full_precision/adam/"
):
    opts = {
        "device": {"id": device},
        "distributed": {
            "world_rank": world_rank,
            "world_size": world_size,
            "allreduce_post_accumulation": True,
            "deepspeed_zero_optimization": {"stage": 1},
        },
        "debug": {"deterministic_compute": True},
    }
    verify_optimizer_state_match(device, opts, checkpoint_dir, world_rank, use_lamb=False)


@distributed_setup
def test_optim_load_to_distributed_zero_mixed_precision_adam(
    world_rank, world_size, device, checkpoint_dir="checkpoint_dir/distributed_zero/mixed_precision/adam/"
):
    opts = {
        "device": {"id": device},
        "mixed_precision": {"enabled": True},
        "distributed": {
            "world_rank": world_rank,
            "world_size": world_size,
            "allreduce_post_accumulation": True,
            "deepspeed_zero_optimization": {"stage": 1},
        },
        "debug": {"deterministic_compute": True},
    }
    verify_optimizer_state_match(device, opts, checkpoint_dir, world_rank, use_lamb=False)


@distributed_setup
def test_optim_load_to_distributed_zero_full_precision_lamb(
    world_rank, world_size, device, checkpoint_dir="checkpoint_dir/distributed_zero/full_precision/lamb/"
):
    opts = {
        "device": {"id": device},
        "distributed": {
            "world_rank": world_rank,
            "world_size": world_size,
            "allreduce_post_accumulation": True,
            "deepspeed_zero_optimization": {"stage": 1},
        },
        "debug": {"deterministic_compute": True},
    }
    verify_optimizer_state_match(device, opts, checkpoint_dir, world_rank, use_lamb=True)


@distributed_setup
def test_optim_load_to_distributed_zero_mixed_precision_lamb(
    world_rank, world_size, device, checkpoint_dir="checkpoint_dir/distributed_zero/mixed_precision/lamb/"
):
    opts = {
        "device": {"id": device},
        "mixed_precision": {"enabled": True},
        "distributed": {
            "world_rank": world_rank,
            "world_size": world_size,
            "allreduce_post_accumulation": True,
            "deepspeed_zero_optimization": {"stage": 1},
        },
        "debug": {"deterministic_compute": True},
    }
    verify_optimizer_state_match(device, opts, checkpoint_dir, world_rank, use_lamb=True)


function_map = {
    # load to zero configs
    "test_optim_load_to_distributed_zero_full_precision_adam": test_optim_load_to_distributed_zero_full_precision_adam,
    "test_optim_load_to_distributed_zero_mixed_precision_adam": test_optim_load_to_distributed_zero_mixed_precision_adam,
    "test_optim_load_to_distributed_zero_mixed_precision_lamb": test_optim_load_to_distributed_zero_mixed_precision_lamb,
    "test_optim_load_to_distributed_zero_full_precision_lamb": test_optim_load_to_distributed_zero_full_precision_lamb,
}
parser = argparse.ArgumentParser(description="Test loading of initial optimizer state for Zero-1")
parser.add_argument(
    "--scenario", choices=function_map.keys(), help="training scenario to test loaded states", required=True
)
parser.add_argument("--checkpoint_dir", help="path to the saved states directory", required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir=args.checkpoint_dir)
