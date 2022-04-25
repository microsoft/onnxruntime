#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

################################################################################
# Refer to orttraining_test_checkpoint.py for an overview about Checkpoint tests
################################################################################

import os
import argparse

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _test_helpers import (
    _train,
    distributed_setup,
    create_initialized_orttrainer,
    split_state_dict,
    global_fp16_fp32_atol,
    verify_model_state,
    verify_opt_state,
    verify_part_info,
)


def test_single_node_full_precision_lamb(device="cuda", checkpoint_dir=""):
    opts_dict = {"device": {"id": device}, "debug": {"deterministic_compute": True}}
    is_mixedprecision = False
    is_zero_run = False

    trainer = create_initialized_orttrainer(device, opts_dict, True)

    expected_state_dict = trainer._training_session.get_state()
    expected_state_dict = split_state_dict(expected_state_dict)

    verify_model_state(trainer, expected_state_dict, is_mixedprecision)

    verify_opt_state(trainer, expected_state_dict)

    verify_part_info(trainer, expected_state_dict, is_mixedprecision, is_zero_run)


@distributed_setup
def test_distributed_zero_full_precision_lamb(world_rank, world_size, device, checkpoint_dir):
    is_mixedprecision = False
    is_zero_run = True
    opts_dict = {
        "device": {"id": device},
        "mixed_precision": {"enabled": is_mixedprecision},
        "distributed": {
            "world_rank": world_rank,
            "world_size": world_size,
            "allreduce_post_accumulation": True,
            "deepspeed_zero_optimization": {"stage": 1},
        },
        "debug": {"deterministic_compute": True},
    }

    trainer = create_initialized_orttrainer(device, opts_dict, True)

    expected_state_dict = trainer._training_session.get_state()
    expected_state_dict = split_state_dict(expected_state_dict)

    verify_model_state(trainer, expected_state_dict, is_mixedprecision)

    verify_opt_state(trainer, expected_state_dict)

    verify_part_info(trainer, expected_state_dict, is_mixedprecision, is_zero_run)


@distributed_setup
def test_distributed_zero_mixed_precision_lamb(world_rank, world_size, device, checkpoint_dir):
    is_mixedprecision = True
    is_zero_run = True
    opts_dict = {
        "device": {"id": device},
        "mixed_precision": {"enabled": is_mixedprecision},
        "distributed": {
            "world_rank": world_rank,
            "world_size": world_size,
            "allreduce_post_accumulation": True,
            "deepspeed_zero_optimization": {"stage": 1},
        },
        "debug": {"deterministic_compute": True},
    }

    trainer = create_initialized_orttrainer(device, opts_dict, True)

    expected_state_dict = trainer._training_session.get_state()
    expected_state_dict = split_state_dict(expected_state_dict)

    verify_model_state(trainer, expected_state_dict, is_mixedprecision)

    verify_opt_state(trainer, expected_state_dict)

    verify_part_info(trainer, expected_state_dict, is_mixedprecision, is_zero_run)


# To run single node test locally, from build directory
# python3 checkpoint/orttraining_test_backend_api.py
# test_single_node_full_precision_lamb()

# To run distributed test locally, from build directory
# mpirun -n 4 -x NCCL_DEBUG=INFO python3 checkpoint/orttraining_test_backend_api.py
# test_distributed_zero_full_precision_lamb(checkpoint_dir='')
# test_distributed_zero_mixed_precision_lamb(checkpoint_dir='')

function_map = {
    "test_single_node_full_precision_lamb": test_single_node_full_precision_lamb,
    "test_distributed_zero_full_precision_lamb": test_distributed_zero_full_precision_lamb,
    "test_distributed_zero_mixed_precision_lamb": test_distributed_zero_mixed_precision_lamb,
}
parser = argparse.ArgumentParser(description="Test saved states of trainers to loaded states")
parser.add_argument(
    "--scenario", choices=function_map.keys(), help="training scenario to test saved and loaded states", required=True
)
parser.add_argument("--checkpoint_dir", help="path to the saved states directory", required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir=args.checkpoint_dir)
