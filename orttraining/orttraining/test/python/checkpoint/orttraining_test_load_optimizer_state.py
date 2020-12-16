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
from _test_helpers import distributed_setup, load_model_optim_state_and_eval, split_state_dict, aggregate_states, global_fp16_fp32_atol
from _test_commons import get_optim_state_from_state_dict

def verify_optimizer_state_match(device, opts, checkpoint_dir,  world_rank, use_lamb=False):
    expected_optim_state, trainer_state = load_model_optim_state_and_eval(device, opts, use_lamb)
    trainer_state = split_state_dict(trainer_state)
    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(trainer_state, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir, "distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        optimizer_config = optim.LambConfig() if use_lamb else optim.AdamConfig()
        actual_optim_state = get_optim_state_from_state_dict(optimizer_states, optimizer_config)
        assert actual_optim_state.keys() == expected_optim_state.keys()
        for param_name, a_state in actual_optim_state.items():
            for k, v in a_state.items():
                assert_allclose(v.reshape(expected_optim_state[param_name][k].shape),
                                expected_optim_state[param_name][k], 
                                err_msg=f"Optimizer state mismatch for param {param_name}, key {k}")

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))


@distributed_setup
def test_optim_load_to_distributed_zero_full_precision_adam(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/adam/'):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    verify_optimizer_state_match(device, opts, checkpoint_dir,  world_rank, use_lamb=False)


@distributed_setup
def test_optim_load_to_distributed_zero_mixed_precision_adam(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/adam/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    verify_optimizer_state_match(device, opts, checkpoint_dir,  world_rank, use_lamb=False)


@distributed_setup
def test_optim_load_to_distributed_zero_full_precision_lamb(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/lamb/'):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    verify_optimizer_state_match(device, opts, checkpoint_dir,  world_rank, use_lamb=True)

@distributed_setup
def test_optim_load_to_distributed_zero_mixed_precision_lamb(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    verify_optimizer_state_match(device, opts, checkpoint_dir,  world_rank, use_lamb=True)


function_map = {
    # load to zero configs
    'test_optim_load_to_distributed_zero_full_precision_adam': test_optim_load_to_distributed_zero_full_precision_adam,
    'test_optim_load_to_distributed_zero_mixed_precision_adam': test_optim_load_to_distributed_zero_mixed_precision_adam,
    'test_optim_load_to_distributed_zero_mixed_precision_lamb': test_optim_load_to_distributed_zero_mixed_precision_lamb,
    'test_optim_load_to_distributed_zero_full_precision_lamb': test_optim_load_to_distributed_zero_full_precision_lamb
}
parser = argparse.ArgumentParser(description='Test loading of initial optimizer state for Zero-1')
parser.add_argument('--scenario', choices=function_map.keys(), help='training scenario to test loaded states', required=True)
parser.add_argument('--checkpoint_dir', help='path to the saved states directory', required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir=args.checkpoint_dir)
