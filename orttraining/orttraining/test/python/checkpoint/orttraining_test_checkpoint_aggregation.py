#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

################################################################################
# Refer to orttraining_test_checkpoint.py for an overview about Checkpoint tests
################################################################################

import os
import pickle
from numpy.testing import assert_allclose
import numpy as np
import argparse
import glob

import torch
import torch.distributed as dist

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime
from onnxruntime.training import checkpoint
from _test_helpers import distributed_setup, create_orttrainer_and_load_checkpoint, split_state_dict, aggregate_states, global_fp16_fp32_atol


def test_zero_aggregation(checkpoint_dir, loaded_state_dict, is_mixedprecision):
    # get aggregated state dict independently
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    aggregate_state_dict = agg_checkpoint.aggregate_checkpoints()

    # verify loaded state and aggregated states match:
    assert aggregate_state_dict.keys() == loaded_state_dict.keys()
    for k, v in loaded_state_dict.items():
        assert_allclose(v, aggregate_state_dict[k])

    # split state for next few checks
    loaded_state_dict = split_state_dict(loaded_state_dict)

    # verify that aggregation is done correctly
    num_states = len(glob.glob1(checkpoint_dir, "state_dict*"))

    sharded_state_rank_offset = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        rank_state_dict = split_state_dict(state['state_dict_'+str(rank)])

        if is_mixedprecision:
            for k, v in rank_state_dict['fp16_param'].items():
                # verify fp16 weights match
                assert_allclose(v, loaded_state_dict['fp16_param'][k])
                # verify rank fp16 weights match loaded fp32 correctly
                fp32_name = k.split('_fp16')[0]
                assert_allclose(v, loaded_state_dict['fp32_param'][fp32_name], atol=global_fp16_fp32_atol)

        for k, v in rank_state_dict['fp32_param'].items():
            if k in loaded_state_dict['fp32_param']:
                assert_allclose(v, loaded_state_dict['fp32_param'][k])
            else:
                assert '_view_' in k
                weight_key = k.split('_view_')[0]
                rank_offset = 0
                if weight_key in sharded_state_rank_offset:
                    rank_offset = sharded_state_rank_offset[weight_key]
                rank_len = v.numel()
                loaded_tensor = loaded_state_dict['fp32_param'][weight_key].view(-1)
                assert rank_offset + rank_len <= loaded_tensor.numel()
                assert_allclose(v, loaded_tensor[rank_offset: rank_offset + rank_len])
                # update offset
                sharded_state_rank_offset[weight_key] = rank_offset + rank_len

        for k, v in rank_state_dict['optimizer'].items():
            if k in loaded_state_dict['optimizer']:
                assert_allclose(v, loaded_state_dict['optimizer'][k])
            else:
                assert '_view_' in k
                if k.startswith('Moment_'):  # verify moment tensors
                    optim_key = k.split('_view_')[0]
                    rank_offset = 0
                    if optim_key in sharded_state_rank_offset:
                        rank_offset = sharded_state_rank_offset[optim_key]
                    rank_len = v.numel()
                    loaded_tensor = loaded_state_dict['optimizer'][optim_key].view(-1)
                    assert rank_offset + rank_len <= loaded_tensor.numel()
                    assert_allclose(v, loaded_tensor[rank_offset: rank_offset + rank_len])

                    sharded_state_rank_offset[optim_key] = rank_offset + rank_len


def test_aggregation_from_distributed_zero_full_precision_adam(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero/full_precision/adam/'):
    opts = {'device': {'id': device},
            'debug': {'deterministic_compute': True}}

    # extract state dictionaries to compare
    loaded_state_dict, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir, use_lamb=False)
    test_zero_aggregation(checkpoint_dir, loaded_state_dict, is_mixedprecision=False)


def test_aggregation_from_distributed_zero_mixed_precision_adam(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero/mixed_precision/adam/'):
    opts = {
                'device': {'id': device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug': {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    loaded_state_dict, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir, use_lamb=False)
    test_zero_aggregation(checkpoint_dir, loaded_state_dict, is_mixedprecision=True)


def test_aggregation_from_distributed_zero_full_precision_lamb(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero/full_precision/lamb/'):
    opts = {'device': {'id': device},
            'debug': {'deterministic_compute': True}}

    # extract state dictionaries to compare
    loaded_state_dict, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir, use_lamb=True)
    test_zero_aggregation(checkpoint_dir, loaded_state_dict, is_mixedprecision=False)


def test_aggregation_from_distributed_zero_mixed_precision_lamb(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero/mixed_precision/lamb/'):
    opts = {
                'device': {'id': device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug': {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    loaded_state_dict, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir, use_lamb=True)
    test_zero_aggregation(checkpoint_dir, loaded_state_dict, is_mixedprecision=True)


function_map = {
    # all config to single node config
    'test_aggregation_from_distributed_zero_full_precision_adam': test_aggregation_from_distributed_zero_full_precision_adam,
    'test_aggregation_from_distributed_zero_mixed_precision_adam': test_aggregation_from_distributed_zero_mixed_precision_adam,
    'test_aggregation_from_distributed_zero_mixed_precision_lamb': test_aggregation_from_distributed_zero_mixed_precision_lamb,
    'test_aggregation_from_distributed_zero_full_precision_lamb': test_aggregation_from_distributed_zero_full_precision_lamb
}
parser = argparse.ArgumentParser(description='Test aggregation of states for Zero-1')
parser.add_argument('--scenario', choices=function_map.keys(), help='training scenario to test saved and loaded states', required=True)
parser.add_argument('--checkpoint_dir', help='path to the saved states directory', required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir=args.checkpoint_dir)
