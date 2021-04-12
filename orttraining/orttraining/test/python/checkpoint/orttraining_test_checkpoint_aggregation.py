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
from _test_helpers import distributed_setup, create_orttrainer_and_load_checkpoint, create_orttrainer_and_load_checkpoint_bart, aggregate_states
from _test_commons import assert_all_states_close_ort


def test_zero_aggregation(checkpoint_dir, loaded_state_dict, is_mixedprecision):
    # get aggregated state dict independently
    aggregate_state_dict_from_checkpoint = \
        checkpoint.aggregate_checkpoints(glob.glob(os.path.join(checkpoint_dir, "checkpoint*.ortcp")), pytorch_format=False)

    # verify loaded state and aggregated states match:
    assert_all_states_close_ort(loaded_state_dict, aggregate_state_dict_from_checkpoint)

    # manually aggregate the states from the previously saved pickle file
    aggregate_state_dict_from_test = aggregate_states(checkpoint_dir)

    # compare state dictionaries between the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregate_state_dict_from_test, aggregate_state_dict_from_checkpoint, reshape_states=True)


def test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision):
    # get aggregated state dict independently
    aggregate_state_dict_from_checkpoint = \
        checkpoint.aggregate_checkpoints(glob.glob(os.path.join(checkpoint_dir, "checkpoint*.ortcp")), pytorch_format=False)

    # verify loaded state and aggregated states match:
    assert_all_states_close_ort(loaded_state_dict, aggregate_state_dict_from_checkpoint)

    #compare with expected state dict
    assert_all_states_close_ort(expected_state_dict, loaded_state_dict)

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


def test_aggregation_from_distributed_megatron_full_precision_adam(device='cuda', checkpoint_dir='checkpoint_dir/distributed_megatron/full_precision/adam/'):
    opts = {'device': {'id': device},
            'debug': {'deterministic_compute': True}}

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=False)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=False)


def test_aggregation_from_distributed_megatron_mixed_precision_adam(device='cuda', checkpoint_dir='checkpoint_dir/distributed_megatron/mixed_precision/adam/'):
    opts = {
                'device': {'id': device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug': {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=False)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=True)


def test_aggregation_from_distributed_megatron_full_precision_lamb(device='cuda', checkpoint_dir='checkpoint_dir/distributed_megatron/full_precision/lamb/'):
    opts = {'device': {'id': device},
            'debug': {'deterministic_compute': True}}

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=True)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=False)


def test_aggregation_from_distributed_megatron_mixed_precision_lamb(device='cuda', checkpoint_dir='checkpoint_dir/distributed_megatron/mixed_precision/lamb/'):
    opts = {
                'device': {'id': device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug': {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=True)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=True)

def test_aggregation_from_distributed_zero_megatron_full_precision_adam(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero_megatron/full_precision/adam/'):
    opts = {'device': {'id': device},
            'debug': {'deterministic_compute': True}}

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=False)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=False)


def test_aggregation_from_distributed_zero_megatron_mixed_precision_adam(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero_megatron/mixed_precision/adam/'):
    opts = {
                'device': {'id': device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug': {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=False)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=True)


def test_aggregation_from_distributed_zero_megatron_full_precision_lamb(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero_megatron/full_precision/lamb/'):
    opts = {'device': {'id': device},
            'debug': {'deterministic_compute': True}}

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=True)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=False)


def test_aggregation_from_distributed_zero_megatron_mixed_precision_lamb(device='cuda', checkpoint_dir='checkpoint_dir/distributed_zero_megatron/mixed_precision/lamb/'):
    opts = {
                'device': {'id': device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug': {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    loaded_state_dict, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir, use_lamb=True)
    test_megatron_aggregation(checkpoint_dir, loaded_state_dict, expected_state_dict, is_mixedprecision=True)


function_map = {
    # all config to single node config
    'test_aggregation_from_distributed_zero_full_precision_adam': test_aggregation_from_distributed_zero_full_precision_adam,
    'test_aggregation_from_distributed_zero_mixed_precision_adam': test_aggregation_from_distributed_zero_mixed_precision_adam,
    'test_aggregation_from_distributed_zero_mixed_precision_lamb': test_aggregation_from_distributed_zero_mixed_precision_lamb,
    'test_aggregation_from_distributed_zero_full_precision_lamb': test_aggregation_from_distributed_zero_full_precision_lamb,
    'test_aggregation_from_distributed_megatron_full_precision_adam': test_aggregation_from_distributed_megatron_full_precision_adam,
    'test_aggregation_from_distributed_megatron_mixed_precision_adam': test_aggregation_from_distributed_megatron_mixed_precision_adam,
    'test_aggregation_from_distributed_megatron_mixed_precision_lamb': test_aggregation_from_distributed_megatron_mixed_precision_lamb,
    'test_aggregation_from_distributed_megatron_full_precision_lamb': test_aggregation_from_distributed_megatron_full_precision_lamb,
    'test_aggregation_from_distributed_zero_megatron_full_precision_adam': test_aggregation_from_distributed_zero_megatron_full_precision_adam,
    'test_aggregation_from_distributed_zero_megatron_mixed_precision_adam': test_aggregation_from_distributed_zero_megatron_mixed_precision_adam,
    'test_aggregation_from_distributed_zero_megatron_mixed_precision_lamb': test_aggregation_from_distributed_zero_megatron_mixed_precision_lamb,
    'test_aggregation_from_distributed_zero_megatron_full_precision_lamb': test_aggregation_from_distributed_zero_megatron_full_precision_lamb
}
parser = argparse.ArgumentParser(description='Test aggregation of states for Zero-1')
parser.add_argument('--scenario', choices=function_map.keys(), help='training scenario to test saved and loaded states', required=True)
parser.add_argument('--checkpoint_dir', help='path to the saved states directory', required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir=args.checkpoint_dir)
