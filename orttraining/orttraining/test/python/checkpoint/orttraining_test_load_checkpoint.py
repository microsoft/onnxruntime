#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

################################################################################
# Refer to orttraining_test_checkpoint.py for an overview about Checkpoint tests
################################################################################

import os
import pickle
import argparse
import glob

import torch
import torch.distributed as dist

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime
from onnxruntime.training import checkpoint
from _test_helpers import distributed_setup, create_orttrainer_and_load_checkpoint, create_orttrainer_and_load_checkpoint_bart, aggregate_states, assert_all_states_close, save_ort_ckpt
from _test_commons import assert_all_states_close_ort, assert_all_states_close_pytorch

def test_load_from_single_node_full_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_single_node_mixed_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_single_node_mixed_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_single_node_full_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_data_parallelism_full_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

def test_load_from_distributed_zero_full_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)

def test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)

def test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)

def test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)

def test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir):
    # compare the expected dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(expected_state_dict, state_dict_post_checkpoint)

    # TODO: aggregate checkpoints previously saved and load it into the pytorch model for comparison, 
    # need to add support to add the bart pytorch model to unit tests instead of current onnx model
    # checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    # agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    # assert_all_states_close_pytorch(agg_state_dict, model)

def test_load_from_distributed_megatron_full_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

def test_load_from_distributed_megatron_mixed_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

def test_load_from_distributed_megatron_mixed_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

def test_load_from_distributed_megatron_full_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

def test_load_from_distributed_zero_megatron_full_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

def test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_full_precision(checkpoint_dir, device = 'cuda'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

def test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

def test_load_from_distributed_zero_megatron_full_precision_into_single_node_mixed_precision(checkpoint_dir, device = 'cuda'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # compare all states
    assert_all_states_close(checkpoint_dir, 'state_dict', state_dict_post_checkpoint, model)

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # manually aggregate states from the previously saved state dictionary in a pickle file
    aggregated_state_dict = aggregate_states(checkpoint_dir)

    # compare the manually aggregated state dictionary with the aggregated state dictionary from the ORTTrainer
    assert_all_states_close_ort(aggregated_state_dict, state_dict_post_checkpoint, reshape_states=True)

    # aggregate checkpoints previously saved and load it into the pytorch model for comparison
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
    agg_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=True)
    assert_all_states_close_pytorch(agg_state_dict, model)


@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)
    test_load_from_megatron_to_non_model_parallel_node(state_dict_post_checkpoint, expected_state_dict, model, checkpoint_dir)

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the single node state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the single node state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the single node state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the single node state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the data parallel state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the data parallel state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the data parallel state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel node trainer to the state dictionary from a zero run:
    # - Save the state dictionaries for each rank for the zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate states from the previously saved state dictionary in a pickle file
        aggregated_state_dict = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the manually aggregated state dictionary with the data parallel state dictionary that was previously saved in a pickle file
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]

    # compare all states for each rank independently
    assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint)

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # To compare state dictionary between two distributed zero node trainers (with different mixed precision parameter):
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because the full precision model weights in a mixed precision trainer are sharded but the same weights
    # are not shareded in a full precision trainer run.
    # Therefore, the state dictionary model weights are different between a mixed precision trainer run and full precision trainer run.
    # Which is why the need to compare the aggregated state dictionary (which returns a single state dictionary with all model and optimizer states)
    # as opposed to comparing the state dictionary for each rank independently.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
        aggregated_state_dict1 = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)
        # aggregate checkpoints from the previous mixed precision zero trainer
        aggregated_state_dict2 = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(aggregated_state_dict2, aggregated_state_dict1, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]

    # compare all states for each rank independently
    assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint)

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    # To compare state dictionary between two distributed zero node trainers (with different mixed precision parameter):
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because the full precision model weights in a mixed precision trainer are sharded but the same weights
    # are not shareded in a full precision trainer run.
    # Therefore, the state dictionary model weights are different between a mixed precision trainer run and full precision trainer run.
    # Which is why the need to compare the aggregated state dictionary (which returns a single state dictionary with all model and optimizer states)
    # as opposed to comparing the state dictionary for each rank independently.

    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.ortcp'))
        aggregated_state_dict1 = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)
        # aggregate checkpoints from the previous mixed precision zero trainer
        aggregated_state_dict2 = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(aggregated_state_dict2, aggregated_state_dict1, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the current zero run in a pickle file (distributed_state_world_rank.pkl)
    # - On rank 0, manually load each state dictionary (distributed_state_world_rank.pkl) and aggregate all of them into a single state dictionary.
    # - Aggregate the checkpoint files from the previous zero run checkpoint files into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the aggregated state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        aggregated_state_dict_loaded = aggregate_states(checkpoint_dir, filename_prefix='distributed_state', state_dict_key_name=None)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))


###########################################################################################################################################
# LOAD TO MEGATRON
###########################################################################################################################################

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run..
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a megatron run:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ , _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]

    # compare all states for each rank independently
    assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint)

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed megatron node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ , _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]

    # compare all states for each rank independently
    assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint)

@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed megatron node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero+megatron node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero+megatron node trainers :
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero+megatron node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : world_size
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed megatron and distributed zero+megatron node trainers:
    # - Save the state dictionaries for each rank for the megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

###########################################################################################################################################
# LOAD TO ZERO+MEGATRON
###########################################################################################################################################

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a single node trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the single node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected single node state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run..
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel node run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _, _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # To compare state dictionary from a data parallel trainer to the state dictionary from a zero+megatron run:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary against the state dictionary previously saved from the data parallel run.
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the manually aggregated state dictionary with the expected data parallel state dictionary
        assert_all_states_close_ort(aggregated_state_dict, state_dict_pre_checkpoint)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current full precision zero trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed megatron node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed megatron node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed megatron node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed megatron node trainers:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ , _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]

    # compare all states for each rank independently
    assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint)

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_full_precision(world_rank, world_size, device, checkpoint_dir):
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero+megatron node trainers with different precisions:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

@distributed_setup
def test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, _ , _ = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]

    # compare all states for each rank independently
    assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint)

@distributed_setup
def test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_mixed_precision(world_rank, world_size, device, checkpoint_dir):
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
                    'horizontal_parallel_size' : int(world_size/2),
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }

    # extract state dictionaries to compare
    state_dict_post_checkpoint, expected_state_dict, model = create_orttrainer_and_load_checkpoint_bart(device, opts, checkpoint_dir)

    # To compare state dictionary between distributed zero+megatron and distributed zero+megatron node trainers with different precisions:
    # - Save the state dictionaries for each rank for the zero+megatron run (distributed_state_world_rank.ort.pt)
    # - On rank 0, manually load each state dictionary and aggregate all of them into a single state dictionary.
    # - Compare the aggregated state dictionary from the current run against the expected state dictionary from the previous run.
    # This is needed because of difference in model-parallel config causing different sharding of model and optimizer states
    filename = 'distributed_state_' + str(world_rank) + '.ort.pt'
    filepath = os.path.join(checkpoint_dir, filename)
    save_ort_ckpt(state_dict_post_checkpoint, filepath)
    dist.barrier()

    if world_rank == 0:
        # manually aggregate the states for the current trainer
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'distributed_state*.ort.pt'))
        aggregated_state_dict_loaded = checkpoint.aggregate_checkpoints(checkpoint_files, pytorch_format=False)

        # compare the two state dictionaries
        assert_all_states_close_ort(expected_state_dict, aggregated_state_dict_loaded, reshape_states=True)

    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.ort.pt'))

function_map = {
    # all config to single node config
    'test_load_from_single_node_full_precision_into_single_node_full_precision': test_load_from_single_node_full_precision_into_single_node_full_precision,
    'test_load_from_single_node_mixed_precision_into_single_node_mixed_precision': test_load_from_single_node_mixed_precision_into_single_node_mixed_precision,
    'test_load_from_single_node_mixed_precision_into_single_node_full_precision': test_load_from_single_node_mixed_precision_into_single_node_full_precision,
    'test_load_from_single_node_full_precision_into_single_node_mixed_precision': test_load_from_single_node_full_precision_into_single_node_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_single_node_full_precision': test_load_from_data_parallelism_full_precision_into_single_node_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision': test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision': test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_single_node_full_precision': test_load_from_distributed_zero_full_precision_into_single_node_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision': test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_single_node_full_precision': test_load_from_distributed_megatron_full_precision_into_single_node_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_single_node_full_precision': test_load_from_distributed_megatron_mixed_precision_into_single_node_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_single_node_mixed_precision': test_load_from_distributed_megatron_mixed_precision_into_single_node_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_single_node_mixed_precision': test_load_from_distributed_megatron_full_precision_into_single_node_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_single_node_full_precision': test_load_from_distributed_zero_megatron_full_precision_into_single_node_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_full_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_megatron_full_precision_into_single_node_mixed_precision,

    # all config to data parallel node config
    'test_load_from_single_node_full_precision_into_data_parallelism_full_precision': test_load_from_single_node_full_precision_into_data_parallelism_full_precision,
    'test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision': test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision,
    'test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision,
    'test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision': test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision': test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision': test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision': test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_data_parallelism_full_precision': test_load_from_distributed_megatron_full_precision_into_data_parallelism_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_data_parallelism_full_precision': test_load_from_distributed_megatron_mixed_precision_into_data_parallelism_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_megatron_mixed_precision_into_data_parallelism_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_megatron_full_precision_into_data_parallelism_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_megatron_full_precision_into_data_parallelism_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_data_parallelism_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_data_parallelism_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_megatron_full_precision_into_data_parallelism_mixed_precision,

    # all config to distributed zero node config
    'test_load_from_single_node_full_precision_into_distributed_zero_full_precision': test_load_from_single_node_full_precision_into_distributed_zero_full_precision,
    'test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision,
    'test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision,
    'test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision': test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_distributed_zero_full_precision': test_load_from_distributed_megatron_full_precision_into_distributed_zero_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_full_precision': test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_megatron_full_precision_into_distributed_zero_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_mixed_precision,

    # all config to distributed megatron node config
    'test_load_from_single_node_full_precision_into_distributed_megatron_full_precision': test_load_from_single_node_full_precision_into_distributed_megatron_full_precision,
    'test_load_from_single_node_mixed_precision_into_distributed_megatron_full_precision': test_load_from_single_node_mixed_precision_into_distributed_megatron_full_precision,
    'test_load_from_single_node_mixed_precision_into_distributed_megatron_mixed_precision': test_load_from_single_node_mixed_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_single_node_full_precision_into_distributed_megatron_mixed_precision': test_load_from_single_node_full_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_distributed_megatron_full_precision': test_load_from_data_parallelism_full_precision_into_distributed_megatron_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_distributed_megatron_full_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_megatron_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_distributed_megatron_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_distributed_megatron_mixed_precision': test_load_from_data_parallelism_full_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_distributed_megatron_full_precision': test_load_from_distributed_zero_full_precision_into_distributed_megatron_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_full_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_distributed_megatron_mixed_precision': test_load_from_distributed_zero_full_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_distributed_megatron_full_precision': test_load_from_distributed_megatron_full_precision_into_distributed_megatron_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_full_precision': test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_mixed_precision': test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_distributed_megatron_mixed_precision': test_load_from_distributed_megatron_full_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_full_precision': test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_full_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_mixed_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_mixed_precision': test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_mixed_precision,
    
    # all config to distributed zero + megatron node config
    'test_load_from_single_node_full_precision_into_distributed_zero_megatron_full_precision': test_load_from_single_node_full_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_full_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_single_node_full_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_single_node_full_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_distributed_zero_megatron_full_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_megatron_full_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_data_parallelism_full_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_full_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_full_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_full_precision': test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_full_precision': test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_full_precision': test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_full_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_full_precision,
    'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision,
    'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_mixed_precision': test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_mixed_precision
}
parser = argparse.ArgumentParser(description='Test saved states of trainers to loaded states')
parser.add_argument('--scenario', choices=function_map.keys(), help='training scenario to test saved and loaded states', required=True)
parser.add_argument('--checkpoint_dir', help='path to the saved states directory', required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir=args.checkpoint_dir)
