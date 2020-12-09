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


def test_load_from_single_node_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_single_node_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_single_node_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

def test_load_from_single_node_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

def test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_distributed_zero_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # model states
        for key, value in state_dict_pre_checkpoint['fp32_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

def test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

def test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb'):
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
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

def test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/lamb/'):
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
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp32 states are not sharded
        for key, value in state_dict_post_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=global_fp16_fp32_atol)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/lamb/'):
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
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # model states
        for key, value in state_dict_pre_checkpoint['fp32_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb'):
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
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb'):
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
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/lamb/'):
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
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = None
        with open(os.path.join(checkpoint_dir, 'state_dict_'+str(rank)+'.pkl'), 'rb') as f:
            state = pickle.load(f)
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp32 states are not sharded
        for key, value in state_dict_post_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp32_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_post_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp32_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_post_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'), "wb") as f:
        pickle.dump(state_dict_post_checkpoint, f)
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = None
            with open(os.path.join(checkpoint_dir, 'distributed_state_'+str(rank)+'.pkl'), 'rb') as f:
                rank_state_dict = pickle.load(f)

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(os.path.join(checkpoint_dir, 'distributed_state_'+str(world_rank)+'.pkl'))

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/lamb/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp32_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    """
    TODO: Uncomment this after Checkpoint redesign. Current implementation does not support it
    fp32 weights pre checkpoint are sharded. But since this is a one to one mapping (from distributed zero to distributed zero albeit
    mixed to full precision), the fp32 weights are not aggregated before copying into the new trainer
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)
    """

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/lamb/'):
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
    state_dict_post_checkpoint, model = create_orttrainer_and_load_checkpoint(device, opts, checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = None
    with open(os.path.join(checkpoint_dir, 'state_dict_'+str(world_rank)+'.pkl'), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    for key, value in state_dict_post_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=global_fp16_fp32_atol)

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

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
    'test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision
}
parser = argparse.ArgumentParser(description='Test saved states of trainers to loaded states')
parser.add_argument('--scenario', choices=function_map.keys(), help='training scenario to test saved and loaded states', required=True)
parser.add_argument('--checkpoint_dir', help='path to the saved states directory', required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir=args.checkpoint_dir)
