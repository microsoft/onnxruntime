import os
import pytest
import pickle
import argparse
from itertools import islice
import torch
import torch.distributed as dist
from onnxruntime import set_seed
from onnxruntime.training import amp, checkpoint, optim, orttrainer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _test_helpers import _train, distributed_setup, generate_model_optimizer_from_training_instance, create_initialized_orttrainer 
from numpy.testing import assert_allclose

def verify_model_state(trainer, init_model_state):
    actual_model_state = trainer._training_session.get_model_state(include_mixed_precision_weights=False)
    for fp_or_mp_key, weights in actual_model_state.items():
        for weight_name, tensor in weights.items():
            expected_tensor = init_model_state[fp_or_mp_key][weight_name]
            assert_allclose(tensor, expected_tensor, 1e-3, 1e-3)

def verify_opt_state(trainer, init_opt_state):
    actual_opt_state = trainer._training_session.get_optimizer_state()
    #print(actual_opt_state)
    for weight_name, weights in actual_opt_state.items():
        for opt_prefix, tensor in weights.items():
            expected_tensor = init_opt_state[weight_name][opt_prefix]
            if tensor.dtype == "int64":
                assert expected_tensor[0] == tensor[0], "step should increment by 1"
            else: 
                assert_allclose(tensor, expected_tensor, 1e-1, 1e-2)

def verify_part_info(trainer, is_zero_run):
    part_info = trainer._training_session.get_partition_info_map()
    for weight_name, weight_info in part_info.items():
        for info, value in weight_info.items():
            assert isinstance(value, list), "get_partition_info_map should return list"
            assert isinstance(value[0], int), "get_partition_info_map should return list of int"
            if is_zero_run:
                if info == "megatron_row_partition":
                    assert value[0] == 0, "megatron_row_partition is 0 (false) if megatron optimization is not on"
                if info == "original_dimension":
                    assert len(value) > 0, "original_dimension should not be empty if zero run"

def test_backend_api(device = 'cuda'):
    opts_dict = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # generate model and optimizer value from a training instance
    init_model_state, init_opt_state = generate_model_optimizer_from_training_instance(device, opts_dict)

    trainer = create_initialized_orttrainer(device, opts_dict, True)

    verify_model_state(trainer, init_model_state)
    
    verify_opt_state(trainer, init_opt_state)

    verify_part_info(trainer, False)

@distributed_setup
def test_zero(world_rank, world_size, device, checkpoint_dir):
#def test_zero(world_rank, world_size, device):
    opts_dict = {
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
    
    # generate model and optimizer value from a training instance
    init_model_state, init_opt_state = generate_model_optimizer_from_training_instance(device, opts_dict)

    trainer = create_initialized_orttrainer(device, opts_dict, True)

    actual_model_state = trainer._training_session.get_model_state(include_mixed_precision_weights=True)
    print(f"On rank {world_rank} ---------------------------------")
    for k, v in actual_model_state.items():
        print(f"{k}====")
        for k2, v2 in v.items():
            print(k2)
    
    verify_model_state(trainer, init_model_state)
    
    verify_opt_state(trainer, init_opt_state)

    part_info = trainer._training_session.get_partition_info_map()
    print(f"On rank {world_rank} ---------------------------------")
    print(part_info)
    verify_part_info(trainer, True)

#test_backend_api()
test_zero(checkpoint_dir='')