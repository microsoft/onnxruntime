import os
import pytest
import pickle
import argparse
from itertools import islice
import torch
import torch.distributed as dist
from onnxruntime import set_seed
from onnxruntime.training import amp, checkpoint, optim, orttrainer
from orttraining_test_orttrainer_frontend import _load_pytorch_transformer_model
from onnxruntime.capi._pybind_state import set_cuda_device_id, get_mpi_context_world_rank, get_mpi_context_world_size

from checkpoint._test_helpers import distributed_setup, create_orttrainer_and_save_checkpoint, create_orttrainer_and_load_checkpoint
from numpy.testing import assert_allclose

def train(trainer, train_data, batcher_fn, total_batch_steps = 5, seed = 1):
    for i in range(total_batch_steps):
        torch.manual_seed(seed)
        set_seed(seed)
        data, targets = batcher_fn(train_data, i*35)
        trainer.train_step(data, targets)
# def distributed_setup(save_function):
#     def setup():
#         world_rank = get_mpi_context_world_rank()
#         world_size = get_mpi_context_world_size()
#         device = 'cuda:' + str(world_rank)
#         os.environ['RANK'] = str(world_rank)
#         os.environ['WORLD_SIZE'] = str(world_size)
#         os.environ['MASTER_ADDR'] = '127.0.0.1'
#         os.environ['MASTER_PORT'] = '29500'
#         set_cuda_device_id(world_rank)
#         dist.init_process_group(backend='nccl', world_size=world_size, rank=world_rank)
#         #save_function(world_rank, world_size, device)
#     return setup

# change to train 1 step to avoid saving and loading checkpoint
# adapted from create_orttrainer_and_load_checkpoint method in checkpoint/_test_helpers.py 
def generate_model_optimizer_from_training_instance(device, trainer_opts, use_lamb=True):
    """Instantiate and load checkpoint into trainer

    - Instantiates the ORTTrainer with given input trainer_opts configuration for a simple transformer model
    - Runs train_step on the trainer so the trainer onnx graph is initialized
    - Returns the trainer state_dict and the pytorch model
    """
    seed = 1
    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model setup
    learning_rate = 1e-10
    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=orttrainer.ORTTrainerOptions(trainer_opts))

    # train 1 step
    train(trainer, train_data, batcher_fn)

    model_state = trainer._training_session.get_model_state(include_mixed_precision_weights=False)
    opt_state = trainer._training_session.get_optimizer_state()

    return model_state, opt_state

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
                assert expected_tensor[0] + 1 == tensor[0], "step should increment by 1"
            else: 
                assert_allclose(tensor, expected_tensor, 1e-1, 1e-2)

def verify_part_info(trainer):
    part_info = trainer._training_session.get_partition_info_map()
    for weight_name, weight_info in part_info.items():
        for info, value in weight_info.items():
            assert isinstance(value, list), "get_partition_info_map should return list"
            assert isinstance(value[0], int), "get_partition_info_map should return list of int"
            assert value[0] == 0, "megatron_row_partition is 0 (false) if megatron optimization is not on"

def test_backend_api(device = 'cuda'):
    learning_rate = 1e-10
    seed = 1
    torch.manual_seed(seed)
    set_seed(seed)
    opts_dict = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # generate model and optimizer value from a training instance
    init_model_state, init_opt_state = generate_model_optimizer_from_training_instance(device, opts_dict)
    opts = orttrainer.ORTTrainerOptions(opts_dict)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    #train(trainer, train_data, batcher_fn)
    train(trainer, train_data, batcher_fn)

    trainer._training_session.load_model_optimizer_state(init_model_state, init_opt_state, False)

    # train one step
    train(trainer, train_data, batcher_fn, 1)
    
    verify_model_state(trainer, init_model_state)
    
    verify_opt_state(trainer, init_opt_state)

    verify_part_info(trainer)

@distributed_setup
def test_zero(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb/'):
    #print("in test_zero")
    learning_rate = 1e-10
    seed = 1
    torch.manual_seed(seed)
    set_seed(seed)
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
    opts = orttrainer.ORTTrainerOptions(opts_dict)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    #train(trainer, train_data, batcher_fn)
    train(trainer, train_data, batcher_fn)

    trainer._training_session.load_model_optimizer_state(init_model_state, init_opt_state, False)

    # train one step
    train(trainer, train_data, batcher_fn, 1)

    actual_model_state = trainer._training_session.get_model_state(include_mixed_precision_weights=True)
    #if world_rank == 0:
    print(f"On rank {world_rank} ---------------------------------")
    for k, v in actual_model_state.items():
        print(f"{k}====")
        for k2, v2 in v.items():
            print(k2)
    
    verify_model_state(trainer, init_model_state)
    
    verify_opt_state(trainer, init_opt_state)

    verify_part_info(trainer)

#test_backend_api()
test_zero(checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/lamb/')