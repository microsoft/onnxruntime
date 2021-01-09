import os
import pickle
from itertools import islice

import torch
import torch.distributed as dist

from onnxruntime import set_seed
from onnxruntime.training import amp, checkpoint, optim, orttrainer
from orttraining_test_orttrainer_frontend import _load_pytorch_transformer_model
from onnxruntime.capi._pybind_state import set_cuda_device_id, get_mpi_context_world_rank, get_mpi_context_world_size

global_fp16_fp32_atol = 1e-3

def _train(trainer, train_data, batcher_fn, total_batch_steps = 5, seed = 1):
    """Runs train_step total_batch_steps number of times on the given trainer"""
    for i in range(total_batch_steps):
        torch.manual_seed(seed)
        set_seed(seed)
        data, targets = batcher_fn(train_data, i*35)
        trainer.train_step(data, targets)

def makedir(checkpoint_dir):
    """Creates a directory if checkpoint_dir does not exist"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok = True)

def _save(trainer, checkpoint_dir, state_dict_key_name):
    """Saves the ORTTrainer checkpoint and the complete state dictionary to the given checkpoint_dir directory""" 

    # save current model parameters as a checkpoint
    makedir(checkpoint_dir)
    checkpoint.experimental_save_checkpoint(trainer, checkpoint_dir)
    state_dict = checkpoint.experimental_state_dict(trainer)
    pickle.dump({state_dict_key_name : state_dict}, open(os.path.join(checkpoint_dir, state_dict_key_name+'.pkl'), "wb"))

def _chunkify(sequence, num_chunks):
    """Breaks down a given sequence into num_chunks chunks"""
    quo, rem = divmod(len(sequence), num_chunks)
    return (sequence[i * quo + min(i, rem):(i + 1) * quo + min(i + 1, rem)] for i in range(num_chunks))

def _setup_test_infra(world_rank, world_size):
    """distributed setup just for testing purposes"""
    os.environ['RANK'] = str(world_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    set_cuda_device_id(world_rank)

    dist.init_process_group(backend='nccl', world_size=world_size, rank=world_rank)

def distributed_setup(func):
    """Decorator function for distributed tests.

    Sets up distributed environment by extracting the following variables from MPI context
    - world_rank
    - world_size
    - device

    Also sets up the infrastructure required for the distributed tests such as setting up the torch distributed initialization
    """
    def setup(checkpoint_dir):
        world_rank = get_mpi_context_world_rank()
        world_size = get_mpi_context_world_size()
        device = 'cuda:' + str(world_rank)

        _setup_test_infra(world_rank, world_size)

        func(world_rank, world_size, device, checkpoint_dir=checkpoint_dir)

    return setup

def create_orttrainer_and_load_checkpoint(device, trainer_opts, checkpoint_dir, use_lamb=True):
    """Instantiate and load checkpoint into trainer

    - Instantiates the ORTTrainer with given input trainer_opts configuration for a simple transformer model
    - Loads the checkpoint from directory checkpoint_dir into the trainer
    - Runs eval_step on the trainer so the trainer onnx graph is initialized
    - Returns the trainer state_dict and the pytorch model
    """
    seed = 1
    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model setup
    learning_rate = 0.1
    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=orttrainer.ORTTrainerOptions(trainer_opts))

    # load checkpoint into trainer
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    return checkpoint.experimental_state_dict(trainer), model

def split_state_dict(state_dict):
    """Given a flat state dictionary, split it into optimizer, fp32_param, fp16_param hierarchical dictionary and return"""

    optimizer_keys = ['Moment_1_', 'Moment_2_', 'Update_Count_', 'Step_']
    split_sd = {'optimizer': {}, 'fp32_param': {}, 'fp16_param': {}}
    for k, v in state_dict.items():
        mode = 'fp32_param'
        for optim_key in optimizer_keys:
            if k.startswith(optim_key):
                mode = 'optimizer'
                break
        if k.endswith('_fp16'):
            mode = 'fp16_param'
        split_sd[mode][k] = v
    return split_sd

def _split_name(name):
    """Splits given state name (model or optimizer state name) into the param_name, optimizer_key, view_num and the fp16_key"""
    name_split = name.split('_view_')
    view_num = None
    if(len(name_split) > 1):
        view_num = int(name_split[1])
    optimizer_key = ''
    fp16_key = ''
    if name_split[0].startswith('Moment_1'):
        optimizer_key = 'Moment_1_'
    elif name_split[0].startswith('Moment_2'):
        optimizer_key = 'Moment_2_'
    elif name_split[0].startswith('Update_Count'):
        optimizer_key = 'Update_Count_'
    elif name_split[0].endswith('_fp16'):
        fp16_key = '_fp16'
    param_name = name_split[0]
    if optimizer_key != '':
        param_name = param_name.split(optimizer_key)[1]
    param_name = param_name.split('_fp16')[0]
    return param_name, optimizer_key, view_num, fp16_key

def aggregate_states(aggregated_states, state_dict):
    """Concatenate existing aggregated state dict values with given state_dict values"""

    for key, value in state_dict.items():
        weight_name, optimizer_key, view_num, fp16_key = _split_name(key)
        if view_num is not None:
            # parameter is sharded
            param_name = optimizer_key + weight_name + fp16_key

            if param_name in aggregated_states and optimizer_key not in ['Update_Count_']:
                # found a previous shard of the param, concatenate shards ordered by ranks
                aggregated_states[param_name] = torch.cat((aggregated_states[param_name], value))
            else:
                aggregated_states[param_name] = value
        else:
            aggregated_states[key] = value

def create_orttrainer_and_save_checkpoint(device, trainer_opts, checkpoint_dir, state_dict_key_name='state_dict', use_lamb=True):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=orttrainer.ORTTrainerOptions(trainer_opts))

    if 'distributed' in trainer_opts:
        train_data = next(islice(_chunkify(train_data, trainer_opts['distributed']['world_size']), trainer_opts['distributed']['world_rank'], None))

    # run train steps
    _train(trainer, train_data, batcher_fn)

    # save current model parameters as a checkpoint
    if checkpoint_dir:
        _save(trainer, checkpoint_dir, state_dict_key_name)
