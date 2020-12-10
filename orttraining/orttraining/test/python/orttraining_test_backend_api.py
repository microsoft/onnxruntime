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
def train(trainer, train_data, batcher_fn, total_batch_steps = 5, seed = 1):
    for i in range(total_batch_steps):
        torch.manual_seed(seed)
        set_seed(seed)
        data, targets = batcher_fn(train_data, i*35)
        trainer.train_step(data, targets)
def makedir(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok = True)
def save(trainer, checkpoint_dir, state_dict_key_name = 'state_dict'):
    # save current model parameters as a checkpoint
    makedir(checkpoint_dir)
    checkpoint.experimental_save_checkpoint(trainer, checkpoint_dir)
    state_dict = checkpoint.experimental_state_dict(trainer)
    pickle.dump({state_dict_key_name : state_dict}, open(checkpoint_dir+state_dict_key_name+'.pkl', "wb"))
def chunkify(sequence, num_chunks):
    quo, rem = divmod(len(sequence), num_chunks)
    return (sequence[i * quo + min(i, rem):(i + 1) * quo + min(i + 1, rem)] for i in range(num_chunks))
def distributed_setup(save_function):
    def setup():
        world_rank = get_mpi_context_world_rank()
        world_size = get_mpi_context_world_size()
        device = 'cuda:' + str(world_rank)
        os.environ['RANK'] = str(world_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        set_cuda_device_id(world_rank)
        dist.init_process_group(backend='nccl', world_size=world_size, rank=world_rank)
        save_function(world_rank, world_size, device)
    return setup

def verify_model_state(trainer, model):
    print(model)
    actual_model_state = trainer._training_session.get_model_state(include_mixed_precision_weights=False)

def single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1
    torch.manual_seed(seed)
    set_seed(seed)
    # PyTorch transformer model as example
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    # run train steps
    train(trainer, train_data, batcher_fn)
    # session_state = trainer._training_session.get_state()
    # print("====get_state")
    # for k, v in session_state.items():
    #     print(k)
    #actual_model_state = trainer._training_session.get_model_state(include_mixed_precision_weights=False)
    verify_model_state(trainer, model)
    # print("====get_model_state")
    # for k, v in model_state.items():
    #     print(k)
    #     for k1, v1 in v.items():
    #         print(k1)
    opt_state = trainer._training_session.get_optimizer_state()
    # print("====opt_state")
    # print(opt_state)
    # for k, v in opt_state.items():
    #     print(k)
    #     for k1, v1 in v.items():
    #         print(k1)
    part_info = trainer._training_session.get_partition_info_map()
    print("====part_info")
    print(part_info)
    for k, v in part_info.items():
        print(k)
        for k1, v1 in v.items():
            print(k1)

    #trainer._training_session.load_model_opt_state(model_state, opt_state, strict=True)
    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir)
single_node_full_precision()