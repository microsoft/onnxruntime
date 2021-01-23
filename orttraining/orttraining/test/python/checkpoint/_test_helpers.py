import os
import pickle
from itertools import islice
import glob
import copy

import torch
import torch.distributed as dist
from onnx import numpy_helper

from onnxruntime import set_seed
from onnxruntime.training import amp, checkpoint, _checkpoint_storage, optim, orttrainer
from onnxruntime.capi._pybind_state import set_cuda_device_id, get_mpi_context_world_rank, get_mpi_context_world_size

from _test_commons import generate_random_input_from_bart_model_desc, generate_dummy_optim_state, \
                          _load_pytorch_transformer_model, _load_bart_model, \
                          assert_all_states_close_ort, assert_all_states_close_pytorch

from numpy.testing import assert_allclose, assert_array_equal

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

def _save(trainer, checkpoint_dir, state_dict_key_name, world_rank=None):
    """Saves the ORTTrainer checkpoint and the complete state dictionary to the given checkpoint_dir directory"""

    # save current model parameters as a checkpoint
    makedir(checkpoint_dir)
    checkpoint_file_name = 'checkpoint{}.ortcp'.format('' if world_rank is None else str(world_rank))
    trainer.save_checkpoint(os.path.join(checkpoint_dir, checkpoint_file_name))
    state_dict = trainer.state_dict()
    with open(os.path.join(checkpoint_dir, state_dict_key_name+'.pkl'), "wb") as f:
        pickle.dump({state_dict_key_name : state_dict}, f)

def save_ort_ckpt(state_dict, filepath):
    _checkpoint_storage.save(state_dict, filepath)

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

def _is_model_parallel_run(trainer_options):
    zero = trainer_options.distributed.deepspeed_zero_optimization.stage > 0
    megatron = trainer_options.distributed.horizontal_parallel_size > 1
    return zero or megatron

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

def create_orttrainer_and_load_checkpoint(device, trainer_opts, checkpoint_dir, use_lamb=True, seed=1, learning_rate=0.1):
    """Instantiate and load checkpoint into trainer

    - Instantiates the ORTTrainer with given input trainer_opts configuration for a simple transformer model
    - Loads the checkpoint from directory checkpoint_dir into the trainer
    - Runs eval_step on the trainer so the trainer onnx graph is initialized
    - Returns the trainer state_dict and the pytorch model
    """
    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model setup    
    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=orttrainer.ORTTrainerOptions(trainer_opts))

    # load checkpoint into trainer
    checkpoint_file_name = 'checkpoint*.ortcp'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, checkpoint_file_name))
    trainer.load_checkpoint(*checkpoint_files)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    return trainer.state_dict(), model

def create_orttrainer_and_load_checkpoint_bart(device, trainer_opts, checkpoint_dir, use_lamb=True, seed=1, learning_rate=0.1):
    """Instantiate and load checkpoint into trainer

    - Instantiates the ORTTrainer with given input trainer_opts configuration for a simple BART model
    - Loads the checkpoint from directory checkpoint_dir into the trainer
    - Runs eval_step on the trainer so the trainer onnx graph is initialized
    - Returns the trainer state_dict, the expected state dict if present, and the onnx model
    """
    torch.manual_seed(seed)
    set_seed(seed)

    # model setup
    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc = _load_bart_model()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=orttrainer.ORTTrainerOptions(trainer_opts))

    # load checkpoint into trainer
    checkpoint_file_name = 'checkpoint*.ortcp'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, checkpoint_file_name))
    trainer.load_checkpoint(*checkpoint_files)

    # run an eval step to innitialize the graph
    src_tokens, prev_output_tokens, target = generate_random_input_from_bart_model_desc(model_desc, seed = seed)
    trainer.eval_step(src_tokens, prev_output_tokens, target)

    expected_state_dict = None
    fname = os.path.join(checkpoint_dir, 'expected_state_dict.pkl')
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            expected_state_dict = pickle.load(f)

    return trainer.state_dict(), expected_state_dict, model

def create_initialized_orttrainer(device, trainer_opts, use_lamb=True, seed=1, learning_rate=1e-10):
    torch.manual_seed(seed)
    set_seed(seed)

    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=orttrainer.ORTTrainerOptions(trainer_opts))

    _train(trainer, train_data, batcher_fn)

    return trainer

def verify_model_state(trainer, expected_state_dict, is_mixedprecision):
    actual_model_state = trainer._training_session.get_model_state(include_mixed_precision_weights=True)
    for fp_or_mp, value in actual_model_state.items():
        for weight_name in value:
            assert weight_name.find('_view_') == -1
    assert len(expected_state_dict['fp32_param']) == len(actual_model_state['full_precision']), \
        "expected and actual should have same number of tensors"
    for weight_name, tensor in expected_state_dict['fp32_param'].items():
        if not weight_name in actual_model_state['full_precision']:
            assert '_view_' in weight_name, \
                "only zero shared weight may not match name"
            weight_name = weight_name.split('_view_')[0]
        assert_allclose(tensor, actual_model_state['full_precision'][weight_name])

    if is_mixedprecision:
        assert 'mixed_precision' in actual_model_state.keys(), "missing 'mixed_precision' key in mixed precision run"
        assert len(expected_state_dict['fp16_param']) == len(actual_model_state['mixed_precision']), \
            "expected and actual should have same number of tensors"
        for weight_name, tensor in expected_state_dict['fp16_param'].items():
            weight_name = weight_name.split('_fp16')[0]
            assert_allclose(tensor, actual_model_state['mixed_precision'][weight_name])

def verify_opt_state(trainer, expected_state_dict):
    actual_opt_state = trainer._training_session.get_optimizer_state()
    actual_opt_count = sum(len(v) for v in actual_opt_state.values())
    assert actual_opt_count == len(expected_state_dict['optimizer'])
    for weight_name in actual_opt_state:
        assert weight_name.find('_view_') == -1
    for opt_name, expected_tensor in expected_state_dict['optimizer'].items():
        if opt_name == "Step":
            actual_tensor = actual_opt_state['shared_optimizer_state']['Step']
        else:
            if opt_name.startswith('Moment_'):
                prefix = opt_name[:len("Moment_0")]
                weight_name = opt_name[len("Moment_0_"):]
                if not weight_name in actual_opt_state:
                    assert '_view_' in weight_name, \
                        "only zero shared weight may not match name"
                    weight_name = weight_name.split('_view_')[0]
            elif opt_name.startswith('Update_Count_'):
                prefix = "Update_Count"
                weight_name = opt_name[len(prefix + 1):]
            actual_tensor = actual_opt_state[weight_name][prefix]
        assert_allclose(actual_tensor, expected_tensor, atol=global_fp16_fp32_atol)

def verify_part_info(trainer, expected_state_dict, is_mixedprecision, is_zero_run):
    part_info = trainer._training_session.get_partition_info_map()
    for weight_name, weight_info in part_info.items():
        for info, value in weight_info.items():
            assert isinstance(value, list), "get_partition_info_map should return list"
            assert isinstance(value[0], int), "get_partition_info_map should return list of int"
            if info == "megatron_row_partition":
                assert len(value) == 1, "megatron_row_partition should only have 1 element"
                if is_zero_run:
                    assert value[0] == -1, "megatron_row_partition is -1 if megatron optimization is not on"
            if info == "original_dim":
                if is_zero_run:
                    assert len(value) > 0, "original_dim should not be empty in zero run"
                    if is_mixedprecision:
                        assert_array_equal(part_info[weight_name]['original_dim'], expected_state_dict['fp16_param'][weight_name + '_fp16'].shape)
                    else:
                        assert_array_equal(part_info[weight_name]['original_dim'], expected_state_dict['fp32_param'][weight_name].shape)

def split_state_dict(state_dict):
    """Given a flat state dictionary, split it into optimizer, fp32_param, fp16_param hierarchical dictionary and return"""

    optimizer_keys = ['Moment_1_', 'Moment_2_', 'Update_Count_', 'Step']
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

def aggregate_states(checkpoint_dir, filename_prefix='state_dict', state_dict_key_name='state_dict'):
    """Aggregate state dictionaries saved in the checkpoint_dir as pickle files with name filename_prefix_world_rank.pkl"""

    aggregated_states = {}
    num_states = len(glob.glob1(checkpoint_dir,"{}*".format(filename_prefix)))
    for rank in range(num_states):
        rank_state_dict = None
        with open(os.path.join(checkpoint_dir, '{}_{}.pkl'.format(filename_prefix, rank)), 'rb') as f:
            rank_state_dict = pickle.load(f)
        # if state_dict_key_name is None, then the rank_state_dictis the loaded object
        if state_dict_key_name:
            # if it has a name, index into the loaded object to extract the rank_state_dict
            rank_state_dict = rank_state_dict['{}_{}'.format(state_dict_key_name, rank)]

        checkpoint._aggregate_model_states(rank_state_dict, {}, aggregated_states, rank_state_dict['trainer_options']['mixed_precision'])

        checkpoint._aggregate_optimizer_states(rank_state_dict, {}, aggregated_states)
    
    return aggregated_states

def create_orttrainer_and_save_checkpoint(device, trainer_opts, checkpoint_dir, state_dict_key_name='state_dict', use_lamb=True, seed=1, learning_rate=0.1):
    torch.manual_seed(seed)
    set_seed(seed)

    ort_trainer_opts = orttrainer.ORTTrainerOptions(trainer_opts)
    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=ort_trainer_opts)

    if 'distributed' in trainer_opts:
        train_data = next(islice(_chunkify(train_data, trainer_opts['distributed']['world_size']), trainer_opts['distributed']['world_rank'], None))

    # run train steps
    _train(trainer, train_data, batcher_fn)

    # save current model parameters as a checkpoint
    if checkpoint_dir:
        if _is_model_parallel_run(ort_trainer_opts):
            _save(trainer, checkpoint_dir, state_dict_key_name, world_rank=ort_trainer_opts.distributed.world_rank)
        else:
            _save(trainer, checkpoint_dir, state_dict_key_name)

def create_orttrainer_and_save_checkpoint_bart(device, trainer_opts, checkpoint_dir, state_dict_key_name='state_dict', use_lamb=True, seed=1, learning_rate=0.1):
    """Instantiate trainer and save checkpoint for BART.

    - Instantiates the ORTTrainer with given input trainer_opts configuration for a simple BART model
    - Loads a dummy optimizer state into the trainer
    - Runs eval_step on the trainer so the trainer onnx graph is initialized
    - Returns the trainer state_dict, the expected state dict if present, and the onnx model
    """
    torch.manual_seed(seed)
    set_seed(seed)

    ort_trainer_opts = orttrainer.ORTTrainerOptions(trainer_opts)
    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc = _load_bart_model()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=ort_trainer_opts)

    # load dummy optimizer state as we are not going to run real training
    dummy_init_state = generate_dummy_optim_state(model, optim_config)
    init_state = copy.deepcopy(dummy_init_state)
    trainer.load_state_dict(dummy_init_state)

    # run an eval step to innitialize the graph
    src_tokens, prev_output_tokens, target = generate_random_input_from_bart_model_desc(model_desc, seed = seed)
    trainer.eval_step(src_tokens, prev_output_tokens, target)

    # save current model parameters as a checkpoint
    if checkpoint_dir:
        if _is_model_parallel_run(ort_trainer_opts):
            _save(trainer, checkpoint_dir, state_dict_key_name, world_rank=ort_trainer_opts.distributed.world_rank)
            # save the initial complete model and optimizer states
            if ort_trainer_opts.distributed.world_rank == 0:
                init_state['model'] = {'full_precision': dict()}
                for initializer in model.graph.initializer:
                    init_state['model']['full_precision'][initializer.name] = numpy_helper.to_array(initializer)
                with open(os.path.join(checkpoint_dir, 'expected_state_dict.pkl'), "wb") as f:
                    pickle.dump(init_state, f)
        else:
            _save(trainer, checkpoint_dir, state_dict_key_name)

def load_model_optim_state_and_eval(device, trainer_opts, use_lamb=True, seed=1, learning_rate=0.1):
    torch.manual_seed(seed)
    set_seed(seed)

    optim_config = optim.LambConfig(lr=learning_rate) if use_lamb else optim.AdamConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=orttrainer.ORTTrainerOptions(trainer_opts))

    # load dummy state
    dummy_init_state = generate_dummy_optim_state(model, optim_config)
    trainer.load_state_dict(dummy_init_state)

    # run an eval step to innitialize the graph
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    optimizer_state_dict = trainer.state_dict()
    del optimizer_state_dict['model']

    return dummy_init_state, optimizer_state_dict

def assert_all_states_close(checkpoint_dir, state_dict_key_name, state_dict_post_checkpoint, pytorch_model=None):
    """Extract previously saved state dictionary from pickle file and compare against state_dict_post_checkpoint for all states"""

    state = None
    with open(os.path.join(checkpoint_dir, '{}.pkl'.format(state_dict_key_name)), 'rb') as f:
        state = pickle.load(f)
    state_dict_pre_checkpoint = state[state_dict_key_name]

    # assert that the ort states are close
    assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint)

    # assert that the pytorch model states are close
    if pytorch_model:
        assert_all_states_close_pytorch(checkpoint._to_pytorch_format(state_dict_pre_checkpoint), pytorch_model)
