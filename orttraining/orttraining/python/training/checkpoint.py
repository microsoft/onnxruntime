import numpy as np
import onnx
import os
import torch
import warnings


################################################################################
# Experimental Checkpoint APIs
################################################################################


def experimental_state_dict(ort_trainer, include_optimizer_state=True):
    if not ort_trainer._training_session:
        warnings.warn("ONNX Runtime training session is not initialized yet. "
                        "Please run train_step or eval_step at least once before calling state_dict().")
        return ort_trainer._state_dict

    # extract trained weights
    session_state = ort_trainer._training_session.get_state()
    torch_state = {}
    for name in session_state:
        torch_state[name] = torch.from_numpy(session_state[name])

    # extract untrained weights and buffer
    for n in ort_trainer._onnx_model.graph.initializer:
        if n.name not in torch_state and n.name in ort_trainer.options.utils.frozen_weights:
            torch_state[n.name] = torch.from_numpy(np.array(onnx.numpy_helper.to_array(n)))

    # Need to remove redundant (optimizer) initializers to map back to original torch state names
    if not include_optimizer_state and ort_trainer._torch_state_dict_keys:
        return {key: torch_state[key] for key in ort_trainer._torch_state_dict_keys if key in torch_state}
    return torch_state


def experimental_load_state_dict(ort_trainer, state_dict, strict=False):
    # Note: It may happen ONNX model has not yet been initialized
    # In this case we cache a reference to desired state and delay the restore until after initialization
    # Unexpected behavior will result if the user changes the reference before initialization
    if not ort_trainer._training_session:
        ort_trainer._state_dict = state_dict
        ort_trainer._load_state_dict_strict = strict
        return

    # Update onnx model from loaded state dict
    cur_initializers_names = [n.name for n in ort_trainer._onnx_model.graph.initializer]
    new_initializers = {}

    for name in state_dict:
        if name in cur_initializers_names:
            new_initializers[name] = state_dict[name].numpy()
        elif strict:
            raise RuntimeError("Checkpoint tensor: {} is not present in the model.".format(name))

    ort_trainer._update_onnx_model_initializers(new_initializers)

    # create new session based on updated onnx model
    ort_trainer._state_dict = None
    ort_trainer._init_session()

    # load training state
    session_state = {name:state_dict[name].numpy() for name in state_dict}
    ort_trainer._training_session.load_state(session_state, strict)


def experimental_save_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix="ORT_checkpoint", checkpoint_state_dict=None, include_optimizer_state=True):
    if checkpoint_state_dict is None:
        checkpoint_state_dict = {'model': experimental_state_dict(ort_trainer, include_optimizer_state)}
    else:
        checkpoint_state_dict.update({'model': experimental_state_dict(ort_trainer, include_optimizer_state)})

    assert os.path.exists(checkpoint_dir), f"checkpoint_dir ({checkpoint_dir}) directory doesn't exist"

    checkpoint_name = _get_checkpoint_name(checkpoint_prefix,
                                           ort_trainer.options.distributed.deepspeed_zero_optimization.stage,
                                           ort_trainer.options.distributed.world_rank,
                                           ort_trainer.options.distributed.world_size)
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.exists(checkpoint_file):
        msg = f"{checkpoint_file} already exists, overwriting."
        warnings.warn(msg)
    torch.save(checkpoint_state_dict, checkpoint_file)


def experimental_load_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix="ORT_checkpoint", strict=False):
    checkpoint_files = _list_checkpoint_files(
        checkpoint_dir, checkpoint_prefix)
    is_partitioned = False
    if len(checkpoint_files) > 1:
        msg = (f"Found more than one file with prefix {checkpoint_prefix} in directory {checkpoint_dir}."
               " Attempting to load ZeRO checkpoint.")
        warnings.warn(msg)
        is_partitioned = True
    if (not ort_trainer.options.distributed.deepspeed_zero_optimization.stage) and is_partitioned:
        return _load_multi_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix, strict)
    else:
        return _load_single_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix, is_partitioned, strict)


################################################################################
# Helper functions
################################################################################


def _load_single_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix, is_partitioned, strict):
    checkpoint_name = _get_checkpoint_name(
        checkpoint_prefix, is_partitioned, ort_trainer.options.distributed.world_rank, ort_trainer.options.distributed.world_size)
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)

    if is_partitioned:
        assert_msg = (f"Couldn't find checkpoint file {checkpoint_file}."
                      " Optimizer partitioning is enabled using ZeRO. Please make sure the checkpoint file exists "
                     f"for rank {ort_trainer.options.distributed.world_rank} of {ort_trainer.options.distributed.world_size}")
    else:
        assert_msg = f"Couldn't find checkpoint file {checkpoint_file}."
    assert os.path.exists(checkpoint_file), assert_msg

    checkpoint_state = torch.load(checkpoint_file, map_location='cpu')
    experimental_load_state_dict(ort_trainer, checkpoint_state['model'], strict=strict)
    del(checkpoint_state['model'])
    return checkpoint_state


def _load_multi_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix, strict):
    checkpoint_files = _list_checkpoint_files(checkpoint_dir, checkpoint_prefix)

    ckpt_agg = _CombineZeroCheckpoint(checkpoint_files)
    aggregate_state_dict = ckpt_agg.aggregate_checkpoints()

    experimental_load_state_dict(ort_trainer, aggregate_state_dict, strict=strict)

    # aggregate other keys in the state_dict.
    # Values will be overwritten for matching keys among workers
    all_checkpoint_states = dict()
    for checkpoint_file in checkpoint_files:
        checkpoint_state = torch.load(checkpoint_file, map_location='cpu')
        del(checkpoint_state['model'])
        all_checkpoint_states.update(checkpoint_state)
    return all_checkpoint_states


def _list_checkpoint_files(checkpoint_dir, checkpoint_prefix, extension='.ort.pt'):
    ckpt_file_names = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix)]
    ckpt_file_names = [f for f in ckpt_file_names if f.endswith(extension)]
    ckpt_file_names = [os.path.join(checkpoint_dir, f) for f in ckpt_file_names]

    assert len(ckpt_file_names) > 0, f"No checkpoint found with prefix '{checkpoint_prefix}' at '{checkpoint_dir}'"
    return ckpt_file_names


def _get_checkpoint_name(prefix, is_partitioned, world_rank=None, world_size=None):
    SINGLE_CHECKPOINT_FILENAME = '{prefix}.ort.pt'
    MULTIPLE_CHECKPOINT_FILENAME = '{prefix}.ZeRO.{world_rank}.{world_size}.ort.pt'

    if is_partitioned:
        filename = MULTIPLE_CHECKPOINT_FILENAME.format(prefix=prefix, world_rank=world_rank, world_size=(world_size-1))
    else:
        filename = SINGLE_CHECKPOINT_FILENAME.format(prefix=prefix)
    return filename


def _split_state_dict(state_dict):
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


class _CombineZeroCheckpoint(object):
    def __init__(self, checkpoint_files, clean_state_dict=None):

        assert len(checkpoint_files) > 0, "No checkpoint files passed"
        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split('ZeRO')[1].split('.')[2]) + 1
        assert len(self.checkpoint_files) == self.world_size, f"Could not find {self.world_size} files"
        self.weight_shape_map = dict()
        self.sharded_params = set()

    def _split_name(self, name):
        name_split = name.split('_view_')
        if(len(name_split) > 1):
            view_num = int(name_split[1])
        else:
            view_num = None
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

    def _update_weight_statistics(self, name, value):
        if name not in self.weight_shape_map:
            self.weight_shape_map[name] = value.size()  # original shape of tensor

    def _reshape_tensor(self, key):
        value = self.aggregate_state_dict[key]
        weight_name, _, _, _ = self._split_name(key)
        set_size = self.weight_shape_map[weight_name]
        self.aggregate_state_dict[key] = value.reshape(set_size)

    def _aggregate(self, param_dict):
        for k, v in param_dict.items():
            weight_name, optimizer_key, view_num, fp16_key = self._split_name(k)
            if view_num is not None:
                # parameter is sharded
                param_name = optimizer_key + weight_name + fp16_key

                if param_name in self.aggregate_state_dict and optimizer_key not in ['Update_Count_']:
                    self.sharded_params.add(param_name)
                    # Found a previous shard of the param, concatenate shards ordered by ranks
                    self.aggregate_state_dict[param_name] = torch.cat((self.aggregate_state_dict[param_name], v))
                else:
                    self.aggregate_state_dict[param_name] = v
            else:
                if k in self.aggregate_state_dict:
                    assert (self.aggregate_state_dict[k] == v).all(), "Unsharded params must have the same value"
                else:
                    self.aggregate_state_dict[k] = v
                self._update_weight_statistics(weight_name, v)

    def aggregate_checkpoints(self):
        checkpoint_prefix = self.checkpoint_files[0].split('.ZeRO')[0]
        self.aggregate_state_dict = dict()

        for i in range(self.world_size):
            checkpoint_name = _get_checkpoint_name(checkpoint_prefix, True, i, self.world_size)
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            if 'model' in rank_state_dict:
                rank_state_dict = rank_state_dict['model']

            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)

            rank_state_dict = _split_state_dict(rank_state_dict)
            self._aggregate(rank_state_dict['fp16_param'])
            self._aggregate(rank_state_dict['fp32_param'])
            self._aggregate(rank_state_dict['optimizer'])

        for k in self.sharded_params:
            self._reshape_tensor(k)
        return self.aggregate_state_dict
