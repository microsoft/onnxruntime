from collections import OrderedDict
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
        if n.name not in torch_state:
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


class _CombineZeroCheckpoint(object):
    def __init__(self, checkpoint_files, clean_state_dict=None):

        assert len(checkpoint_files) > 0, "No checkpoint files passed"
        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split('ZeRO')[1].split('.')[2]) + 1
        assert len(self.checkpoint_files) == self.world_size, f"Could not find {self.world_size} files"
        self.weight_shape_map = dict()

    def _is_sharded(self, name):
        if '_view_' in name:
            return True
        return False

    def _has_fp16_weights(self, state_dict):
        for k in state_dict.keys():
            if k.endswith('_fp16'):
                return True
        return False

    def _split_moment_name(self, name):
        name_split = name.split('_view_')
        if(len(name_split) > 1):
            view_num = int(name_split[1])
        else:
            view_num = None
        weight_name = name_split[0].split('Moment_')[1][2:]
        moment_num = int(name_split[0].split('Moment_')[1][0])
        return moment_num, weight_name, view_num

    def _update_weight_statistics(self, name, value):
        self.weight_shape_map[name] = value.size()  # original shape of tensor

    def _reshape_tensors(self, state_dict, fp16):
        for k, v in state_dict.items():
            if k.startswith('Moment_'):
                _, weight_name, _ = self._split_moment_name(k)
                set_size = self.weight_shape_map[weight_name]
                state_dict[k] = v.reshape(set_size)
                state_dict[weight_name] = state_dict[weight_name].reshape(set_size)
        return state_dict

    def aggregate_checkpoints(self):
        checkpoint_dir = os.path.dirname(self.checkpoint_files[0])
        checkpoint_prefix = self.checkpoint_files[0].split('.ZeRO')[0]
        self.aggregate_state_dict = dict()

        is_fp16 = False
        weight_offset = dict()
        for i in range(self.world_size):
            checkpoint_name = _get_checkpoint_name(checkpoint_prefix, True, i, self.world_size)
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            if 'model' in rank_state_dict:
                rank_state_dict = rank_state_dict['model']

            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)

            if i == 0:
                is_fp16 = self._has_fp16_weights(rank_state_dict)

            for k, v in rank_state_dict.items():
                if k.startswith('Moment_'):
                    moment_num, weight_name, view_num = self._split_moment_name(k)

                    if self._is_sharded(k):
                        clean_name = 'Moment_' + str(moment_num) + '_' + weight_name
                        if clean_name in self.aggregate_state_dict:
                            # Found a previous shard of the moment, concatenate shards ordered by ranks
                            self.aggregate_state_dict[clean_name] = torch.cat((self.aggregate_state_dict[clean_name], v), 0)
                        else:
                            self.aggregate_state_dict[clean_name] = v
                    else:
                        # Moment is not sharded, add as is
                        self.aggregate_state_dict[k] = v

                    if is_fp16 and moment_num == 1:
                        # FP32 weights are sharded, patch together based on moments
                        if view_num == 0:
                            # This FP32 weight's first shard is present on this rank,
                            # flatten and add the weight's first view
                            self.aggregate_state_dict[weight_name] = rank_state_dict[weight_name].view(-1)
                            self._update_weight_statistics(weight_name, rank_state_dict[weight_name])
                            weight_offset[weight_name] = v.numel()

                        elif view_num == 1:
                            # This FP32 weight is carryforward from previous rank
                            # Get start and end of weight slice to be updated from this rank
                            weight_start = weight_offset[weight_name]
                            weight_end = weight_start + v.numel()

                            if weight_start:
                                old_value = self.aggregate_state_dict[weight_name]
                                new_value = rank_state_dict[weight_name].view(-1)
                                # patch the weight together
                                self.aggregate_state_dict[weight_name] = torch.cat((old_value[:weight_start],
                                                                                    new_value[weight_start:weight_end],
                                                                                    old_value[weight_end:]), 0)

                            # update offset for next view
                            weight_offset[weight_name] = weight_end

                elif k.startswith('Update_Count'):
                    clean_name = k.split('_view_')[0]
                    # add a single copy of the 'Update_Count' tensor for current weight
                    if clean_name not in self.aggregate_state_dict:
                        self.aggregate_state_dict[clean_name] = v

                else:
                    if k not in self.aggregate_state_dict:
                        self.aggregate_state_dict[k] = v
                        if not (k.endswith('_fp16') or k == 'Step'):
                            # FP32 Weight
                            self._update_weight_statistics(k, v)

        final_state_dict = self._reshape_tensors(
            self.aggregate_state_dict, is_fp16)
        return final_state_dict
