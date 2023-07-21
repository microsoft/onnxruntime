import os
import tempfile
import warnings
from enum import Enum

import numpy as np
import onnx
import torch

from . import _checkpoint_storage, _utils

################################################################################
# Experimental Checkpoint APIs
################################################################################


def experimental_state_dict(ort_trainer, include_optimizer_state=True):
    warnings.warn(
        "experimental_state_dict() will be deprecated soon. Please use ORTTrainer.state_dict() instead.",
        DeprecationWarning,
    )

    if not ort_trainer._training_session:
        warnings.warn(
            "ONNX Runtime training session is not initialized yet. "
            "Please run train_step or eval_step at least once before calling state_dict()."
        )
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
    warnings.warn(
        "experimental_load_state_dict() will be deprecated soon. Please use ORTTrainer.load_state_dict() instead.",
        DeprecationWarning,
    )

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
            raise RuntimeError(f"Checkpoint tensor: {name} is not present in the model.")

    ort_trainer._update_onnx_model_initializers(new_initializers)

    # create new session based on updated onnx model
    ort_trainer._state_dict = None
    ort_trainer._init_session()

    # load training state
    session_state = {name: state_dict[name].numpy() for name in state_dict}
    ort_trainer._training_session.load_state(session_state, strict)


def experimental_save_checkpoint(
    ort_trainer,
    checkpoint_dir,
    checkpoint_prefix="ORT_checkpoint",
    checkpoint_state_dict=None,
    include_optimizer_state=True,
):
    warnings.warn(
        "experimental_save_checkpoint() will be deprecated soon. Please use ORTTrainer.save_checkpoint() instead.",
        DeprecationWarning,
    )

    if checkpoint_state_dict is None:
        checkpoint_state_dict = {"model": experimental_state_dict(ort_trainer, include_optimizer_state)}
    else:
        checkpoint_state_dict.update({"model": experimental_state_dict(ort_trainer, include_optimizer_state)})

    assert os.path.exists(checkpoint_dir), f"checkpoint_dir ({checkpoint_dir}) directory doesn't exist"

    checkpoint_name = _get_checkpoint_name(
        checkpoint_prefix,
        ort_trainer.options.distributed.deepspeed_zero_optimization.stage,
        ort_trainer.options.distributed.world_rank,
        ort_trainer.options.distributed.world_size,
    )
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.exists(checkpoint_file):
        msg = f"{checkpoint_file} already exists, overwriting."
        warnings.warn(msg)
    torch.save(checkpoint_state_dict, checkpoint_file)


def experimental_load_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix="ORT_checkpoint", strict=False):
    warnings.warn(
        "experimental_load_checkpoint() will be deprecated soon. Please use ORTTrainer.load_checkpoint() instead.",
        DeprecationWarning,
    )

    checkpoint_files = _list_checkpoint_files(checkpoint_dir, checkpoint_prefix)
    is_partitioned = False
    if len(checkpoint_files) > 1:
        msg = (
            f"Found more than one file with prefix {checkpoint_prefix} in directory {checkpoint_dir}."
            " Attempting to load ZeRO checkpoint."
        )
        warnings.warn(msg)
        is_partitioned = True
    if (not ort_trainer.options.distributed.deepspeed_zero_optimization.stage) and is_partitioned:
        return _load_multi_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix, strict)
    else:
        return _load_single_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix, is_partitioned, strict)


class _AGGREGATION_MODE(Enum):  # noqa: N801
    Zero = 0
    Megatron = 1


def _order_paths(paths, D_groups, H_groups):
    """Reorders the given paths in order of aggregation of ranks for D and H parallellism respectively
    and returns the ordered dict"""

    trainer_options_path_tuples = []
    world_rank = _utils.state_dict_trainer_options_world_rank_key()

    for path in paths:
        trainer_options_path_tuples.append(  # noqa: PERF401
            (_checkpoint_storage.load(path, key=_utils.state_dict_trainer_options_key()), path)
        )

    # sort paths according to rank
    sorted_paths = [
        path
        for _, path in sorted(
            trainer_options_path_tuples, key=lambda trainer_options_path_pair: trainer_options_path_pair[0][world_rank]
        )
    ]

    ordered_paths = dict()
    ordered_paths["D"] = [[sorted_paths[i] for i in D_groups[group_id]] for group_id in range(len(D_groups))]
    ordered_paths["H"] = [[sorted_paths[i] for i in H_groups[group_id]] for group_id in range(len(H_groups))]

    return ordered_paths


def _add_or_update_sharded_key(
    state_key, state_value, state_sub_dict, model_state_key, state_partition_info, sharded_states_original_dims, mode
):
    """Add or update the record for the sharded state_key in the state_sub_dict"""

    # record the original dimension for this state
    original_dim = _utils.state_dict_original_dimension_key()
    sharded_states_original_dims[model_state_key] = state_partition_info[original_dim]

    axis = 0
    if mode == _AGGREGATION_MODE.Megatron and state_partition_info["megatron_row_partition"] == 0:
        axis = -1

    if state_key in state_sub_dict:
        # state_dict already contains a record for this state
        # since this state is sharded, concatenate the state value to
        # the record in the state_dict
        state_sub_dict[state_key] = np.concatenate((state_sub_dict[state_key], state_value), axis)
    else:
        # create a new entry for this state in the state_dict
        state_sub_dict[state_key] = state_value


def _add_or_validate_unsharded_key(state_key, state_value, state_sub_dict, mismatch_error_string):
    """Add or validate the record for the unsharded state_key in the state_sub_dict"""

    if state_key in state_sub_dict:
        # state_dict already contains a record for this unsharded state.
        # assert that all values are the same for this previously loaded state
        assert (state_sub_dict[state_key] == state_value).all(), mismatch_error_string
    else:
        # create a new entry for this state in the state_sub_dict
        state_sub_dict[state_key] = state_value


def _aggregate_model_states(
    rank_state_dict, sharded_states_original_dims, state_dict, mixed_precision_enabled, mode=_AGGREGATION_MODE.Zero
):
    """Aggregates all model states from the rank_state_dict into state_dict"""

    model = _utils.state_dict_model_key()
    full_precision = _utils.state_dict_full_precision_key()
    partition_info = _utils.state_dict_partition_info_key()

    # if there are no model states in the rank_state_dict, no model aggregation is needed
    if model not in rank_state_dict:
        return

    if model not in state_dict:
        state_dict[model] = {}

    if full_precision not in state_dict[model]:
        state_dict[model][full_precision] = {}

    # iterate over all model state keys
    for model_state_key, model_state_value in rank_state_dict[model][full_precision].items():
        # ZERO: full precision model states are sharded only when they exist in the partition_info subdict and mixed
        # precision training was enabled. for full precision training, full precision model states are not sharded
        # MEGATRON : full precision model states are sharded when they exist in the partition_info subdict
        if (model_state_key in rank_state_dict[partition_info]) and (
            mode == _AGGREGATION_MODE.Megatron or mixed_precision_enabled
        ):
            # this model state is sharded
            _add_or_update_sharded_key(
                model_state_key,
                model_state_value,
                state_dict[model][full_precision],
                model_state_key,
                rank_state_dict[partition_info][model_state_key],
                sharded_states_original_dims,
                mode,
            )
        else:
            # this model state is not sharded since a record for it does not exist in the partition_info subdict
            _add_or_validate_unsharded_key(
                model_state_key,
                model_state_value,
                state_dict[model][full_precision],
                f"Value mismatch for model state {model_state_key}",
            )


def _aggregate_optimizer_states(rank_state_dict, sharded_states_original_dims, state_dict, mode=_AGGREGATION_MODE.Zero):
    """Aggregates all optimizer states from the rank_state_dict into state_dict"""

    optimizer = _utils.state_dict_optimizer_key()
    partition_info = _utils.state_dict_partition_info_key()
    sharded_optimizer_keys = _utils.state_dict_sharded_optimizer_keys()

    # if there are no optimizer states in the rank_state_dict, no optimizer aggregation is needed
    if optimizer not in rank_state_dict:
        return

    if optimizer not in state_dict:
        state_dict[optimizer] = {}

    # iterate over all optimizer state keys
    for model_state_key, optimizer_dict in rank_state_dict[optimizer].items():
        for optimizer_key, optimizer_value in optimizer_dict.items():
            if model_state_key not in state_dict[optimizer]:
                state_dict[optimizer][model_state_key] = {}

            if optimizer_key in sharded_optimizer_keys and model_state_key in rank_state_dict[partition_info]:
                # this optimizer state is sharded since a record exists in the partition_info subdict
                _add_or_update_sharded_key(
                    optimizer_key,
                    optimizer_value,
                    state_dict[optimizer][model_state_key],
                    model_state_key,
                    rank_state_dict[partition_info][model_state_key],
                    sharded_states_original_dims,
                    mode,
                )
            else:
                # this optimizer state is not sharded since a record for it does not exist in the partition_info subdict
                # or this optimizer key is not one of the sharded optimizer keys
                _add_or_validate_unsharded_key(
                    optimizer_key,
                    optimizer_value,
                    state_dict[optimizer][model_state_key],
                    f"Value mismatch for model state {model_state_key} and optimizer state {optimizer_key}",
                )


def _reshape_states(sharded_states_original_dims, state_dict, mixed_precision_enabled):
    """Reshape model and optimizer states in the state_dict according to dimensions in sharded_states_original_dims"""

    model = _utils.state_dict_model_key()
    full_precision = _utils.state_dict_full_precision_key()
    optimizer = _utils.state_dict_optimizer_key()
    sharded_optimizer_keys = _utils.state_dict_sharded_optimizer_keys()

    for sharded_state_key, original_dim in sharded_states_original_dims.items():
        # reshape model states to original_dim only when mixed precision is enabled
        if mixed_precision_enabled and (model in state_dict):
            state_dict[model][full_precision][sharded_state_key] = state_dict[model][full_precision][
                sharded_state_key
            ].reshape(original_dim)

        # reshape optimizer states to original_dim
        if optimizer in state_dict:
            for optimizer_key, optimizer_value in state_dict[optimizer][sharded_state_key].items():
                if optimizer_key in sharded_optimizer_keys:
                    state_dict[optimizer][sharded_state_key][optimizer_key] = optimizer_value.reshape(original_dim)


def _aggregate_trainer_options(rank_state_dict, state_dict, partial_aggregation):
    """Extracts trainer options from rank_state_dict and loads them accordingly on state_dict"""
    trainer_options = _utils.state_dict_trainer_options_key()
    state_dict[trainer_options] = {}

    mixed_precision = _utils.state_dict_trainer_options_mixed_precision_key()
    zero_stage = _utils.state_dict_trainer_options_zero_stage_key()
    world_rank = _utils.state_dict_trainer_options_world_rank_key()
    world_size = _utils.state_dict_trainer_options_world_size_key()
    optimizer_name = _utils.state_dict_trainer_options_optimizer_name_key()
    D_size = _utils.state_dict_trainer_options_data_parallel_size_key()  # noqa: N806
    H_size = _utils.state_dict_trainer_options_horizontal_parallel_size_key()  # noqa: N806

    state_dict[trainer_options][mixed_precision] = rank_state_dict[trainer_options][mixed_precision]
    state_dict[trainer_options][zero_stage] = 0
    state_dict[trainer_options][world_rank] = rank_state_dict[trainer_options][world_rank] if partial_aggregation else 0
    state_dict[trainer_options][world_size] = 1
    state_dict[trainer_options][optimizer_name] = rank_state_dict[trainer_options][optimizer_name]
    state_dict[trainer_options][D_size] = 1
    state_dict[trainer_options][H_size] = 1


def _aggregate_megatron_partition_info(rank_state_dict, state_dict):
    """Extracts partition_info from rank_state_dict and loads on state_dict for megatron-partitioned weights"""
    partition_info = _utils.state_dict_partition_info_key()
    if partition_info not in state_dict:
        state_dict[partition_info] = {}

    rank_partition_info = rank_state_dict[partition_info]
    for model_state_key, partition_info_dict in rank_partition_info.items():
        if model_state_key not in state_dict[partition_info]:
            # add partition info only if weight is megatron partitioned
            if partition_info_dict["megatron_row_partition"] >= 0:
                state_dict[partition_info][model_state_key] = partition_info_dict


def _to_pytorch_format(state_dict):
    """Convert ORT state dictionary schema (hierarchical structure) to PyTorch state dictionary schema (flat structure)"""

    pytorch_state_dict = {}
    for model_state_key, model_state_value in state_dict[_utils.state_dict_model_key()][
        _utils.state_dict_full_precision_key()
    ].items():
        # convert numpy array to a torch tensor
        pytorch_state_dict[model_state_key] = torch.tensor(model_state_value)
    return pytorch_state_dict


def _get_parallellism_groups(data_parallel_size, horizontal_parallel_size, world_size):
    """Returns the D and H groups for the given sizes"""
    num_data_groups = world_size // data_parallel_size
    data_groups = []
    for data_group_id in range(num_data_groups):
        data_group_ranks = []
        for r in range(data_parallel_size):
            data_group_ranks.append(data_group_id + horizontal_parallel_size * r)  # noqa: PERF401
        data_groups.append(data_group_ranks)

    num_horizontal_groups = world_size // horizontal_parallel_size
    horizontal_groups = []
    for hori_group_id in range(num_horizontal_groups):
        hori_group_ranks = []
        for r in range(horizontal_parallel_size):
            hori_group_ranks.append(hori_group_id * horizontal_parallel_size + r)  # noqa: PERF401
        horizontal_groups.append(hori_group_ranks)

    return data_groups, horizontal_groups


def _aggregate_over_ranks(
    ordered_paths,
    ranks,
    sharded_states_original_dims=None,
    mode=_AGGREGATION_MODE.Zero,
    partial_aggregation=False,
    pytorch_format=True,
):
    """Aggregate checkpoint files over set of ranks and return a single state dictionary

    Args:
        ordered_paths: list of paths in the order in which they must be aggregated
        ranks: list of ranks that are to be aggregated
        sharded_states_original_dims: dict containing the original dims for sharded states that are persisted over
                                        multiple calls to _aggregate_over_ranks()
        mode: mode of aggregation: Zero or Megatron
        partial_aggregation: boolean flag to indicate whether to produce a partially
                                aggregated state which can be further aggregated over
        pytorch_format: boolean flag to select either ONNX Runtime or PyTorch state schema of the returned state_dict
    Returns:
        state_dict that can be loaded into an ORTTrainer or into a PyTorch model
    """
    state_dict = {}
    if sharded_states_original_dims is None:
        sharded_states_original_dims = dict()
    world_rank = _utils.state_dict_trainer_options_world_rank_key()
    mixed_precision = _utils.state_dict_trainer_options_mixed_precision_key()
    zero_stage = _utils.state_dict_trainer_options_zero_stage_key()
    world_size = _utils.state_dict_trainer_options_world_size_key()
    optimizer_name = _utils.state_dict_trainer_options_optimizer_name_key()

    loaded_mixed_precision = None
    loaded_world_size = None
    loaded_zero_stage = None
    loaded_optimizer_name = None

    for i, path in enumerate(ordered_paths):
        rank_state_dict = _checkpoint_storage.load(path)

        assert _utils.state_dict_partition_info_key() in rank_state_dict, "Missing information: partition_info"
        assert _utils.state_dict_trainer_options_key() in rank_state_dict, "Missing information: trainer_options"
        assert (
            ranks[i] == rank_state_dict[_utils.state_dict_trainer_options_key()][world_rank]
        ), "Unexpected rank in file at path {}. Expected {}, got {}".format(
            path, rank, rank_state_dict[_utils.state_dict_trainer_options_key()][world_rank]  # noqa: F821
        )
        if loaded_mixed_precision is None:
            loaded_mixed_precision = rank_state_dict[_utils.state_dict_trainer_options_key()][mixed_precision]
        else:
            assert (
                loaded_mixed_precision == rank_state_dict[_utils.state_dict_trainer_options_key()][mixed_precision]
            ), f"Mixed precision state mismatch among checkpoint files. File: {path}"
        if loaded_world_size is None:
            loaded_world_size = rank_state_dict[_utils.state_dict_trainer_options_key()][world_size]
        else:
            assert (
                loaded_world_size == rank_state_dict[_utils.state_dict_trainer_options_key()][world_size]
            ), f"World size state mismatch among checkpoint files. File: {path}"
        if loaded_zero_stage is None:
            loaded_zero_stage = rank_state_dict[_utils.state_dict_trainer_options_key()][zero_stage]
        else:
            assert (
                loaded_zero_stage == rank_state_dict[_utils.state_dict_trainer_options_key()][zero_stage]
            ), f"Zero stage mismatch among checkpoint files. File: {path}"
        if loaded_optimizer_name is None:
            loaded_optimizer_name = rank_state_dict[_utils.state_dict_trainer_options_key()][optimizer_name]
        else:
            assert (
                loaded_optimizer_name == rank_state_dict[_utils.state_dict_trainer_options_key()][optimizer_name]
            ), f"Optimizer name mismatch among checkpoint files. File: {path}"

        # aggregate all model states
        _aggregate_model_states(rank_state_dict, sharded_states_original_dims, state_dict, loaded_mixed_precision, mode)

        if not pytorch_format:
            # aggregate all optimizer states if pytorch_format is False
            _aggregate_optimizer_states(rank_state_dict, sharded_states_original_dims, state_dict, mode)

            # for D+H aggregation scenario, the first pass of aggregation(partial aggregation) is over D groups
            # to aggregate over Zero, and another pass to aggregate Megatron partitioned
            # states. Preserve the relevant partition info only for weights that are megatron partitioned for
            # a partial aggregation call
            if partial_aggregation:
                _aggregate_megatron_partition_info(rank_state_dict, state_dict)

            # entry for trainer_options in the state_dict to perform other sanity checks
            if _utils.state_dict_trainer_options_key() not in state_dict:
                _aggregate_trainer_options(rank_state_dict, state_dict, partial_aggregation)

            # entry for user_dict in the state_dict if not already present
            if (
                _utils.state_dict_user_dict_key() not in state_dict
                and _utils.state_dict_user_dict_key() in rank_state_dict
            ):
                state_dict[_utils.state_dict_user_dict_key()] = rank_state_dict[_utils.state_dict_user_dict_key()]

    # for a partial aggregation scenario, we might not have the entire tensor aggregated yet, thus skip reshape
    if not partial_aggregation:
        # reshape all the sharded tensors based on the original dimensions stored in sharded_states_original_dims
        _reshape_states(sharded_states_original_dims, state_dict, loaded_mixed_precision)

    # return a flat structure for PyTorch model in case pytorch_format is True
    # else return the hierarchical structure for ORTTrainer
    return _to_pytorch_format(state_dict) if pytorch_format else state_dict


def _aggregate_over_D_H(ordered_paths, D_groups, H_groups, pytorch_format):  # noqa: N802
    """Aggregate checkpoint files and return a single state dictionary for the D+H
    (Zero+Megatron) partitioning strategy.
    For D+H aggregation scenario, the first pass of aggregation(partial aggregation) is over D groups
    to aggregate over Zero, and another pass over the previously aggregated states
    to aggregate Megatron partitioned states.
    """
    sharded_states_original_dims = {}
    aggregate_data_checkpoint_files = []

    # combine for Zero over data groups and save to temp file
    with tempfile.TemporaryDirectory() as save_dir:
        for group_id, d_group in enumerate(D_groups):
            aggregate_state_dict = _aggregate_over_ranks(
                ordered_paths["D"][group_id],
                d_group,
                sharded_states_original_dims,
                partial_aggregation=True,
                pytorch_format=False,
            )

            filename = "ort.data_group." + str(group_id) + ".ort.pt"
            filepath = os.path.join(save_dir, filename)
            _checkpoint_storage.save(aggregate_state_dict, filepath)
            aggregate_data_checkpoint_files.append(filepath)

        assert len(aggregate_data_checkpoint_files) > 0

        # combine for megatron:
        aggregate_state = _aggregate_over_ranks(
            aggregate_data_checkpoint_files,
            H_groups[0],
            sharded_states_original_dims,
            mode=_AGGREGATION_MODE.Megatron,
            pytorch_format=pytorch_format,
        )

    return aggregate_state


def aggregate_checkpoints(paths, pytorch_format=True):
    """Aggregate checkpoint files and return a single state dictionary

    Aggregates checkpoint files specified by paths and loads them one at a time, merging
    them into a single state dictionary.
    The checkpoint files represented by paths must be saved through ORTTrainer.save_checkpoint() function.
    The schema of the state_dict returned will be in the same as the one returned by ORTTrainer.state_dict()

    Args:
        paths: list of more than one file represented as strings where the checkpoint is saved
        pytorch_format: boolean flag to select either ONNX Runtime or PyTorch state schema of the returned state_dict
    Returns:
        state_dict that can be loaded into an ORTTrainer or into a PyTorch model
    """

    loaded_trainer_options = _checkpoint_storage.load(paths[0], key=_utils.state_dict_trainer_options_key())
    D_size = _utils.state_dict_trainer_options_data_parallel_size_key()  # noqa: N806
    H_size = _utils.state_dict_trainer_options_horizontal_parallel_size_key()  # noqa: N806
    world_size = _utils.state_dict_trainer_options_world_size_key()

    D_size = loaded_trainer_options[D_size]  # noqa: N806
    H_size = loaded_trainer_options[H_size]  # noqa: N806
    world_size = loaded_trainer_options[world_size]
    D_groups, H_groups = _get_parallellism_groups(D_size, H_size, world_size)  # noqa: N806

    combine_zero = loaded_trainer_options[_utils.state_dict_trainer_options_zero_stage_key()] > 0
    combine_megatron = len(H_groups[0]) > 1

    # order the paths in the order of groups in which they must be aggregated according to
    # data-parallel groups and H-parallel groups obtained
    # eg: {'D': [[path_0, path_2],[path_1, path_3]], 'H': [[path_0, path_1],[path_2, path_3]]}
    ordered_paths = _order_paths(paths, D_groups, H_groups)

    aggregate_state = None
    if combine_zero and combine_megatron:
        aggregate_state = _aggregate_over_D_H(ordered_paths, D_groups, H_groups, pytorch_format)
    elif combine_zero:
        aggregate_state = _aggregate_over_ranks(
            ordered_paths["D"][0], D_groups[0], mode=_AGGREGATION_MODE.Zero, pytorch_format=pytorch_format
        )
    elif combine_megatron:
        aggregate_state = _aggregate_over_ranks(
            ordered_paths["H"][0], H_groups[0], mode=_AGGREGATION_MODE.Megatron, pytorch_format=pytorch_format
        )

    return aggregate_state


################################################################################
# Helper functions
################################################################################


def _load_single_checkpoint(ort_trainer, checkpoint_dir, checkpoint_prefix, is_partitioned, strict):
    checkpoint_name = _get_checkpoint_name(
        checkpoint_prefix,
        is_partitioned,
        ort_trainer.options.distributed.world_rank,
        ort_trainer.options.distributed.world_size,
    )
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)

    if is_partitioned:
        assert_msg = (
            f"Couldn't find checkpoint file {checkpoint_file}."
            " Optimizer partitioning is enabled using ZeRO. Please make sure the checkpoint file exists "
            f"for rank {ort_trainer.options.distributed.world_rank} of {ort_trainer.options.distributed.world_size}"
        )
    else:
        assert_msg = f"Couldn't find checkpoint file {checkpoint_file}."
    assert os.path.exists(checkpoint_file), assert_msg

    checkpoint_state = torch.load(checkpoint_file, map_location="cpu")
    experimental_load_state_dict(ort_trainer, checkpoint_state["model"], strict=strict)
    del checkpoint_state["model"]
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
        checkpoint_state = torch.load(checkpoint_file, map_location="cpu")
        del checkpoint_state["model"]
        all_checkpoint_states.update(checkpoint_state)
    return all_checkpoint_states


def _list_checkpoint_files(checkpoint_dir, checkpoint_prefix, extension=".ort.pt"):
    ckpt_file_names = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix)]
    ckpt_file_names = [f for f in ckpt_file_names if f.endswith(extension)]
    ckpt_file_names = [os.path.join(checkpoint_dir, f) for f in ckpt_file_names]

    assert len(ckpt_file_names) > 0, f"No checkpoint found with prefix '{checkpoint_prefix}' at '{checkpoint_dir}'"
    return ckpt_file_names


def _get_checkpoint_name(prefix, is_partitioned, world_rank=None, world_size=None):
    SINGLE_CHECKPOINT_FILENAME = "{prefix}.ort.pt"  # noqa: N806
    MULTIPLE_CHECKPOINT_FILENAME = "{prefix}.ZeRO.{world_rank}.{world_size}.ort.pt"  # noqa: N806

    if is_partitioned:
        filename = MULTIPLE_CHECKPOINT_FILENAME.format(
            prefix=prefix, world_rank=world_rank, world_size=(world_size - 1)
        )
    else:
        filename = SINGLE_CHECKPOINT_FILENAME.format(prefix=prefix)
    return filename


def _split_state_dict(state_dict):
    optimizer_keys = ["Moment_1_", "Moment_2_", "Update_Count_", "Step"]
    split_sd = {"optimizer": {}, "fp32_param": {}, "fp16_param": {}}
    for k, v in state_dict.items():
        mode = "fp32_param"
        for optim_key in optimizer_keys:
            if k.startswith(optim_key):
                mode = "optimizer"
                break
        if k.endswith("_fp16"):
            mode = "fp16_param"
        split_sd[mode][k] = v
    return split_sd


class _CombineZeroCheckpoint:
    def __init__(self, checkpoint_files, clean_state_dict=None):
        assert len(checkpoint_files) > 0, "No checkpoint files passed"
        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split("ZeRO")[1].split(".")[2]) + 1
        assert len(self.checkpoint_files) == self.world_size, f"Could not find {self.world_size} files"
        self.weight_shape_map = {}
        self.sharded_params = set()

    def _split_name(self, name: str):
        name_split = name.split("_view_")
        view_num = None
        if len(name_split) > 1:
            view_num = int(name_split[1])
        optimizer_key = ""
        mp_suffix = ""
        if name_split[0].startswith("Moment_1"):
            optimizer_key = "Moment_1_"
        elif name_split[0].startswith("Moment_2"):
            optimizer_key = "Moment_2_"
        elif name_split[0].startswith("Update_Count"):
            optimizer_key = "Update_Count_"
        elif name_split[0].endswith("_fp16"):
            mp_suffix = "_fp16"
        param_name = name_split[0]
        if optimizer_key:
            param_name = param_name.split(optimizer_key)[1]
        param_name = param_name.split("_fp16")[0]
        return param_name, optimizer_key, view_num, mp_suffix

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
            weight_name, optimizer_key, view_num, mp_suffix = self._split_name(k)
            if view_num is not None:
                # parameter is sharded
                param_name = optimizer_key + weight_name + mp_suffix

                if param_name in self.aggregate_state_dict and optimizer_key not in ["Update_Count_"]:
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
        warnings.warn(
            "_CombineZeroCheckpoint.aggregate_checkpoints() will be deprecated soon. "
            "Please use aggregate_checkpoints() instead.",
            DeprecationWarning,
        )

        checkpoint_prefix = self.checkpoint_files[0].split(".ZeRO")[0]
        self.aggregate_state_dict = dict()

        for i in range(self.world_size):
            checkpoint_name = _get_checkpoint_name(checkpoint_prefix, True, i, self.world_size)
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            if "model" in rank_state_dict:
                rank_state_dict = rank_state_dict["model"]

            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)

            rank_state_dict = _split_state_dict(rank_state_dict)
            self._aggregate(rank_state_dict["fp16_param"])
            self._aggregate(rank_state_dict["fp32_param"])
            self._aggregate(rank_state_dict["optimizer"])

        for k in self.sharded_params:
            self._reshape_tensor(k)
        return self.aggregate_state_dict
