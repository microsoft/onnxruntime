#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import os
import shutil
import sys
from checkpoint._test_helpers import makedir
from _test_commons import _single_run, _distributed_run

checkpoint_dir = os.path.abspath('checkpoint/checkpoint_dir/')
makedir(checkpoint_dir)

# test workflow:
# - there are a total of three files that are used for checkpointing tests:
#     - orttraining_test_checkpoint.py: co-ordinating all the checkpoint tests
#     - orttraining_test_save_checkpoint.py: responsible for saving all checkpoint files and trained states
#     - orttraining_test_load_checkpoint.py: loading the saved checkpoints and the saved states and asserting whether
#       the saved states match the loaded states.
# - and tests encompassing checkpointing tests for scenarios:
#     - from [onnxruntime orttrainer][full_precision, mixed_precision][single node training, data parallel training, distributed zero, distributed megatron, distributed zero+megatron training] to
#       [onnxruntime orttrainer, pytorch][full_precision, mixed_precision][single node training, data parallel training, distributed zero, distributed megatron, distributed zero+megatron training]
# - all tests cannot be written in the same process because:
#     - some of them require to be run in a distributed environment (using mpirun) while others can be run using a single process.
#     - there is a known limitation where the distributed training run context is implemented as a singleton, so in the same process, no more than one
#       orttrainer can be instantiated. Hence the need to run these tests in different processes one at a time.
# - workflow:
#     - orttraining_test_checkpoint.py calls orttraining_test_save_checkpoint.py to save following files to disk
#         - ORTTrainer checkpoint files through the ORTTrainer.save_checkpoint method
#         - ORTTrainer states through pickle after extracting all the states of the ORTTrainer through the ORTTrainer.state_dict method
#         - for each configuration across [onnxruntime orttrainer][full_precision, mixed_precision][single node training, data parallel training, distributed zero training]
#     - orttraining_test_checkpoint.py calls orttraining_test_load_checkpoint.py to load each checkpoint into each orttrainer configuration
#         - Saved ORTTrainer checkpoint files are loaded into an ORTTrainer using the ORTTrainer.load_checkpoint method for each ORTTrainer configuration.
#         - Saved states are loaded into a python dictionary (called the state dictionary) through pickle
#         - state dictionary is extracted from the ORTTrainer after it has loaded the checkpoint file and the onnx graph has been initialized (by calling eval_step)
#           through the ORTTrainer.state_dict method.
#         - the loaded state dictionary (through pickle) is compared against the extracted state dictionary for:
#             - equality (or near equality) of model states
#             - equality (or near equality) of optimizer states
#         - In some cases the comparison is not directly possible; for example single node trainer to a distributed zero trainer because the extracted state
#           dictionary is a distributed one and cannot be compared against a single node trainer directly.
#             - First these states are saved using pickle for each rank to a file on disk
#             - Wait for all ranks to complete writing the file to disk using barrier()
#             - Load all states and aggregate them into 1 state dictionary
#             - Compare this aggregated state dictionary against the original one loaded from disk.
#         - Similarly, it is not possible to compare mixed precision zero trainer state_dict against full precision zero trainer state_dict because the
#           full precision states are sharded in the mixed precision trainer run and not shareded in the full precision trainer run. To compare these two state_dicts:
#             - Both state_dicts (mixed precision and full precision) are saved to file for all ranks.
#             - Wait for all ranks to complete writing the file to disk using barrier()
#             - Load all states and aggregate them into 1 state dictionary fpr both the configs.
#             - Compare this aggregated state dictionaries against one another.

save_checkpoint_file = os.path.join('checkpoint', 'orttraining_test_save_checkpoint.py')
load_checkpoint_file = os.path.join('checkpoint', 'orttraining_test_load_checkpoint.py')
aggregate_checkpoint_file = os.path.join('checkpoint', 'orttraining_test_checkpoint_aggregation.py')
optim_state_file = os.path.join('checkpoint', 'orttraining_test_load_optimizer_state.py')
backend_api_file = os.path.join('checkpoint', 'orttraining_test_backend_api.py')

single_node_full_precision_path = os.path.join(checkpoint_dir, 'single_node', 'full_precision')
single_node_mixed_precision_path = os.path.join(checkpoint_dir, 'single_node', 'mixed_precision')
distributed_zero_full_precision_lamb_path = os.path.join(checkpoint_dir, 'distributed_zero', 'full_precision', 'lamb')
distributed_zero_mixed_precision_lamb_path = os.path.join(checkpoint_dir, 'distributed_zero', 'mixed_precision', 'lamb')

# megatron saving and loading uses a different model
single_node_full_precision_bart_path = os.path.join(checkpoint_dir, 'bart', 'single_node', 'full_precision')
single_node_mixed_precision_bart_path = os.path.join(checkpoint_dir, 'bart', 'single_node', 'mixed_precision')
distributed_zero_full_precision_lamb_bart_path = os.path.join(checkpoint_dir, 'bart', 'distributed_zero', 'full_precision', 'lamb')
distributed_zero_mixed_precision_lamb_bart_path = os.path.join(checkpoint_dir, 'bart', 'distributed_zero', 'mixed_precision', 'lamb')
distributed_megatron_full_precision_lamb_path = os.path.join(checkpoint_dir, 'bart', 'distributed_megatron', 'full_precision', 'lamb')
distributed_megatron_mixed_precision_lamb_path = os.path.join(checkpoint_dir, 'bart', 'distributed_megatron', 'mixed_precision', 'lamb')
distributed_zero_megatron_full_precision_adam_path = os.path.join(checkpoint_dir, 'bart', 'distributed_zero_megatron', 'full_precision', 'adam')
distributed_zero_megatron_mixed_precision_adam_path = os.path.join(checkpoint_dir, 'bart', 'distributed_zero_megatron', 'mixed_precision', 'adam')
distributed_zero_megatron_full_precision_lamb_path = os.path.join(checkpoint_dir, 'bart', 'distributed_zero_megatron', 'full_precision', 'lamb')
distributed_zero_megatron_mixed_precision_lamb_path = os.path.join(checkpoint_dir, 'bart', 'distributed_zero_megatron', 'mixed_precision', 'lamb')

# save all checkpoint files (pre-checkpoint)
_single_run(save_checkpoint_file, 'single_node_full_precision', single_node_full_precision_path)
_single_run(save_checkpoint_file, 'single_node_mixed_precision', single_node_mixed_precision_path)
_distributed_run(save_checkpoint_file, 'distributed_zero_full_precision_lamb', distributed_zero_full_precision_lamb_path)
_distributed_run(save_checkpoint_file, 'distributed_zero_mixed_precision_lamb', distributed_zero_mixed_precision_lamb_path)

_single_run(save_checkpoint_file, 'single_node_full_precision_bart', single_node_full_precision_bart_path)
_single_run(save_checkpoint_file, 'single_node_mixed_precision_bart', single_node_mixed_precision_bart_path)
_distributed_run(save_checkpoint_file, 'distributed_zero_full_precision_lamb_bart', distributed_zero_full_precision_lamb_bart_path)
_distributed_run(save_checkpoint_file, 'distributed_zero_mixed_precision_lamb_bart', distributed_zero_mixed_precision_lamb_bart_path)

_distributed_run(save_checkpoint_file, 'distributed_megatron_full_precision_lamb', distributed_megatron_full_precision_lamb_path)
_distributed_run(save_checkpoint_file, 'distributed_megatron_mixed_precision_lamb', distributed_megatron_mixed_precision_lamb_path)
_distributed_run(save_checkpoint_file, 'distributed_zero_megatron_full_precision_lamb', distributed_zero_megatron_full_precision_lamb_path)
_distributed_run(save_checkpoint_file, 'distributed_zero_megatron_mixed_precision_lamb', distributed_zero_megatron_mixed_precision_lamb_path)

# load checkpoint files (post-checkpoint)
# going to single node trainer
_single_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_single_node_full_precision', single_node_full_precision_path)
_single_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_single_node_full_precision', single_node_mixed_precision_path)
_single_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_single_node_mixed_precision', single_node_mixed_precision_path)
_single_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_single_node_mixed_precision', single_node_full_precision_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_single_node_full_precision', distributed_zero_full_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision', distributed_zero_mixed_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision', distributed_zero_mixed_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision', distributed_zero_full_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_single_node_full_precision', distributed_megatron_full_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_single_node_full_precision', distributed_megatron_mixed_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_single_node_mixed_precision', distributed_megatron_mixed_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_single_node_mixed_precision', distributed_megatron_full_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_single_node_full_precision', distributed_zero_megatron_full_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_full_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_single_node_mixed_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_single_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_single_node_mixed_precision', distributed_zero_megatron_full_precision_lamb_path)

# going to distributed zero trainer
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_distributed_zero_full_precision', single_node_full_precision_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision', single_node_mixed_precision_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision', single_node_mixed_precision_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision', single_node_full_precision_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision', distributed_zero_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision', distributed_zero_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision', distributed_zero_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision', distributed_zero_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_distributed_zero_full_precision', distributed_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_full_precision', distributed_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_mixed_precision', distributed_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_distributed_zero_mixed_precision', distributed_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_full_precision', distributed_zero_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_full_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_mixed_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_mixed_precision', distributed_zero_megatron_full_precision_lamb_path)

# going to distributed zero+megatron trainer
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_distributed_megatron_full_precision', single_node_full_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_distributed_megatron_full_precision', single_node_mixed_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_distributed_megatron_mixed_precision', single_node_mixed_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_distributed_megatron_mixed_precision', single_node_full_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_distributed_megatron_full_precision', distributed_zero_full_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_full_precision', distributed_zero_mixed_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_distributed_megatron_mixed_precision', distributed_zero_mixed_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_distributed_megatron_mixed_precision', distributed_zero_full_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_distributed_megatron_full_precision', distributed_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_full_precision', distributed_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_distributed_megatron_mixed_precision', distributed_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_distributed_megatron_mixed_precision', distributed_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_full_precision', distributed_zero_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_full_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_megatron_mixed_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_distributed_megatron_mixed_precision', distributed_zero_megatron_full_precision_lamb_path)

# going to distributed zero+megatron trainer
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_distributed_zero_megatron_full_precision', single_node_full_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_full_precision', single_node_mixed_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_mixed_precision_into_distributed_zero_megatron_mixed_precision', single_node_mixed_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_single_node_full_precision_into_distributed_zero_megatron_mixed_precision', single_node_full_precision_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_full_precision', distributed_zero_full_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_full_precision', distributed_zero_mixed_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_megatron_mixed_precision', distributed_zero_mixed_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_full_precision_into_distributed_zero_megatron_mixed_precision', distributed_zero_full_precision_lamb_bart_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_full_precision', distributed_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_full_precision', distributed_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision', distributed_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_megatron_full_precision_into_distributed_zero_megatron_mixed_precision', distributed_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_full_precision', distributed_zero_megatron_full_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_full_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_mixed_precision_into_distributed_zero_megatron_mixed_precision', distributed_zero_megatron_mixed_precision_lamb_path)
_distributed_run(load_checkpoint_file, 'test_load_from_distributed_zero_megatron_full_precision_into_distributed_zero_megatron_mixed_precision', distributed_zero_megatron_full_precision_lamb_path)

shutil.rmtree(checkpoint_dir)
