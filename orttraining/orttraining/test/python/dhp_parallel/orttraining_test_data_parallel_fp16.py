# Add onnxruntime path as system path.
# Otherwise, "import onnxruntime" may fail.
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime
import torch
from onnxruntime.capi._pybind_state import get_mpi_context_world_rank, get_mpi_context_world_size
from onnxruntime.training import amp, ORTTrainer, ORTTrainerOptions, optim
from torch import nn

# Get MPI setting.
rank = get_mpi_context_world_rank()
total_ranks = get_mpi_context_world_size()

torch.manual_seed(0)
cuda_device = 'cuda:' + str(rank)

model, model_desc, my_loss, batcher_fn, train_data, _, _ = _test_commons._load_pytorch_transformer_model(cuda_device)
optim_config = optim.LambConfig(lr=0.001, max_norm_clip=max_norm_clip)
gradient_accumulation_steps = 8
total_steps = 12

# Specify configuration for pipeline parallel training.
options = ORTTrainerOptions({
    'batch': {
        'gradient_accumulation_steps': gradient_accumulation_steps
    },
    'device': {
        'id': cuda_device
    },
    'mixed_precision': {
        'enabled': True,
        'loss_scaler': amp.DynamicLossScaler()
    },
    'distributed': {
        'world_size': total_ranks,
        'world_rank': rank,
        'data_parallel_size': int(total_ranks),
        'horizontal_parallel_size': 1,
        'allreduce_post_accumulation': True，
    },
    'debug'： {'deterministic_compute' : True}
})

trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=options)

# Training loop
actual_loss = []
for i in range(total_steps):
    data, targets = batcher_fn(train_data, i)
    loss, _ = trainer.train_step(data, targets)
    actual_loss.append(loss.cpu())

expected_loss = [0.9420, 0.6608, 0.8944, 1.2279, 1.1173]
rtol = 1e-5
_test_helpers.assert_model_outputs(expected_loss, actual_loss, rtol=rtol)
