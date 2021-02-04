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

class Mlp(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(Mlp, self).__init__()
        self.linear0 = nn.Linear(d_in, d_hidden)
        self.linear1 = nn.Linear(d_hidden, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.linear3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h0 = self.linear0(x)
        h1 = h0.relu()
        h2 = self.linear1(h1)
        h3 = h2.relu()
        h4 = self.linear2(h3)
        h5 = h4.relu()
        h6 = self.linear3(h5)
        return h6


n = 10
d_in = 2
d_hidden = 2
d_out = 2

# Input.
x = torch.rand((n, d_in))

# Output
y = torch.randint(0, d_out, (n,))

# Modeling.
loss = nn.CrossEntropyLoss(reduction='sum')
model = Mlp(d_in, d_hidden, d_out)


def apply_loss(p, y):
    return loss(p, y)


# Number of micro-batches.
num_pipeline_steps = 10
# Number of pipeline stages.
num_pipeline_stages = 2

# Compute batch size for micro-batches.
n_slice = int(n / num_pipeline_steps)

cuda_device = 'cuda:' + str(rank)
# Schema used when running the original batch.
schema = {'inputs': [('x', ['n', 'd_in']), ('target', ['n'])], 'outputs': [
    ('loss', [], True), ('output', ['n', d_out])]}
# Actual schema used when running micro-batches.
pipeline_schema = {'x': [n_slice, d_in], 'target': [
    n_slice], 'output': [n_slice, d_out], 'loss': []}
# Describe which axis to slice along for each sliced tensor.
sliced_axes = {'x': 0, 'target': 0, 'output': 0}

adam_config = optim.AdamConfig(lr=0.1)

# Specify configuration for pipeline parallel training.
trainer_config = ORTTrainerOptions({
    'batch': {
        'gradient_accumulation_steps': num_pipeline_steps
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
        'data_parallel_size': int(total_ranks / num_pipeline_stages),
        'horizontal_parallel_size': 1,
        'pipeline_parallel': {
            'pipeline_parallel_size': int(num_pipeline_stages),
            'num_pipeline_micro_batches': num_pipeline_steps,
            'sliced_schema': pipeline_schema,
            'sliced_axes': sliced_axes,
            'sliced_tensor_names': ['x', 'target', 'output'],
            # Define pipeline stage partition by specifying cut points.
            # 2-stage cut. It's a cut on tensor "12".
            'pipeline_cut_info_string': '12'
        },
        'allreduce_post_accumulation': True
    }
})

trainer = ORTTrainer(model, schema, adam_config, apply_loss, trainer_config)

loss_history = []
for i in range(5):
    l, p = trainer.train_step(x.to(cuda_device), y.to(cuda_device))
    loss_history.append(l)

print('loss history: ', loss_history)

# Valid ranks are [0, 1, 2, 3].
# [0, 2] forms the 2-stage pipeline in the 1st data parallel group.
# [1, 3] forms the 2-stage pipeline in the 2nd data parallel group.
last_pipeline_stage_ranks = [2, 3]

# The loss values computed at the last pipeline stages. Note that intermediate
# stages may not have valid loss values, so we don't check them.
expected_loss_history = [0.9420, 0.6608, 0.8944, 1.2279, 1.1173]
if rank in last_pipeline_stage_ranks:
    for result, expected in zip(loss_history, expected_loss_history):
        assert torch.allclose(result.cpu(), torch.Tensor([expected], device='cpu'), 1e-03)
