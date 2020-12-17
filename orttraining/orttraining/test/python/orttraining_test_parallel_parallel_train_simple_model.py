import argparse
import time
from onnxruntime.training import ORTTrainer, ORTTrainerOptions, optim
from torch import nn
import torch
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total_ranks = comm.Get_size()

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s', type=int, help='number of pipeline steps')
parser.add_argument('-n', type=int, help='number of pipeline stages')
args = parser.parse_args()


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
d_in = 8
d_hidden = 8
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


# Load number of stages from command line args.
# # of micro-batches.
num_pipeline_steps = args.s
# Compute batch size for sub-batches.
n_slice = int(n / num_pipeline_steps)

cuda_device = 'cuda:' + str(rank)
# Schema used when running the original batch.
schema = {'inputs': [('x', ['n', 'd_in']), ('target', ['n'])], 'outputs': [
    ('loss', [], True), ('output', ['n', d_out])]}
# Actual schema used when running sub-batches.
pipeline_schema = {'x': [n_slice, d_in], 'target': [
    n_slice], 'output': [n_slice, d_out], 'loss': []}
# Describe which axis to slice along for each sliced tensor.
sliced_axes = {'x': 0, 'target': 0, 'output': 0}

adam_config = optim.AdamConfig(lr=0.1)
num_pipeline_stages = args.n

# # Specify configuration for pipeline parallel training.
trainer_config = ORTTrainerOptions({
    'batch': {
        'gradient_accumulation_steps': num_pipeline_steps
    },
    'device': {
        'id': cuda_device
    },
    'distributed': {
        'world_size': total_ranks,
        'world_rank': rank,
        'data_parallel_size': int(total_ranks / num_pipeline_stages),
        'horizontal_parallel_size': 1,
        'pipeline_parallel_size': int(num_pipeline_stages),
        'num_pipeline_micro_batches': num_pipeline_steps,
        'sliced_schema': pipeline_schema,
        'sliced_axes': sliced_axes,
        'sliced_tensor_names': ['x', 'target', 'output'],
        'allreduce_post_accumulation': True
    }
})


# Define pipeline stage partition by specifying cut points.
if num_pipeline_stages == 2:
    # 2-stage cut. It's a cut on tensor "12".
    trainer_config.distributed.pipeline_cut_info_string = '12'
elif num_pipeline_stages == 3:
    # 3-stage cut. There is one cut on tensor "11" and other cut on tensor "15".
    trainer_config.distributed.pipeline_cut_info_string = '11,15'

trainer = ORTTrainer(model, schema, adam_config, apply_loss, trainer_config)

loss_history = []
for i in range(5):
    l, p = trainer.train_step(x.to(cuda_device), y.to(cuda_device))
    loss_history.append(l)

# Valid ranks are [0, 1, 2, 3].
# [0, 2] forms the 2-stage pipeline in the 1st data parallel group.
# [1, 3] forms the 2-stage pipeline in the 2nd data parallel group.
last_pipeline_stage_ranks = [2, 3]
# The loss values computed at the last pipeline stages. Note that intermediate
# stages may not have valid loss values, so we don't check them.
expected_loss_history = [0.8659626245, 0.6528335810, 0.2935703397, 0.0308648348, 0.0003252029]
if rank in last_pipeline_stage_ranks:
    for result, expected in zip(loss_history, expected_loss_history):
        assert torch.allclose(result.cpu(), torch.Tensor([expected], device='cpu'))