import torch
from torch import nn, optim
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.utils import RepeatingLoader

import onnxruntime
from onnxruntime.training import ORTModule

import argparse

#import pdb; pdb.set_trace()

# USAGE:
# pip install deepspeed
# deepspeed orttraining_test_ortmodule_deepspeed_pipeline_parallel.py --deepspeed_config=ds_config_ortmodule_deepspeed_pipeline_parallel.json --pipeline-parallel-size 2 --steps=100


class SampleData(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        print("size of data ",y.size(), x.size())
        return x.size()[0] #10
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=0, help='PRNG seed')
    parser.add_argument('--fp16',type=bool,default=False,help='fp16 run')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

n = 20 #10
d_in = 4
d_hidden = 8
d_out = 3
args = get_args()

torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

dist.init_process_group(backend=args.backend)
torch.manual_seed(args.seed)
# Model.
# ORT module
model = nn.Sequential(
			ORTModule(nn.Linear(d_in, d_hidden).to(device)),     # Stage 1
			ORTModule(nn.ReLU().to(device)),                     # Stage 1
			ORTModule(nn.Linear(d_hidden, d_hidden).to(device)), # Stage 1
			ORTModule(nn.ReLU().to(device)),                     # Stage 1
			ORTModule(nn.Linear(d_hidden, d_hidden).to(device)), # Stage 2
			ORTModule(nn.ReLU().to(device)),                     # Stage 2
			ORTModule(nn.Linear(d_hidden, d_out).to(device))     # Stage 2
			)




# Enable interactions with Deepspeed LayerSpec for future
# model = [ 
#        LayerSpec(nn.Linear,d_in, d_hidden),     # Stage 1
#	LayerSpec(nn.ReLU),                     # Stage 1
#	LayerSpec(nn.Linear,d_hidden, d_hidden), # Stage 1
#	LayerSpec(nn.ReLU),                     # Stage 1
#	LayerSpec(nn.Linear,d_hidden, d_hidden), # Stage 2
#	LayerSpec(nn.ReLU),                     # Stage 2
#	LayerSpec(nn.Linear,d_hidden, d_out)     # Stage 2
#        ]

model = PipelineModule(layers=model,
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     num_stages=args.pipeline_parallel_size,
                     partition_method='uniform', #'parameters',
                     activation_checkpoint_interval=0
                     )

params = [p for p in model.parameters() if p.requires_grad]

# Input.
x = torch.rand((n, d_in))
if args.fp16:
    x = x.half()

# Output
y = torch.randint(0, d_out, (n,))

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=params,
                                                     training_data=ds #(x,y)#
                                                     )
for step in range(args.steps):
    loss = model_engine.train_batch()
