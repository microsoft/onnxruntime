# adapted from run_glue.py of huggingface transformers

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import unittest
import numpy as np
from numpy.testing import assert_allclose

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import torch
import argparse

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

import sys
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

class TestDataset(Dataset):
    def __init__(self):
        self.items_ = []
        for i in range(0, 100):
            self.items_.append(torch.tensor([i]))

    def __len__(self):
        return len(self.items_)

    def __getitem__(self, item):
        index = item % len(self.items_)
        return self.items_[index]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("sys.argv: ", sys.argv)
    args = parse_arguments()
    if args.local_rank == -1:
        local_rank = args.local_rank
        # # mpi launch
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # local_rank = comm.Get_rank()
        # world_rank = comm.Get_rank()
        # world_size = comm.Get_size()
        # print("mpirun local_rank: ", local_rank)
        # print("mpirun world_rank: ", world_rank)
        # print("mpirun world_size: ", world_size)
        # torch.distributed.init_process_group(backend="mpi", world_size=world_size, rank=world_rank)
        torch.distributed.init_process_group(backend="mpi")
    else:
        local_rank = args.local_rank
        print("torch.distributed.launch local_rank: ", local_rank)
        torch.distributed.init_process_group(backend="nccl")

    train_dataset = TestDataset()
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=3, sampler=sampler)
    for epoch in range(0, 3):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, inputs in enumerate(epoch_iterator):
            print(inputs)

