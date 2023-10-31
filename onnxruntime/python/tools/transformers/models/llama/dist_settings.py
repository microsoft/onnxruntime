import os

import torch
import torch.distributed as dist
from mpi4py import MPI


def init_dist():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:7647", world_size=world_size, rank=rank)
    device = torch.device(local_rank)
    return device


comm = MPI.COMM_WORLD


def get_rank():
    return comm.Get_rank()


def get_size():
    return comm.Get_size()


def barrier():
    comm.Barrier()


def print_out(*args):
    if get_rank() == 0:
        print(*args)
