import os

import torch.distributed as dist

comm = None


def init_dist():
    if "LOCAL_RANK" in os.environ:
        int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:7645", world_size=world_size, rank=rank)
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD  # noqa: F841

        int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:7647", world_size=world_size, rank=rank)
    else:
        # don't need to do init for single process
        pass


def get_rank():
    return comm.Get_rank() if comm is not None else 0


def get_size():
    return comm.Get_size() if comm is not None else 1


def barrier():
    if comm is not None:
        comm.Barrier()


def print_out(*args):
    if get_rank() == 0:
        print(*args)
