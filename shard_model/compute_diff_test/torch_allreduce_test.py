import torch
import os
import argparse
import numpy as np
import pickle

INPUT_DATA_FILE = 'input-data-{}.pkl'

def main(args):
    rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    print(f'world_size: {world_size}, rank: {rank}')

    torch.distributed.init_process_group('nccl', init_method="tcp://127.0.0.1:12345", world_size=world_size, rank=rank)

    # load input data of each rank
    input_datas = []
    for i in range(args.size):
        with open(INPUT_DATA_FILE.format(i), 'rb') as fp:
            input_datas.append(pickle.load(fp))

    # call allreduce of torch
    torch_data = torch.tensor(input_datas[rank], device=rank)

    # calc true result
    need_y = input_datas[0]
    for i in input_datas[1:]:
        need_y = need_y.astype(np.float32) + i.astype(np.float32)

    need_y = need_y.astype(args.type)


    torch.distributed.all_reduce(torch_data, op=torch.distributed.ReduceOp.SUM)
    y = torch_data.cpu().numpy()

    if np.allclose(need_y, y):
        print(f'SAME. shape: {y[0].shape}')
    else:
        diff = abs(need_y - y)
        rel_diff = abs(diff / y)
        print(f'NOT SAME. shape: {y.shape} diff: {diff.max()} rel-diff: {rel_diff.max()}')


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--type', type=str, default='float16')
    parser.add_argument('--size', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)

