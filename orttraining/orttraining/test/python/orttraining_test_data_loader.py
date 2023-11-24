import random
from enum import Enum

import torch
from torch.utils.data import DataLoader, Dataset

global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


def generate_sample(desc, device=None):
    """Generate a sample based on the description"""
    # symbolic dimensions are described with strings. set symbolic dimensions to be 1
    size = [s if isinstance(s, (int)) else 1 for s in desc.shape_]
    if desc.num_classes_:
        return torch.randint(0, desc.num_classes_, size, dtype=desc.dtype_).to(device)
    else:
        return torch.randn(size, dtype=desc.dtype_).to(device)


class OrtTestDataset(Dataset):
    def __init__(self, input_desc, seq_len, dataset_len, device):
        import copy

        self.input_desc_ = copy.deepcopy(input_desc)
        for input_desc in self.input_desc_:
            shape_ = []
            for i, axis in enumerate(input_desc.shape_):
                if axis == "max_seq_len_in_batch":
                    shape_ = [*shape_, seq_len]
                elif axis != "batch":
                    shape_ = input_desc.shape_[i]
            input_desc.shape_ = shape_
        self.dataset_len_ = dataset_len
        self.device_ = device

    def __len__(self):
        return self.dataset_len_

    def __getitem__(self, item):
        input_batch = []
        for input_desc in self.input_desc_:
            input_sample = generate_sample(input_desc, self.device_)
            input_batch.append(input_sample)
        return input_batch


def create_ort_test_dataloader(input_desc, batch_size, seq_len, dataset_len, device):
    dataset = OrtTestDataset(input_desc, seq_len, dataset_len, device)
    return DataLoader(dataset, batch_size=batch_size)


class BatchArgsOption(Enum):
    List = 1
    Dict = 2
    ListAndDict = 3


def split_batch(batch, input_desc, args_count):
    total_argument_count = len(input_desc)
    # batch=[input_ids[batch, seglen], attention_mask[batch, seglen], token_type_ids[batch,seglen], token_type_ids[batch, seglen]]
    args = []  # (input_ids[batch, seglen], attention_mask[batch, seglen])
    kwargs = {}  # {'token_type_ids': token_type_ids[batch,seglen], 'position_ids': token_type_ids[batch, seglen]}
    for i in range(args_count):
        args = [*args, batch[i]]

    for i in range(args_count, total_argument_count):
        kwargs[input_desc[i].name_] = batch[i]

    return args, kwargs
