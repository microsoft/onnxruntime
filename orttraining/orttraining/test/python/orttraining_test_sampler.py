# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_sampler.py

import torch
from onnxruntime.training.utils.data import sampler
import random


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def test_load_balancing_data_sampler_balances_load():
    samples_and_complexities = [(torch.FloatTensor([val]), torch.randint(0, 100, (1,)).item()) for val in range(100)]
    dataset = MyDataset(samples_and_complexities)

    def complexity_fn(sample):
        return sample[1]

    data_sampler0 = sampler.LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=2, rank=0, shuffle=False
    )
    data_sampler1 = sampler.LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=2, rank=1, shuffle=False
    )

    largest_complexity = -1
    for index in data_sampler0:
        assert samples_and_complexities[index][1] >= largest_complexity
        largest_complexity = samples_and_complexities[index][1]

    largest_complexity = -1
    for index in data_sampler1:
        assert samples_and_complexities[index][1] >= largest_complexity
        largest_complexity = samples_and_complexities[index][1]


def test_load_balancing_data_sampler_shuffles_and_balances_load():
    complexities = []
    for i in range(50):
        c = torch.randint(0, 100, (1,)).item()
        complexities.append(c)
        complexities.append(c)
    random.shuffle(complexities)

    samples = [torch.FloatTensor([val]) for val in range(100)]
    samples_and_complexities = list(zip(samples, complexities))
    dataset = MyDataset(samples_and_complexities)

    def complexity_fn(sample):
        return sample[1]

    data_sampler0 = sampler.LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=2, rank=0, shuffle=True
    )
    data_sampler1 = sampler.LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=2, rank=1, shuffle=True
    )

    for index0, index1 in zip(data_sampler0, data_sampler1):
        assert samples_and_complexities[index0][1] == samples_and_complexities[index1][1]


def test_load_balancing_data_sampler_sorts_in_groups():
    samples_and_complexities = [(torch.FloatTensor([val]), torch.randint(0, 100, (1,)).item()) for val in range(100)]
    dataset = MyDataset(samples_and_complexities)

    def complexity_fn(sample):
        return sample[1]

    group_size = 8
    samples_and_complexities_sorted = samples_and_complexities.copy()
    for begin_index in range(0, len(samples_and_complexities), group_size):
        end_index = min(begin_index + group_size, len(samples_and_complexities))
        samples_and_complexities_sorted[begin_index:end_index] = sorted(
            samples_and_complexities_sorted[begin_index:end_index], key=lambda x: x[1]
        )

    data_sampler = sampler.LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=1, rank=0, shuffle=False, group_size=8
    )

    for index, sorted_sample in zip(data_sampler, samples_and_complexities_sorted):
        assert samples_and_complexities[index][1] == sorted_sample[1]


def test_load_balancing_data_sampler_sorts_and_shuffles_in_groups():
    samples_and_complexities = [(torch.FloatTensor([val]), torch.randint(0, 100, (1,)).item()) for val in range(100)]
    dataset = MyDataset(samples_and_complexities)

    def complexity_fn(sample):
        return sample[1]

    group_size = 8
    samples_and_complexities_sorted = samples_and_complexities.copy()
    for begin_index in range(0, len(samples_and_complexities), group_size):
        end_index = min(begin_index + group_size, len(samples_and_complexities))
        samples_and_complexities_sorted[begin_index:end_index] = sorted(
            samples_and_complexities_sorted[begin_index:end_index], key=lambda x: x[1]
        )

    samples_and_complexities_sorted_and_shuffled = samples_and_complexities_sorted.copy()
    shuffled_group_order = torch.randperm(
        (len(samples_and_complexities) + group_size - 1) // group_size, generator=torch.Generator().manual_seed(0)
    ).tolist()
    end = 0
    for group_index in shuffled_group_order:
        original_begin = group_index * group_size
        original_end = min(original_begin + group_size, len(samples_and_complexities))
        begin = end
        end = begin + (original_end - original_begin)
        samples_and_complexities_sorted_and_shuffled[begin:end] = samples_and_complexities_sorted[
            original_begin:original_end
        ]

    data_sampler = sampler.LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=1, rank=0, shuffle=True, group_size=8
    )

    for index, sorted_and_shuffled_sample in zip(data_sampler, samples_and_complexities_sorted_and_shuffled):
        assert samples_and_complexities[index][1] == sorted_and_shuffled_sample[1]


def test_load_balancing_batch_sampler_uses_data_sampler():
    samples_and_complexities = [(torch.FloatTensor([val]), torch.randint(0, 100, (1,)).item()) for val in range(100)]
    dataset = MyDataset(samples_and_complexities)

    def complexity_fn(sample):
        return sample[1]

    data_sampler = sampler.LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=1, rank=0, shuffle=False
    )

    batch_size = 12

    def batch_fn(indices):
        nonlocal batch_size
        batches = []
        for batch_index_begin in range(0, len(indices), batch_size):
            batch_index_end = min(batch_index_begin + batch_size, len(indices))
            batches.append(indices[batch_index_begin:batch_index_end])
        return batches

    batch_sampler = sampler.LoadBalancingDistributedBatchSampler(data_sampler, batch_fn)

    for batch in batch_sampler:
        assert len(batch) == batch_size or len(batch) == len(samples_and_complexities) % batch_size
