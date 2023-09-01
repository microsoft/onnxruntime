# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# sampler.py

import math
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


def _shard_wrapped_indices_across_workers(dataset_index_list, num_shards, num_samples_per_shard):
    """Yield successive num_shards-sized chunks from dataset_index_list."""
    num_samples = max(1, num_samples_per_shard)
    num_elements = num_samples * num_shards
    current_lst = []
    for i in range(num_elements):
        current_lst.append(dataset_index_list[i % len(dataset_index_list)])
        if len(current_lst) == num_shards:
            yield current_lst
            current_lst = []


def shard_wrapped_indices_for_worker(dataset_index_list, shard_id, num_shards):
    """Shard wrapped around dataset_index_list across num_shards and return the indices for this shard_id"""
    num_samples_per_worker = (len(dataset_index_list) + num_shards - 1) // num_shards
    sharded_indices = list(
        _shard_wrapped_indices_across_workers(dataset_index_list, num_shards, num_samples_per_worker)
    )
    return [sharded_indices[i][shard_id] for i in range(len(sharded_indices))]


# Implementation is adapted from bagua/load_balancing_data_loader.py
# https://github.com/BaguaSys/bagua/blob/01874a7c3f90904c37c5612a9db866b5d4b8b5ed/bagua/torch_api/contrib/load_balancing_data_loader.py#L12
class LoadBalancingDistributedSampler:
    r"""Sampler that balances the data load across workers based on the sample's complexity.
    This sampler uses a :attr:`complexity_fn` to calculate each sample's computational
    complexity and make each batch get similar computational complexity.
    This is useful in scenarios like speech and NLP, where each batch has variable
    length and distributed training suffers from straggler problem. In such scenarios,
    the complexity function could be defined to return the length of the input sample sequence.
    The usage is similar to `torch.utils.data.DistributedSampler`, where each process loads a
    subset of the original dataset that is exclusive to it.
    The sampler sorts the dataset in increasing order of complexity. If the :attr:`group_size` is
    provided, the sorting happens within dataset groups of size :attr:`group_size` before the
    group order is shuffled followed by sharding of data across workers. If :attr:`group_size`
    is not provided, the data is distributed across workers before the data indices for each worker
    is shuffled deterministically.
    .. note::
        Dataset is assumed to be of constant size (map-style dataset).
    Args:
        dataset: Dataset (map-style) used for sampling.
        complexity_fn(Callable): A function whose input is a sample and output is an integer as a
            measure of the computational complexity of the sample.
        world_size (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`world_size`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices within the dataset if :attr:`group_size` is None, else will
            shuffle the groups if :attr:`group_size` is not None.
        group_size (int, optional): If provided, the dataset will be broken down into
            :attr:`group_size` sized groups. Indices will only be sorted within the groups
            and not across the entire dataset. If :attr:`shuffle` is ```True``` and
            :attr:`group_size` is not ```None```, the position of each group in the dataset
            will be shuffled. Default: ```None```
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: 0.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            shards. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the shards. Default: ``False``.
        random_level (float, optional): A float varies from 0 and 1 that controls the extent
            of load balance. 0 means the best load balance, while 1 means the opposite.
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the `torch.utils.data.DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        Define your :attr:`complexity_fn`, which accepts a dataset sample as its input and produces an integer
        as the sample's computational complexity:
        >>> dataset = MyVariableSequenceLengthDataset(dataset_samples)
        >>> complexity_fn = lambda x: len(x)
        Below is the usage of :class:`LoadBalancingDistributedSampler`
        and `torch.utils.data.DataLoader`:
        >>> sampler = onnxruntime.training.utils.data.LoadBalancingDistributedSampler(
        ...     dataset,
        ...     complexity_fn=complexity_fn)
        >>> loader = torch.utils.data.DataLoader(dataset,
        ...     sampler=sampler)
        >>>
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        complexity_fn: Callable[..., int],
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        group_size: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        random_level: float = 0,
    ) -> None:
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]")
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.group_size = group_size

        # If the dataset length is evenly divisible by number of shards, then there
        # is no need to drop any data, since the dataset will be split equally.
        dataset_len = len(self.dataset)
        if self.drop_last and dataset_len % self.world_size != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = dataset_len // self.world_size
        else:
            self.num_samples = math.ceil(dataset_len / self.world_size)
        self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed

        self.complexity_fn = complexity_fn
        self.sample_complexities = None
        self.ordered_sample_complexities = None

        if random_level < 0.0 or random_level > 1.0:
            raise ValueError(f"Invalid random level {random_level}, shoule be in the range [0.0, 1.0]")

        self.random_level = random_level
        self.random_number = None

    def _sort_shard_and_shuffle_dataset(self):
        # This method returns a list of dataset sample indices after
        # the dataset has been sorted, sharded and shuffled.
        # The sorting of the dataset happens based on the group_size and complexities
        # of each sample.
        # Sharding happens across the number of workers.
        # Shuffling is done either before sharding on the group indices (if group_size is provided)
        # or on the dataset sample indices if the group_size is not provided.

        def sort_in_groups(sample_complexities, group_size):
            """Sort the dataset samples indices inside each group of size group_size."""
            # If the group_size is None, the entire dataset is considered as a single group
            if group_size is None:
                group_size = len(sample_complexities)
            # Sort the dataset samples inside each group of the dataset based on sample complexity.
            for group_begin_index in range(0, len(sample_complexities), group_size):
                group_end_index = min(group_begin_index + group_size, len(sample_complexities))
                sorted_indices = group_begin_index + np.argsort(
                    sample_complexities[group_begin_index:group_end_index, 1]
                )
                sample_complexities[group_begin_index:group_end_index, :] = sample_complexities[sorted_indices]
            return sample_complexities

        # Get the samples and their complexities from the complexity_fn
        if not self.sample_complexities:
            self.sample_complexities = np.empty((len(self.dataset), 2), dtype=np.int64)
            for sample_index in range(len(self.dataset)):
                self.sample_complexities[sample_index][0] = sample_index
                self.sample_complexities[sample_index][1] = self.complexity_fn(self.dataset[sample_index])

        if self.random_number is None:
            max_complexity = max(self.sample_complexities, key=lambda t: t[1])[1]
            min_complexity = min(self.sample_complexities, key=lambda t: t[1])[1]
            self.random_number = int((max_complexity - min_complexity) * self.random_level + 1)

        sample_complexities = self.sample_complexities.copy()

        # Control the degree of load balancing by modifying the complexities of
        # all samples using the random_number.
        g = torch.Generator()
        g = g.manual_seed(self.seed + self.epoch)

        if self.random_number > 1:
            complexity_random_ints = torch.randint(
                self.random_number, (len(sample_complexities),), generator=g
            ).tolist()

            for index, random_int in enumerate(complexity_random_ints):
                sample_complexities[index][1] += random_int

        # Sort the data based on the computed complexities and group sizes.
        # Sort only once if random_number <= 1 else sort everytime
        if self.ordered_sample_complexities is None or self.random_number > 1:
            self.ordered_sample_complexities = sort_in_groups(sample_complexities, self.group_size)
        ordered_sample_complexities = self.ordered_sample_complexities

        # If group_size is not None, shuffle the index of each group instead
        # of shuffling the data indices.
        if self.shuffle and self.group_size is not None:
            num_groups = (len(self.sample_complexities) + self.group_size - 1) // self.group_size
            group_order = torch.randperm(num_groups, generator=g).tolist()
            end = 0
            sample_complexities_copy = ordered_sample_complexities.copy()
            for group_index in group_order:
                original_list_begin_index = self.group_size * group_index
                original_list_end_index = min(original_list_begin_index + self.group_size, len(sample_complexities))
                begin = end
                end = begin + (original_list_end_index - original_list_begin_index)
                sample_complexities_copy[begin:end, :] = sample_complexities[
                    original_list_begin_index:original_list_end_index, :
                ]
            ordered_sample_complexities = sample_complexities_copy

        # Shard the data across the different workers.
        index_chunks = list(
            _shard_wrapped_indices_across_workers(
                [index_complexity_tuple[0] for index_complexity_tuple in ordered_sample_complexities],
                self.world_size,
                self.num_samples,
            )
        )

        # Shuffle the sharded data indices deterministically based on epoch and seed.
        chunk_indices = list(range(len(index_chunks)))
        if self.shuffle and self.group_size is None:
            chunk_indices = torch.randperm(len(index_chunks), generator=g).tolist()

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.num_samples - len(chunk_indices)
            if padding_size <= len(chunk_indices):
                chunk_indices += chunk_indices[:padding_size]
            else:
                chunk_indices += (chunk_indices * math.ceil(padding_size / len(chunk_indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible.
            chunk_indices = chunk_indices[: self.num_samples]

        assert len(chunk_indices) == self.num_samples
        return index_chunks, chunk_indices

    def __iter__(self) -> Iterator:
        index_chunks, chunk_indices = self._sort_shard_and_shuffle_dataset()
        # Extract indices based on current rank.
        indices = [index_chunks[i][self.rank] for i in chunk_indices]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Sets the epoch for this sampler.
        When :attr:`shuffle=True`, this ensures all shards use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class LoadBalancingDistributedBatchSampler(Sampler):
    r"""Wraps another load balance sampler to yield variable sized mini-batches.
    Args:
        sampler (LoadBalancingDistributedSampler): Load balance sampler.
        batch_fn (Callable): Callable to yield mini-batch indices.
        drop_last (bool): If ``True``, the sampler will drop the last few batches exceeding
            the least number of batches among replicas, otherwise, the number of batches
            on each replica will be padded to the same.
    :attr:`batch_fn` will have the signature of::
        def batch_fn(indices: List[int]) -> List[List[int]]
    Example::
        >>> from onnxruntime.training.utils.data import LoadBalancingDistributedSampler, \
        ...     LoadBalancingDistributedBatchSampler
        >>>
        >>> sampler = LoadBalancingDistributedSampler(dataset, complexity_fn=complexity_fn)
        >>> batch_sampler = LoadBalancingDistributedBatchSampler(sampler, batch_fn=batch_fn)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
        >>>
        >>> for epoch in range(start_epoch, n_epochs):
        ...     batch_sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        sampler: LoadBalancingDistributedSampler,
        batch_fn,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(sampler, LoadBalancingDistributedSampler):
            raise ValueError("sampler should be of LoadBalancingDistributedSampler type.")

        if sampler.drop_last:
            raise ValueError("drop_last of sampler should be False")

        self.sampler = sampler
        self.batch_fn = batch_fn
        self.drop_last = drop_last

        self.world_size = self.sampler.world_size
        self.rank = self.sampler.rank

        self.generate_batches()

    def generate_batches(self):
        index_chunks, chunk_indices = self.sampler._sort_shard_and_shuffle_dataset()

        batches = []
        for rank in range(self.world_size):
            sub_indices = [index_chunks[i][rank] for i in chunk_indices]
            batches.append(self.batch_fn(sub_indices))

        self.total_batch = max([len(b) for b in batches]) if not self.drop_last else min([len(b) for b in batches])

        # here {len(batches[self.rank]) - self.total_batch} batches dropped for
        # rank {self.rank}
        if self.total_batch < len(batches[self.rank]):
            pass

        self.padded_batches = [batch + batch[: self.total_batch - len(batch)] for batch in batches]

    def __iter__(self):
        return iter(self.padded_batches[self.rank])

    def __len__(self):
        return self.total_batch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.generate_batches()
