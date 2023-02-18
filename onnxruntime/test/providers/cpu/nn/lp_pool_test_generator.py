# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from torch import nn


# use this code to generate test data for PoolTest.LpPool1d and PoolTest.LpPool2d
def generate_lppool_1d_test_cases() -> None:
    p = 2
    x = np.array(
        [
            [
                [1, 2, 3, 4],
            ]
        ]
    ).astype(np.float32)

    print(x)
    kernel_sizes = [2, 3]
    strides = [[1], [2]]
    for kernel_size in kernel_sizes:
        for stride in strides:
            print(kernel_size)
            print(stride)
            model = nn.LPPool1d(p, kernel_size=kernel_size, stride=stride)
            pt_y = model(torch.from_numpy(x))
            print(torch.flatten(pt_y))
            print(pt_y.shape)


def generate_lppool_2d_test_cases() -> None:
    p = 2
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ]
    ).astype(np.float32)

    print(x)
    kernel_sizes = [[2, 2], [3, 3]]
    strides = [[1, 1], [2, 2]]
    for kernel_size in kernel_sizes:
        for stride in strides:
            model = nn.LPPool2d(p, kernel_size=kernel_size, stride=stride)
            pt_y = model(torch.from_numpy(x))
            print(kernel_size)
            print(stride)
            print(torch.flatten(pt_y))
            print(pt_y.shape)


generate_lppool_1d_test_cases()
generate_lppool_2d_test_cases()
