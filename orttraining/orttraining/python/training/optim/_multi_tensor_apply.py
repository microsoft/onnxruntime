# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# multi_tensor_apply.py
# This file has been adapted from microsoft/DeepSpeed

"""
Copyright 2020 The Microsoft DeepSpeed Team

Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex, commit a109f85
"""


class MultiTensorApply:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
