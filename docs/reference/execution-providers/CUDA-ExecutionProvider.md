---
title: CUDA
parent: Execution Providers
grand_parent: Reference
nav_order: 1
---

# CUDA Execution Provider

The CUDA Execution Provider enables hardware accelerated computation on CUDA-enabled GPUs.

## Build
For build instructions, please see the [BUILD page](../../how-to/build.md#CUDA).

## Configuration Options
The CUDA EP supports the following configuration options:

### device_id
The device ID.

### cuda_mem_limit
The memory limit of the device memory arena in bytes.

### arena_extend_strategy
The strategy for extending the device memory arena.
Valid values:
- kNextPowerOfTwo (0)
- kSameAsRequested (1)

### cudnn_conv_algo_search
The type of search done for cuDNN convolution algorithms.
Valid values:
- EXHAUSTIVE (0)
- HEURISTIC (1)
- DEFAULT (2)

### do_copy_in_default_stream
Whether to do copies in the default stream or use separate streams.
