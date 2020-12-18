# CUDA Execution Provider

**TODO This is an initial version of the CUDA EP documentation so we have a place to list valid configuration options. Need to add more detail.**

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#CUDA).

## Configuration Options
The CUDA EP supports the following configuration options:

Name | Description
-|-
device_id | The device ID.
cuda_mem_limit | The memory limit of the device memory arena in bytes.
arena_extend_strategy | The strategy for extending the device memory arena.
cudnn_conv_algo_search | The type of search done for cuDNN convolution algorithms.
do_copy_in_default_stream | Whether to do copies in the default stream instead of using separate streams.
