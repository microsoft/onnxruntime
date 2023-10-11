// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/resource.h"

#define ORT_CUDA_RESOUCE_VERSION 2

enum CudaResource : int {
  cuda_stream_t = cuda_resource_offset,
  cudnn_handle_t,
  cublas_handle_t,
  deferred_cpu_allocator_t,
};