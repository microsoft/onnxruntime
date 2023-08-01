// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_CUDA_RESOUCE_VERSION 1

enum CudaResource : int {
  cuda_stream_t = 10000,
  cudnn_handle_t,
  cublas_handle_t
};
