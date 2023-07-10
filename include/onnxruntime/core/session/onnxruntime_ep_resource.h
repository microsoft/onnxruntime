// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_CUDA_RESOUCE_VERSION 1

//todo - switch to enum class?
enum CudaResource: int {
  cuda_stream_t = 0,
  cudnn_handle_t,
  cublas_handle_t
};

#define ORT_DML_RESOUCE_VERSION 1

enum DmlResource : int {
  dml_device_t = 0,
  d3d12_device_t,
  cmd_list_t
};