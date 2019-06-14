// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/gpu_data_transfer.h"

namespace onnxruntime {

Status CopyTensorFromCudaToCudaPinned(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id) {
  // copying from GPU to pinned memory, this is non-blocking
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, streams_[exec_queue_id]));
  return Status::OK();
}

Status CopyTensorFromCudaToCpu(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id) {
  // copying from GPU to CPU memory, this is blocking
  CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
  return Status::OK();
}

Status CopyTensorFromCudaPinnedToCuda(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id) {
  // copy from pinned memory to GPU, this is non-blocking
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, streams_[exec_queue_id]));
  return Status::OK();
}

Status CopyTensorFromCudaToCuda(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id) {
  // copying between GPU, this is non-blocking
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, streams_[kCudaStreamDefault]));
  return Status::OK();
}

Status CopyTensorFromCpuToCuda(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id) {
  // copy from other CPU memory to GPU, this is blocking
  CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
  return Status::OK();
}

}  // namespace onnxruntime
