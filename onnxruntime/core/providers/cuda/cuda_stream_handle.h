// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/framework/stream_handles.h"

namespace onnxruntime {

struct CudaStream : Stream {
  CudaStream(cudaStream_t stream,
             const OrtDevice& device,
             AllocatorPtr cpu_allocator,
             bool release_cpu_buffer_on_cuda_stream,
             bool own_flag,
             cudnnHandle_t external_cudnn_handle,
             cublasHandle_t external_cublass_handle);

  ~CudaStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  Status CleanUpOnRunEnd() override;

  void EnqueDeferredCPUBuffer(void* cpu_buffer);

  bool own_stream_{true};

  cudnnHandle_t cudnn_handle_{};

  cublasHandle_t cublas_handle_{};

 private:
  std::vector<void*> deferred_cpu_buffers_;
  AllocatorPtr cpu_allocator_;
  bool release_cpu_buffer_on_cuda_stream_{true};
};

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               AllocatorPtr cpu_allocator,
                               bool release_cpu_buffer_on_cuda_stream,
                               cudaStream_t external_stream,
                               bool use_existing_stream,
                               cudnnHandle_t external_cudnn_handle,
                               cublasHandle_t external_cublass_handle);
void WaitCudaNotificationOnDevice(Stream& stream, synchronize::Notification& notification);
}  // namespace onnxruntime
