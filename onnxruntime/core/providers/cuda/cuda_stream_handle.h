// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/framework/stream_handles.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"

namespace onnxruntime {

struct CudaStream;
void WaitCudaNotificationOnDevice(Stream* stream, synchronize::Notification& notification);

struct DeferredCpuAllocator : public OrtAllocator {
  DeferredCpuAllocator(CudaStream&);
  CudaStream& cuda_stream_;
};

struct CudaStream : Stream {
  CudaStream(cudaStream_t stream,
             const OrtDevice& device,
             AllocatorPtr cpu_allocator,
             bool release_cpu_buffer_on_cuda_stream,
             bool own_flag,
             cudnnHandle_t external_cudnn_handle,
             cublasHandle_t external_cublass_handle,
             const CUDAExecutionProviderInfo& ep_info);

  ~CudaStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  Status CleanUpOnRunEnd() override;

  void EnqueDeferredCPUBuffer(void* cpu_buffer);

  bool own_stream_{true};

  cudnnHandle_t cudnn_handle_{};

  cublasHandle_t cublas_handle_{};

  void* GetResource(int version, int id) const override;

  onnxruntime::IAllocator* GetCpuAllocator() const { return cpu_allocator_.get(); }

 private:
  // Releases the pinned staging buffers that were retained because the caller was capturing.
  void ReleaseCaptureRetainedCpuBuffers();

  std::vector<void*> deferred_cpu_buffers_;
  // Pinned staging buffers that a caller-initiated capture recorded as the host source of a
  // cudaMemcpyAsync node. A captured node keeps that host address for every replay, so these are
  // owned until this stream is destroyed rather than until the next run end. See CleanUpOnRunEnd.
  std::vector<void*> capture_retained_cpu_buffers_;
  AllocatorPtr cpu_allocator_;
  bool release_cpu_buffer_on_cuda_stream_{true};
  DeferredCpuAllocator deferred_cpu_allocator_;
  const CUDAExecutionProviderInfo ep_info_;
};

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               AllocatorPtr cpu_allocator,
                               bool release_cpu_buffer_on_cuda_stream,
                               cudaStream_t external_stream,
                               bool use_existing_stream,
                               cudnnHandle_t external_cudnn_handle,
                               cublasHandle_t external_cublass_handle,
                               const CUDAExecutionProviderInfo& ep_info);
}  // namespace onnxruntime
