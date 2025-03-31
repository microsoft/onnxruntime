// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/rocm_pch.h"
// #include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"
#include "core/framework/stream_handles.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"

namespace onnxruntime {

struct RocmStream;
void WaitRocmNotificationOnDevice(Stream& stream, synchronize::Notification& notification);

struct DeferredCpuAllocator : public OrtAllocator {
  DeferredCpuAllocator(RocmStream&);
  RocmStream& rocm_stream_;
};

struct RocmStream : Stream {
  RocmStream(hipStream_t stream,
             const OrtDevice& device,
             AllocatorPtr cpu_allocator,
             bool release_cpu_buffer_on_rocm_stream,
             bool own_flag,
             miopenHandle_t external_miopen_handle,
             hipblasHandle_t external_hipblas_handle,
             const ROCMExecutionProviderInfo& ep_info);

  ~RocmStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  Status CleanUpOnRunEnd() override;

  void EnqueDeferredCPUBuffer(void* cpu_buffer);

  bool own_stream_{true};

  miopenHandle_t miopen_handle_{};

  hipblasHandle_t hipblas_handle_{};

  void* GetResource(int version, int id) const override;

  onnxruntime::IAllocator* GetCpuAllocator() const { return cpu_allocator_.get(); }

  WaitNotificationFn GetWaitNotificationFn() const override { return WaitRocmNotificationOnDevice; }

 private:
  std::vector<void*> deferred_cpu_buffers_;
  AllocatorPtr cpu_allocator_;
  bool release_cpu_buffer_on_rocm_stream_{true};
  DeferredCpuAllocator deferred_cpu_allocator_;
  const ROCMExecutionProviderInfo ep_info_;
};

void RegisterRocmStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               AllocatorPtr cpu_allocator,
                               bool release_cpu_buffer_on_rocm_stream,
                               hipStream_t external_stream,
                               bool use_existing_stream,
                               miopenHandle_t external_miopen_handle,
                               hipblasHandle_t external_hipblas_handle,
                               const ROCMExecutionProviderInfo& ep_info);
}  // namespace onnxruntime
