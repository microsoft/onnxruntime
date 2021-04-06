// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <vector>

#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/rocm/gpu_data_transfer.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"
#include "core/providers/rocm/rocm_pch.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {

// Logical device representation.
class ROCMExecutionProvider : public IExecutionProvider {
 public:
  explicit ROCMExecutionProvider(const ROCMExecutionProviderInfo& info);
  virtual ~ROCMExecutionProvider();

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

  Status Sync() const override;

  Status OnRunStart() override;

  Status OnRunEnd() override;

  const void* GetExecutionHandle() const noexcept override {
    // The HIP interface does not return anything interesting.
    return nullptr;
  }

  Status SetComputeStream(void* stream) override {
    if (stream != stream_) {
      if (stream_) {
        HIP_CALL(hipStreamDestroy(stream_));
      }

      external_stream_ = true;
      stream_ = static_cast<hipStream_t>(stream);
    }
    return Status::OK();
  }

  void* GetComputeStream() const override { return static_cast<void*>(stream_); }

  rocblas_handle PerThreadRocblasHandle() {
    return GetPerThreadContext().RocblasHandle();
  }

  miopenHandle_t PerThreadMiopenHandle() {
    return GetPerThreadContext().MiopenHandle();
  }

  template <typename T>
  const T* GetConstOnes(size_t count) {
    return GetPerThreadContext().template GetConstOnes<T>(count);
  }

  void AddDeferredReleaseCPUPtr(void* p);

  template <typename T>
  IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(info_.device_id, OrtMemTypeDefault), count_or_bytes);
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  int GetDeviceId() const override { return info_.device_id; }
  const hipDeviceProp_t& GetDeviceProp() const { return device_prop_; };

  ProviderOptions GetProviderOptions() const override {
    return ROCMExecutionProviderInfo::ToProviderOptions(info_);
  }
  
  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;
  static AllocatorPtr CreateRocmAllocator(OrtDevice::DeviceId device_id, size_t rocm_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                                          ROCMExecutionProviderExternalAllocatorInfo external_alloc_info);

 private:
  ROCMExecutionProviderInfo info_;
  hipDeviceProp_t device_prop_;
  bool external_stream_ = false;
  hipStream_t stream_ = nullptr;

  struct DeferredReleaseCPUPtrs {
    bool recorded = false;
    std::vector<void*> cpu_ptrs;
  };

  std::unordered_map<hipEvent_t, DeferredReleaseCPUPtrs> deferred_release_cpu_ptr_;
  OrtMutex deferred_release_cpu_ptr_mutex_;

  class PerThreadContext final {
   public:
    PerThreadContext(OrtDevice::DeviceId device_id, hipStream_t stream, size_t gpu_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                     ROCMExecutionProviderExternalAllocatorInfo external_alloc_info);
    ~PerThreadContext();

    rocblas_handle RocblasHandle() const {
      return rocblas_handle_;
    }

    miopenHandle_t MiopenHandle() const {
      return miopen_handle_;
    }

    hipEvent_t& GetCurrentDeferredReleaseEvent() {
      return current_deferred_release_event_;
    }

    template <typename T>
    const T* GetConstOnes(size_t count) {
      if (std::is_same<T, float>::value) {
        if (!constant_ones_float_) {
          constant_ones_float_ = rocm::CreateConstantOnes<float>();
        }
        return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(stream_, count));
      } else if (std::is_same<T, double>::value) {
        if (!constant_ones_double_) {
          constant_ones_double_ = rocm::CreateConstantOnes<double>();
        }
        return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(stream_, count));
      } else if (std::is_same<T, half>::value) {
        if (!constant_ones_half_) {
          constant_ones_half_ = rocm::CreateConstantOnes<half>();
        }
        return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(stream_, count));
      } else {
        return nullptr;
      }
    }

    AllocatorPtr GetAllocator() const {
      return allocator_;
    }

   private:
    hipStream_t stream_ = nullptr;
    rocblas_handle rocblas_handle_ = nullptr;
    miopenHandle_t miopen_handle_ = nullptr;

    // deferred release for temporary CPU pinned memory used in hipMemcpyAsync
    // note that hipEvent will be assigned at OnRunEnd() when PerThreadContext destory
    // so the ownership is passed to deferred_release_cpu_ptr_
    hipEvent_t current_deferred_release_event_ = nullptr;

    std::unique_ptr<rocm::IConstantBuffer<float>> constant_ones_float_;
    std::unique_ptr<rocm::IConstantBuffer<double>> constant_ones_double_;
    std::unique_ptr<rocm::IConstantBuffer<half>> constant_ones_half_;

    AllocatorPtr allocator_;
  };

  using PerThreadContextMap = std::unordered_map<const ROCMExecutionProvider*, std::weak_ptr<PerThreadContext>>;
  // thread local PerThreadContext cache
  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache() {
    thread_local const auto per_thread_context_cache = std::make_shared<PerThreadContextMap>();
    return per_thread_context_cache;
  }

  struct PerThreadContextState {
    // contexts that are currently active
    std::set<std::shared_ptr<PerThreadContext>, std::owner_less<std::shared_ptr<PerThreadContext>>> active_contexts;
    // contexts available for reuse
    std::vector<std::shared_ptr<PerThreadContext>> retired_context_pool;
    // weak references to thread local caches from which this ROCMExecutionProvider instance's entry should be removed
    // upon destruction
    std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
        caches_to_update_on_destruction;
    // synchronizes access to PerThreadContextState members
    OrtMutex mutex;
  };

  // The execution provider maintains the PerThreadContexts in this structure.
  // Synchronization is required to update the contained structures.
  // On the other hand, access to an individual PerThreadContext is assumed to be from a single thread at a time,
  // so synchronization is not required for that.
  mutable PerThreadContextState context_state_;

  PerThreadContext& GetPerThreadContext() const;
  void ReleasePerThreadContext() const;
};

}  // namespace onnxruntime
