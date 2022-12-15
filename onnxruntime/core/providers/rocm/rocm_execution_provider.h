// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <vector>

#include "core/framework/allocatormgr.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"
#include "core/providers/rocm/rocm_pch.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {

void RunOnUnload(std::function<void()> function);

// Logical device representation.
class ROCMExecutionProvider : public IExecutionProvider {
 public:
  explicit ROCMExecutionProvider(const ROCMExecutionProviderInfo& info);
  virtual ~ROCMExecutionProvider();

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

  Status Sync() const override;

  Status OnRunStart() override;

  Status OnRunEnd(bool sync_stream) override;

  const void* GetExecutionHandle() const noexcept override {
    // The ROCM interface does not return anything interesting.
    return nullptr;
  }

  rocblas_handle PerThreadRocblasHandle() {
    return GetPerThreadContext().RocblasHandle();
  }

  miopenHandle_t PerThreadMiopenHandle() {
    return GetPerThreadContext().MiopenHandle();
  }

  template <typename T>
  const T* GetConstOnes(size_t count, hipStream_t stream) {
    return GetPerThreadContext().template GetConstOnes<T>(count, stream);
  }

  template <typename T>
  IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, Stream* stream, WaitNotificationFn wait_fn) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(info_.device_id, OrtMemTypeDefault), count_or_bytes, false, stream, wait_fn);
  }

  template <typename T>
  IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(info_.device_id, OrtMemTypeDefault), count_or_bytes, true);
  }

  template <typename T>
  IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
    // Note that OrtMemTypeCPU and OrtMemTypeCPUOutput are the same. See onnxruntime_c_api.h.
    // In some ROCm async
    if (count_or_bytes == 0)
      return nullptr;
    return IAllocator::MakeUniquePtr<T>(GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPUOutput),
                                        count_or_bytes);
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const IKernelLookup& kernel_lookup) const override;

  int GetDeviceId() const override { return info_.device_id; }
  const hipDeviceProp_t& GetDeviceProp() const { return device_prop_; };
  int GetMiopenConvExhaustiveSearch() const { return info_.miopen_conv_exhaustive_search; }
  bool DoCopyOnDefaultStream() const { return info_.do_copy_in_default_stream; }

  bool GetMiopenConvUseMaxWorkspace() const { return info_.miopen_conv_use_max_workspace; }

  ProviderOptions GetProviderOptions() const override {
    return ROCMExecutionProviderInfo::ToProviderOptions(info_);
  }

  void RegisterAllocator(AllocatorManager& allocator_manager) override;
  static AllocatorPtr CreateRocmAllocator(OrtDevice::DeviceId device_id, size_t rocm_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                                          ROCMExecutionProviderExternalAllocatorInfo external_alloc_info, OrtArenaCfg* arena_cfg);

  void EnableTunableOp();
  void DisableTunableOp();
  bool IsTunableOpEnabled() const;

  std::unique_ptr<profiling::EpProfiler> GetProfiler() override;

  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry) const override;

 private:
  ROCMExecutionProviderInfo info_;
  hipDeviceProp_t device_prop_;
  bool external_stream_ = false;
  hipStream_t stream_ = nullptr;

  bool use_ep_level_unified_stream_ = false;

  class PerThreadContext final {
   public:
    PerThreadContext(OrtDevice::DeviceId device_id, hipStream_t stream, size_t rocm_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                     ROCMExecutionProviderExternalAllocatorInfo external_alloc_info, OrtArenaCfg* arena_cfg);
    ~PerThreadContext();

    rocblas_handle RocblasHandle() const {
      return rocblas_handle_;
    }

    miopenHandle_t MiopenHandle() const {
      return miopen_handle_;
    }

    template <typename T>
    const T* GetConstOnes(size_t count, hipStream_t stream) {
      if (std::is_same<T, float>::value) {
        if (!constant_ones_float_) {
          constant_ones_float_ = rocm::CreateConstantOnes<float>();
        }
        return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(stream, count));
      } else if (std::is_same<T, double>::value) {
        if (!constant_ones_double_) {
          constant_ones_double_ = rocm::CreateConstantOnes<double>();
        }
        return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(stream, count));
      } else if (std::is_same<T, half>::value) {
        if (!constant_ones_half_) {
          constant_ones_half_ = rocm::CreateConstantOnes<half>();
        }
        return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(stream, count));
      } else {
        return nullptr;
      }
    }

   private:
    rocblas_handle rocblas_handle_ = nullptr;
    miopenHandle_t miopen_handle_ = nullptr;

    std::unique_ptr<rocm::IConstantBuffer<float>> constant_ones_float_;
    std::unique_ptr<rocm::IConstantBuffer<double>> constant_ones_double_;
    std::unique_ptr<rocm::IConstantBuffer<half>> constant_ones_half_;
  };

  using PerThreadContextMap = std::unordered_map<const ROCMExecutionProvider*, std::weak_ptr<PerThreadContext>>;
  // thread local PerThreadContext cache

  struct ContextCacheHolder {
    ContextCacheHolder() {
      // Keep a weak pointer to the object, if the weak pointer can be locked, then the shared pointer is still around, so we can reset it
      RunOnUnload([&, weak_p_ = std::weak_ptr<PerThreadContextMap>(p)] {
        if (auto lock = weak_p_.lock())
          p.reset();
      });
    }
    std::shared_ptr<PerThreadContextMap> p = std::make_shared<PerThreadContextMap>();
  };

  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache() {
    thread_local const ContextCacheHolder per_thread_context_cache;
    return per_thread_context_cache.p;
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
