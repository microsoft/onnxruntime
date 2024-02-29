// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <vector>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"
#include "core/providers/rocm/rocm_graph.h"
#include "core/providers/rocm/rocm_pch.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"
#include "core/providers/rocm/tunable/rocm_tuning_context.h"

namespace onnxruntime {

void RunOnUnload(std::function<void()> function);

// Logical device representation.
class ROCMExecutionProvider : public IExecutionProvider {
 public:
  explicit ROCMExecutionProvider(const ROCMExecutionProviderInfo& info);
  virtual ~ROCMExecutionProvider();

  Status Sync() const override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  const void* GetExecutionHandle() const noexcept override {
    // The ROCM interface does not return anything interesting.
    return nullptr;
  }

  rocblas_handle PerThreadDefaultRocblasHandle() {
    return GetPerThreadContext().RocblasHandle();
  }

  miopenHandle_t PerThreadDefaultMiopenHandle() {
    return GetPerThreadContext().MiopenHandle();
  }

  template <typename T>
  const T* GetConstOnes(size_t count, hipStream_t stream) {
    return GetPerThreadContext().template GetConstOnes<T>(count, stream);
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

  static AllocatorPtr CreateRocmAllocator(OrtDevice::DeviceId device_id, size_t rocm_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                                          ROCMExecutionProviderExternalAllocatorInfo external_alloc_info, const OrtArenaCfg* arena_cfg);

  ITuningContext* GetTuningContext() const override;

  std::unique_ptr<profiling::EpProfiler> GetProfiler() override;

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured() const override;
  Status ReplayGraph() override;
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

 private:
  ROCMExecutionProviderInfo info_;
  hipDeviceProp_t device_prop_;
  bool external_stream_ = false;
  // only used when set user external stream or hip graph
  hipStream_t stream_ = nullptr;

  bool use_ep_level_unified_stream_ = false;

  // the tuning context might be altered when calling into a TunableOp
  mutable rocm::tunable::RocmTuningContext tuning_context_;

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
      constexpr bool is_float = std::is_same<T, float>::value;
      constexpr bool is_double = std::is_same<T, double>::value;
      constexpr bool is_half = std::is_same<T, half>::value;
      constexpr bool is_BFloat16 = std::is_same<T, BFloat16>::value;
      if (is_float) {
        if (!constant_ones_float_) {
          constant_ones_float_ = rocm::CreateConstantOnes<float>();
        }
        return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(stream, count));
      } else if (is_double) {
        if (!constant_ones_double_) {
          constant_ones_double_ = rocm::CreateConstantOnes<double>();
        }
        return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(stream, count));
      } else if (is_half) {
        if (!constant_ones_half_) {
          constant_ones_half_ = rocm::CreateConstantOnes<half>();
        }
        return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(stream, count));
      } else if (is_BFloat16) {
        if (!constant_ones_bfloat16_) {
          constant_ones_bfloat16_ = rocm::CreateConstantOnes<BFloat16>();
        }
        return reinterpret_cast<const T*>(constant_ones_bfloat16_->GetBuffer(stream, count));
      } else {
        return nullptr;
      }
    }

    bool IsGraphCaptureAllowed() const;
    void CaptureBegin();
    void CaptureEnd();
    bool IsGraphCaptured() const;
    Status ReplayGraph();
    void IncrementRegularRunCountBeforeGraphCapture();

   private:
    rocblas_handle rocblas_handle_ = nullptr;
    miopenHandle_t miopen_handle_ = nullptr;

    std::unique_ptr<rocm::IConstantBuffer<float>> constant_ones_float_;
    std::unique_ptr<rocm::IConstantBuffer<double>> constant_ones_double_;
    std::unique_ptr<rocm::IConstantBuffer<half>> constant_ones_half_;
    std::unique_ptr<rocm::IConstantBuffer<BFloat16>> constant_ones_bfloat16_;

    // Hip graph with multi threads will be supported in the future, so hip_graph_
    // is put under PerThreadContext.
    ROCMGraph hip_graph_;
    bool is_graph_captured_ = false;
    int regular_run_count_before_graph_capture_ = 0;

    // There is chance that the second regular run allocates GPU memory for causes like:
    // (1) memory pattern is enabled. (2) arena allocation for stream.
    // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
    // to allocate enough memory in Arena before graph capturing.
    const int min_num_runs_before_hip_graph_capture_ = 2;  // required min regular runs before graph capture for the necessary memory allocations.
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
