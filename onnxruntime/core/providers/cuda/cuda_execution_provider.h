// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <vector>

#include "core/framework/allocatormgr.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_graph.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/tunable/cuda_tuning_context.h"

namespace onnxruntime {

void RunOnUnload(std::function<void()> function);

// Logical device representation.
class CUDAExecutionProvider : public IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const CUDAExecutionProviderInfo& info);
  virtual ~CUDAExecutionProvider();

  AllocatorPtr GetAllocator(OrtMemType mem_type) const override;

  Status Sync() const override;

  Status OnRunStart() override;

  Status OnRunEnd(bool sync_stream) override;

  const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  cublasHandle_t PerThreadDefaultCublasHandle() {
    return GetPerThreadContext().CublasHandle();
  }

  cublasLtHandle_t PerThreadCublasLtHandle() {
    return GetPerThreadContext().CublasLtHandle();
  }

  cudnnHandle_t PerThreadDefaultCudnnHandle() {
    return GetPerThreadContext().CudnnHandle();
  }

  template <typename T>
  const T* GetConstOnes(size_t count, cudaStream_t stream) {
    return GetPerThreadContext().template GetConstOnes<T>(count, stream);
  }

  // GPU scratch buffer need to be allocated on stream
  template <typename T>
  IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, Stream* stream, WaitNotificationFn wait_fn) const {
    if (count_or_bytes == 0)
      return nullptr;
    return IAllocator::MakeUniquePtr<T>(GetAllocator(OrtMemTypeDefault), count_or_bytes, false, stream, wait_fn);
  }

  template <typename T>
  IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(OrtMemTypeDefault), count_or_bytes, true);
  }

  template <typename T>
  IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
    // Note that OrtMemTypeCPU and OrtMemTypeCPUOutput are the same. See onnxruntime_c_api.h.
    // In some CUDA async
    if (count_or_bytes == 0)
      return nullptr;
    return IAllocator::MakeUniquePtr<T>(GetAllocator(OrtMemTypeCPU),
                                        count_or_bytes);
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const IKernelLookup& kernel_lookup) const override;

  int GetDeviceId() const override { return info_.device_id; }
  const cudaDeviceProp& GetDeviceProp() const { return device_prop_; };
  int GetCudnnConvAlgo() const { return info_.cudnn_conv_algo_search; }
  bool DoCopyOnDefaultStream() const { return info_.do_copy_in_default_stream; }
  bool GetCudnnConvUseMaxWorkspace() const { return info_.cudnn_conv_use_max_workspace; }
  bool GetCudnnConv1dPadToNc1d() const { return info_.cudnn_conv1d_pad_to_nc1d; }

  ProviderOptions GetProviderOptions() const override {
    return CUDAExecutionProviderInfo::ToProviderOptions(info_);
  }

  void RegisterAllocator(AllocatorManager& allocator_manager) override;
  static AllocatorPtr CreateCudaAllocator(OrtDevice::DeviceId device_id, size_t cuda_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                                          CUDAExecutionProviderExternalAllocatorInfo external_alloc_info, OrtArenaCfg* arena_cfg);

  ITuningContext* GetTuningContext() const override;

  std::unique_ptr<profiling::EpProfiler> GetProfiler() override;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured() const override;
  Status ReplayGraph() override;
#endif
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry) const override;

  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;

 private:
  CUDAExecutionProviderInfo info_;
  cudaDeviceProp device_prop_;
  bool external_stream_ = false;
  // only used when set user external stream or cuda graph
  cudaStream_t stream_ = nullptr;

  bool use_ep_level_unified_stream_ = false;

  // the tuning context might be altered when calling into a TunableOp
  mutable cuda::tunable::CudaTuningContext tuning_context_;

  class PerThreadContext final {
   public:
    PerThreadContext(OrtDevice::DeviceId device_id, cudaStream_t stream, size_t cuda_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                     CUDAExecutionProviderExternalAllocatorInfo external_alloc_info, OrtArenaCfg* arena_cfg);
    ~PerThreadContext();

    cublasHandle_t CublasHandle() const {
      return cublas_handle_;
    }

    cudnnHandle_t CudnnHandle() const {
      return cudnn_handle_;
    }

    cublasLtHandle_t CublasLtHandle() const {
      return cublas_lt_handle_;
    }

    template <typename T>
    const T* GetConstOnes(size_t count, cudaStream_t stream) {
      if (std::is_same<T, float>::value) {
        if (!constant_ones_float_) {
          constant_ones_float_ = cuda::CreateConstantOnes<float>();
        }
        return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(stream, count));
      } else if (std::is_same<T, double>::value) {
        if (!constant_ones_double_) {
          constant_ones_double_ = cuda::CreateConstantOnes<double>();
        }
        return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(stream, count));
      } else if (std::is_same<T, half>::value) {
        if (!constant_ones_half_) {
          constant_ones_half_ = cuda::CreateConstantOnes<half>();
        }
        return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(stream, count));
      } else if (std::is_same<T, BFloat16>::value) {
        if (!constant_ones_bfloat16_) {
          constant_ones_bfloat16_ = cuda::CreateConstantOnes<BFloat16>();
        }
        return reinterpret_cast<const T*>(constant_ones_bfloat16_->GetBuffer(stream, count));
      } else {
        return nullptr;
      }
    }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
    bool IsGraphCaptureAllowed() const;
    void CaptureBegin();
    void CaptureEnd();
    bool IsGraphCaptured() const;
    Status ReplayGraph();
    void IncrementRegularRunCountBeforeGraphCapture();
#endif

   private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    cublasLtHandle_t cublas_lt_handle_ = nullptr;

    std::unique_ptr<cuda::IConstantBuffer<float>> constant_ones_float_;
    std::unique_ptr<cuda::IConstantBuffer<double>> constant_ones_double_;
    std::unique_ptr<cuda::IConstantBuffer<half>> constant_ones_half_;
    std::unique_ptr<cuda::IConstantBuffer<BFloat16>> constant_ones_bfloat16_;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
    // Cuda graph with multi threads will be supported in the future, so cuda_graph_
    // is put under PerThreadContext.
    CUDAGraph cuda_graph_;
    bool is_graph_captured_ = false;
    int regular_run_count_before_graph_capture_ = 0;
    const int min_num_runs_before_cuda_graph_capture_ = 1;  // required min regular runs before graph capture for the necessary memory allocations.

#endif
  };

  using PerThreadContextMap = std::unordered_map<const CUDAExecutionProvider*, std::weak_ptr<PerThreadContext>>;
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
    // weak references to thread local caches from which this CUDAExecutionProvider instance's entry should be removed
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
