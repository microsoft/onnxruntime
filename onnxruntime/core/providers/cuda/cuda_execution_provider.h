// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <vector>

#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {

// Logical device representation.
class CUDAExecutionProvider : public IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const CUDAExecutionProviderInfo& info);
  virtual ~CUDAExecutionProvider();

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

  Status Sync() const override;

  Status OnRunStart() override;

  Status OnRunEnd() override;

  const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  Status SetComputeStream(void* stream) override;

  void* GetComputeStream() const override { return static_cast<void*>(stream_); }

  cublasHandle_t PerThreadCublasHandle() {
    return GetPerThreadContext().CublasHandle();
  }

  cudnnHandle_t PerThreadCudnnHandle() {
    return GetPerThreadContext().CudnnHandle();
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
  const cudaDeviceProp& GetDeviceProp() const { return device_prop_; };
  int GetCudnnConvAlgo() const { return info_.cudnn_conv_algo_search; }

  ProviderOptions GetProviderOptions() const override {
    return CUDAExecutionProviderInfo::ToProviderOptions(info_);
  }

  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;
  static AllocatorPtr CreateCudaAllocator(OrtDevice::DeviceId device_id, size_t cuda_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                                          CUDAExecutionProviderExternalAllocatorInfo external_alloc_info, OrtArenaCfg* arena_cfg);

 private:
  CUDAExecutionProviderInfo info_;
  cudaDeviceProp device_prop_;
  bool external_stream_ = false;
  cudaStream_t stream_ = nullptr;
  bool shrink_default_memory_allocator_on_every_run_end_ = false;

  struct DeferredReleaseCPUPtrs {
    bool recorded = false;
    std::vector<void*> cpu_ptrs;
  };

  std::unordered_map<cudaEvent_t, DeferredReleaseCPUPtrs> deferred_release_cpu_ptr_;
  OrtMutex deferred_release_cpu_ptr_mutex_;

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

    cudaEvent_t& GetCurrentDeferredReleaseEvent() {
      return current_deferred_release_event_;
    }

    template <typename T>
    const T* GetConstOnes(size_t count) {
      if (std::is_same<T, float>::value) {
        if (!constant_ones_float_) {
          constant_ones_float_ = cuda::CreateConstantOnes<float>();
        }
        return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(stream_, count));
      } else if (std::is_same<T, double>::value) {
        if (!constant_ones_double_) {
          constant_ones_double_ = cuda::CreateConstantOnes<double>();
        }
        return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(stream_, count));
      } else if (std::is_same<T, half>::value) {
        if (!constant_ones_half_) {
          constant_ones_half_ = cuda::CreateConstantOnes<half>();
        }
        return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(stream_, count));
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
      } else if (std::is_same<T, nv_bfloat16>::value) {
        if (!constant_ones_bfloat16_) {
          constant_ones_bfloat16_ = cuda::CreateConstantOnes<nv_bfloat16>();
        }
        return reinterpret_cast<const T*>(constant_ones_bfloat16_->GetBuffer(stream_, count));
#endif
      } else {
        return nullptr;
      }
    }

    AllocatorPtr GetAllocator() const {
      return allocator_;
    }

   private:
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;

    // deferred release for temporary CPU pinned memory used in cudaMemcpyAsync
    // note that cudaEvent will be assigned at OnRunEnd() when PerThreadContext destory
    // so the ownership is passed to deferred_release_cpu_ptr_
    cudaEvent_t current_deferred_release_event_ = nullptr;

    std::unique_ptr<cuda::IConstantBuffer<float>> constant_ones_float_;
    std::unique_ptr<cuda::IConstantBuffer<double>> constant_ones_double_;
    std::unique_ptr<cuda::IConstantBuffer<half>> constant_ones_half_;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    std::unique_ptr<cuda::IConstantBuffer<nv_bfloat16>> constant_ones_bfloat16_;
#endif

    AllocatorPtr allocator_;
  };

  using PerThreadContextMap = std::unordered_map<const CUDAExecutionProvider*, std::weak_ptr<PerThreadContext>>;
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