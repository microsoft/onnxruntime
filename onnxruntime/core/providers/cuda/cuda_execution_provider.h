// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <vector>

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

  Status Sync() const override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  DataLayout GetPreferredLayout() const override;

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

  cudaStream_t ComputeStream() {
    // this will return the CUDA EP level stream which can differ from the actual compute tasks stream
    // the compute task stream is supplied within OpKernelContext during inference
    return stream_;
  }

  template <typename T>
  const T* GetConstOnes(size_t count, cudaStream_t stream) {
    return GetPerThreadContext().template GetConstOnes<T>(count, stream);
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
  bool IsSkipLayerNormInStrictMode() const { return info_.enable_skip_layer_norm_strict_mode; }
  bool IsNHWCPreferred() const { return info_.prefer_nhwc; }
  bool UseTF32() const { return info_.use_tf32; }

  ProviderOptions GetProviderOptions() const override {
    return CUDAExecutionProviderInfo::ToProviderOptions(info_);
  }

  static AllocatorPtr CreateCudaAllocator(OrtDevice::DeviceId device_id, size_t cuda_mem_limit, ArenaExtendStrategy arena_extend_strategy,
                                          CUDAExecutionProviderExternalAllocatorInfo external_alloc_info, const OrtArenaCfg* arena_cfg);

  ITuningContext* GetTuningContext() const override;

  std::unique_ptr<profiling::EpProfiler> GetProfiler() override;

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured(CudaGraphAnnotation_t graph_annotation_id) const override;
  Status ReplayGraph(CudaGraphAnnotation_t graph_annotation_id) override;
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

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
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PerThreadContext);

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
      if constexpr (std::is_same<T, float>::value) {
        if (!constant_ones_float_) {
          constant_ones_float_ = cuda::CreateConstantOnes<float>();
        }
        return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(stream, count));
      } else if constexpr (std::is_same<T, double>::value) {
        if (!constant_ones_double_) {
          constant_ones_double_ = cuda::CreateConstantOnes<double>();
        }
        return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(stream, count));
      } else if constexpr (std::is_same<T, half>::value) {
        if (!constant_ones_half_) {
          constant_ones_half_ = cuda::CreateConstantOnes<half>();
        }
        return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(stream, count));
      } else if constexpr (std::is_same<T, BFloat16>::value) {
        if (!constant_ones_bfloat16_) {
          constant_ones_bfloat16_ = cuda::CreateConstantOnes<BFloat16>();
        }
        return reinterpret_cast<const T*>(constant_ones_bfloat16_->GetBuffer(stream, count));
#if !defined(DISABLE_FLOAT8_TYPES)
      } else if constexpr (std::is_same<T, Float8E4M3FN>::value) {
        if (!constant_ones_float8e4m3fn_) {
          constant_ones_float8e4m3fn_ = cuda::CreateConstantOnes<Float8E4M3FN>();
        }
        return reinterpret_cast<const T*>(constant_ones_float8e4m3fn_->GetBuffer(stream, count));
      } else if constexpr (std::is_same<T, Float8E5M2>::value) {
        if (!constant_ones_float8e5m2_) {
          constant_ones_float8e5m2_ = cuda::CreateConstantOnes<Float8E5M2>();
        }
        return reinterpret_cast<const T*>(constant_ones_float8e5m2_->GetBuffer(stream, count));
#endif
      } else {
        return nullptr;
      }
    }

    bool IsGraphCaptureAllowed(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    void CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id);
    void CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id);
    bool IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    CudaGraphAnnotation_t GetCudaGraphAnnotationId(const onnxruntime::RunOptions& run_options) const;
    Status ReplayGraph(CudaGraphAnnotation_t cuda_graph_annotation_id);
    void IncrementRegularRunCountBeforeGraphCapture();

   private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    cublasLtHandle_t cublas_lt_handle_ = nullptr;

    std::unique_ptr<cuda::IConstantBuffer<float>> constant_ones_float_;
    std::unique_ptr<cuda::IConstantBuffer<double>> constant_ones_double_;
    std::unique_ptr<cuda::IConstantBuffer<half>> constant_ones_half_;
    std::unique_ptr<cuda::IConstantBuffer<BFloat16>> constant_ones_bfloat16_;
#if !defined(DISABLE_FLOAT8_TYPES)
    std::unique_ptr<cuda::IConstantBuffer<Float8E4M3FN>> constant_ones_float8e4m3fn_;
    std::unique_ptr<cuda::IConstantBuffer<Float8E5M2>> constant_ones_float8e5m2_;
#endif

    // Cuda graph with multi threads will be supported in the future, so cuda_graph_
    // is put under PerThreadContext.
    CUDAGraph cuda_graph_;
    int regular_run_count_before_graph_capture_ = 0;

    // There is chance that the second regular run allocates GPU memory for causes like:
    // (1) memory pattern is enabled. (2) arena allocation for stream.
    // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
    // to allocate enough memory in Arena before graph capturing.
    const int min_num_runs_before_cuda_graph_capture_ = 2;  // required min regular runs before graph capture for the necessary memory allocations.
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
