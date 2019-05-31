// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cuda_pch.h"
#include "core/platform/ort_mutex.h"
#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "shared_inc/cuda_utils.h"
#include <deque>

namespace onnxruntime {

// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderInfo {
  int device_id{0};
};

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

// Logical device representation.
class CUDAExecutionProvider : public IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const CUDAExecutionProviderInfo& info);
  virtual ~CUDAExecutionProvider();

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type = OrtMemTypeDefault) const override;

  Status Sync() const override;

  Status OnRunStart() override;

  Status OnRunEnd() override;

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

  cublasHandle_t PerThreadCublasHandle() {
    // Assure each thread has its TLS context.
    if (!per_thread_context_)
      per_thread_context_ = std::make_shared<PerThreadContext>(device_id_);
    return per_thread_context_->CublasHandle();
  }

  cudnnHandle_t PerThreadCudnnHandle() {
    // Assure each thread has its TLS context.
    // TODO: improve its performance when calling cuda functions from multiple threads.
    if (!per_thread_context_)
      per_thread_context_ = std::make_shared<PerThreadContext>(device_id_);
    return per_thread_context_->CudnnHandle();
  }

  cudaStream_t GetStream(int queue_id) const {
    ORT_ENFORCE(queue_id >= 0 && queue_id < kTotalCudaStreams);
    return streams_[queue_id];
  }

  template <typename T>
  const T* GetConstOnes(size_t count) {
    // Assure each thread has its TLS context.
    if (!per_thread_context_)
      per_thread_context_ = std::make_shared<PerThreadContext>(device_id_);
    return per_thread_context_->template GetConstOnes<T>(count);
  }

  void AddDeferredReleaseCPUPtr(void* p);

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(OrtMemTypeDefault), count_or_bytes);
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  virtual std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

 private:
  cudaStream_t streams_[kTotalCudaStreams];
  int device_id_;

  struct DeferredReleaseCPUPtrs {
    bool recorded = false;
    std::vector<void*> cpu_ptrs;
  };
  std::unordered_map<cudaEvent_t, DeferredReleaseCPUPtrs> deferred_release_cpu_ptr_;
  OrtMutex deferred_release_cpu_ptr_mutex_;

  class PerThreadContext final {
   public:
    PerThreadContext(int device_id);
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
        return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(count));
      } else if (std::is_same<T, double>::value) {
        if (!constant_ones_double_) {
          constant_ones_double_ = cuda::CreateConstantOnes<double>();
        }
        return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(count));
      } else if (std::is_same<T, half>::value) {
        if (!constant_ones_half_) {
          constant_ones_half_ = cuda::CreateConstantOnes<half>();
        }
        return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(count));
      } else {
        return nullptr;
      }
    }

   private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;

    // deferred release for temporary CPU pinned memory used in cudaMemcpyAsync
    // note that cudaEvent will be assigned at OnRunEnd() when PerThreadContext destory
    // so the ownership is passed to deferred_release_cpu_ptr_
    cudaEvent_t current_deferred_release_event_ = nullptr;

    std::unique_ptr<cuda::IConstantBuffer<float>> constant_ones_float_;
    std::unique_ptr<cuda::IConstantBuffer<double>> constant_ones_double_;
    std::unique_ptr<cuda::IConstantBuffer<half>> constant_ones_half_;
  };

  // thread local context during execution
  static thread_local std::shared_ptr<PerThreadContext> per_thread_context_;

  // thread local GPU memory allocator. could be used before execution
  static thread_local AllocatorPtr per_thread_default_allocator_;

  // reuse thread local GPU memory allocator for memory pattern
  mutable std::deque<AllocatorPtr> default_allocator_pool_;
  mutable OrtMutex default_allocator_pool_mutex_;

  // reuse thread local context
  mutable std::deque<std::shared_ptr<PerThreadContext>> context_pool_;
  mutable OrtMutex context_pool_mutex_;

  void ReleasePerThreadStuffs() const;
};

}  // namespace onnxruntime
