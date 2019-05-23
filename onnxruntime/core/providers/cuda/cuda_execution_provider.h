// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cuda_pch.h"
#include "core/platform/ort_mutex.h"
#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "shared_inc/cuda_utils.h"
#include "cuda_allocator.h"
#include <deque>
#include <thread>

namespace onnxruntime {

// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderInfo {
  int device_id{0};
};

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams = 30,
};

// T could be a cublas_handle_ or cudnn_handle_
//template <typename T>
//class HandleLockWrapper {
// public:
//  CudnnHandle(OrtMutex* mutex, const T& handle) : mutex_(mutex), handle_(handle) { mutex_->lock(); }
//  ~CudnnHandle() { mutex_->unlock(); }
//
//  T Handle() const { return handle_; }
//
// private:
//  OrtMutex* mutex_;
//  T handle_;
//};

class CublasHandle {
 public:
  CublasHandle(OrtMutex* mutex, const cublasHandle_t& handle) : mutex_(mutex), handle_(handle) { mutex_->lock(); }
  ~CublasHandle() { mutex_->unlock(); }

  cublasHandle_t Handle() const { return handle_; }

 private:
  OrtMutex* mutex_;
  cublasHandle_t handle_;
};

class PerThreadContext final {
 public:
  PerThreadContext(int device_id);
  ~PerThreadContext();

  cublasHandle_t GetCublasHandle(const cudaStream_t& stream) {
    cublasSetStream(cublas_handle_, stream);
    return cublas_handle_;
  }

  cudnnHandle_t GetCudnnHandle(const cudaStream_t& stream) {
    cudnnSetStream(cudnn_handle_, stream);
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
  OrtMutex cublas_handle_mutex_;
  OrtMutex cudnn_handle_mutex_;

  // deferred release for temporary CPU pinned memory used in cudaMemcpyAsync
  // note that cudaEvent will be assigned at OnRunEnd() when PerThreadContext destory
  // so the ownership is passed to deferred_release_cpu_ptr_
  cudaEvent_t current_deferred_release_event_ = nullptr;

  std::unique_ptr<cuda::IConstantBuffer<float>> constant_ones_float_;
  std::unique_ptr<cuda::IConstantBuffer<double>> constant_ones_double_;
  std::unique_ptr<cuda::IConstantBuffer<half>> constant_ones_half_;
};

class CudaContextPool {
 public:
  CudaContextPool(int device_id) : device_id_(device_id) {}

  std::shared_ptr<PerThreadContext> GetPerThreadContext(int exec_id) {
    auto find = context_in_use_.find(exec_id);
    if (find != context_in_use_.end()) {
      return context_in_use_[exec_id];
    }

    std::lock_guard<OrtMutex> ctx_lock(context_pool_mutex_);
    if (context_available_.empty()) {
      auto per_thread_context_ = std::make_shared<PerThreadContext>(device_id_);
      context_in_use_[exec_id] = per_thread_context_;
    } else {
      context_in_use_[exec_id] = context_available_.back();
      context_available_.pop_back();
    }

    return context_in_use_[exec_id];
  }

  void ReleasePerThreadContext(int exec_id) {
    std::lock_guard<OrtMutex> ctx_lock(context_pool_mutex_);
    if (context_in_use_.find(exec_id) == context_in_use_.end()) {
      return;
    }
    context_available_.push_back(context_in_use_[exec_id]);
    context_in_use_.erase(exec_id);
  }

 private:
  std::deque<std::shared_ptr<PerThreadContext>> context_available_;
  // map<stream_id, PerThreadContext>
  std::unordered_map<int, std::shared_ptr<PerThreadContext>> context_in_use_;
  OrtMutex context_pool_mutex_;
  int device_id_;
};

class CudaResourcePool {
 public:
  CudaResourcePool(int device_id) : device_id_(device_id) {}

  AllocatorPtr GetCudaResource(int stream_id) {
    std::lock_guard<OrtMutex> lock(cuda_resource_mutex_);

    auto find = in_used_resource_.find(stream_id);
    if (find != in_used_resource_.end()) {
      return in_used_resource_[stream_id];
    }

    if (available_resources_.empty()) {
      DeviceAllocatorRegistrationInfo default_allocator_info(
          {OrtMemTypeDefault,
           [](int id) { return std::make_unique<CUDAAllocator>(id); }, std::numeric_limits<size_t>::max()});
      in_used_resource_[stream_id] = CreateAllocator(default_allocator_info, device_id_);
    } else {
      in_used_resource_[stream_id] = available_resources_.back();
      available_resources_.pop_back();
    }

    return in_used_resource_[stream_id];
  }

  void ReleaseCudaResource(int stream_id) {
    std::lock_guard<OrtMutex> ctx_lock(cuda_resource_mutex_);
    if (in_used_resource_.find(stream_id) == in_used_resource_.end()) {
      return;
    }
    available_resources_.push_back(in_used_resource_[stream_id]);
    in_used_resource_.erase(stream_id);
  }

 private:
  std::deque<AllocatorPtr> available_resources_;
  // map<stream_id, AllocatorPtr>
  std::unordered_map<int, AllocatorPtr> in_used_resource_;
  OrtMutex cuda_resource_mutex_;
  int device_id_;
};

class CudaEventPool {
 public:
  CudaEventPool() {}
  ~CudaEventPool();
  cudaEvent_t GetCudaEvent();
  void ReleaseCudaEvent();
  Status StreamSync(cudaStream_t stream, std::vector<cudaStream_t> dep_stream_ids);
  void SetEventRecordFlag(cudaEvent_t cuda_event) {
    std::lock_guard<OrtMutex> lock(event_mutex_);
    if (in_used_events_.find(cuda_event) != in_used_events_.end()) {
      in_used_events_[cuda_event] = true;
    }
  }

 private:
  cudaEvent_t GetOrCreateCudaEvent();

 private:
  std::deque<cudaEvent_t> available_events_;
  // map<event, event_recorded>
  std::unordered_map<cudaEvent_t, bool> in_used_events_;
  OrtMutex event_mutex_;
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

  const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  cublasHandle_t PerThreadCublasHandle(int execution_id) {
    // Assure each thread has its TLS context.
    auto execution_stream = GetStream(execution_id);
    auto per_thread_context = cuda_context_pool_.GetPerThreadContext(execution_id);
    if (!per_thread_context)
      per_thread_context = std::make_shared<PerThreadContext>(device_id_);
    return per_thread_context->GetCublasHandle(execution_stream);
  }

  cudnnHandle_t PerThreadCudnnHandle(int execution_id) {
    // Assure each thread has its TLS context.
    // TODO: improve its performance when calling cuda functions from multiple threads.
    auto execution_stream = GetStream(execution_id);
    auto per_thread_context = cuda_context_pool_.GetPerThreadContext(execution_id);
    return per_thread_context->GetCudnnHandle(execution_stream);
  }

  template <typename T>
  const T* GetConstOnes(int exec_queue_id, size_t count) {
    // Assure each thread has its TLS context.
    auto per_thread_context = cuda_context_pool_.GetPerThreadContext(exec_queue_id);
    if (!per_thread_context)
      per_thread_context = std::make_shared<PerThreadContext>(device_id_);
    return per_thread_context->template GetConstOnes<T>(count);
  }

  void AddDeferredReleaseCPUPtr(void* p, cudaStream_t stream);

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

  cudaStream_t GetStream(int queue_id) const {
    ORT_ENFORCE(queue_id >= 0 && queue_id < kTotalCudaStreams);
    return streams_[queue_id];
  }

  virtual const int GetQueueID() override {
    std::lock_guard<OrtMutex> lock(queue_ids_mutex_);
    cuda_stream_id_pos_ = cuda_stream_id_pos_ % cuda_stream_ids_.size();
    return cuda_stream_ids_.at(cuda_stream_id_pos_++);
  }

  virtual void ReleaseQueueID(int queue_id) override {
    default_allocator_pool_.ReleaseCudaResource(queue_id);
    cuda_context_pool_.ReleasePerThreadContext(queue_id);
  }

  Status StreamSync(int stream_id, std::vector<int> dep_stream_ids) {
    if (0 == stream_id || dep_stream_ids.size() < 1) {
      return Status::OK();
    }
    cuda_event_pool_.ReleaseCudaEvent();
    auto stream = GetStream(stream_id);
    std::vector<cudaStream_t> dep_streams(dep_stream_ids.size());
    for (int i = 0; i < dep_stream_ids.size(); ++i) {
      dep_streams[i] = GetStream(dep_stream_ids[i]);
    }
    return cuda_event_pool_.StreamSync(stream, dep_streams);
  }

  void TryReleaseCudaEvent() {
    cuda_event_pool_.ReleaseCudaEvent();
  }

 private:
  int GetQueueIDByThreadId(std::thread::id& thread_id) {
    std::lock_guard<OrtMutex> lock(thread_queue_id_map_mutex_);
    auto queue_id = thead_id_2_queue_id_.find(thread_id);
    if (queue_id == thead_id_2_queue_id_.end()) {
      auto new_queue_id = GetQueueID();
      thead_id_2_queue_id_.emplace(thread_id, new_queue_id);
    }
    return thead_id_2_queue_id_[thread_id];
  }

 private:
  cudaStream_t streams_[kTotalCudaStreams];
  std::vector<int> cuda_stream_ids_;
  int cuda_stream_id_pos_;
  int device_id_;
  mutable OrtMutex queue_ids_mutex_;

  // Merge the deferred_release_cpu_ptr with CudaEventPool?
  struct DeferredReleaseCPUPtrs {
    bool recorded = false;
    cudaEvent_t cuda_event;
    std::vector<void*> cpu_ptrs;
  };
  std::unordered_map<cudaStream_t, DeferredReleaseCPUPtrs> deferred_release_cpu_ptr_;
  OrtMutex deferred_release_cpu_ptr_mutex_;

  // reuse thread local GPU memory allocator for memory pattern
  CudaResourcePool default_allocator_pool_;
  //mutable OrtMutex default_allocator_pool_mutex_;

  // reuse thread local context
  CudaContextPool cuda_context_pool_;
  //mutable std::deque<std::shared_ptr<PerThreadContext>> context_pool_;
  //mutable OrtMutex context_pool_mutex_;

  // mapping from thread id to queue id, use for scenario that cuda execution provide code executed in multi-thread RunStart/End
  std::unordered_map<std::thread::id, int> thead_id_2_queue_id_;
  OrtMutex thread_queue_id_map_mutex_;

  CudaEventPool cuda_event_pool_;

  void ReleasePerThreadStuffs();

  bool RNNNeedFallbackToCPU(const onnxruntime::Node& node, const std::vector<std::string> activations_supported, const std::string& op_type) const;
  bool ConvNeedFallbackToCPU(const onnxruntime::Node& node) const;
};

}  // namespace onnxruntime
