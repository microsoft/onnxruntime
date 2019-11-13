// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_common.h"
#include "cuda_execution_provider.h"
#include "core/framework/memcpy.h"
#include "cuda_fence.h"
#include "cuda_allocator.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/providers/cuda/gpu_data_transfer.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cuda_contrib_kernels.h"
#endif

using namespace onnxruntime::common;

namespace onnxruntime {

namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .ExecQueueId(kCudaStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .ExecQueueId(kCudaStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

}  // namespace cuda

thread_local std::unique_ptr<CUDAExecutionProvider::PerThreadContextMap> CUDAExecutionProvider::per_thread_context_map_;

CUDAExecutionProvider::PerThreadContext::PerThreadContext(int device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id));
  CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
  CUDNN_CALL_THROW(cudnnCreate(&cudnn_handle_));

  DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault,
       [](int id) { return onnxruntime::make_unique<CUDAAllocator>(id, CUDA); }, std::numeric_limits<size_t>::max()});
  allocator_ = CreateAllocator(default_memory_info, device_id);
}

CUDAExecutionProvider::PerThreadContext::~PerThreadContext() {
  CUBLAS_CALL_THROW(cublasDestroy(cublas_handle_));
  CUDNN_CALL_THROW(cudnnDestroy(cudnn_handle_));
}

CUDAExecutionProvider::CUDAExecutionProvider(const CUDAExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kCudaExecutionProvider}, device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));

  DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault, [](int device_id) { return onnxruntime::make_unique<CUDAAllocator>(device_id, CUDA); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(default_memory_info, device_id_));

  DeviceAllocatorRegistrationInfo pinned_memory_info(
      {OrtMemTypeCPUOutput, [](int device_id) { return onnxruntime::make_unique<CUDAPinnedAllocator>(device_id, CUDA_PINNED); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(pinned_memory_info, CPU_ALLOCATOR_DEVICE_ID));

  // TODO: this is actually used for the cuda kernels which explicitly ask for inputs from CPU.
  // This will be refactored/removed when allocator and execution provider are decoupled.
  DeviceAllocatorRegistrationInfo cpu_memory_info({OrtMemTypeCPUInput,
                                                   [](int device_id) { return onnxruntime::make_unique<CPUAllocator>(
                                                                           onnxruntime::make_unique<OrtMemoryInfo>(
                                                                               "CUDA_CPU",
                                                                               OrtAllocatorType::OrtDeviceAllocator,
                                                                               OrtDevice(),
                                                                               device_id,
                                                                               OrtMemTypeCPUInput)); },
                                                   std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(cpu_memory_info, CPU_ALLOCATOR_DEVICE_ID));
}

CUDAExecutionProvider::~CUDAExecutionProvider() {
  auto cpu_alloc = GetAllocator(CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
  std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
  auto it = deferred_release_cpu_ptr_.begin();
  while (it != deferred_release_cpu_ptr_.end()) {
    auto& e = it->first;
    auto& v = it->second;
    if (v.recorded)
      CUDA_CALL_THROW(cudaEventSynchronize(e));
    for (auto p : v.cpu_ptrs) {
      cpu_alloc->Free(p);
    }
    CUDA_CALL_THROW(cudaEventDestroy(e));
    it = deferred_release_cpu_ptr_.erase(it);
  }
}

CUDAExecutionProvider::PerThreadContext& CUDAExecutionProvider::GetPerThreadContext() const {
  if (per_thread_context_map_ == nullptr) {
    per_thread_context_map_ = onnxruntime::make_unique<PerThreadContextMap>();
  }

  auto* p = per_thread_context_map_.get();
  if (p->count(this) == 0) {
    std::lock_guard<OrtMutex> lock(context_pool_mutex_);
    unsigned int tid = logging::GetThreadId();
    auto inuse_iter = inuse_contexts_.find(tid);
    std::shared_ptr<PerThreadContext> ptc;
    if (inuse_iter == inuse_contexts_.end()) {
      if (retired_context_pool_.empty()) {
        ptc = std::make_shared<PerThreadContext>(device_id_);
      } else {
        ptc = retired_context_pool_.back();
        retired_context_pool_.pop_back();
      }
      inuse_contexts_.insert(std::make_pair(tid, ptc));
    } else {
      ptc = inuse_iter->second;
    }
    p->insert(std::make_pair(this, ptc));
  }
  return *(p->at(this));
}

void CUDAExecutionProvider::ReleasePerThreadStuffs() const {
  ORT_ENFORCE(per_thread_context_map_ != nullptr);
  auto iter_ctx = per_thread_context_map_->find(this);
  ORT_ENFORCE(iter_ctx != per_thread_context_map_->end());

  std::lock_guard<OrtMutex> lock(context_pool_mutex_);
  unsigned int tid = logging::GetThreadId();
  if (inuse_contexts_.count(tid)) {
    inuse_contexts_.erase(tid);
  }
  retired_context_pool_.push_back(iter_ctx->second);
  per_thread_context_map_->erase(iter_ctx);
  // Release TLS if empty to avoid memory leak report
  if (per_thread_context_map_->empty()) {
    per_thread_context_map_.reset(nullptr);
  }
}

AllocatorPtr CUDAExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  // Pinned memory allocator is shared between threads, but CUDA memory allocator is per-thread or it may cause result changes
  // A hypothesis is that arena allocator is not aligned with CUDA output cache, and data from different kernel writes may
  // cause cacheline to contain dirty data.
  if (mem_type == OrtMemTypeDefault) {
    return GetPerThreadContext().GetAllocator();
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

Status CUDAExecutionProvider::Sync() const {
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  return Status::OK();
}

void CUDAExecutionProvider::AddDeferredReleaseCPUPtr(void* p) {
  // when not running in InferenceSession (e.g. Test)
  // it's OK to not remember the deferred release ptr
  // as the actual memory will be cleaned in arena allocator dtor
  auto current_deferred_release_event = GetPerThreadContext().GetCurrentDeferredReleaseEvent();
  if (current_deferred_release_event) {
    std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
    auto iter = deferred_release_cpu_ptr_.find(current_deferred_release_event);
    ORT_ENFORCE(iter != deferred_release_cpu_ptr_.end());
    iter->second.cpu_ptrs.push_back(p);
  }
}

Status CUDAExecutionProvider::OnRunStart() {
  auto cpu_alloc = GetAllocator(0, OrtMemTypeCPU);
  // check if cudaEvents has passed for deferred release
  // note that we need to take a mutex in case of multi-threaded Run()
  std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
  auto it = deferred_release_cpu_ptr_.begin();
  while (it != deferred_release_cpu_ptr_.end()) {
    auto& e = it->first;
    auto& v = it->second;
    // note that cudaEventQuery returns cudaSucess before first cudaEventRecord
    if (v.recorded && cudaSuccess == cudaEventQuery(e)) {
      for (auto p : v.cpu_ptrs) {
        cpu_alloc->Free(p);
      }
      cudaEvent_t expired_event = it->first;
      it = deferred_release_cpu_ptr_.erase(it);
      CUDA_RETURN_IF_ERROR(cudaEventDestroy(expired_event));
    } else {
      ++it;
    }
  }

  auto& current_deferred_release_event = GetPerThreadContext().GetCurrentDeferredReleaseEvent();
  CUDA_RETURN_IF_ERROR(cudaEventCreate(&current_deferred_release_event, cudaEventDisableTiming));
  deferred_release_cpu_ptr_.emplace(current_deferred_release_event, DeferredReleaseCPUPtrs());
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunEnd() {
  // record deferred release event on default stream, and release per_thread_context
  auto current_deferred_release_event = GetPerThreadContext().GetCurrentDeferredReleaseEvent();
  CUDA_RETURN_IF_ERROR(cudaEventRecord(current_deferred_release_event, nullptr));
  ReleasePerThreadStuffs();
  std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
  deferred_release_cpu_ptr_[current_deferred_release_event].recorded = true;
  return Status::OK();
}

namespace cuda {
// opset 1 to 9
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, double, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, int32_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, int64_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, uint32_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, uint64_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, float, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, double, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, int32_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, int64_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, uint32_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, uint64_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, float, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, double, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, MLFloat16, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, float, Max);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, double, Max);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, float, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, double, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, MLFloat16, Max);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, float, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, double, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, float, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, double, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, MLFloat16, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Equal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Clip);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, bool, Not);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, bool, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, bool, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape_1);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Shape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, float, Slice);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, Compress);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, int32_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, uint8_t, Upsample);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, Split);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, ConstantOfShape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int8_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int16_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint16_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Shrink);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Less);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, EyeLike);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scatter);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, bool, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, NonZero);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, TopK);

// opset 10
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, AveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int32_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, uint8_t, Resize);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, ReverseSequence);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int32_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int64_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, TopK);

// opset 11
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, NonMaxSuppression);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Range);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, ScatterElements);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Split);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, TopK);

static void RegisterCudaKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, int32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, int64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, uint32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, uint64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, float, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, double, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, int32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, int64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, uint32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, uint64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, float, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, double, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, MLFloat16, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, float, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, double, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, float, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, double, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, MLFloat16, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, float, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, double, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, float, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, double, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, MLFloat16, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, bool, Not)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape_1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, float, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, uint8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int8_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int16_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint16_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, EyeLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scatter)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, TopK)>,

      // opset 10
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, ReverseSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, TopK)>,

      // opset 11
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, TopK)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetCudaKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterCudaKernels(*kernel_registry);

#ifndef DISABLE_CONTRIB_OPS
  ::onnxruntime::contrib::cuda::RegisterCudaContribKernels(*kernel_registry);
#endif

  return kernel_registry;
}

}  // namespace cuda

std::shared_ptr<KernelRegistry> CUDAExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::cuda::GetCudaKernelRegistry();
  return kernel_registry;
}

static bool RNNNeedFallbackToCPU(const onnxruntime::Node& node,
                                 const std::vector<std::string> activations_supported,
                                 const std::string& op_type) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto attr_name = attr.first;
    auto attr_value = attr.second;

    if ("activation_alpha" == attr_name || "activation_beta" == attr_name || "clip" == attr_name) {
      return true;
    }

    if ("activations" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS == attr_value.type()) {
      for (int i = 0; i < attr_value.strings_size(); ++i) {
        std::string activation_lowercase(attr_value.strings(i));
        std::transform(activation_lowercase.begin(), activation_lowercase.end(), activation_lowercase.begin(),
                       [](const unsigned char i) { return static_cast<char>(::tolower(i)); });
        if (activations_supported[i] != activation_lowercase) {
          return true;
        }
      }
    }

    if ("LSTM" == op_type &&
        "input_forget" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT == attr_value.type()) {
      if (0 != attr_value.i()) {
        return true;
      }
    }

    if ("GRU" == op_type &&
        "linear_before_reset" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT == attr_value.type()) {
      // cudnn GRU only support linear_before_reset = 1
      if (1 != attr_value.i()) {
        return true;
      }
    }
  }

  if ("LSTM" == op_type) {
    // cudnn LSTM not support peephole
    auto input_defs = node.InputDefs();
    if (8 == input_defs.size()) {
      auto peephole = input_defs.at(7);
      if (peephole->Exists()) {
        return true;
      }
    }
  }
  return false;
}

static bool ConvNeedFallbackToCPU(const onnxruntime::Node& node) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto attr_name = attr.first;
    auto attr_value = attr.second;

    //cudnn only supports symmetric padding
    if ("pads" == attr_name && ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS == attr_value.type()) {
      auto pads = attr_value.ints();
      int pads_size = pads.size();
      ORT_ENFORCE(pads_size % 2 == 0);
      int rank = pads_size / 2;
      for (int i = 0; i < rank; i++) {
        if (pads.Get(i) != pads.Get(i + rank)) {
          return true;
        }
      }
    }
  }

  return false;
}

static bool CastNeedFallbackToCPU(const onnxruntime::Node& node) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto attr_name = attr.first;
    auto attr_value = attr.second;

    // string is not supported
    if ("to" == attr_name && ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT == attr_value.type()) {
      auto to_type = attr_value.i();
      if (to_type == ::ONNX_NAMESPACE::TensorProto_DataType_STRING)
        return true;
    }
  }

  return false;
}

std::unique_ptr<onnxruntime::IDataTransfer> CUDAExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<onnxruntime::GPUDataTransfer>();
}

std::vector<std::unique_ptr<ComputeCapability>>
CUDAExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                     const std::vector<const KernelRegistry*>&) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::unordered_set<const NodeArg*> defs_outside_cuda;

  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    const auto* cuda_kernel_def = GetKernelRegistry()->TryFindKernel(node, Type());
    if (cuda_kernel_def == nullptr || !node.GetExecutionProviderType().empty()) {
      // node is not in cuda exeuction provider if no kernel def found,
      // or if other execution provider already assigned to it
      defs_outside_cuda.insert(node.OutputDefs().cbegin(), node.OutputDefs().cend());
      continue;
    }

    bool not_supported = false;
    bool force_outside = false;
    bool force_inside = false;  // for some compute heavy ops, we'll force it to run inside CUDA
    if ("LSTM" == node.OpType()) {
      // the supported activations covers the bidirectional mode
      std::vector<std::string> activations_supported{"sigmoid", "tanh", "tanh", "sigmoid", "tanh", "tanh"};
      not_supported = RNNNeedFallbackToCPU(node, activations_supported, node.OpType());
      force_inside = !not_supported;
    } else if ("RNN" == node.OpType()) {
      std::vector<std::string> activations_supported{"tanh", "tanh"};
      not_supported = RNNNeedFallbackToCPU(node, activations_supported, node.OpType());
      force_inside = !not_supported;
    } else if ("GRU" == node.OpType()) {
      std::vector<std::string> activations_supported{"sigmoid", "tanh", "sigmoid", "tanh"};
      not_supported = RNNNeedFallbackToCPU(node, activations_supported, node.OpType());
      force_inside = !not_supported;
    } else if ("Conv" == node.OpType()) {
      not_supported = ConvNeedFallbackToCPU(node);
      force_inside = !not_supported;
    } else if ("Cast" == node.OpType()) {
      not_supported = CastNeedFallbackToCPU(node);
      // cast is not compute heavy, and may be placed outside
    }

    if (!not_supported && !force_inside) {
      // Note that nodes with only inputs from initializer would not be place on CUDA
      // Ideally, those nodes should be eliminated in constant folding
      bool should_force_outside = true;
      bool all_input_are_initializer = true;
      node.ForEachWithIndex(
          node.InputDefs(),
          [&](const NodeArg& def, size_t index) {
            const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
            // The input is not a initializer and the input is from CPU
            // or the input declared as CPU memory and is from CPU
            // in that case we should still keep the node on CUDA
            bool initializer_input = graph.GetInitializedTensor(def.Name(), initializer);
            if ((!initializer_input && !defs_outside_cuda.count(&def)) ||
                (defs_outside_cuda.count(&def) && cuda_kernel_def->kernel_def->IsInputOnCpu(index)))
              should_force_outside = false;
            if (!initializer_input) {
              all_input_are_initializer = false;
            }
            return Status::OK();
          });
      // If all the inputs are initialier, we shouldn't force it to CPU
      if (should_force_outside && !all_input_are_initializer) {
        force_outside = true;
      }
    }

    if (!force_inside && (not_supported || force_outside)) {
      defs_outside_cuda.insert(node.OutputDefs().cbegin(), node.OutputDefs().cend());
      if (not_supported) {
        LOGS_DEFAULT(WARNING) << "CUDA kernel not supported. Fallback to CPU execution provider for Op type: " << node.OpType() << " node name: " << node.Name();
      } else if (force_outside) {
        LOGS_DEFAULT(INFO) << "Force fallback to CPU execution provider for Op type: " << node.OpType() << " node name: " << node.Name();
      }
    } else {
      // for nodes placed on CUDA, check if its output is on CPU
      node.ForEachWithIndex(
          node.OutputDefs(),
          [&](const NodeArg& def, size_t out_index) {
            if (cuda_kernel_def->kernel_def->OutputMemoryType(out_index) != OrtMemTypeDefault)
              defs_outside_cuda.insert(&def);
            return Status::OK();
          });
      std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }
  return result;
}

}  // namespace onnxruntime
