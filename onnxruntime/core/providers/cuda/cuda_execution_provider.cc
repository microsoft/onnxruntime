// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_fwd.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/cuda_profiler.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cuda/cuda_contrib_kernels.h"
#endif

#ifdef ENABLE_TRAINING_OPS
#include "orttraining/training_ops/cuda/cuda_training_kernels.h"
#endif

#include "core/providers/cuda/cuda_stream_handle.h"

using namespace onnxruntime::common;

namespace onnxruntime {

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    auto X_type = ctx->InputType(0);
    if (X_type->IsTensorType()) {
      const auto* X = ctx->Input<Tensor>(0);
      ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
      Tensor* Y = ctx->Output(0, X->Shape());
      ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
      // do we support async copy?
      // The cudaMemCpyAsync will handle the pinned memory and non-pinned memory,
      // so we don't need the check here.
      auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);
      ORT_RETURN_IF_ERROR(gpu_data_transfer->CopyTensorAsync(*X, *Y, *ctx->GetComputeStream()));
      return Status::OK();
    } else {
      if (X_type->IsSparseTensorType()) {
        // TODO: support aysnc copy for sparse tensor
        // sync the stream first, since it is a sync memory copy
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
        const auto* X = ctx->Input<SparseTensor>(0);
        ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
        SparseTensor* Y = ctx->OutputSparse(0, X->DenseShape());
        ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output sparse tensor.");
        return X->Copy(Info().GetDataTransferManager(), *Y);
      } else if (X_type->IsTensorSequenceType()) {
        const TensorSeq* X = ctx->Input<TensorSeq>(0);
        ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor sequence is nullptr.");
        TensorSeq* Y = ctx->Output<TensorSeq>(0);
        ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor sequence.");
        auto X_dtype = X->DataType();
        Y->SetType(X_dtype);
        AllocatorPtr alloc;

        // If we are copying contents to CUDA, the allocator to use
        // to allocate the buffers of the new tensors in the sequence
        // can be temp space allocator associated with the CUDA EP
        if (Node().OpType() == "MemcpyFromHost") {
          auto status = ctx->GetTempSpaceAllocator(&alloc);
          if (!status.IsOK()) {
            return Status(common::ONNXRUNTIME, common::FAIL,
                          "Memcpy cuda: unable to get an allocator.");
          }
        } else {
          // If we are copying contents to CPU (op type is "MemcpyToHost"),
          // the allocator to use to allocate the buffers of the new tensors
          // in the sequence will be the allocator from the CPU EP
          auto status = ctx->GetTempSpaceCPUAllocator(&alloc);
          if (!status.IsOK()) {
            return Status(common::ONNXRUNTIME, common::FAIL,
                          "Memcpy cuda: unable to get the CPU allocator.");
          }
        }
        auto X_size = X->Size();
        for (size_t i = 0; i < X_size; ++i) {
          const Tensor& source_tensor = X->Get(i);
          std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(), source_tensor.Shape(), alloc);
          auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(source_tensor.Location().device,
                                                                                    target_tensor->Location().device);
          ORT_RETURN_IF_ERROR(gpu_data_transfer->CopyTensorAsync(source_tensor, *target_tensor, *ctx->GetComputeStream()));
          Y->Add(std::move(*target_tensor));
        }
        return Status::OK();
      }
      return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
    }
  }
};

namespace cuda {
ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

}  // namespace cuda

AllocatorPtr CUDAExecutionProvider::CreateCudaAllocator(OrtDevice::DeviceId device_id,
                                                        size_t gpu_mem_limit,
                                                        ArenaExtendStrategy arena_extend_strategy,
                                                        CUDAExecutionProviderExternalAllocatorInfo external_allocator_info,
                                                        OrtArenaCfg* default_memory_arena_cfg) {
  if (external_allocator_info.UseExternalAllocator()) {
    AllocatorCreationInfo default_memory_info(
        [external_allocator_info](OrtDevice::DeviceId id) {
          return std::make_unique<CUDAExternalAllocator>(id, CUDA,
                                                         external_allocator_info.alloc,
                                                         external_allocator_info.free,
                                                         external_allocator_info.empty_cache);
        },
        device_id,
        false);

    return CreateAllocator(default_memory_info);
  } else {
    AllocatorCreationInfo default_memory_info(
        [](OrtDevice::DeviceId id) {
          return std::make_unique<CUDAAllocator>(id, CUDA);
        },
        device_id,
        true,
        {default_memory_arena_cfg ? *default_memory_arena_cfg
                                  : OrtArenaCfg(gpu_mem_limit, static_cast<int>(arena_extend_strategy), -1, -1, -1)},
        // make it stream aware
        true,
        // enable cross stream sharing?
        false);

    // CUDA malloc/free is expensive so always use an arena
    return CreateAllocator(default_memory_info);
  }
}

CUDAExecutionProvider::PerThreadContext::PerThreadContext(OrtDevice::DeviceId device_id, cudaStream_t stream, size_t /*gpu_mem_limit*/,
                                                          ArenaExtendStrategy /*arena_extend_strategy*/, CUDAExecutionProviderExternalAllocatorInfo /*external_allocator_info*/,
                                                          OrtArenaCfg* /*default_memory_arena_cfg*/) {
  CUDA_CALL_THROW(cudaSetDevice(device_id));

  CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
  CUBLAS_CALL_THROW(cublasLtCreate(&cublas_lt_handle_));
  CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, stream));

  CUDNN_CALL_THROW(cudnnCreate(&cudnn_handle_));
  CUDNN_CALL_THROW(cudnnSetStream(cudnn_handle_, stream));

#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  cuda_graph_.SetStream(stream);
#else
  ORT_UNUSED_PARAMETER(stream);
#endif
}

CUDAExecutionProvider::PerThreadContext::~PerThreadContext() {
  ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasDestroy(cublas_handle_)));
  ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasLtDestroy(cublas_lt_handle_)));
  ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnDestroy(cudnn_handle_)));
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
bool CUDAExecutionProvider::PerThreadContext::IsGraphCaptureAllowed() const {
  return regular_run_count_before_graph_capture_ >= min_num_runs_before_cuda_graph_capture_;
}

void CUDAExecutionProvider::PerThreadContext::CaptureBegin() {
  cuda_graph_.Reset();
  cuda_graph_.CaptureBegin();
}

void CUDAExecutionProvider::PerThreadContext::CaptureEnd() {
  cuda_graph_.CaptureEnd();
  is_graph_captured_ = true;
}

bool CUDAExecutionProvider::PerThreadContext::IsGraphCaptured() const {
  return is_graph_captured_;
}

Status CUDAExecutionProvider::PerThreadContext::ReplayGraph() {
  ORT_ENFORCE(IsGraphCaptured());
  return cuda_graph_.Replay();
}

void CUDAExecutionProvider::PerThreadContext::IncrementRegularRunCountBeforeGraphCapture() {
  ++regular_run_count_before_graph_capture_;
}
#endif

void OverrideTunableOpInfoByEnv(CUDAExecutionProviderInfo& info) {
  auto env_tunable_op_enabled = onnxruntime::ParseTestOnlyEnvironmentVariable<bool>(
      "ORT_CUDA_TUNABLE_OP_ENABLED", {"0", "1"}, "Use provider_options \"tunable_op_enabled\" instead.");
  if (env_tunable_op_enabled.has_value() && env_tunable_op_enabled != info.tunable_op.enabled) {
    LOGS_DEFAULT(INFO) << "ORT_CUDA_TUNABLE_OP_ENABLED is set to " << *env_tunable_op_enabled;
    info.tunable_op.enabled = *env_tunable_op_enabled;
  }
}

CUDAExecutionProvider::CUDAExecutionProvider(const CUDAExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kCudaExecutionProvider},
      info_{info} {
  CUDA_CALL_THROW(cudaSetDevice(info_.device_id));

  // must wait GPU idle, otherwise cudaGetDeviceProperties might fail
  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaGetDeviceProperties(&device_prop_, info_.device_id));

  // This scenario is not supported.
  ORT_ENFORCE(!(info.has_user_compute_stream && info.external_allocator_info.UseExternalAllocator()));

  if (info.has_user_compute_stream) {
    external_stream_ = true;
    use_ep_level_unified_stream_ = true;
    stream_ = static_cast<cudaStream_t>(info.user_compute_stream);
  } else {
    if (info.external_allocator_info.UseExternalAllocator()) {
      use_ep_level_unified_stream_ = true;
      stream_ = nullptr;
    } else if (info.enable_cuda_graph) {
      // current cuda graph implementation only works with single stream
      // use EP level unified stream for all the reqeust
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
      use_ep_level_unified_stream_ = true;
    } else {
      stream_ = nullptr;
    }
  }

  size_t free = 0;
  size_t total = 0;
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));

  OverrideTunableOpInfoByEnv(info_);
}

CUDAExecutionProvider::~CUDAExecutionProvider() {
  // clean up thread local context caches
  {
    std::lock_guard<OrtMutex> lock(context_state_.mutex);
    for (const auto& cache_weak : context_state_.caches_to_update_on_destruction) {
      const auto cache = cache_weak.lock();
      if (!cache) continue;
      ORT_IGNORE_RETURN_VALUE(cache->erase(this));
    }
  }

  if (!external_stream_ && stream_) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(stream_)));
  }
}

void CUDAExecutionProvider::EnableTunableOp() {
  LOGS_DEFAULT(INFO) << "Enable TunableOp for CUDA Execution Provider";
  info_.tunable_op.enabled = true;
}

void CUDAExecutionProvider::DisableTunableOp() {
  LOGS_DEFAULT(INFO) << "Disable TunableOp for CUDA Execution Provider";
  info_.tunable_op.enabled = false;
}

bool CUDAExecutionProvider::IsTunableOpEnabled() const {
  return info_.tunable_op.enabled;
}

std::unique_ptr<profiling::EpProfiler> CUDAExecutionProvider::GetProfiler() {
  return std::make_unique<profiling::CudaProfiler>();
}

// Suppressing warning "C26816: The pointer points to memory allocated on the stack." for
// CUDAExecutionProvider::GetPerThreadContext().
// While CUDAExecutionProvider::GetPerThreadContext() does return the result of dereferencing a local
// std::shared_ptr<PerThreadContext>, the underlying PerThreadContext is owned by the CUDA EP and is not local to this
// function.
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 26816)
#endif

CUDAExecutionProvider::PerThreadContext& CUDAExecutionProvider::GetPerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  // try to use cached context
  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    ORT_ENFORCE(cached_context);
    return *cached_context;
  }

  // get context and update cache
  std::shared_ptr<PerThreadContext> context;
  {
    std::lock_guard<OrtMutex> lock(context_state_.mutex);

    // get or create a context
    if (context_state_.retired_context_pool.empty()) {
      context = std::make_shared<PerThreadContext>(info_.device_id, stream_, info_.gpu_mem_limit,
                                                   info_.arena_extend_strategy, info_.external_allocator_info, info_.default_memory_arena_cfg);
    } else {
      context = context_state_.retired_context_pool.back();
      context_state_.retired_context_pool.pop_back();
    }

    // insert into active_contexts, should not already be present
    const auto active_contexts_insert_result = context_state_.active_contexts.insert(context);
    ORT_ENFORCE(active_contexts_insert_result.second);

    // insert into caches_to_update_on_destruction, may already be present
    ORT_IGNORE_RETURN_VALUE(context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache));
  }

  per_thread_context_cache->insert(std::make_pair(this, context));

  return *context;
}

#ifdef _WIN32
#pragma warning(pop)
#endif

void CUDAExecutionProvider::ReleasePerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  ORT_ENFORCE(cached_context_it != per_thread_context_cache->end());
  auto cached_context = cached_context_it->second.lock();
  ORT_ENFORCE(cached_context);

  {
    std::lock_guard<OrtMutex> lock(context_state_.mutex);
    context_state_.active_contexts.erase(cached_context);
    context_state_.retired_context_pool.push_back(cached_context);
  }

  per_thread_context_cache->erase(cached_context_it);
}

AllocatorPtr CUDAExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    auto cuda_alloc = IExecutionProvider::GetAllocator(id, mem_type);
    if (!cuda_alloc) {
      // this means the program invoke GetAllocator before RegsiterAllocators,
      // which only happnens in some UTs.
      // here is a hack to return another allocator instance.
      // need to fix this in the future.
      return CreateCudaAllocator(info_.device_id, info_.gpu_mem_limit, info_.arena_extend_strategy,
                                 info_.external_allocator_info, info_.default_memory_arena_cfg);
    }
  }

  return IExecutionProvider::GetAllocator(id, mem_type);
}

Status CUDAExecutionProvider::Sync() const {
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunStart() {
  // always set CUDA device when session::Run() in case it runs in a worker thread
  CUDA_RETURN_IF_ERROR(cudaSetDevice(GetDeviceId()));
  if (IsGraphCaptureEnabled() && GetPerThreadContext().IsGraphCaptureAllowed() && !GetPerThreadContext().IsGraphCaptured()) {
    LOGS_DEFAULT(INFO) << "Capturing the cuda graph for this model";
    GetPerThreadContext().CaptureBegin();
  }
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunEnd(bool sync_stream) {
  if (IsGraphCaptureEnabled() && !GetPerThreadContext().IsGraphCaptured()) {
    if (GetPerThreadContext().IsGraphCaptureAllowed()) {
      GetPerThreadContext().CaptureEnd();
      // CUDA work issued to a capturing stream doesn’t actually run on the GPU,
      // so run the captured graph here to actually execute the work.
      ORT_RETURN_IF_ERROR(GetPerThreadContext().ReplayGraph());
    } else {
      GetPerThreadContext().IncrementRegularRunCountBeforeGraphCapture();
    }
  }

  if (sync_stream) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
  }

  // The reason of !IsGraphCaptureEnabled():
  //  If cuda graph is enabled, the per thread context will not be released
  //  because the per thread cuda graph needs to be maintained and replayed for
  //  the next run.
  // The reason of PerThreadContextCache()->find(this) != PerThreadContextCache()->end():
  //  In extreme cases (e.g., 1-op graph and that op fallbacks to CPU),
  //  PerThreadContext won't be created and there isbe nothing to
  //  release. This didn't happen before because we always call
  //  GetPerThreadContext in OnRunStart.
  if (!IsGraphCaptureEnabled() &&
      PerThreadContextCache()->find(this) != PerThreadContextCache()->end()) {
    ReleasePerThreadContext();
  }

  return Status::OK();
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
bool CUDAExecutionProvider::IsGraphCaptureEnabled() const {
  return info_.enable_cuda_graph;
}

bool CUDAExecutionProvider::IsGraphCaptured() const {
  return GetPerThreadContext().IsGraphCaptured();
}

Status CUDAExecutionProvider::ReplayGraph() {
  return GetPerThreadContext().ReplayGraph();
}
#endif

namespace cuda {
// opset 1 to 9
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, 10, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, double, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, float, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, double, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Tanh);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Tanh);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, float, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, double, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, MLFloat16, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, Sum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Sum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Max);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Max);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Min);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint16_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 10, float, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Erf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Erf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, bool, Not);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, float, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, double, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSumExp);
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
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, 12, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, Tile);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Tile);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Compress);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, float, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, double, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, MLFloat16, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, int32_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, uint8_t, Upsample);
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
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Less);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, EyeLike);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scatter);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, uint8_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, NonZero);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, TopK);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 8, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, SpaceToDepth);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomNormal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomNormalLike);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomUniform);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomUniformLike);

// opset 10
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 11, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, uint8_t, Resize);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, ReverseSequence);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, TopK);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, Mod);

// opset 11
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ArgMin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Compress);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Gather);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, GatherElements);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int64_t, GatherND);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Loop);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, NonMaxSuppression);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Range);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 15, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterElements);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, TopK);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceAt);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceConstruct);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceEmpty);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceLength);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, ConcatFromSequence);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceErase);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceInsert);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, QuantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, uint8_t, QuantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 13, CumSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int64_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int64_t_float_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int32_t_float_int32_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int64_t_MLFloat16_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int32_t_MLFloat16_int32_t, OneHot);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace);

// OpSet 12
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Clip);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, double, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, MLFloat16, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, int8_t, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, uint8_t, MaxPool);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Pow);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, uint8_t, ReduceMax);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, uint8_t, ReduceMin);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, GatherND);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Dropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, Einsum);

// OpSet 13
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Add);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Expand);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Sum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Max);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Size);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Transpose);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, ScatterElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LogSoftmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LogSoftmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Split);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Unsqueeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Gather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint8_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, GatherND);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Resize);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Loop);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LRN);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, ScatterND);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, Pad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, SpaceToDepth);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, DepthToSpace);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, ReduceSum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Mod);

// OpSet 14
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, CumSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Div);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 14, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 14, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 14, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, ReduceMin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Trilu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Relu);

// OpSet 15
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, double, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, Shape);

// Opset 16
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, PRelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, Scan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint8_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LessOrEqual);

// Opset 17
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, float, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, double, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, BFloat16, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, MLFloat16, LayerNormalization);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  return {};
}

static Status RegisterCudaKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, float, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, double, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 10, float, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, float, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, double, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, float, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, double, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, bool, Not)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, float, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, double, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSumExp)>,
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
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, 12, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, double, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, MLFloat16, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, uint8_t, Upsample)>,
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
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, EyeLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scatter)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, uint8_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 8, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomNormal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomNormalLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomUniform)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, RandomUniformLike)>,

      // opset 10
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 11, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, ReverseSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, uint8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, uint8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, Mod)>,

      // opset 11
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 15, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceAt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceConstruct)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceEmpty)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceLength)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, ConcatFromSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceErase)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceInsert)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, float, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, double, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, MLFloat16, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 13, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int64_t_int64_t_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int64_t_float_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int32_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int64_t_MLFloat16_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, int32_t_MLFloat16_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace)>,

      // OpSet 12
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Clip)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, int8_t, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, uint8_t, MaxPool)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Pow)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, uint8_t, ReduceMax)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, uint8_t, ReduceMin)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, GatherND)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, Einsum)>,

      // OpSet 13
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, DepthToSpace)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Mod)>,

      // OpSet 14
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 14, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 14, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 14, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Trilu)>,

      // OpSet 15
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, Shape)>,

      // Opset 16
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint8_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LessOrEqual)>,

      // Opset 17
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, float, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, double, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, BFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, MLFloat16, LayerNormalization)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

#ifndef DISABLE_CONTRIB_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::cuda::RegisterCudaContribKernels(kernel_registry));
#endif

#ifdef ENABLE_TRAINING_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::cuda::RegisterCudaTrainingKernels(kernel_registry));
#endif

  return Status::OK();
}

}  // namespace cuda

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void InitializeRegistry() {
  s_kernel_registry = KernelRegistry::Create();
  ORT_THROW_IF_ERROR(cuda::RegisterCudaKernels(*s_kernel_registry));
}

void DeleteRegistry() {
  s_kernel_registry.reset();
}

std::shared_ptr<KernelRegistry> CUDAExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

static bool RNNNeedFallbackToCPU(const onnxruntime::Node& node,
                                 const std::vector<std::string> activations_supported,
                                 const std::string& op_type) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto& attr_name = attr.first;
    auto& attr_value = attr.second;

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

static bool ConvTransposeNeedFallbackToCPU(const onnxruntime::Node& node) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto& attr_name = attr.first;
    auto& attr_value = attr.second;

    // cudnn only supports symmetric padding, so drop the node down to CPU if the padding provided is asymmetric
    // TODO: Check if we can adopt a similar approach to deal with asymmetric pads in 'ConvTranspose'
    // as we did for 'Conv' to circumvent the cudnn limitation
    if ("pads" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS == attr_value.type()) {
      auto& pads = attr_value.ints();
      int pads_size = pads.size();
      ORT_ENFORCE(pads_size % 2 == 0);
      int rank = pads_size / 2;
      for (int i = 0; i < rank; i++) {
        if (pads.Get(i) != pads.Get(i + rank)) {
          LOGS_DEFAULT(WARNING) << "Dropping the ConvTranspose node: " << node.Name()
                                << " to CPU because it requires asymmetric padding which the CUDA EP"
                                << " currently does not support";
          return true;
        }
      }
    }

    if ("auto_pad" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING == attr_value.type()) {
      auto& auto_pad_attr = attr_value.s();
      ORT_ENFORCE(auto_pad_attr == "SAME_UPPER" || auto_pad_attr == "SAME_LOWER" ||
                      auto_pad_attr == "VALID" || auto_pad_attr == "NOTSET",
                  "auto_pad must be either NOTSET, VALID, SAME_UPPER, SAME_LOWER");

      // If auto_pad is SAME_UPPER or SAME_LOWER, pads will be computed dynamically at runtime
      // based on the provided input shape. This may or may not lead to symmetric padding.
      // If it turns out to be asymmetric padding, CuDNN will return a cryptic unfriendly error message.
      // So drop down the node to CPU if auto_pad is SAME_UPPER or SAME_LOWER even if it may lead to
      // symmetric padding.
      // TODO: Remove this after we have supported asymmetric padding in the CUDA ConvTranspose kernel
      if (auto_pad_attr == "SAME_UPPER" || auto_pad_attr == "SAME_LOWER") {
        LOGS_DEFAULT(WARNING) << "Dropping the ConvTranspose node: " << node.Name()
                              << " to CPU because it uses the auto_pad attribute which may lead to asymmetric padding which"
                              << " the CUDA EP currently does not support";
        return true;
      }
    }
  }

  return false;
}

static bool CastNeedFallbackToCPU(const onnxruntime::Node& node) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto& attr_name = attr.first;
    auto& attr_value = attr.second;

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
  return std::make_unique<onnxruntime::GPUDataTransfer>();
}

std::vector<std::unique_ptr<ComputeCapability>>
CUDAExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                     const IKernelLookup& kernel_lookup) const {
  InlinedVector<NodeIndex> candidates;
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      continue;
    }

    const KernelCreateInfo* cuda_kernel_def = kernel_lookup.LookUpKernel(node);
    // none of the provided registries has a CUDA kernel for this node
    if (cuda_kernel_def == nullptr) {
      LOGS_DEFAULT(INFO) << "CUDA kernel not found in registries for Op type: " << node.OpType() << " node name: " << node.Name();
      continue;
    }

    bool not_supported = false;
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
    } else if ("ConvTranspose" == node.OpType()) {
      not_supported = ConvTransposeNeedFallbackToCPU(node);
      force_inside = !not_supported;
    } else if ("Cast" == node.OpType()) {
      not_supported = CastNeedFallbackToCPU(node);
      // cast is not compute heavy, and may be placed outside
    }

    if (!force_inside && not_supported) {
      if (not_supported) {
        LOGS_DEFAULT(WARNING) << "CUDA kernel not supported. Fallback to CPU execution provider for Op type: " << node.OpType() << " node name: " << node.Name();
      }
    } else {
      candidates.push_back(node.Index());
    }
  }

  // For CUDA EP, exclude the subgraph that is preferred to be placed in CPU
  // These are usually shape related computation subgraphs
  // Following logic can be extended for other EPs
  auto cpu_nodes = GetCpuPreferredNodes(graph, kernel_lookup, candidates);
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    if (cpu_nodes.count(node_index) > 0)
      continue;

    auto sub_graph = IndexedSubGraph::Create();
    sub_graph->Nodes().push_back(node_index);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }
  /*
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    auto sub_graph = IndexedSubGraph::Create();
    sub_graph->Nodes().push_back(node_index);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }*/
  return result;
}

void CUDAExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
  OrtDevice cuda_device{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, info_.device_id};
  OrtDevice pinned_device{OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};
  OrtDevice cpu_device{OrtDevice::CPU, OrtDevice::MemType::DEFAULT, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};

  // setup CUDA allocator
  // if EP is used in multiple inference sessions we may already have an allocator. if so use that.
  // NOTE: We call IExecutionProvider::GetAllocator as CUDAExecutionProvider::GetAllocator will return
  //       a per-thread allocator for OrtMemTypeDefault.
  auto cuda_alloc = IExecutionProvider::GetAllocator(cuda_device.Id(), OrtMemTypeDefault);
  if (!cuda_alloc) {
    // use shared allocator if available
    cuda_alloc = allocator_manager.GetAllocator(OrtMemTypeDefault, cuda_device);

    if (!cuda_alloc) {
      cuda_alloc = CreateCudaAllocator(info_.device_id, info_.gpu_mem_limit, info_.arena_extend_strategy,
                                       info_.external_allocator_info, info_.default_memory_arena_cfg);
      // enable sharing of our allocator
      allocator_manager.InsertAllocator(cuda_alloc);
    }

    InsertAllocator(cuda_alloc);
  }

  // OrtMemTypeCPUOutput -- allocated by cudaMallocHost, used to copy CUDA device memory to CPU
  // Use pinned memory instead of pageable memory make the data transfer faster
  // Used by node MemcpyToHost only
  auto cuda_pinned_alloc = IExecutionProvider::GetAllocator(pinned_device.Id(), OrtMemTypeCPUOutput);
  if (!cuda_pinned_alloc) {
    cuda_pinned_alloc = allocator_manager.GetAllocator(OrtMemTypeCPUOutput, pinned_device);

    if (!cuda_pinned_alloc) {
      AllocatorCreationInfo pinned_memory_info(
          [](OrtDevice::DeviceId device_id) {
            return std::make_unique<CUDAPinnedAllocator>(device_id, CUDA_PINNED);
          },
          // TODO: should we use info_.device_id instead of DEFAULT_CPU_ALLOCATOR_DEVICE_ID?
          // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb
          // says the pinned memory allocated by cudaMallocHost is associated with a specific device, so it may be more
          // correct to use the GPU device id, unless we wanted to share the pinned memory allocator across devices,
          // at the risk the lifetime isn't managed correctly if one of those devices go away.
          pinned_device.Id());

      cuda_pinned_alloc = CreateAllocator(pinned_memory_info);
      allocator_manager.InsertAllocator(cuda_pinned_alloc);
    }

    InsertAllocator(cuda_pinned_alloc);
  }

  // OrtMemTypeCPUInput -- op place the input on CPU and will not be accessed by CUDA kernel, no sync issue
  auto cuda_cpu_alloc = IExecutionProvider::GetAllocator(cpu_device.Id(), OrtMemTypeCPUInput);
  if (!cuda_cpu_alloc) {
    cuda_cpu_alloc = allocator_manager.GetAllocator(OrtMemTypeCPUInput, cpu_device);

    if (!cuda_cpu_alloc) {
      // TODO: this is actually used for the cuda kernels which explicitly ask for inputs from CPU.
      // This will be refactored/removed when allocator and execution provider are decoupled.
      // Need to move the OrtMemoryType out of Allocator, that's one thing blocking us to share it with CPU EP
      // CPUAllocator is OrtMemTypeDefault for CPU EP
      AllocatorCreationInfo cpu_memory_info(
          [](int device_id) {
            return std::make_unique<CPUAllocator>(
                OrtMemoryInfo("CUDA_CPU", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), device_id,
                              OrtMemTypeCPUInput));
          },
          cpu_device.Id());

      cuda_cpu_alloc = CreateAllocator(cpu_memory_info);
      allocator_manager.InsertAllocator(cuda_cpu_alloc);
    }

    InsertAllocator(cuda_cpu_alloc);
  }
}

void CUDAExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry) const {
  // This allocator must be the same to the allocator
  // used in AllocateBufferOnCPUPinned.
  auto allocator = GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
  if (use_ep_level_unified_stream_)
    RegisterCudaStreamHandles(stream_handle_registry,
                              OrtDevice::GPU,
                              allocator,
                              !IsGraphCaptureEnabled(),
                              stream_,
                              use_ep_level_unified_stream_,
                              GetPerThreadContext().CudnnHandle(),
                              GetPerThreadContext().CublasHandle());
  else
    RegisterCudaStreamHandles(stream_handle_registry,
                              OrtDevice::GPU,
                              allocator,
                              !IsGraphCaptureEnabled(),
                              stream_,
                              use_ep_level_unified_stream_,
                              GetPerThreadContext().CudnnHandle(),
                              GetPerThreadContext().CublasHandle());
}

}  // namespace onnxruntime
