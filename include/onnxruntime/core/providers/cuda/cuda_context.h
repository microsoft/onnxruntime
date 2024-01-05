// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This header is to expose a context for cuda custom ops.
// By the context, a custom cuda operator could fetch existing resources,
// such as cuda stream and cudnn handle, for reusing.

// For concrete usage, pls find page here:
// https://onnxruntime.ai/docs/reference/operators/add-custom-op.html#custom-ops-for-cuda-and-rocm

#pragma once

#define ORT_CUDA_CTX

#include "cuda_resource.h"
#include "core/providers/custom_op_context.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

namespace Ort {

namespace Custom {

struct CudaContext : public CustomOpContext {
  cudaStream_t cuda_stream = {};
  cudnnHandle_t cudnn_handle = {};
  cublasHandle_t cublas_handle = {};
  OrtAllocator* deferred_cpu_allocator = {};
  // below are cuda ep options
  int16_t device_id = 0;
  bool has_user_compute_stream = false;
  size_t gpu_mem_limit = 0;
  int32_t arena_extend_strategy = 0;
  int32_t cudnn_conv_algo_search = 0;
  bool do_copy_in_default_stream = true;
  void* gpu_external_alloc = {};
  void* gpu_external_free = {};
  void* gpu_external_empty_cache = {};
  bool cudnn_conv_use_max_workspace = true;
  bool enable_cuda_graph = false;
  bool cudnn_conv1d_pad_to_nc1d = false;
  bool tunable_op_enable = false;
  bool tunable_op_tuning_enable = false;
  int32_t tunable_op_max_tuning_duration_ms = 0;
  bool enable_skip_layer_norm_strict_mode = false;
  bool prefer_nhwc = false;
  bool use_ep_level_unified_stream = false;

  void Init(const OrtKernelContext& kernel_ctx) {
    cuda_stream = FetchResource<cudaStream_t>(kernel_ctx, CudaResource::cuda_stream_t);
    cudnn_handle = FetchResource<cudnnHandle_t>(kernel_ctx, CudaResource::cudnn_handle_t);
    cublas_handle = FetchResource<cublasHandle_t>(kernel_ctx, CudaResource::cublas_handle_t);
    deferred_cpu_allocator = FetchResource<OrtAllocator*>(kernel_ctx, CudaResource::deferred_cpu_allocator_t);

    device_id = FetchResource<int16_t>(kernel_ctx, CudaResource::device_id_t);
    has_user_compute_stream = FetchResource<bool>(kernel_ctx, CudaResource::has_user_compute_stream_t);
    gpu_mem_limit = FetchResource<size_t>(kernel_ctx, CudaResource::gpu_mem_limit_t);
    arena_extend_strategy = FetchResource<int32_t>(kernel_ctx, CudaResource::arena_extend_strategy_t);

    cudnn_conv_algo_search = FetchResource<int32_t>(kernel_ctx, CudaResource::cudnn_conv_algo_search_t);
    do_copy_in_default_stream = FetchResource<bool>(kernel_ctx, CudaResource::do_copy_in_default_stream_t);
    gpu_external_alloc = FetchResource<void*>(kernel_ctx, CudaResource::gpu_external_alloc_t);
    gpu_external_free = FetchResource<void*>(kernel_ctx, CudaResource::gpu_external_free_t);

    gpu_external_empty_cache = FetchResource<void*>(kernel_ctx, CudaResource::gpu_external_empty_cache_t);
    cudnn_conv_use_max_workspace = FetchResource<bool>(kernel_ctx, CudaResource::cudnn_conv_use_max_workspace_t);
    enable_cuda_graph = FetchResource<bool>(kernel_ctx, CudaResource::enable_cuda_graph_t);
    cudnn_conv1d_pad_to_nc1d = FetchResource<bool>(kernel_ctx, CudaResource::cudnn_conv1d_pad_to_nc1d_t);

    tunable_op_enable = FetchResource<bool>(kernel_ctx, CudaResource::tunable_op_enable_t);
    tunable_op_tuning_enable = FetchResource<bool>(kernel_ctx, CudaResource::tunable_op_tuning_enable_t);
    tunable_op_max_tuning_duration_ms = FetchResource<int32_t>(kernel_ctx, CudaResource::tunable_op_max_tuning_duration_ms_t);
    enable_skip_layer_norm_strict_mode = FetchResource<bool>(kernel_ctx, CudaResource::enable_skip_layer_norm_strict_mode_t);

    prefer_nhwc = FetchResource<bool>(kernel_ctx, CudaResource::prefer_nhwc_t);
    use_ep_level_unified_stream = FetchResource<bool>(kernel_ctx, CudaResource::use_ep_level_unified_stream_t);
  }

  template<typename T>
  T FetchResource(const OrtKernelContext& kernel_ctx, CudaResource resource_type) {
    const auto& ort_api = Ort::GetApi();
    void* resource = {};
    OrtStatus* status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_CUDA_RESOUCE_VERSION, resource_type, &resource);
    if (status) {
      ORT_CXX_API_THROW("Failed to fetch cuda ep resource, resouce type: " + std::to_string(resource_type), OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    return static_cast<T>(*reinterpret_cast<T*>(&resource));
  }

  void* AllocDeferredCpuMem(size_t size) const {
    if (0 == size) {
      return {};
    }
    const auto& ort_api = Ort::GetApi();
    void* mem = {};
    auto status = ort_api.AllocatorAlloc(deferred_cpu_allocator, size, &mem);
    if (status) {
      ORT_CXX_API_THROW("failed to allocate deferred cpu memory", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    return mem;
  }

  void FreeDeferredCpuMem(void* mem) const {
    if (mem) {
      const auto& ort_api = Ort::GetApi();
      auto status = ort_api.AllocatorFree(deferred_cpu_allocator, mem);
      if (status) {
        ORT_CXX_API_THROW("failed to free deferred cpu memory", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
      }
    }
  }
};

}  // namespace Custom
}  // namespace Ort
