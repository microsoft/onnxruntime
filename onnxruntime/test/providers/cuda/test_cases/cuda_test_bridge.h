// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#if !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
#include <cudnn_version.h>
#endif

#include "core/framework/external_data_loader.h"

// Include after the caller's core or provider headers have declared Node.
#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"
#include "contrib_ops/cuda/bert/packed_attention_workspace_estimate.h"
#endif
#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM
#include "contrib_ops/cuda/quantization/matmul_nbits_workspace_estimate.h"
#endif

namespace onnxruntime::test {

#if !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN) && CUDNN_MAJOR >= 9
struct ConvPlanCacheSnapshot {
  std::shared_ptr<void> conv_plan;
  size_t cached_plan_count = 0;
  size_t workspace_bytes = 0;
  size_t plan_workspace_bytes = 0;
  std::vector<int64_t> last_x_dims;
  bool conv_plan_matches_inputs = false;
  bool bias_fused = false;
  const void* x_binding = nullptr;
  const void* w_binding = nullptr;
  const void* b_binding = nullptr;
  const void* y_binding = nullptr;
  const void* z_binding = nullptr;
};

ConvPlanCacheSnapshot GetConvPlanCacheForTest(const void* kernel, int32_t element_type);
#endif

// Node and Tensor are borrowed core objects. The provider-side accessors forward
// their addresses back to ProviderHost; neither type crosses this link boundary.
#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
std::optional<contrib::cuda::GQAWorkspaceEstimateConfig> GetGroupQueryAttentionWorkspaceEstimateConfigForTest(
    const void* node, bool head_sink_is_constant_initializer = false,
    int64_t max_total_sequence_length = 0);

std::optional<contrib::cuda::GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspaceForTest(
    const void* node, gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop, const AttentionKernelOptions& kernel_options,
    bool head_sink_is_constant_initializer = false,
    int64_t max_total_sequence_length = 0);

std::optional<contrib::cuda::PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspaceForTest(
    const void* node, gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop, const AttentionKernelOptions& kernel_options);
#endif

#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM
std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemoryForTest(
    const void* node, const cudaDeviceProp& device_prop,
    contrib::cuda::MatMulNBitsMemoryEstimateOptions options = {});
std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemoryForTest(
    const void* node, gsl::span<const int64_t> input_shape, const cudaDeviceProp& device_prop,
    contrib::cuda::MatMulNBitsMemoryEstimateOptions options = {});
std::optional<size_t> EstimateMatMulNBitsWorkspaceForTest(
    const void* node, const cudaDeviceProp& device_prop);
std::optional<size_t> EstimateMatMulNBitsWorkspaceForTest(
    const void* node, gsl::span<const int64_t> input_shape, const cudaDeviceProp& device_prop);
#endif

common::Status LoadCudaExternalDataForTest(
    const void* loader, const Env& env, const std::filesystem::path& path,
    FileOffsetType offset, SafeInt<size_t> length, void* tensor);

}  // namespace onnxruntime::test
