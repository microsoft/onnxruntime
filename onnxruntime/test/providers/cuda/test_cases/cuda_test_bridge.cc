// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_external_data_loader.h"
#include "test/providers/cuda/test_cases/cuda_test_bridge.h"

#if !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN) && CUDNN_MAJOR >= 9
#include "core/providers/cuda/nn/conv.h"

namespace onnxruntime::cuda {

struct ConvPlanCacheTestPeer {
  template <typename T>
  static test::ConvPlanCacheSnapshot Snapshot(const void* kernel) {
    const auto& state = static_cast<const Conv<T, false>*>(kernel)->s_;
    test::ConvPlanCacheSnapshot snapshot;
    snapshot.conv_plan = state.conv_plan;
    snapshot.cached_plan_count = state.cached_conv_plans.size();
    snapshot.last_x_dims.assign(state.last_x_dims.GetDims().begin(), state.last_x_dims.GetDims().end());
    snapshot.conv_plan_matches_inputs = state.conv_plan_matches_inputs;
    if (state.conv_plan && state.conv_plan_matches_inputs) {
      snapshot.workspace_bytes = state.workspace_bytes;
      snapshot.plan_workspace_bytes = state.conv_plan->workspace_bytes;
      snapshot.bias_fused = state.conv_plan->bias_fused;
      snapshot.x_binding = state.variant_pack.at(state.conv_plan->X);
      snapshot.w_binding = state.variant_pack.at(state.conv_plan->W);
      snapshot.y_binding = state.variant_pack.at(state.conv_plan->Y);
      if (state.conv_plan->bias_fused && state.b_data) {
        snapshot.b_binding = state.variant_pack.at(state.conv_plan->B);
      }
    }
    return snapshot;
  }
};

}  // namespace onnxruntime::cuda
#endif

namespace onnxruntime::test {

#if !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN) && CUDNN_MAJOR >= 9
ConvPlanCacheSnapshot GetConvPlanCacheForTest(const void* kernel, bool bfloat16) {
  return bfloat16 ? cuda::ConvPlanCacheTestPeer::Snapshot<BFloat16>(kernel)
                  : cuda::ConvPlanCacheTestPeer::Snapshot<float>(kernel);
}
#endif

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
std::optional<contrib::cuda::GQAWorkspaceEstimateConfig> GetGroupQueryAttentionWorkspaceEstimateConfigForTest(
    const void* node, bool head_sink_is_constant_initializer,
    int64_t max_total_sequence_length) {
  return contrib::cuda::GetGroupQueryAttentionWorkspaceEstimateConfig(
      *static_cast<const Node*>(node), head_sink_is_constant_initializer,
      max_total_sequence_length);
}

std::optional<contrib::cuda::GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspaceForTest(
    const void* node, gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop, const AttentionKernelOptions& kernel_options,
    bool head_sink_is_constant_initializer,
    int64_t max_total_sequence_length) {
  return contrib::cuda::EstimateGroupQueryAttentionWorkspace(
      *static_cast<const Node*>(node), input_shapes, device_prop, kernel_options,
      head_sink_is_constant_initializer, max_total_sequence_length);
}

std::optional<contrib::cuda::PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspaceForTest(
    const void* node, gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop, const AttentionKernelOptions& kernel_options) {
  return contrib::cuda::EstimatePackedAttentionWorkspace(
      *static_cast<const Node*>(node), input_shapes, device_prop, kernel_options);
}
#endif

#if !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM
std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemoryForTest(
    const void* node, const cudaDeviceProp& device_prop,
    contrib::cuda::MatMulNBitsMemoryEstimateOptions options) {
  return contrib::cuda::EstimateMatMulNBitsMemory(*static_cast<const Node*>(node), device_prop, options);
}

std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemoryForTest(
    const void* node, gsl::span<const int64_t> input_shape, const cudaDeviceProp& device_prop,
    contrib::cuda::MatMulNBitsMemoryEstimateOptions options) {
  return contrib::cuda::EstimateMatMulNBitsMemory(
      *static_cast<const Node*>(node), input_shape, device_prop, options);
}

std::optional<size_t> EstimateMatMulNBitsWorkspaceForTest(
    const void* node, const cudaDeviceProp& device_prop) {
  return contrib::cuda::EstimateMatMulNBitsWorkspace(*static_cast<const Node*>(node), device_prop);
}

std::optional<size_t> EstimateMatMulNBitsWorkspaceForTest(
    const void* node, gsl::span<const int64_t> input_shape, const cudaDeviceProp& device_prop) {
  return contrib::cuda::EstimateMatMulNBitsWorkspace(*static_cast<const Node*>(node), input_shape, device_prop);
}
#endif

common::Status LoadCudaExternalDataForTest(
    const void* loader, const Env& env, const std::filesystem::path& path,
    FileOffsetType offset, SafeInt<size_t> length, void* tensor) {
  std::unique_ptr<RandomAccessFile> file;
  ORT_RETURN_IF_ERROR(env.OpenRandomAccessFile(path.c_str(), file));
  return static_cast<const cuda::ExternalDataLoader*>(loader)->LoadTensor(
      *file, offset, length, *static_cast<Tensor*>(tensor));
}

}  // namespace onnxruntime::test
