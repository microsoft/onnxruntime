// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_external_data_loader.h"
#include "test/providers/cuda/test_cases/cuda_test_bridge.h"

namespace onnxruntime::test {

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
std::optional<contrib::cuda::GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspaceForTest(
    const void* node, gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop, const AttentionKernelOptions& kernel_options,
    bool head_sink_is_constant_initializer) {
  return contrib::cuda::EstimateGroupQueryAttentionWorkspace(
      *static_cast<const Node*>(node), input_shapes, device_prop, kernel_options,
      head_sink_is_constant_initializer);
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
