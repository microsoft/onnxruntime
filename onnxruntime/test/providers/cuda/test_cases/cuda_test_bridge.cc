// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_external_data_loader.h"
#include "test/providers/cuda/test_cases/cuda_test_bridge.h"

#if !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN) && CUDNN_MAJOR >= 9
#include "core/providers/cuda/nn/conv.h"

namespace onnxruntime::cuda {

struct ConvPlanCacheTestPeer {
  template <typename T, bool Layout = false>
  static test::ConvPlanCacheSnapshot Snapshot(const void* kernel) {
    const auto& conv = *static_cast<const Conv<T, Layout>*>(kernel);
    const auto& state = conv.s_;
    test::ConvPlanCacheSnapshot snapshot;
    snapshot.channels_last = Layout;
    snapshot.weights_in_nhwc = conv.W_ != nullptr || conv.W_already_nhwc;
    if (conv.W_) {
      const auto dims = conv.W_->Shape().GetDims();
      snapshot.prepacked_weight_dims.assign(dims.begin(), dims.end());
      snapshot.prepacked_weight_data = conv.W_->DataRaw();
    }
    snapshot.conv_plan = state.conv_plan;
    snapshot.cached_plan_count = state.cached_conv_plans.size();
    snapshot.plan_key_storage = state.conv_plan_key.data();
    snapshot.last_x_dims.assign(state.last_x_dims.GetDims().begin(), state.last_x_dims.GetDims().end());
    snapshot.conv_plan_matches_inputs = state.conv_plan_matches_inputs;
    if (state.conv_plan && state.conv_plan_matches_inputs) {
      snapshot.workspace_bytes = state.workspace_bytes;
      snapshot.plan_workspace_bytes = state.conv_plan->workspace_bytes;
      snapshot.bias_fused = state.conv_plan->bias_fused;
      const auto& bindings = state.conv_plan->variant_pack;
      snapshot.binding_count = bindings.size();
      snapshot.x_binding_slot = &bindings.at(state.conv_plan->X);
      snapshot.x_binding = bindings.at(state.conv_plan->X);
      snapshot.w_binding = bindings.at(state.conv_plan->W);
      snapshot.y_binding = bindings.at(state.conv_plan->Y);
      if (state.conv_plan->bias_fused && state.b_data) {
        snapshot.b_binding = bindings.at(state.conv_plan->B);
      }
      if (state.conv_plan->bias_fused && state.conv_plan->Z) {
        snapshot.z_binding = bindings.at(state.conv_plan->Z);
      }
    }
    return snapshot;
  }
};

}  // namespace onnxruntime::cuda
#endif

namespace onnxruntime::test {

#if !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN) && CUDNN_MAJOR >= 9
ConvPlanCacheSnapshot GetConvPlanCacheForTest(const void* kernel, int32_t element_type, bool channels_last) {
#ifdef ENABLE_CUDA_NHWC_OPS
  if (channels_last) {
    switch (element_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        return cuda::ConvPlanCacheTestPeer::Snapshot<float, true>(kernel);
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        return cuda::ConvPlanCacheTestPeer::Snapshot<MLFloat16, true>(kernel);
      default:
        ORT_THROW("Unsupported NHWC Conv plan cache test element type: ", element_type);
    }
  }
#else
  ORT_ENFORCE(!channels_last, "NHWC Conv kernels are not enabled.");
#endif
  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return cuda::ConvPlanCacheTestPeer::Snapshot<float>(kernel);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return cuda::ConvPlanCacheTestPeer::Snapshot<MLFloat16>(kernel);
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return cuda::ConvPlanCacheTestPeer::Snapshot<BFloat16>(kernel);
    default:
      ORT_THROW("Unsupported Conv plan cache test element type: ", element_type);
  }
}
#endif

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
std::optional<contrib::cuda::GQAWorkspaceEstimateConfig> GetGroupQueryAttentionWorkspaceEstimateConfigForTest(
    const void* node, bool head_sink_is_constant_initializer,
    int64_t max_total_sequence_length,
    KernelDispatchPolicy dispatch_policy) {
  return contrib::cuda::GetGroupQueryAttentionWorkspaceEstimateConfig(
      *static_cast<const Node*>(node), head_sink_is_constant_initializer,
      max_total_sequence_length, dispatch_policy);
}

std::optional<contrib::cuda::GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspaceForTest(
    const void* node, gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop, const AttentionKernelOptions& kernel_options,
    bool head_sink_is_constant_initializer,
    int64_t max_total_sequence_length,
    KernelDispatchPolicy dispatch_policy) {
  return contrib::cuda::EstimateGroupQueryAttentionWorkspace(
      *static_cast<const Node*>(node), input_shapes, device_prop, kernel_options,
      head_sink_is_constant_initializer, max_total_sequence_length, dispatch_policy);
}

std::optional<contrib::cuda::PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspaceForTest(
    const void* node, gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop, const AttentionKernelOptions& kernel_options) {
  return contrib::cuda::EstimatePackedAttentionWorkspace(
      *static_cast<const Node*>(node), input_shapes, device_prop, kernel_options);
}
#endif

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
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
    const void* loader, const RandomAccessFile& file,
    FileOffsetType offset, SafeInt<size_t> length, void* tensor) {
#if !defined(ORT_MINIMAL_BUILD) && !defined(USE_CUDA_MINIMAL)
  return static_cast<const cuda::ExternalDataLoader*>(loader)->LoadTensor(
      file, offset, length, *static_cast<Tensor*>(tensor));
#else
  ORT_UNUSED_PARAMETER(loader);
  ORT_UNUSED_PARAMETER(file);
  ORT_UNUSED_PARAMETER(offset);
  ORT_UNUSED_PARAMETER(length);
  ORT_UNUSED_PARAMETER(tensor);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Not supported in minimal builds.");
#endif
}

common::Status LoadCudaExternalDataForTest(
    const void* loader, const Env& env, const std::filesystem::path& path,
    FileOffsetType offset, SafeInt<size_t> length, void* tensor) {
#if !defined(ORT_MINIMAL_BUILD) && !defined(USE_CUDA_MINIMAL)
  std::unique_ptr<RandomAccessFile> file;
  ORT_RETURN_IF_ERROR(env.OpenRandomAccessFile(path.c_str(), file));
  return LoadCudaExternalDataForTest(loader, *file, offset, length, tensor);
#else
  ORT_UNUSED_PARAMETER(loader);
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(path);
  ORT_UNUSED_PARAMETER(offset);
  ORT_UNUSED_PARAMETER(length);
  ORT_UNUSED_PARAMETER(tensor);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Not supported in minimal builds.");
#endif
}

}  // namespace onnxruntime::test
