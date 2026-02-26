// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep.h"
#include "cuda_ep_factory.h"
#include "core/providers/cuda/plugin/cuda_kernel_adapter.h"

namespace onnxruntime {
namespace cuda_plugin {

CudaEp::CudaEp(CudaEpFactory& factory, const Config& config, const OrtLogger& logger)
    : OrtEp{},
      factory_(factory),
      name_(factory.GetEpName()),
      config_(config),
      logger_(logger) {
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers for kernel-registry-based EP
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;

  // Not a compile-based EP
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;

  const OrtApi& ort_api = factory_.GetOrtApi();
  Ort::Status log_status(ort_api.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_INFO,
                                                   "CUDA Plugin EP created",
                                                   ORT_FILE, __LINE__, __FUNCTION__));

  // Seed adapter-level runtime options for migrated kernels.
  onnxruntime::cuda::SetCudaKernelAdapterRuntimeConfig(
      config_.use_tf32, config_.device_id, config_.enable_skip_layer_norm_strict_mode,
      config_.cudnn_conv_algo, config_.cudnn_conv1d_pad_to_nc1d);
}

CudaEp::~CudaEp() = default;

/*static*/
const char* ORT_API_CALL CudaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const CudaEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetCapabilityImpl(
    OrtEp* this_ptr, const OrtGraph* ort_graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  const OrtEpApi& ep_api = ep->factory_.GetEpApi();

  Ort::ConstGraph graph{ort_graph};
  std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

  if (all_nodes.empty()) {
    return nullptr;
  }

  // For each node, check if we have a registered kernel
  for (const auto& node : all_nodes) {
    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(
        graph_support_info, node, &kernel_def));

    if (kernel_def != nullptr) {
      RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_AddSingleNode(
          graph_support_info, node));
    }
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetKernelRegistryImpl(
    OrtEp* this_ptr,
    const OrtKernelRegistry** kernel_registry) noexcept {
  auto* ep = static_cast<CudaEp*>(this_ptr);
  *kernel_registry = nullptr;

  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetPreferredDataLayoutImpl(
    OrtEp* this_ptr, OrtEpDataLayout* preferred_data_layout) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
  *preferred_data_layout = ep->config_.prefer_nhwc ? OrtEpDataLayout_NHWC : OrtEpDataLayout_NCHW;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunStartImpl(
    OrtEp* /*this_ptr*/, const ::OrtRunOptions* /*run_options*/) noexcept {
  // Stub: will later manage CUDA Graph capture state
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunEndImpl(
    OrtEp* /*this_ptr*/, const ::OrtRunOptions* /*run_options*/, bool /*sync_stream*/) noexcept {
  // Stub: will later manage CUDA Graph replay state
  return nullptr;
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
