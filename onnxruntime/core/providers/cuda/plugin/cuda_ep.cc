// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep.h"
#include "cuda_ep_factory.h"
#include "cuda_stream_plugin.h"
#include "core/providers/cuda/plugin/cuda_kernel_adapter.h"
#include "ep/get_capability_utils.h"

#include <cstring>
#include <string>
#include <string_view>
#include <unordered_set>

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
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  OnRunStart = nullptr;
  OnRunEnd = nullptr;

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

  // Phase 1: Collect tentative nodes — those for which we have a registered kernel.
  std::vector<const OrtNode*> tentative_nodes;
  tentative_nodes.reserve(all_nodes.size());

  for (const auto& node : all_nodes) {
    // Skip nodes already assigned to another EP.
    std::string ep_name = node.GetEpName();
    if (!ep_name.empty()) {
      continue;
    }

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(
        graph_support_info, node, &kernel_def));

    if (kernel_def != nullptr) {
      tentative_nodes.push_back(node);
    }
  }

  // Phase 2: Filter out CPU-preferred nodes (e.g., Shape, NonZero, small compute ops
  // that would be cheaper on CPU than incurring device-to-host copy overhead).
  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  RETURN_IF_ERROR(ep::GetCpuPreferredNodes(
      *ort_graph, *graph_support_info, ep->logger_,
      gsl::span<const OrtNode* const>(tentative_nodes.data(), tentative_nodes.size()),
      cpu_preferred_nodes));

  // Phase 3: Add final supported nodes (tentative minus CPU-preferred).
  for (const OrtNode* ort_node : tentative_nodes) {
    if (cpu_preferred_nodes.count(ort_node) == 0) {
      Ort::ConstNode node{ort_node};
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
OrtStatus* ORT_API_CALL CudaEp::ShouldConvertDataLayoutForOpImpl(
    OrtEp* this_ptr, const char* domain, const char* op_type,
    OrtEpDataLayout target_data_layout, int* should_convert) noexcept {
  (void)this_ptr;

  // Only convert to NHWC; for any other target layout, let ORT decide.
  if (target_data_layout != OrtEpDataLayout_NHWC) {
    *should_convert = -1;  // Let ORT decide
    return nullptr;
  }

  // ONNX domain ops that have NHWC kernel registrations.
  static const std::unordered_set<std::string_view> cuda_nhwc_onnx_ops{
      "BatchNormalization",
      "Conv",
      "ConvTranspose",
      "GlobalMaxPool",
      "MaxPool",
      "GlobalAveragePool",
      "AveragePool",
      "GridSample",
      "DepthToSpace",
      "SpaceToDepth",
      "LRN",
  };

  // Check ONNX domain (empty string) or MS domain (com.microsoft)
  bool is_onnx_domain = (domain[0] == '\0');
  bool is_ms_domain = (std::strcmp(domain, "com.microsoft") == 0);

  if (is_onnx_domain && cuda_nhwc_onnx_ops.count(op_type) > 0) {
    *should_convert = 1;  // Convert
    return nullptr;
  }

  if (is_ms_domain && std::strcmp(op_type, "GridSample") == 0) {
    *should_convert = 1;  // Convert
    return nullptr;
  }

  *should_convert = -1;  // Let ORT decide for other ops
  return nullptr;
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
