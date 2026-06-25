// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include "factory.h"

#include "core/framework/run_options.h"
#include "core/framework/kernel_registry.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

#include "ep/get_capability_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

using onnxruntime::ep::Api;

// Constructor
Ep::Ep(std::unique_ptr<IExecutionProvider> impl, Factory& factory, const OrtLogger& logger, const Config& config)
    : onnxruntime::ep::adapter::Ep{std::move(impl), config.cpu_allocator, config.device_allocator},
      factory_{factory},
      logger_{logger},
      config_{config} {
  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = nullptr;  // Per-kernel EP does not use Compile
  ReleaseNodeComputeInfos = nullptr;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  SetDynamicOptions = nullptr;  // Not implemented
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;
  CreateAllocator = CreateAllocatorImpl;
  CreateSyncStreamForDevice = nullptr;          // Not stream aware
  GetCompiledModelCompatibilityInfo = nullptr;  // Not a compiled EP
  IsConcurrentRunSupported = IsConcurrentRunSupportedImpl;
}

// OrtEp interface implementations
const char* ORT_API_CALL Ep::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->factory_.GetName(&ep->factory_);
}

OrtStatus* ORT_API_CALL Ep::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                              OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN

  auto& ep = *static_cast<WebGpuExecutionProvider*>(static_cast<Ep*>(this_ptr)->EpImpl());
  Ort::ConstGraph ort_graph{graph};

  // Get all nodes in the graph
  std::vector<Ort::ConstNode> all_nodes = ort_graph.GetNodes();

  if (all_nodes.empty()) {
    return nullptr;  // No nodes to process
  }

  std::vector<const OrtNode*> candidate_nodes;
  std::vector<const OrtNode*> tentative_candidate_nodes;

  // For each node, check if we have a registered kernel for it
  for (const auto& node : all_nodes) {
    std::string ep_name = node.GetEpName();

    if (ep_name == kWebGpuExecutionProvider) {
      candidate_nodes.push_back(node);
      continue;
    }

    // Reject nodes already assigned to a different (non-CPU) EP
    if (!ep_name.empty() && ep_name != kCpuExecutionProvider) {
      continue;
    }

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def));

    if (kernel_def == nullptr) {
      LOGS(ep.GetEpLogger(), INFO) << "webgpu kernel not found in registries for Op type: "
                                   << node.GetOperatorType() << " node name: " << node.GetName();
      continue;
    }

    auto cpu_node_names = ep.GetForceCpuNodeNames();
    if (std::find(cpu_node_names.begin(),
                  cpu_node_names.end(),
                  node.GetName()) != cpu_node_names.end()) {
      LOGS(ep.GetEpLogger(), INFO) << "Force CPU execution for node: " << node.GetName();
      continue;
    }

    //
    // The following code checks if the node is really supported by webgpu EP
    //

#define FALLBACK_TO_CPU_IF_EXIST_INPUT(idx)            \
  if (inputs.size() > idx && inputs[idx] != nullptr) { \
    continue;                                          \
  }

#define FALLBACK_TO_CPU_IF_EXIST_OUTPUT(idx)             \
  if (outputs.size() > idx && outputs[idx] != nullptr) { \
    continue;                                            \
  }

    // Check for Attention
    if (node.GetOperatorType() == "Attention" && node.GetDomain() == kMSDomain) {
      const auto& inputs = node.GetInputs();
      const auto& outputs = node.GetOutputs();

      // Current implementation does not support mask_index(input[3]), past(input[4]) and past_seq_len(input[6])
      FALLBACK_TO_CPU_IF_EXIST_INPUT(3);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(4);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(6);

      // Current implementation does not support present(output[1])
      FALLBACK_TO_CPU_IF_EXIST_OUTPUT(1);

      // If attribute past_present_share_buffer is set, fallback to CPU
      bool has_past_present_share_buffer = false;
      for (const auto& attr : node.GetAttributes()) {
        if (attr.GetName() == "past_present_share_buffer") {
          int64_t val = 0;
          RETURN_IF_ERROR(attr.GetValue(val));
          if (val != 0) {
            has_past_present_share_buffer = true;
          }
          break;
        }
      }
      if (has_past_present_share_buffer) {
        continue;
      }
    }

    candidate_nodes.push_back(node);
    tentative_candidate_nodes.push_back(node);
  }

  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  RETURN_IF_ERROR(onnxruntime::ep::GetCpuPreferredNodes(*ort_graph,
                                                        *graph_support_info,
                                                        static_cast<Ep*>(this_ptr)->GetOrtLogger(),
                                                        tentative_candidate_nodes,
                                                        cpu_preferred_nodes));

  for (const auto& node : candidate_nodes) {
    if (cpu_preferred_nodes.count(node) == 0) {
      RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_AddSingleNode(graph_support_info, node));
    }
  }

  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr,
    _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN

  *kernel_registry = nullptr;

  // For the WebGPU EP, delegate to the CreateKernelRegistry function
  // which properly constructs a registry using only public APIs
  auto* ep = static_cast<Ep*>(this_ptr);

  auto& webgpu_ep = *static_cast<WebGpuExecutionProvider*>(ep->EpImpl());

  *kernel_registry = *webgpu_ep.GetKernelRegistryImpl();
  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                       _Out_ OrtEpDataLayout* preferred_data_layout) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // Delegate to the underlying WebGPU EP's GetPreferredLayout()
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  *preferred_data_layout = static_cast<OrtEpDataLayout>(ep->EpImpl()->GetPreferredLayout());
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                             _In_z_ const char* domain,
                                                             _In_z_ const char* op_type,
                                                             _In_ OrtEpDataLayout target_data_layout,
                                                             _Outptr_ int* should_convert) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  auto result = ep->EpImpl()->ShouldConvertDataLayoutForOp(domain, op_type,
                                                           static_cast<DataLayout>(target_data_layout));
  if (result.has_value()) {
    *should_convert = result.value() ? 1 : 0;
  } else {
    *should_convert = -1;
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::OnRunStartImpl(_In_ OrtEp* this_ptr,
                                           _In_ const OrtRunOptions* run_options) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  onnxruntime::RunOptions options{};
  // currently only option "gpu_graph_id" is used
  auto graph_annotation_str = Api().ort.GetRunConfigEntry(run_options, kOrtRunOptionsConfigCudaGraphAnnotation);
  if (graph_annotation_str != nullptr) {
    auto status = options.config_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, graph_annotation_str);
    if (!status.IsOK()) {
      return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                    status.ErrorMessage().c_str());
    }
  }
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->OnRunStart(options);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::OnRunEndImpl(_In_ OrtEp* this_ptr,
                                         _In_ const OrtRunOptions* /*run_options*/,
                                         _In_ bool sync_stream) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->OnRunEnd(sync_stream, {});
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::IsConcurrentRunSupportedImpl(_In_ OrtEp* /*this_ptr*/, _Out_ bool* is_concurrent_run_supported) noexcept {
  *is_concurrent_run_supported = false;
  return nullptr;
}

OrtStatus* ORT_API_CALL Ep::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                _In_ const OrtMemoryInfo* memory_info,
                                                _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  Ort::ConstMemoryInfo ort_memory_info{memory_info};
  if (ort_memory_info.GetAllocatorType() == OrtReadOnlyAllocator) {
    *allocator = new onnxruntime::ep::adapter::Allocator(memory_info, ep->config_.initializer_allocator);
  } else {
    *allocator = new onnxruntime::ep::adapter::Allocator(memory_info, ep->config_.device_allocator);
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
