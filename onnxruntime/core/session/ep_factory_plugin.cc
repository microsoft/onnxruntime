// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_factory_plugin.h"

#include "core/framework/compute_capability.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/model_editor_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/allocator_adapters.h"

namespace onnxruntime {

PluginExecutionProviderFactory::PluginExecutionProviderFactory(OrtEpFactory& ep_factory,
                                                               gsl::span<const OrtEpDevice* const> ep_devices)
    : ep_factory_{ep_factory} {
  devices_.reserve(ep_devices.size());
  ep_metadata_.reserve(ep_devices.size());

  for (const auto* ep_device : ep_devices) {
    devices_.push_back(ep_device->device);
    ep_metadata_.push_back(&ep_device->ep_metadata);
  }
}

std::unique_ptr<IExecutionProvider>
PluginExecutionProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                               const OrtLogger& session_logger) {
  OrtEp* ort_ep = nullptr;
  OrtStatus* status = ep_factory_.CreateEp(&ep_factory_, devices_.data(), ep_metadata_.data(), devices_.size(),
                                           &session_options, &session_logger, &ort_ep);
  if (status != nullptr) {
    ORT_THROW("Error creating execution provider: ", ToStatus(status).ToString());
  }

  return std::make_unique<PluginExecutionProvider>(UniqueOrtEp(ort_ep, OrtEpDeleter(ep_factory_)));
}

PluginExecutionProvider::PluginExecutionProvider(UniqueOrtEp ep)
    : IExecutionProvider(ep->GetName(ep.get()), OrtDevice()),  // TODO: What to do about OrtDevice for plugins?
      ort_ep_(std::move(ep)) {
}

std::vector<std::unique_ptr<ComputeCapability>>
PluginExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup,
                                       const GraphOptimizerRegistry& graph_optimizer_registry,
                                       IResourceAccountant* resource_accountant) const {
  ORT_UNUSED_PARAMETER(graph_optimizer_registry);  // TODO: Add support
  ORT_UNUSED_PARAMETER(resource_accountant);       // TODO: Add support? Not used by prioritized EPs
  ORT_UNUSED_PARAMETER(kernel_lookup);             // TODO: Add support? Not used by prioritized EPs, so probably not needed?

  onnxruntime::EpGraph ep_graph(graph_viewer);
  OrtEpGraphSupportInfo api_graph_support_info(ep_graph);

  ort_ep_->GetCapability(ort_ep_.get(), ep_graph.ToExternal(), &api_graph_support_info);

  std::vector<std::unique_ptr<ComputeCapability>> result;
  if (api_graph_support_info.subgraphs.empty()) {
    return result;
  }

  // TODO: Create ComputeCapability instances from OrtEpGraphSupportInfo::Subgraph instances.

  return result;
}
}  // namespace onnxruntime
