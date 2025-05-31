// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_factory_plugin.h"

#include "core/framework/compute_capability.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/allocator_adapters.h"
#include "core/providers/partitioning_utils.h"

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

/// <summary>
/// Functor used to generate a Metadef name for a subgraph supported by a plugin EP.
/// The generated name is a concatenation of the subgraph name provided by the EP with
/// the model's hash and a unique ID.
/// </summary>
struct PluginEpMetaDefNameFunctor {
  explicit PluginEpMetaDefNameFunctor(const ModelMetadefIdGenerator& generator,
                                      const GraphViewer& graph_viewer,
                                      const std::string& prefix)
      : generator_(generator), graph_viewer_(graph_viewer), prefix_(prefix) {}

  std::string operator()() {
    uint64_t model_hash = 0;
    int id = generator_.GenerateId(graph_viewer_, model_hash);
    return MakeString(prefix_, "_", model_hash, "_", id);
  }

  const ModelMetadefIdGenerator& generator_;
  const GraphViewer& graph_viewer_;
  const std::string& prefix_;
};

std::vector<std::unique_ptr<ComputeCapability>>
PluginExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup,
                                       const GraphOptimizerRegistry& graph_optimizer_registry,
                                       IResourceAccountant* resource_accountant) const {
  ORT_UNUSED_PARAMETER(graph_optimizer_registry);  // TODO: Add support
  ORT_UNUSED_PARAMETER(resource_accountant);       // TODO: Add support? Not used by prioritized EPs
  ORT_UNUSED_PARAMETER(kernel_lookup);             // TODO: Add support? Not used by prioritized EPs, so probably not needed?

  EpGraph ep_graph(graph_viewer);
  OrtEpGraphSupportInfo api_graph_support_info(ep_graph);
  Status status = ToStatus(ort_ep_->GetCapability(ort_ep_.get(), ep_graph.ToExternal(), &api_graph_support_info));

  // GetCapability is not supposed to fail. If there's an error, return an empty result to ensure this EP is not
  // assigned any nodes and log an error.
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "OrtEp::GetCapability() failed with error: " << status.ToString();
    return {};
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  result.reserve(api_graph_support_info.subgraphs.size());
  if (api_graph_support_info.subgraphs.empty()) {
    return {};
  }

  ModelMetadefIdGenerator generator;

  // Create ComputeCapability instances from OrtEpGraphSupportInfo::Subgraph instances.
  for (const OrtEpGraphSupportInfo::Subgraph& subgraph : api_graph_support_info.subgraphs) {
    std::unordered_set<const Node*> node_set;
    node_set.reserve(subgraph.nodes.size());
    for (const EpNode* ep_node : subgraph.nodes) {
      node_set.insert(&ep_node->node);
    }

    std::vector<std::unique_ptr<ComputeCapability>> capabilities = utils::CreateSupportedPartitions(
        graph_viewer, node_set, /*stop_ops*/ {}, PluginEpMetaDefNameFunctor(generator, graph_viewer, this->Type()),
        this->Type(), this->Type(), /*node_unit_map*/ nullptr);

    for (auto& capability : capabilities) {
      // capability->hardware_device = subgraph.hardware_device;  // Would allow app to query which EP+HW runs a subgraph
      capability->use_subgraph_name_as_fused_node_name = true;
      result.push_back(std::move(capability));
    }
  }

  return result;
}

common::Status PluginExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  ORT_UNUSED_PARAMETER(node_compute_funcs);
  const size_t num_graphs = fused_nodes_and_graphs.size();
  std::vector<std::unique_ptr<EpGraph>> api_graphs_holder;
  std::vector<const OrtGraph*> api_graphs;
  std::vector<OrtNodeComputeFunctions*> api_compute_funcs(num_graphs, nullptr);

  api_graphs_holder.reserve(num_graphs);
  api_graphs.reserve(num_graphs);

  for (const FusedNodeAndGraph& node_and_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = node_and_graph.filtered_graph;
    const Node& fused_node = node_and_graph.fused_node;
    ORT_ENFORCE(graph_viewer.Name() == fused_node.Name());  // Should be equal for plugin EPs.

    auto ep_graph = std::make_unique<EpGraph>(node_and_graph.filtered_graph);
    api_graphs.push_back(ep_graph->ToExternal());
    api_graphs_holder.push_back(std::move(ep_graph));
  }

  // Call plugin EP's Compile(). Expect an error for now.
  Status status = ToStatus(ort_ep_->Compile(ort_ep_.get(), api_graphs.data(), num_graphs, api_compute_funcs.data()));

  // TODO: Initialize node_compute_funcs as wrappers to api_compute_funcs.
  // TODO: Store api_compute_funcs and call ort_ep_->ReleaseNodeComputeFunctions() in ~PluginExecutionProvider().

  return status;
}
}  // namespace onnxruntime
