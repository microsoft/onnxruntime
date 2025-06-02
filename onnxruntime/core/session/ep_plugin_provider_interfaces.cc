// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_plugin_provider_interfaces.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "core/framework/compute_capability.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/abi_logger.h"
#include "core/session/allocator_adapters.h"
#include "core/providers/partitioning_utils.h"

namespace onnxruntime {

//
// PluginExecutionProviderFactory
//

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

  auto ep_wrapper = std::make_unique<PluginExecutionProvider>(UniqueOrtEp(ort_ep, OrtEpDeleter(ep_factory_)));
  ep_wrapper->SetLogger(session_logger.ToInternal());

  return ep_wrapper;
}

/// <summary>
/// Functor used to generate a Metadef name for a subgraph supported by a plugin EP.
/// The generated name is a concatenation of a prefix (i.e., the EP name) with
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

//
// PluginExecutionProvider
//

PluginExecutionProvider::PluginExecutionProvider(UniqueOrtEp ep)
    : IExecutionProvider(ep->GetName(ep.get()), OrtDevice()),  // TODO: What to do about OrtDevice for plugins?
      ort_ep_(std::move(ep)) {
}

PluginExecutionProvider::~PluginExecutionProvider() {
  if (ort_ep_ && !api_node_compute_infos_.empty()) {
    ort_ep_->ReleaseNodeComputeInfos(ort_ep_.get(), api_node_compute_infos_.data(),
                                     api_node_compute_infos_.size());
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
PluginExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup,
                                       const GraphOptimizerRegistry& graph_optimizer_registry,
                                       IResourceAccountant* resource_accountant) const {
  ORT_UNUSED_PARAMETER(graph_optimizer_registry);  // TODO: Add support
  ORT_UNUSED_PARAMETER(resource_accountant);       // TODO: Add support? Not used by prioritized EPs
  ORT_UNUSED_PARAMETER(kernel_lookup);             // TODO: Add support? Not used by prioritized EPs, so probably not needed?

  auto ep_graph = EpGraph::Create(graph_viewer);
  OrtEpGraphSupportInfo api_graph_support_info(*ep_graph);
  Status status = ToStatus(ort_ep_->GetCapability(ort_ep_.get(), ep_graph->ToExternal(), &api_graph_support_info));

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
                                                std::vector<NodeComputeInfo>& node_compute_infos) {
  const logging::Logger* logger = GetLogger();
  const size_t num_graphs = fused_nodes_and_graphs.size();
  std::vector<std::unique_ptr<EpGraph>> api_graphs_holder;
  std::vector<const OrtGraph*> api_graphs;
  std::vector<OrtNodeComputeInfo*> api_node_compute_infos(num_graphs, nullptr);

  api_graphs_holder.reserve(num_graphs);
  api_graphs.reserve(num_graphs);

  // Wrap GraphViewers into OrtGraphs
  for (const FusedNodeAndGraph& node_and_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = node_and_graph.filtered_graph;
    const Node& fused_node = node_and_graph.fused_node;
    ORT_ENFORCE(graph_viewer.Name() == fused_node.Name());  // Should be equal for plugin EPs.

    auto ep_graph = EpGraph::Create(node_and_graph.filtered_graph);
    api_graphs.push_back(ep_graph->ToExternal());
    api_graphs_holder.push_back(std::move(ep_graph));
  }

  // Call plugin EP's Compile(). Expect an error for now.
  ORT_RETURN_IF_ERROR(ToStatus(ort_ep_->Compile(ort_ep_.get(), api_graphs.data(), num_graphs,
                                                api_node_compute_infos.data())));

  // Save OrtNodeComputeInfo created by OrtEp instance. They're freed when this IExecutionProvider
  // is destroyed.
  api_node_compute_infos_.reserve(api_node_compute_infos_.size() + num_graphs);
  for (size_t i = 0; i < num_graphs; i++) {
    if (api_node_compute_infos[i] != nullptr) {
      api_node_compute_infos_.push_back(api_node_compute_infos[i]);
    }
  }

  // Initialize node_compute_infos as wrappers to api_node_compute_infos.
  for (size_t i = 0; i < num_graphs; i++) {
    OrtNodeComputeInfo* api_node_compute_info = api_node_compute_infos[i];
    ORT_RETURN_IF(api_node_compute_info == nullptr, "OrtEp::Compile() did not set a valid OrtNodeComputeInfo ",
                  "instance for graph at index ", i);

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [api_node_compute_info, logger](ComputeContext* context,
                                                                     FunctionState* compute_state) -> int {
      Status status = ToStatus(api_node_compute_info->CreateComputeState(api_node_compute_info,
                                                                         reinterpret_cast<OrtNodeComputeContext*>(context),
                                                                         compute_state));
      if (!status.IsOK()) {
        LOGS(*logger, ERROR) << "OrtNodeComputeInfo::CreateComputeState() failed with error: "
                             << status.ErrorMessage();
      }
      return status.IsOK() ? 0 : 1;
    };

    compute_info.release_state_func = [api_node_compute_info](FunctionState compute_state) -> void {
      api_node_compute_info->DestroyComputeState(api_node_compute_info, compute_state);
    };

    compute_info.compute_func = [api_node_compute_info](FunctionState compute_state,
                                                        const OrtApi* c_api,
                                                        OrtKernelContext* kernel_context) -> Status {
      return ToStatus(api_node_compute_info->Compute(api_node_compute_info, compute_state, c_api, kernel_context));
    };

    node_compute_infos.push_back(std::move(compute_info));
  }

  return Status::OK();
}
}  // namespace onnxruntime
