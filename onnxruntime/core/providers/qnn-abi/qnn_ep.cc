#include "qnn_ep.h"

#include "qnn_ep_factory.h"

#if BUILD_QNN_EP_STATIC_LIB
#include "qnn_execution_provider.h"
#include "core/graph/ep_api_types.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#else
// For shared library build, use the bridge layer to avoid type conflicts
#include "qnn_ep_bridge.h"
#endif

#include "core/providers/qnn/ort_api.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <unordered_set>
#include <iostream>

namespace onnxruntime {

QnnEp::QnnEp(const QnnEpFactory& factory, const std::string& name,
           const Config& config, const OrtLogger* logger)
    : OrtEp{},
    ApiPtrs{static_cast<const ApiPtrs&>(factory)},
    factory_{factory},
    name_{name},
    config_{config},
    logger_{logger},
    context_cache_enabled_{config.enable_ep_context},
    share_ep_contexts_{config.share_ep_contexts}{
        std::cout << "DEBUG: QnnEp constructor called with name: " << name << std::endl;
        GetName = GetNameImpl;
        GetCapability = GetCapabilityImpl;
}
QnnEp::~QnnEp() = default;

const char* ORT_API_CALL QnnEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* qnn_ep = static_cast<const QnnEp*>(this_ptr);
  return qnn_ep->name_.c_str();
}

OrtStatus* ORT_API_CALL QnnEp::GetCapabilityImpl(OrtEp* this_ptr,
                                                const OrtGraph* graph,
                                                OrtEpGraphSupportInfo* graph_support_info) {
    std::cout << "DEBUG: GetCapabilityImpl called!" << std::endl;
    QnnEp* ep = static_cast<QnnEp*>(this_ptr);

#if BUILD_QNN_EP_STATIC_LIB
    // Static library build - use direct access to QNNExecutionProvider

    // Convert OrtGraph to GraphViewer
    const EpGraph* ep_graph = EpGraph::ToInternal(graph);
    if (ep_graph == nullptr) {
        return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid OrtGraph instance");
    }

    const GraphViewer& graph_viewer = ep_graph->GetGraphViewer();

    // Create logging::Logger from OrtLogger
    const logging::Logger* logger = reinterpret_cast<const logging::Logger*>(ep->logger_);

    // Create NodeUnit map
    std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
    std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
    std::tie(node_unit_holder, node_unit_map) = GetQDQNodeUnits(graph_viewer, *logger);

    // Create a QNNExecutionProvider instance to call GetSupportedNodes
    // Use empty provider options - this may need adjustment based on your needs
    ProviderOptions provider_options;
    auto qnn_ep_instance = std::make_unique<QNNExecutionProvider>(provider_options, nullptr);

    // Call GetSupportedNodes to get supported nodes
    std::unordered_set<const Node*> supported_nodes_set =
        qnn_ep_instance->GetSupportedNodes(graph_viewer, node_unit_map, node_unit_holder.size(), *logger);

    // Convert std::unordered_set<const Node*> to std::vector<const OrtNode*>
    std::vector<const OrtNode*> supported_nodes;
    supported_nodes.reserve(supported_nodes_set.size());

    for (const Node* node : supported_nodes_set) {
        // Convert Node* to OrtNode* using the EpGraph
        const OrtNode* ort_node = ep_graph->GetOrtNode(node);
        if (ort_node) {
            supported_nodes.push_back(ort_node);
        }
    }

#else
    // Shared library build - use bridge layer for IsOpSupported validation
    std::vector<const OrtNode*> supported_nodes =
        qnn::QnnEpBridge::ValidateNodesWithIsOpSupported(graph, ep->ort_api, ep->logger_, graph_support_info);

#endif

    if (supported_nodes.empty()) {
        return nullptr;
    }

    OrtNodeFusionOptions node_fusion_options = {};
    node_fusion_options.ort_version_supported = ORT_API_VERSION;
    node_fusion_options.drop_constant_initializers = true;
    RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                    supported_nodes.data(),
                                                                    supported_nodes.size(),
                                                                    &node_fusion_options));
    return nullptr;

}



}
