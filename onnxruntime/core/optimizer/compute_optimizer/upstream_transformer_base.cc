// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE

#include <onnx/defs/attr_proto_util.h>
#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

// Put some utils in anonymous namespace
namespace {

/**
 * @brief Check all inputs/outputs have shapes for given node.
 *
 * @return true when all shapes exist, false otherwise.
 */
bool EnforceNodeAllInputOutputHaveShapes(const Node& node) {
  for (const auto* input_def : node.InputDefs()) {
    if (!input_def->Shape()) {
      return false;
    }
  }

  for (const auto* output_def : node.OutputDefs()) {
    if (!output_def->Shape()) {
      return false;
    }
  }
  return true;
}

}  // namespace

template <typename T1, typename T2>
Status UpStreamGraphTransformerBase<T1, T2>::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                       const logging::Logger& logger)
    const {
  LOG_DEBUG_INFO(logger, "Enter UpStreamGraphTransformerBase");
  bool reordered = false;
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  const auto& graph_outputs = graph.GetOutputs();

  size_t reordered_node_count = 0;  // For summary
  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    std::optional<T1> op_info = IsSupportedForUpstream(graph, node, logger);
    if (!op_info.has_value()) {
      continue;
    }

    auto& output_arg = node.MutableOutputDefs()[0];
    if (std::find(graph_outputs.begin(), graph_outputs.end(), output_arg) != graph_outputs.end()) {
      continue;
    }

    std::deque<T1> queue;
    queue.push_back(std::move(op_info.value()));

    std::string node_name = node.Name();
    std::string node_type = node.OpType();
    std::string log_prefix = "Entry node " + node_name + " (" + node_type + ") ";
    LOG_DEBUG_INFO(logger, log_prefix + " starts re-ordering check");

    // DON'T operate on `node` once this loop starts, as it may be removed from the graph.
    while (!queue.empty()) {
      T1 info = queue.front();
      Node* node_to_upstream = info.node_ptr;
      queue.pop_front();
      Node* input_tensor_producer_node =
          graph.GetMutableProducerNode(node_to_upstream->MutableInputDefs()[0]->Name());
      if (input_tensor_producer_node == nullptr) {
        break;
      }

      if (graph.GetConsumerNodes(input_tensor_producer_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        LOG_DEBUG_INFO(logger, log_prefix + " stops at node " + input_tensor_producer_node->Name() +
                                   " since multiple consumer found");
        continue;
      }

      auto ret = Upstream(graph, queue, *input_tensor_producer_node, info, logger, node_name);
      if (ret) {
        LOG_DEBUG_INFO(logger, log_prefix + " moves up across node " + input_tensor_producer_node->Name());
        modified = true;
        reordered = true;
      } else {
        LOG_DEBUG_INFO(logger, log_prefix + " stops when handling " + input_tensor_producer_node->Name());
      }
    }

    if (reordered) {
      ++reordered_node_count;
    }
  }

  LOG_DEBUG_INFO(logger, "Exit UpStreamGraphTransformerBase, reordered " + std::to_string(reordered_node_count) +
                             " nodes");
  return Status::OK();
}

template <typename T1, typename T2>
bool UpStreamGraphTransformerBase<T1, T2>::Upstream(Graph& graph, std::deque<T1>& queue,
                                                    Node& current_node, T1& info,
                                                    const logging::Logger& logger,
                                                    std::string& entry_node_name) const {
  const std::string op_type = GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());
  if (allowed_passthrough_ops_.count(op_type)) {
    auto& pass_through_config = allowed_passthrough_ops_.at(op_type);
    LOG_DEBUG_INFO(logger, "Enter reorder handle for node " + current_node.Name() + "(" + op_type + ")");

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(current_node, current_node.OpType(),
                                                        pass_through_config.opsets, current_node.Domain())) {
      LOG_DEBUG_INFO(logger, "Unsupported opset for " + current_node.Name() + "(" + op_type + ") since version: " +
                                 std::to_string(current_node.SinceVersion()));
      return false;
    }

    if (!EnforceNodeAllInputOutputHaveShapes(current_node)) {
      LOG_DEBUG_INFO(logger, "Some inputs/outputs' shape not found for node " + current_node.Name() + "(" +
                                 op_type + ")");
      return false;
    }

    return UpStreamInternal(graph, queue, current_node, info, pass_through_config, logger, entry_node_name);
  } else {
    LOG_DEBUG_INFO(logger, "op_type not supported for " + current_node.Name() + "(" + op_type + ")");
    return false;
  }
}

template class UpStreamGraphTransformerBase<SliceInfo, UpStreamGatherOperatorActorBase>;

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
