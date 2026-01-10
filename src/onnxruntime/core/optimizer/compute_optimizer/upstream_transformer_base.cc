// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include <onnx/defs/attr_proto_util.h>
#include "core/common/safeint.h"
#include "core/common/string_utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"
#include "core/optimizer/compute_optimizer/upstream_reshape_actors.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

// Put some utils in an anonymous namespace
namespace {

/**
 * @brief Check all inputs/outputs have shapes for the given node.
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

  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  const auto& graph_outputs = graph.GetOutputs();

  [[maybe_unused]] size_t reordered_node_count = 0;  // For summary
  [[maybe_unused]] size_t passthrough_count = 0;

  for (const auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    bool reordered = false;
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
    std::string log_prefix = "Entry node " + node_name + " (" + node_type + ")";
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

      // This condition implies few things:
      // 1. The node's outputs are only used once (that's the node to upstream).
      // 2. If the node has more than one outputs, only one of the outputs has output edge
      //   and that output is used by the node to upstream.
      if (input_tensor_producer_node->GetOutputEdgesCount() > 1) {
        LOG_DEBUG_INFO(logger, log_prefix + " stops at node " + input_tensor_producer_node->Name() +
                                   " since multiple consumers found");
        continue;
      }

      auto ret = Upstream(graph, queue, *input_tensor_producer_node, info, logger);
      if (ret) {
        LOG_DEBUG_INFO(logger, log_prefix + " moves up across node " + input_tensor_producer_node->Name());
        modified = true;
        reordered = true;
        passthrough_count += 1;
      } else {
        LOG_DEBUG_INFO(logger, log_prefix + " stops when handling " + input_tensor_producer_node->Name());
      }
    }

    if (reordered) {
      ++reordered_node_count;
    }
  }

  // For `Node A` -> 'Entry Node B', one `passthrough` means, entry node B is moved ahead of node A on
  // its every input branch. `passthrough_count` is the total number of times we move entry node B.
  LOG_DEBUG_INFO(logger, "Exit UpStreamGraphTransformerBase, reordered " + std::to_string(reordered_node_count) +
                             " nodes, total passthrough count (how many times we re-order the nodes): " +
                             std::to_string(passthrough_count));
  return Status::OK();
}

template <typename T1, typename T2>
bool UpStreamGraphTransformerBase<T1, T2>::Upstream(Graph& graph, std::deque<T1>& queue,
                                                    Node& current_node, T1& info,
                                                    const logging::Logger& logger) const {
  const std::string op_type = utils::GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());
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

    return UpStreamInternal(graph, queue, current_node, info, pass_through_config, logger);
  } else {
    LOG_DEBUG_INFO(logger, "op_type not supported for " + current_node.Name() + "(" + op_type + ")");
    return false;
  }
}

template class UpStreamGraphTransformerBase<SliceInfo, UpStreamGatherOperatorActorBase>;
template class UpStreamGraphTransformerBase<ReshapeInfo, UpStreamReshapeOperatorActorBase>;

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
