// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "orttraining/core/optimizer/scheduler_optimizer.h"

namespace onnxruntime {

Status SchedulerOptimizer::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                     const logging::Logger& logger)
    const {
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder(onnxruntime::ExecutionOrder::PRIORITY_BASED);

  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr) { /* skip removed nodes*/
      continue;
    }

    if (p_node->OpType() == "SoftmaxCrossEntropyLossInternalGrad") {
      const NodeArg* node_arg = p_node->OutputDefs()[0];
      if (node_arg == nullptr) {
        continue;
      }

      for (auto it = p_node->OutputNodesBegin(); it != p_node->OutputNodesEnd(); ++it) {
        Node* output_node = graph.GetNode(it->Index());
        if (output_node == nullptr) {
          continue;
        }

        output_node->SetPriority(static_cast<int>(ExecutionPriority::LOCAL_HIGH));
        modified = true;
        LOGS(logger, WARNING) << "Set priority of node " << output_node->Name() << " to " << static_cast<int>(ExecutionPriority::LOCAL_HIGH);
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
