#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/pre_shape_node_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status PreShapeNodeElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool PreShapeNodeElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  auto output_nodes = graph.GetConsumerNodes(node.OutputDefs()[0]->Name());

  if (output_nodes.empty()) {
    return false;
  }

  const Node* next_node = output_nodes[0];
  const auto& op_type = node.OpType();

  // Check if the current node is Cast and the next node is Shape
  if (op_type == "Cast") {
    return next_node->OpType() == "Shape";
  }

  // Check if the current node is Transpose and the next node is Shape
  if (op_type == "Transpose") {
    // Check if the dimensions of the input to Transpose are the same
    const auto& transpose_input_shape = node.InputDefs()[0]->Shape();

    if (transpose_input_shape->dim_size() >= 2) {
      const int64_t dim_size = transpose_input_shape->dim_size();
      const int64_t first_dim = transpose_input_shape->dim(0).dim_value();
      for (int64_t i = 1; i < dim_size; ++i) {
        if (!transpose_input_shape->dim(i).has_dim_value() ||
            transpose_input_shape->dim(i).dim_value() != first_dim) {
          return false;
        }
      }

      return next_node->OpType() == "Shape";
    }
  }

  return false;
}

}  // namespace onnxruntime
