// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/shape_to_initializer.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

Status ShapeToInitializer::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const {
  // Create an execution frame to get access to the memory allocator.
  OptimizerExecutionFrame::Info info({&node}, InitializedTensorSet());

  // Store the statically inferred shape of the input to the Shape operator.
  const ONNX_NAMESPACE::TensorShapeProto* input_shape_proto = node.InputDefs()[0]->Shape();
  std::vector<int64_t> input_dims;
  int num_dimensions = input_shape_proto->dim_size();
  for (int i = 0; i < num_dimensions; i++) {
    ORT_ENFORCE(input_shape_proto->dim(i).has_dim_value());
    input_dims.push_back(gsl::narrow_cast<int64_t>(input_shape_proto->dim(i).dim_value()));
  }

  // Create a tensor that corresponds to the output of the Shape operator.
  const auto shape_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int64_t>(),
                                                     TensorShape({gsl::narrow_cast<int64_t>(num_dimensions)}),
                                                     input_dims.data(), info.GetAllocator()->Info());

  // Create the TensorProto that will be used as initializer in place of the Shape operator.
  ONNX_NAMESPACE::TensorProto shape_initializer_proto;
  const auto* shape_out_def = node.OutputDefs()[0];
  graph_utils::BuildTensorProtoForInitializer(*shape_tensor, *shape_out_def, shape_initializer_proto);

  // Remove the output edges of the Shape node, then remove the node itself, and replace it with the initializer.
  graph_utils::RemoveNodeOutputEdges(graph, node);

  if (graph.RemoveNode(node.Index())) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  graph.AddInitializedTensor(shape_initializer_proto);

  return Status::OK();
}

bool ShapeToInitializer::SatisfyCondition(const Graph& graph, const Node& node) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Shape", {1}) ||
      // Making sure we are not left with a graph with no nodes.
      graph.IsNodeOutputsInGraphOutputs(node) ||
      // The shape of the input has to be statically known for applying the rule.
      !node.InputDefs()[0]->Shape()) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
