// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/shape_to_initializer.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

Status ShapeToInitializer::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const {
  // Store the statically inferred shape of the input to the Shape operator.
  const ONNX_NAMESPACE::TensorShapeProto* input_shape_proto = node.InputDefs()[0]->Shape();
  std::vector<int64_t> input_dims;
  int num_dimensions = input_shape_proto->dim_size();
  for (int i = 0; i < num_dimensions; i++) {
    input_dims.push_back(gsl::narrow_cast<int64_t>(input_shape_proto->dim(i).dim_value()));
  }

  // Create the TensorProto that will be used as initializer in place of the Shape operator.
  const auto* shape_out_def = node.OutputDefs()[0];

  ONNX_NAMESPACE::TensorProto shape_initializer_proto;

  shape_initializer_proto.set_name(shape_out_def->Name());

  TensorShape tensor_shape({gsl::narrow_cast<int64_t>(num_dimensions)});
  for (auto& dim : tensor_shape.GetDims()) {
    shape_initializer_proto.add_dims(dim);
  }

  auto tensor_proto_data_type = shape_out_def->TypeAsProto()->tensor_type().elem_type();

  shape_initializer_proto.set_data_type(tensor_proto_data_type);

  // Here we expect little-indian format to set raw data of the TensorProto.
  shape_initializer_proto.set_raw_data(input_dims.data(),
                                       input_dims.size() * sizeof(decltype(input_dims)::value_type));

  // Remove the output edges of the Shape node, then remove the node itself, and replace it with the initializer.
  graph_utils::RemoveNodeOutputEdges(graph, node);

  if (graph.RemoveNode(node.Index())) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
    graph.AddInitializedTensor(shape_initializer_proto);
  }

  return Status::OK();
}

bool ShapeToInitializer::SatisfyCondition(const Graph& graph, const Node& node) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Shape", {1}) ||
      // Making sure we are not left with a graph with no nodes.
      graph.IsNodeOutputsInGraphOutputs(node)) {
    return false;
  }

  // The shape of the input has to be statically known. Moreover, each dimension should have a meaningful value
  // (the rule cannot be applied if one of the dimension is a symbolic variable (or)
  // if the dimension has an alias value whose real value can only be inferred at inference run time (e.g.) '-1').
  const auto* input_shape = node.InputDefs()[0]->Shape();
  if (!input_shape) {
    return false;
  }

  for (int i = 0, num_dims = input_shape->dim_size(); i < num_dims; i++) {
    const auto& input_dim = input_shape->dim(i);
    if (!input_dim.has_dim_value() || input_dim.dim_value() < 0) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime
