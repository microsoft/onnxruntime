// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/unsqueeze_elimination.h"
#include "core/common/logging/logging.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

Status UnsqueezeElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
  NodeArg& input_def = *node.MutableInputDefs()[0];
  const auto& tensor_proto = *graph_utils::GetConstantInitializer(graph, input_def.Name());

  auto new_name = graph.GenerateNodeArgName("UnsqueezeElimination_" + input_def.Name());
  if (!graph_utils::CanReplaceNodeWithInitializer(graph, node, new_name, logger)) {
    LOGS(logger, WARNING) << "UnsqueezeElimination cannot remove node " << node.Name();
    return Status::OK();
  }

  InlinedVector<int64_t> axes;
  if (!graph_utils::GetRepeatedNodeAttributeValues(node, "axes", axes)) {
    // missing 'axes'. should have failed at model load but just in case...
    return Status::OK();
  }

  const int64_t output_rank = narrow<int64_t>(axes.size() + tensor_proto.dims().size());

  // handle any negative axis values and validate range
  for (auto& axis : axes) {
    if (!IsAxisInRange(axis, output_rank)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "'axes' has an out of range axis value ", axis,
                             " for output rank ", output_rank,
                             ". This is an invalid model. Node: ", node.Name());
    }
    if (axis < 0) {
      axis += output_rank;
    }
  }

  // Generate new dims. Mark axes positions with 1, fill the rest from input dims.
  InlinedVector<int64_t> new_dims(narrow<size_t>(output_rank), 0);
  for (int64_t axis : axes) {
    const size_t idx = narrow<size_t>(axis);
    if (new_dims[idx] != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "'axes' has a duplicate axis value ", axis,
                             ". This is an invalid model. Node: ", node.Name());
    }
    new_dims[idx] = 1;
  }

  auto begin = tensor_proto.dims().cbegin();
  for (auto& dim : new_dims) {
    if (dim == 0) {
      assert(begin != tensor_proto.dims().cend());
      dim = *begin++;
    }
  }
  assert(begin == tensor_proto.dims().cend());

  Initializer initializer(graph, tensor_proto, graph.ModelPath(), /*check_outer_scope=*/false);
  ONNX_NAMESPACE::TensorProto new_tensor_proto;
  OrtValue ort_value;
  initializer.ToProtoWithOrtValue(new_tensor_proto, ort_value);

  // Update shape of tensor proto.
  new_tensor_proto.set_name(new_name);
  new_tensor_proto.clear_dims();

  for (const auto& dim : new_dims) {
    new_tensor_proto.add_dims(dim);
  }

  if (utils::HasExternalDataInMemory(new_tensor_proto)) {
    ORT_ENFORCE(ort_value.IsAllocated());
    TensorShape new_shape(new_tensor_proto.dims());
    ort_value.GetMutable<Tensor>()->Reshape(new_shape);
  }

  auto& new_node_arg = graph_utils::AddInitializerWithOrtValue(graph, new_tensor_proto, ort_value);
  graph_utils::ReplaceNodeWithInitializer(graph, node, new_node_arg);

  // Remove the Unsqueeze node and replace it with the initializer.
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool UnsqueezeElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  // Attempt to remove an Unsqueeze operator only if it gets a constant initializer as input.
  return graph_utils::IsConstantInitializer(graph, node.InputDefs()[0]->Name());
}

}  // namespace onnxruntime
