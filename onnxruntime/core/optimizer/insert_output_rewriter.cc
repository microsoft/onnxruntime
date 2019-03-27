// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/insert_output_rewriter.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

Status InsertMaxPoolOutput::Apply(Graph& graph, Node& node, bool& modified, bool& deleted) {
  auto& outputs = node.MutableOutputDefs();
  const NodeArg* Y = outputs[0];

  TypeProto t;
  t.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  t.mutable_tensor_type()->mutable_shape()->CopyFrom(*Y->Shape());

  NodeArg& node_arg = graph.GetOrCreateNodeArg(Y->Name() + "_mask", &t);

  outputs.push_back(&node_arg);

  modified = true;
  deleted = false;
  return Status::OK();
}

bool InsertMaxPoolOutput::SatisfyCondition(const Graph& /*graph*/, const Node& node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", 8) &&
      node.OutputDefs().size() == 1) {
    return true;
  }
  return false;
}

}  // namespace onnxruntime
