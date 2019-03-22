// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/insert_output_transformer.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

Status InsertOutputTransformer::ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level) const {
  for (auto& node : graph.Nodes()) {
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    if (!utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", 8)) {
      continue;
    }

    auto& outputs = node.MutableOutputDefs();

    if (outputs.size() > 1) {
      continue;
    }

    const NodeArg* Y = outputs[0];

    TypeProto t;
    t.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    t.mutable_tensor_type()->mutable_shape()->CopyFrom(*Y->Shape());

    NodeArg& node_arg = graph.GetOrCreateNodeArg(Y->Name() + "_mask", &t);

    outputs.push_back(&node_arg);

    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime
