// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/function_utils.h"

namespace onnxruntime {
namespace function_utils {
std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const Graph& graph,
    const IndexedSubGraph& nodes_to_fuse) {
  const auto* meta_def = nodes_to_fuse.GetMetaDef();
  auto op_schema = std::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema->SetName(meta_def->name);
  op_schema->SetDomain(meta_def->domain);
  op_schema->SetDoc(meta_def->doc_string);
  op_schema->SinceVersion(meta_def->since_version);

  if (meta_def->type_and_shape_inference_function) {
    op_schema->TypeAndShapeInferenceFunction(meta_def->type_and_shape_inference_function);
  }

  int i = 0;

  for (auto& input : meta_def->inputs) {
    auto input_arg = graph.GetNodeArg(input);
    // inputs must have a type. can be inferred for outputs.
    ORT_ENFORCE(input_arg->Type() != nullptr);
    op_schema->Input(i, input, "", *input_arg->Type());
    ++i;
  }
  i = 0;
  for (auto& output : meta_def->outputs) {
    auto output_arg = graph.GetNodeArg(output);
    op_schema->Output(i, output, "", *output_arg->Type());
    ++i;
  }
  op_schema->Finalize();

  return op_schema;
}

}
}