// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/training/loss_function_builder.h"
#include "core/training/training_session.h"

// TODO: solve the op version issue in the entire training framework
// Here is to reference GRADIENT_OP_VERSION temporarily for global version control.
#include "core/training/gradient_op_schema.h"

namespace onnxruntime {
namespace training {

GraphAugmenter::GraphDefs LossFunctionBuilder::Build(const Graph& graph,
                                                     const LossFunctionInfo& loss_func_info) const {
  auto& registry = LossFunctionRegistry::GetInstance();

  // If not in the loss function registry.
  if (!registry.Contains(loss_func_info.name_)) {
    //If is a valid op, add to the loss function registry
    //TODO: Better handle op version and domain.
    if (ONNX_NAMESPACE::OpSchemaRegistry::Instance()->GetSchema(loss_func_info.name_,
                                                                GRADIENT_OP_VERSION,
                                                                ONNX_NAMESPACE::ONNX_DOMAIN) ||
        ONNX_NAMESPACE::OpSchemaRegistry::Instance()->GetSchema(loss_func_info.name_,
                                                                1,
                                                                kMSDomain)) {
      registry.RegisterOperatorLossFunction(loss_func_info.name_);
    } else {
      ORT_THROW(loss_func_info.name_, "is not a system provided loss function nor a valid op");
    }
  }

  auto loss_func = registry.MakeUnique(loss_func_info.name_);
  ORT_ENFORCE(loss_func != nullptr,
              "Fail to create loss function from registry", loss_func_info.name_);

  // Generate GraphDefs, without worrying about label's TypeProto.
  auto graph_defs = loss_func->GetDefs(loss_func_info);

  // Now let's set label's TypeProto from prediction's.
  // TODO: see if we want to relax this.
  const auto* node_arg = graph.GetNodeArg(loss_func_info.prediction_name_);
  ORT_ENFORCE(node_arg != nullptr,
              "Fail to get prediction's node arg", loss_func_info.prediction_name_);
  const auto* prediction_type_proto = node_arg->TypeAsProto();

  for (auto& node : graph_defs.NodeDefs()) {
    for (auto& arg_def : node.input_args) {
      if (arg_def.name == loss_func_info.label_name_) {
        arg_def.type_proto = prediction_type_proto;
      }
    }
  }

  return graph_defs;
}
}  // namespace training
}  // namespace onnxruntime
