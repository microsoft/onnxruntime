// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "orttraining/core/graph/loss_function_builder.h"
#include "orttraining/core/session/training_session.h"

// TODO: solve the op version issue in the entire training framework
#define GRADIENT_OP_VERSION 9

namespace onnxruntime {
namespace training {

std::unique_ptr<ILossFunction> LossFunctionBuilder::Build(const std::string& loss_func_type) {
  auto& registry = LossFunctionRegistry::GetInstance();

  // If not in the loss function registry.
  if (!registry.Contains(loss_func_type)) {
    //If is a valid op, add to the loss function registry
    //TODO: Better handle op version and domain.
    if (ONNX_NAMESPACE::OpSchemaRegistry::Instance()->GetSchema(loss_func_type,
                                                                GRADIENT_OP_VERSION,
                                                                ONNX_NAMESPACE::ONNX_DOMAIN) ||
        ONNX_NAMESPACE::OpSchemaRegistry::Instance()->GetSchema(loss_func_type,
                                                                1,
                                                                kMSDomain)) {
      registry.RegisterOperatorLossFunction(loss_func_type);
    } else {
      ORT_THROW(loss_func_type, "is not a system provided loss function nor a valid op");
    }
  }

  auto loss_func = registry.MakeUnique(loss_func_type);
  ORT_ENFORCE(loss_func != nullptr,
              "Fail to create loss function from registry", loss_func_type);

  return loss_func;
}
}  // namespace training
}  // namespace onnxruntime
