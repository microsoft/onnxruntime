// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_registry.h"

namespace onnxruntime {
namespace training {

GradientDef GetGradientForOp(const Node* node,
                             const std::unordered_set<std::string>& output_args_need_grad,
                             const std::unordered_set<std::string>& input_args_need_grad) {
  auto gradient_builder_func = GradientBuilderRegistry::GetGradientBuilderRegistry().GetGradientBuilderFunc(node->OpType());
  auto gradient_builder = gradient_builder_func(node, output_args_need_grad, input_args_need_grad);

  auto gradient_def = gradient_builder->GetGradientDefs();

  if (gradient_builder->CopyAttributes()) {
    // TODO: Figure out the correct default copy behavior
    // modify the GradientDef returned by GetGradiendDefs()
  }

  return gradient_def;
}

}  // namespace training
}  // namespace onnxruntime
