// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/gradient_builder.h"
#include "core/training/gradient_builder_registry.h"

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

void RegisterGradientBuilders() {
  REGISTER_GRADIENT_BUILDER("Sin", GetSinGradient);
  REGISTER_GRADIENT_BUILDER("MatMul", GetMatmulGradient);
  REGISTER_GRADIENT_BUILDER("Split", GetSplitGradient);
  REGISTER_GRADIENT_BUILDER("Relu", GetReluGradient);
  REGISTER_GRADIENT_BUILDER("Pow", GetPowGradient);
  REGISTER_GRADIENT_BUILDER("ReduceMean", GetReduceMeanGradient);
  REGISTER_GRADIENT_BUILDER("Add", GetAddGradient);
  REGISTER_GRADIENT_BUILDER("Sub", GetSubGradient);
}

}  // namespace training
}  // namespace onnxruntime
