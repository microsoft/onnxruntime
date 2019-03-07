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

  // TODO: Figure out the correct default copy behavior
  // modify the GradientDef returned by GetGradiendDefs()
  if (gradient_builder->CopyAttributes() && node->GetAttributes().size() > 0) {
    for (NodeDef& node_def : gradient_def) {
      node_def.attributes = node->GetAttributes();
    }
  }

  return gradient_def;
}

void RegisterGradientBuilders() {
  REGISTER_GRADIENT_BUILDER("Sin", GetSinGradient);
  REGISTER_GRADIENT_BUILDER("MatMul", GetMatMulGradient);
  REGISTER_GRADIENT_BUILDER("Split", GetSplitGradient);
  REGISTER_GRADIENT_BUILDER("Relu", GetReluGradient);
  REGISTER_GRADIENT_BUILDER("Pow", GetPowGradient);
  REGISTER_GRADIENT_BUILDER("ReduceMean", GetReduceMeanGradient);
  REGISTER_GRADIENT_BUILDER("Add", GetAddGradient);
  REGISTER_GRADIENT_BUILDER("Sub", GetSubGradient);
  REGISTER_GRADIENT_BUILDER("Concat", GetConcatGradient);
  REGISTER_GRADIENT_BUILDER("Reshape", GetReshapeGradient);
  REGISTER_GRADIENT_BUILDER("Gemm", GetGemmGradient);
  REGISTER_GRADIENT_BUILDER("AveragePool", GetPoolGradient);
  REGISTER_GRADIENT_BUILDER("MaxPool", GetPoolGradient);
  REGISTER_GRADIENT_BUILDER("LRN", GetLRNGradient);
  REGISTER_GRADIENT_BUILDER("Dropout", GetDropoutGradient);
  REGISTER_GRADIENT_BUILDER("Conv", GetConvGradient);
  REGISTER_GRADIENT_BUILDER("Softmax", GetSoftmaxGradient);
};

}  // namespace training
}  // namespace onnxruntime
