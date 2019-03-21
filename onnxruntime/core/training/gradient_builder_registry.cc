// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/gradient_builder_registry.h"
#include "core/training/gradient_builder.h"
#include "core/training/gradient_op_schema.h"

namespace onnxruntime {
namespace training {

GradientDef GetGradientForOp(const Node* node,
                             const std::unordered_set<std::string>& output_args_need_grad,
                             const std::unordered_set<std::string>& input_args_need_grad) {
  ORT_ENFORCE(
      node->Op()->SinceVersion() <= GRADIENT_OP_VERSION,
      "Gradients are supported for opset version" + std::to_string(node->Op()->SinceVersion()) +
          "Upgrade your model to use opset" + std::to_string(GRADIENT_OP_VERSION));

  auto gradient_builder = GradientBuilderRegistry::GetInstance().MakeUnique(node->OpType(),
                                                                            node,
                                                                            output_args_need_grad,
                                                                            input_args_need_grad);

  ORT_ENFORCE(gradient_builder != nullptr,
              "The gradient builder has not been registered:", node->OpType());

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

#define REGISTER_GRADIENT_BUILDER(op, gradientbuilder) \
  GradientBuilderRegistry::GetInstance().Register<gradientbuilder>(op);

#define NO_GRADIENT(op) REGISTER_GRADIENT_BUILDER(op, EmptyGradientBuilder)

// There are some operators which are not really computation operators and one shouldn't attempt to
// request one for such operators.
#define SHOULD_NOT_DO_GRADIENT(op) REGISTER_GRADIENT_BUILDER(op, UnSupportedGradientBuilder)

void GradientBuilderRegistry::RegisterGradientBuilders() {
  // Register gradient builders here.
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
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropy", GetSoftmaxCrossEntropyGradient);
  REGISTER_GRADIENT_BUILDER("GlobalAveragePool", GetGlobalAveragePoolGradient);
};

}  // namespace training
}  // namespace onnxruntime
