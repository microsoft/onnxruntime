// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_schema_defs.h"
#include "gradient_builder_registry.h"
#include "gradient_builder.h"

namespace onnxruntime {
namespace training {

GradientDef GetGradientForOp(const Node* node,
                             const std::unordered_set<std::string>& output_args_need_grad,
                             const std::unordered_set<std::string>& input_args_need_grad) {
  // REVIEW(mzs): The below condition does not seem correct, it needs to be >= GRADIENT_OP_VERSION
  // but changing it will break bunch of tests since many operators like sqrt are version 6,
  // yet have a grad operator. However changing the opset requires changing the operator
  // so merely adding the gradient operator does not warrant a version update. If we leave
  // the condition as it is it will not work for operators with gradient that have a version
  // higher than 9, example Slice has version 1, 10 and 11. The grad operator is defined
  // for version 10 and 11.
  //
  // REVIEW(bahuang): We don't have a version control for forward to backward op mapping.
  // Current SliceGrad(kMSDomain, 1) only supports Slice(kOnnxDomain, 10/11) because adding grad operator for versions
  // less than 9 is not supported and for Slice we have Slice-1, Slice-10 and Slice-11.

  /*ORT_ENFORCE(
      node->Op()->SinceVersion() <= GRADIENT_OP_VERSION,
      "Gradients are supported for opset version" + std::to_string(node->Op()->SinceVersion()) +
          "Upgrade your model to use opset" + std::to_string(GRADIENT_OP_VERSION));
          */
  auto gradient_builder = GradientBuilderRegistry::GetInstance().MakeUnique(node->OpType(),
                                                                            node,
                                                                            output_args_need_grad,
                                                                            input_args_need_grad);

  ORT_ENFORCE(gradient_builder != nullptr,
              "The gradient builder has not been registered:", node->OpType());

  return gradient_builder->GetGradientDefs();
}

#define REGISTER_GRADIENT_BUILDER(op, gradientbuilder) \
  GradientBuilderRegistry::GetInstance().Register<gradientbuilder>(op);

#define NO_GRADIENT(op) REGISTER_GRADIENT_BUILDER(op, EmptyGradientBuilder)

// There are some operators which are not really computation operators and one shouldn't attempt to
// request one for such operators.
#define SHOULD_NOT_DO_GRADIENT(op) REGISTER_GRADIENT_BUILDER(op, UnSupportedGradientBuilder)

void GradientBuilderRegistry::RegisterGradientBuilders() {
  // Register gradient builders here.
  REGISTER_GRADIENT_BUILDER("Cast", GetCastGradient);
  REGISTER_GRADIENT_BUILDER("Sin", GetSinGradient);
  REGISTER_GRADIENT_BUILDER("Tanh", GetTanhGradient);
  REGISTER_GRADIENT_BUILDER("Sqrt", GetSqrtGradient);
  REGISTER_GRADIENT_BUILDER("Erf", GetErfGradient);
  REGISTER_GRADIENT_BUILDER("MatMul", GetMatMulGradient);
  REGISTER_GRADIENT_BUILDER("Split", GetSplitGradient);
  REGISTER_GRADIENT_BUILDER("Relu", GetReluGradient);
  REGISTER_GRADIENT_BUILDER("Pow", GetPowGradient);
  REGISTER_GRADIENT_BUILDER("ReduceMean", GetReduceMeanGradient);
  REGISTER_GRADIENT_BUILDER("ReduceSum", GetReduceSumGradient);
  REGISTER_GRADIENT_BUILDER("Add", GetAddSubGradient);
  REGISTER_GRADIENT_BUILDER("Sub", GetAddSubGradient);
  REGISTER_GRADIENT_BUILDER("Mul", GetMulGradient);
  REGISTER_GRADIENT_BUILDER("Div", GetDivGradient);
  REGISTER_GRADIENT_BUILDER("Concat", GetConcatGradient);
  REGISTER_GRADIENT_BUILDER("Reshape", GetReshapeGradient);
  REGISTER_GRADIENT_BUILDER("Transpose", GetTransposeGradient);
  REGISTER_GRADIENT_BUILDER("Gemm", GetGemmGradient);
  REGISTER_GRADIENT_BUILDER("MaxPool", GetMaxPoolGradient);
  REGISTER_GRADIENT_BUILDER("Gather", GetGatherGradient);
  REGISTER_GRADIENT_BUILDER("Conv", GetConvGradient);
  REGISTER_GRADIENT_BUILDER("Squeeze", GetSqueezeGradient);
  REGISTER_GRADIENT_BUILDER("Unsqueeze", GetUnsqueezeGradient);
  REGISTER_GRADIENT_BUILDER("Softmax", GetSoftmaxGradient);
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropy", GetSoftmaxCrossEntropyGradient);
  REGISTER_GRADIENT_BUILDER("SparseSoftmaxCrossEntropy", GetSparseSoftmaxCrossEntropyGradient);
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropyLoss", GetSoftmaxCrossEntropyLossGradient);
  REGISTER_GRADIENT_BUILDER("GlobalAveragePool", GetGlobalAveragePoolGradient);
  REGISTER_GRADIENT_BUILDER("AveragePool", GetAveragePoolGradient);
  REGISTER_GRADIENT_BUILDER("Dropout", GetDropoutGradient)
  REGISTER_GRADIENT_BUILDER("TrainableDropout", GetTrainableDropoutGradient)
  REGISTER_GRADIENT_BUILDER("GatherND", GetGatherNDGradient)
  REGISTER_GRADIENT_BUILDER("GatherElements", GetGatherElementsGradient)
  REGISTER_GRADIENT_BUILDER("Gelu", GetGeluGradient)
  REGISTER_GRADIENT_BUILDER("BiasGelu", GetBiasGeluGradient);
  REGISTER_GRADIENT_BUILDER("FastGelu", GetFastGeluGradient);
  REGISTER_GRADIENT_BUILDER("LayerNormalization", GetLayerNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("BatchNormalization", GetBatchNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("MegatronF", GetMegatronFGradient);
  REGISTER_GRADIENT_BUILDER("MegatronG", GetMegatronGGradient);
  REGISTER_GRADIENT_BUILDER("Slice", GetSliceGradient);
  REGISTER_GRADIENT_BUILDER("Where", GetWhereGradient);
  REGISTER_GRADIENT_BUILDER("Send", GetSendGradient);
  REGISTER_GRADIENT_BUILDER("Recv", GetRecvGradient);
  REGISTER_GRADIENT_BUILDER("Expand", GetExpandGradient);
};

}  // namespace training
}  // namespace onnxruntime
