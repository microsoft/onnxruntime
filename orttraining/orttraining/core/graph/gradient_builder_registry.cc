// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/gradient_builder.h"
#include "orttraining/core/graph/gradient_config.h"

namespace onnxruntime {
namespace training {

GradientDef GetGradientForOp(const GradientGraphConfiguration& gradient_graph_config,
                             Graph* graph,
                             const Node* node,
                             const std::unordered_set<std::string>& output_args_need_grad,
                             const std::unordered_set<std::string>& input_args_need_grad,
                             const logging::Logger& logger) {
  // REVIEW(bahuang): We don't have a version control for forward to backward op mapping.
  // Current SliceGrad(kMSDomain, 1) only supports Slice(kOnnxDomain, 10/11) because adding grad operator for versions
  // less than 9 is not supported and for Slice we have Slice-1, Slice-10 and Slice-11.

  auto gradient_builder = GradientBuilderRegistry::GetInstance().MakeUnique(node->OpType(),
                                                                            gradient_graph_config,
                                                                            graph,
                                                                            node,
                                                                            output_args_need_grad,
                                                                            input_args_need_grad,
                                                                            logger);

  ORT_ENFORCE(gradient_builder != nullptr,
              "The gradient builder has not been registered:", node->OpType(), " for node ", node->Name());

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
  REGISTER_GRADIENT_BUILDER("Log", GetLogGradient);
  REGISTER_GRADIENT_BUILDER("Tanh", GetTanhGradient);
  REGISTER_GRADIENT_BUILDER("Sqrt", GetSqrtGradient);
  REGISTER_GRADIENT_BUILDER("Erf", GetErfGradient);
  REGISTER_GRADIENT_BUILDER("MatMul", GetMatMulGradient);
  REGISTER_GRADIENT_BUILDER("Split", GetSplitGradient);
  REGISTER_GRADIENT_BUILDER("Relu", GetReluGradient);
  REGISTER_GRADIENT_BUILDER("Pow", GetPowGradient);
  REGISTER_GRADIENT_BUILDER("ReduceMean", GetReduceMeanGradient);
  REGISTER_GRADIENT_BUILDER("ReduceSum", GetReduceSumGradient);
  REGISTER_GRADIENT_BUILDER("ReduceLogSumExp", GetReduceLogSumExpGradient);
  REGISTER_GRADIENT_BUILDER("ReduceL2", GetReduceL2Gradient);
  REGISTER_GRADIENT_BUILDER("Add", GetAddSubGradient);
  REGISTER_GRADIENT_BUILDER("Sub", GetAddSubGradient);
  REGISTER_GRADIENT_BUILDER("Mul", GetMulGradient);
  REGISTER_GRADIENT_BUILDER("Div", GetDivGradient);
  REGISTER_GRADIENT_BUILDER("Neg", GetNegGradient);
  REGISTER_GRADIENT_BUILDER("Concat", GetConcatGradient);
  REGISTER_GRADIENT_BUILDER("ConcatTraining", GetConcatTrainingGradient);
  REGISTER_GRADIENT_BUILDER("Reshape", GetReshapeGradient);
  REGISTER_GRADIENT_BUILDER("Transpose", GetTransposeGradient);
  REGISTER_GRADIENT_BUILDER("Gemm", GetGemmGradient);
  REGISTER_GRADIENT_BUILDER("MaxPool", GetMaxPoolGradient);
  REGISTER_GRADIENT_BUILDER("Gather", GetGatherGradient);
  REGISTER_GRADIENT_BUILDER("Conv", GetConvGradient);
  REGISTER_GRADIENT_BUILDER("Squeeze", GetSqueezeGradient);
  REGISTER_GRADIENT_BUILDER("Unsqueeze", GetUnsqueezeGradient);
  REGISTER_GRADIENT_BUILDER("Sigmoid", GetSigmoidGradient);
  REGISTER_GRADIENT_BUILDER("Softmax", GetSoftmaxGradient);
  REGISTER_GRADIENT_BUILDER("LogSoftmax", GetLogSoftmaxGradient);
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropy", GetSoftmaxCrossEntropyGradient);
  REGISTER_GRADIENT_BUILDER("SparseSoftmaxCrossEntropy", GetSparseSoftmaxCrossEntropyGradient);
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropyLoss", GetSoftmaxCrossEntropyLossGradient);
  REGISTER_GRADIENT_BUILDER("GlobalAveragePool", GetGlobalAveragePoolGradient);
  REGISTER_GRADIENT_BUILDER("AveragePool", GetAveragePoolGradient);
  REGISTER_GRADIENT_BUILDER("Dropout", GetDropoutGradient)
  REGISTER_GRADIENT_BUILDER("GatherND", GetGatherNDGradient)
  REGISTER_GRADIENT_BUILDER("GatherElements", GetGatherElementsGradient)
  REGISTER_GRADIENT_BUILDER("Gelu", GetGeluGradient)
  REGISTER_GRADIENT_BUILDER("BiasGelu", GetBiasGeluGradient);
  REGISTER_GRADIENT_BUILDER("FastGelu", GetFastGeluGradient);
  REGISTER_GRADIENT_BUILDER("LayerNormalization", GetLayerNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("SimplifiedLayerNormalization", GetSimplifiedLayerNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("BatchNormInternal", GetBatchNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("MegatronF", GetMegatronFGradient);
  REGISTER_GRADIENT_BUILDER("MegatronG", GetMegatronGGradient);
  REGISTER_GRADIENT_BUILDER("Slice", GetSliceGradient);
  REGISTER_GRADIENT_BUILDER("Where", GetWhereGradient);
  REGISTER_GRADIENT_BUILDER("Send", GetSendGradient);
  REGISTER_GRADIENT_BUILDER("Recv", GetRecvGradient);
  REGISTER_GRADIENT_BUILDER("Expand", GetExpandGradient);
  REGISTER_GRADIENT_BUILDER("Exp", GetExpGradient);
  REGISTER_GRADIENT_BUILDER("Flatten", GetFlattenGradient);
  REGISTER_GRADIENT_BUILDER("TopK", GetTopKGradient);
  REGISTER_GRADIENT_BUILDER("Clip", GetClipGradient);
  REGISTER_GRADIENT_BUILDER("Abs", GetAbsGradient);
  REGISTER_GRADIENT_BUILDER("Min", GetMinMaxGradient);
  REGISTER_GRADIENT_BUILDER("Max", GetMinMaxGradient);
  REGISTER_GRADIENT_BUILDER("Tile", GetTileGradient);
  REGISTER_GRADIENT_BUILDER("ATenOp", GetATenOpGradient);
  REGISTER_GRADIENT_BUILDER("Pad", GetPadGradient);
  REGISTER_GRADIENT_BUILDER("Identity", GetIdentityGradient);
  REGISTER_GRADIENT_BUILDER("PythonOp", GetPythonOpGradient);
};

}  // namespace training
}  // namespace onnxruntime
