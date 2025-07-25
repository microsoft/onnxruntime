// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <unordered_map>
#include <string>

#include <core/graph/graph.h>

#include "op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

static OpBuilderRegistrations CreateOpBuilderRegistrations() {
  OpBuilderRegistrations op_registrations;

  // Activations
  CreateActivationOpBuilder("Sigmoid", op_registrations);
  CreateActivationOpBuilder("Tanh", op_registrations);
  CreateActivationOpBuilder("Relu", op_registrations);
  CreateActivationOpBuilder("PRelu", op_registrations);
  CreateActivationOpBuilder("LeakyRelu", op_registrations);
  CreateActivationOpBuilder("Gelu", op_registrations);

  // Unary ops
  CreateUnaryOpBuilder("Erf", op_registrations);
  CreateUnaryOpBuilder("Reciprocal", op_registrations);
  CreateUnaryOpBuilder("Round", op_registrations);
  CreateUnaryOpBuilder("Sqrt", op_registrations);

  // Binary elementwise ops
  CreateBinaryOpBuilder("Add", op_registrations);
  CreateBinaryOpBuilder("Div", op_registrations);
  CreateBinaryOpBuilder("Mul", op_registrations);
  CreateBinaryOpBuilder("Max", op_registrations);
  CreateBinaryOpBuilder("Pow", op_registrations);
  CreateBinaryOpBuilder("Sub", op_registrations);

  // Pooling ops
  CreatePoolOpBuilder("AveragePool", op_registrations);
  CreatePoolOpBuilder("GlobalAveragePool", op_registrations);
  CreatePoolOpBuilder("GlobalMaxPool", op_registrations);
  CreatePoolOpBuilder("MaxPool", op_registrations);

  // Reduction ops
  CreateReductionOpBuilder("ReduceMean", op_registrations);
  CreateReductionOpBuilder("ReduceMin", op_registrations);
  CreateReductionOpBuilder("ReduceMax", op_registrations);
  CreateReductionOpBuilder("ReduceProd", op_registrations);
  CreateReductionOpBuilder("ReduceSum", op_registrations);

  // Normalization ops
  CreateBatchNormalizationOpBuilder("BatchNormalization", op_registrations);
  CreateNormalizationOpBuilder("GroupNormalization", op_registrations);
  CreateNormalizationOpBuilder("InstanceNormalization", op_registrations);
  CreateNormalizationOpBuilder("LayerNormalization", op_registrations);

  CreateArgMaxOpBuilder("ArgMax", op_registrations);
  CreateCastOpBuilder("Cast", op_registrations);
  CreateClipOpBuilder("Clip", op_registrations);
  CreateConcatOpBuilder("Concat", op_registrations);
  CreateConvOpBuilder("Conv", op_registrations);
  CreateConvTransposeOpBuilder("ConvTranspose", op_registrations);
  CreateDepthToSpaceOpBuilder("DepthToSpace", op_registrations);
  CreateFlattenOpBuilder("Flatten", op_registrations);
  CreateGatherOpBuilder("Gather", op_registrations);
  CreateGemmOpBuilder("Gemm", op_registrations);
  CreateGridSampleOpBuilder("GridSample", op_registrations);
  CreateLRNOpBuilder("LRN", op_registrations);
  CreateGemmOpBuilder("MatMul", op_registrations);
  CreatePadOpBuilder("Pad", op_registrations);
  CreateReshapeOpBuilder("Reshape", op_registrations);
  CreateResizeOpBuilder("Resize", op_registrations);
  CreateShapeOpBuilder("Shape", op_registrations);
  CreateSliceOpBuilder("Slice", op_registrations);
  CreateSplitOpBuilder("Split", op_registrations);
  CreateSoftmaxOpBuilder("Softmax", op_registrations);
  CreateSqueezeOpBuilder("Squeeze", op_registrations);
  CreateTransposeOpBuilder("Transpose", op_registrations);
  CreateSqueezeOpBuilder("Unsqueeze", op_registrations);

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

}  // namespace coreml
}  // namespace onnxruntime
