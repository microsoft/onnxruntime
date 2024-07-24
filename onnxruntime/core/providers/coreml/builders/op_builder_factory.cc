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

  // Unary ops
  CreateUnaryOpBuilder("Sqrt", op_registrations);
  CreateUnaryOpBuilder("Reciprocal", op_registrations);

  // Binary elementwise ops
  CreateBinaryOpBuilder("Add", op_registrations);
  CreateBinaryOpBuilder("Mul", op_registrations);
  CreateBinaryOpBuilder("Pow", op_registrations);
  CreateBinaryOpBuilder("Sub", op_registrations);
  CreateBinaryOpBuilder("Div", op_registrations);

  // Activations
  CreateActivationOpBuilder("Sigmoid", op_registrations);
  CreateActivationOpBuilder("Tanh", op_registrations);
  CreateActivationOpBuilder("Relu", op_registrations);
  CreateActivationOpBuilder("PRelu", op_registrations);
  CreateActivationOpBuilder("LeakyRelu", op_registrations);

  // Pooling ops
  CreatePoolOpBuilder("GlobalAveragePool", op_registrations);
  CreatePoolOpBuilder("GlobalMaxPool", op_registrations);
  CreatePoolOpBuilder("AveragePool", op_registrations);
  CreatePoolOpBuilder("MaxPool", op_registrations);

  // Reduction ops
  CreateReductionOpBuilder("ReduceMean", op_registrations);
  CreateReductionOpBuilder("ReduceSum", op_registrations);

  CreateArgMaxOpBuilder("ArgMax", op_registrations);
  CreateBatchNormalizationOpBuilder("BatchNormalization", op_registrations);
  CreateCastOpBuilder("Cast", op_registrations);
  CreateClipOpBuilder("Clip", op_registrations);
  CreateConcatOpBuilder("Concat", op_registrations);
  CreateConvOpBuilder("Conv", op_registrations);
  CreateConvTransposeOpBuilder("ConvTranspose", op_registrations);
  CreateDepthToSpaceOpBuilder("DepthToSpace", op_registrations);
  CreateFlattenOpBuilder("Flatten", op_registrations);
  CreateGatherOpBuilder("Gather", op_registrations);
  CreateGemmOpBuilder("Gemm", op_registrations);
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

  CreateGridSampleOpBuilder("GridSample", op_registrations);

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

}  // namespace coreml
}  // namespace onnxruntime
