// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_builder_factory.h"

#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"

namespace onnxruntime {
namespace nnapi {

static OpBuilderRegistrations CreateOpBuilderRegistrations() {
  OpBuilderRegistrations op_registrations;

  {
    // Builders handle a single op
    CreateBatchNormalizationOpBuilder("BatchNormalization", op_registrations);
    CreateCastOpBuilder("Cast", op_registrations);
    CreateClipOpBuilder("Clip", op_registrations);
    CreateConcatOpBuilder("Concat", op_registrations);
    CreateDepthToSpaceOpBuilder("DepthToSpace", op_registrations);
    CreateDequantizeLinearOpBuilder("DequantizeLinear", op_registrations);
    CreateEluOpBuilder("Elu", op_registrations);
    CreateFlattenOpBuilder("Flatten", op_registrations);
    CreateGatherOpBuilder("Gather", op_registrations);
    CreateIdentityOpBuilder("Identity", op_registrations);
    CreateLeakyReluOpBuilder("LeakyRelu", op_registrations);
    CreateLRNOpBuilder("LRN", op_registrations);
    CreatePadOpBuilder("Pad", op_registrations);
    CreateQuantizeLinearOpBuilder("QuantizeLinear", op_registrations);
    CreateReluOpBuilder("Relu", op_registrations);
    CreateReshapeOpBuilder("Reshape", op_registrations);
    CreateResizeOpBuilder("Resize", op_registrations);
    CreateSliceOpBuilder("Slice", op_registrations);
    CreateSoftMaxOpBuilder("Softmax", op_registrations);
    CreateSqueezeOpBuilder("Squeeze", op_registrations);
    CreateTransposeOpBuilder("Transpose", op_registrations);
    CreateUnsqueezeOpBuilder("Unsqueeze", op_registrations);
  }

  // Builders shared among similar ops
  {
    CreateBinaryOpBuilder("Add", op_registrations);
    CreateBinaryOpBuilder("Div", op_registrations);
    CreateBinaryOpBuilder("Mul", op_registrations);
    CreateBinaryOpBuilder("Pow", op_registrations);
    CreateBinaryOpBuilder("PRelu", op_registrations);
    CreateBinaryOpBuilder("QLinearAdd", op_registrations);
    CreateBinaryOpBuilder("QLinearMul", op_registrations);
    CreateBinaryOpBuilder("Sub", op_registrations);
  }

  {
    CreatePoolOpBuilder("AveragePool", op_registrations);
    CreatePoolOpBuilder("GlobalAveragePool", op_registrations);
    CreatePoolOpBuilder("GlobalMaxPool", op_registrations);
    CreatePoolOpBuilder("MaxPool", op_registrations);
    CreatePoolOpBuilder("QLinearAveragePool", op_registrations);
  }

  {
    CreateConvOpBuilder("Conv", op_registrations);
    CreateConvOpBuilder("QLinearConv", op_registrations);
  }

  {
    CreateGemmOpBuilder("Gemm", op_registrations);
    CreateGemmOpBuilder("MatMul", op_registrations);
    CreateGemmOpBuilder("QLinearMatMul", op_registrations);
  }

  {
    CreateUnaryOpBuilder("Abs", op_registrations);
    CreateUnaryOpBuilder("Exp", op_registrations);
    CreateUnaryOpBuilder("Floor", op_registrations);
    CreateUnaryOpBuilder("Log", op_registrations);
    CreateUnaryOpBuilder("Neg", op_registrations);
    CreateUnaryOpBuilder("QLinearSigmoid", op_registrations);
    CreateUnaryOpBuilder("Sigmoid", op_registrations);
    CreateUnaryOpBuilder("Sin", op_registrations);
    CreateUnaryOpBuilder("Sqrt", op_registrations);
    CreateUnaryOpBuilder("Tanh", op_registrations);
  }

  {
    CreateMinMaxOpBuilder("Max", op_registrations);
    CreateMinMaxOpBuilder("Min", op_registrations);
  }

  {
    CreateReductionOpBuilder("ReduceMean", op_registrations);
  }

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

}  // namespace nnapi
}  // namespace onnxruntime
