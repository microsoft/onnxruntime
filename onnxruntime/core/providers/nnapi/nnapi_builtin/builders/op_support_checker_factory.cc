// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <string>

#include "op_support_checker_factory.h"

namespace onnxruntime {
namespace nnapi {

static OpSupportCheckerRegistrations CreateOpSupportCheckerRegistrations() {
  OpSupportCheckerRegistrations op_registrations;

  // OpSupportCheckers handle a single op
  {
    CreateBatchNormalizationOpSupportChecker("BatchNormalization", op_registrations);
    CreateCastOpSupportChecker("Cast", op_registrations);
    CreateClipOpSupportChecker("Clip", op_registrations);
    CreateConcatOpSupportChecker("Concat", op_registrations);
    CreateDepthToSpaceOpSupportChecker("DepthToSpace", op_registrations);
    CreateDequantizeLinearOpSupportChecker("DequantizeLinear", op_registrations);
    CreateEluOpSupportChecker("Elu", op_registrations);
    CreateFlattenOpSupportChecker("Flatten", op_registrations);
    CreateGatherOpSupportChecker("Gather", op_registrations);
    CreateLRNOpSupportChecker("LRN", op_registrations);
    CreatePadOpSupportChecker("Pad", op_registrations);
    CreateQuantizeLinearOpSupportChecker("QuantizeLinear", op_registrations);
    CreateReshapeOpSupportChecker("Reshape", op_registrations);
    CreateResizeOpSupportChecker("Resize", op_registrations);
    CreateSliceOpSupportChecker("Slice", op_registrations);
    CreateSoftMaxOpSupportChecker("Softmax", op_registrations);
    CreateSqueezeOpSupportChecker("Squeeze", op_registrations);
    CreateTransposeOpSupportChecker("Transpose", op_registrations);
    CreateUnsqueezeOpSupportChecker("Unsqueeze", op_registrations);
  }

  {
    // Identity and Relu are always supported, we use BaseOpSupportChecker as default
    CreateBaseOpSupportChecker("Identity", op_registrations);
    CreateBaseOpSupportChecker("Relu", op_registrations);
  }

  // OpSupportCheckers shared among similar ops
  {
    CreateBinaryOpSupportChecker("Add", op_registrations);
    CreateBinaryOpSupportChecker("Div", op_registrations);
    CreateBinaryOpSupportChecker("Mul", op_registrations);
    CreateBinaryOpSupportChecker("Pow", op_registrations);
    CreateBinaryOpSupportChecker("PRelu", op_registrations);
    CreateBinaryOpSupportChecker("QLinearAdd", op_registrations);
    CreateBinaryOpSupportChecker("QLinearMul", op_registrations);
    CreateBinaryOpSupportChecker("Sub", op_registrations);
  }

  {
    CreateConvOpSupportChecker("Conv", op_registrations);
    CreateConvOpSupportChecker("QLinearConv", op_registrations);
  }

  {
    CreateGemmOpSupportChecker("Gemm", op_registrations);
    CreateGemmOpSupportChecker("MatMul", op_registrations);
    CreateGemmOpSupportChecker("QLinearMatMul", op_registrations);
  }

  {
    CreateMinMaxOpSupportChecker("Max", op_registrations);
    CreateMinMaxOpSupportChecker("Min", op_registrations);
  }

  {
    CreatePoolOpSupportChecker("AveragePool", op_registrations);
    CreatePoolOpSupportChecker("GlobalAveragePool", op_registrations);
    CreatePoolOpSupportChecker("GlobalMaxPool", op_registrations);
    CreatePoolOpSupportChecker("MaxPool", op_registrations);
    CreatePoolOpSupportChecker("QLinearAveragePool", op_registrations);
  }

  {
    CreateUnaryOpSupportChecker("Abs", op_registrations);
    CreateUnaryOpSupportChecker("Exp", op_registrations);
    CreateUnaryOpSupportChecker("Floor", op_registrations);
    CreateUnaryOpSupportChecker("Log", op_registrations);
    CreateUnaryOpSupportChecker("Neg", op_registrations);
    CreateUnaryOpSupportChecker("QLinearSigmoid", op_registrations);
    CreateUnaryOpSupportChecker("Sigmoid", op_registrations);
    CreateUnaryOpSupportChecker("Sin", op_registrations);
    CreateUnaryOpSupportChecker("Sqrt", op_registrations);
    CreateUnaryOpSupportChecker("Tanh", op_registrations);
  }

  return op_registrations;
}

const std::unordered_map<std::string, const IOpSupportChecker*>& GetOpSupportCheckers() {
  static const OpSupportCheckerRegistrations op_registrations = CreateOpSupportCheckerRegistrations();
  return op_registrations.op_support_checker_map;
}

}  // namespace nnapi
}  // namespace onnxruntime
