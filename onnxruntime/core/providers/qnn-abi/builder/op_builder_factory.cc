// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/op_builder_factory.h"

#include <vector>
#include <unordered_map>
#include <string>

namespace onnxruntime {
namespace qnn {

static OpBuilderRegistrations op_registrations;

OpBuilderRegistrations::OpBuilderRegistrations() {
  {
    CreateArgMaxMinOpBuilder("ArgMax", *this);
    CreateArgMaxMinOpBuilder("ArgMin", *this);
  }

  {
    CreateBatchNormOpBuilder("BatchNormalization", *this);
  }

  {
    CreateCastOpBuilder("Cast", *this);
  }

  {
    CreateClipOpBuilder("Clip", *this);
  }

  {
    CreateConvOpBuilder("Conv", *this);
    CreateConvOpBuilder("ConvTranspose", *this);
  }

  {
    CreateCumSumOpBuilder("CumSum", *this);
  }

  {
    CreateEinsumOpBuilder("Einsum", *this);
  }

  {
    CreateExpandOpBuilder("Expand", *this);
  }

  {
    CreateGatherOpBuilder("Gather", *this);
    CreateGatherOpBuilder("GatherElements", *this);
  }

  {
    CreateGemmOpBuilder("Gemm", *this);
  }

  {
    CreateInstanceNormOpBuilder("InstanceNormalization", *this);
  }

  {
    CreateLayerNormOpBuilder("LayerNormalization", *this);
  }

  {
    CreateLRNOpBuilder("LRN", *this);
  }

  {
    CreateLSTMOpBuilder("LSTM", *this);
  }

  {
    CreateMatMulOpBuilder("MatMul", *this);
  }

  {
    CreateMeanOpBuilder("Mean", *this);
  }

  {
    CreatePadOpBuilder("Pad", *this);
  }

  {
    CreatePoolOpBuilder("GlobalAveragePool", *this);
    CreatePoolOpBuilder("AveragePool", *this);
    CreatePoolOpBuilder("MaxPool", *this);
    CreatePoolOpBuilder("GlobalMaxPool", *this);
  }

  {
    CreateReciprocalOpBuilder("Reciprocal", *this);
  }

  {
    CreateReduceOpBuilder("ReduceMax", *this);
    CreateReduceOpBuilder("ReduceMean", *this);
    CreateReduceOpBuilder("ReduceMin", *this);
    CreateReduceOpBuilder("ReduceProd", *this);
    CreateReduceOpBuilder("ReduceSum", *this);
    CreateReduceOpBuilder("ReduceL2", *this);
  }

  {
    CreateReshapeOpBuilder("Reshape", *this);
    CreateReshapeOpBuilder("Flatten", *this);
    CreateReshapeOpBuilder("Squeeze", *this);
    CreateReshapeOpBuilder("Unsqueeze", *this);
  }

  {
    CreateResizeOpBuilder("Resize", *this);
  }

  {
    CreateSimpleOpBuilder("Add", *this);
    CreateSimpleOpBuilder("Asin", *this);
    CreateSimpleOpBuilder("Atan", *this);
    CreateSimpleOpBuilder("Mul", *this);
    CreateSimpleOpBuilder("Abs", *this);
    CreateSimpleOpBuilder("And", *this);
    CreateSimpleOpBuilder("Ceil", *this);
    CreateSimpleOpBuilder("Cos", *this);
    CreateSimpleOpBuilder("Sign", *this);
    CreateSimpleOpBuilder("Div", *this);
    CreateSimpleOpBuilder("Equal", *this);
    CreateSimpleOpBuilder("Exp", *this);
    CreateSimpleOpBuilder("Floor", *this);
    CreateSimpleOpBuilder("Greater", *this);
    CreateSimpleOpBuilder("GreaterOrEqual", *this);
    CreateSimpleOpBuilder("LeakyRelu", *this);
    CreateSimpleOpBuilder("Less", *this);
    CreateSimpleOpBuilder("LessOrEqual", *this);
    CreateSimpleOpBuilder("Log", *this);
    CreateSimpleOpBuilder("Max", *this);
    CreateSimpleOpBuilder("Min", *this);
    CreateSimpleOpBuilder("Neg", *this);
    CreateSimpleOpBuilder("Not", *this);
    CreateSimpleOpBuilder("Or", *this);
    CreateSimpleOpBuilder("Pow", *this);
    CreateSimpleOpBuilder("PRelu", *this);
    CreateSimpleOpBuilder("Relu", *this);
    CreateSimpleOpBuilder("Gelu", *this);
    CreateSimpleOpBuilder("Elu", *this);
    CreateSimpleOpBuilder("Round", *this);
    CreateSimpleOpBuilder("Where", *this);
    CreateSimpleOpBuilder("ScatterND", *this);
    CreateSimpleOpBuilder("Sigmoid", *this);
    CreateSimpleOpBuilder("Sin", *this);
    CreateSimpleOpBuilder("Sqrt", *this);
    CreateSimpleOpBuilder("Sub", *this);
    CreateSimpleOpBuilder("Sum", *this);
    CreateSimpleOpBuilder("Tanh", *this);

    CreateSimpleOpBuilder("Concat", *this);

    CreateSimpleOpBuilder("QuantizeLinear", *this);
    CreateSimpleOpBuilder("DequantizeLinear", *this);

    CreateSimpleOpBuilder("HardSwish", *this);
    CreateSimpleOpBuilder("HardSigmoid", *this);

    CreateSimpleOpBuilder("DepthToSpace", *this);
    CreateSimpleOpBuilder("SpaceToDepth", *this);

    CreateSimpleOpBuilder("GridSample", *this);

    CreateSimpleOpBuilder("LpNormalization", *this);
  }

  {
    CreateSliceOpBuilder("Slice", *this);
  }

  {
    CreateSoftmaxOpBuilder("Softmax", *this);
    CreateSoftmaxOpBuilder("LogSoftmax", *this);
  }

  {
    CreateSplitOpBuilder("Split", *this);
  }

  {
    CreateTileOpBuilder("Tile", *this);
  }

  {
    CreateTopKOpBuilder("TopK", *this);
  }

  {
    CreateTransposeOpBuilder("Transpose", *this);
  }

  {
    CreateUpsampleOpBuilder("Upsample", *this);
  }
}

void RegisterUDOBuilder(const std::string& op_type, const std::string& op_package) {
  CreateUDOBuilder(op_type, op_package, op_registrations);
}

const IOpBuilder* GetOpBuilder(const std::string& onnx_op_type) {
  return op_registrations.GetOpBuilderByOnnxOpType(onnx_op_type);
}

}  // namespace qnn
}  // namespace onnxruntime
