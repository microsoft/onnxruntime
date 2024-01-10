// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <string>

#include <core/graph/graph.h>

#include "op_builder_factory.h"

namespace onnxruntime {
namespace webnn {

static OpBuilderRegistrations CreateOpBuilderRegistrations() {
  OpBuilderRegistrations op_registrations;

  {  // Unary
    CreateUnaryOpBuilder("Abs", op_registrations);
    CreateUnaryOpBuilder("Ceil", op_registrations);
    CreateUnaryOpBuilder("Cos", op_registrations);
    CreateUnaryOpBuilder("Erf", op_registrations);
    CreateUnaryOpBuilder("Exp", op_registrations);
    CreateUnaryOpBuilder("Floor", op_registrations);
    CreateUnaryOpBuilder("Identity", op_registrations);
    CreateUnaryOpBuilder("Log", op_registrations);
    CreateUnaryOpBuilder("Neg", op_registrations);
    CreateUnaryOpBuilder("Not", op_registrations);
    CreateUnaryOpBuilder("Reciprocal", op_registrations);
    CreateUnaryOpBuilder("Sin", op_registrations);
    CreateUnaryOpBuilder("Sqrt", op_registrations);
    CreateUnaryOpBuilder("Tan", op_registrations);
  }

  {  // Binary
    CreateBinaryOpBuilder("Add", op_registrations);
    CreateBinaryOpBuilder("Sub", op_registrations);
    CreateBinaryOpBuilder("Mul", op_registrations);
    CreateBinaryOpBuilder("Div", op_registrations);
    CreateBinaryOpBuilder("Pow", op_registrations);
    CreateBinaryOpBuilder("PRelu", op_registrations);
  }

  {  // Ternary
    CreateTernaryOpBuilder("Where", op_registrations);
  }

  {  // Activations
    CreateActivationOpBuilder("Elu", op_registrations);
    CreateActivationOpBuilder("HardSigmoid", op_registrations);
    CreateActivationOpBuilder("HardSwish", op_registrations);
    CreateActivationOpBuilder("LeakyRelu", op_registrations);
    CreateActivationOpBuilder("Relu", op_registrations);
    CreateActivationOpBuilder("Sigmoid", op_registrations);
    CreateActivationOpBuilder("Softplus", op_registrations);
    CreateActivationOpBuilder("Softsign", op_registrations);
    CreateActivationOpBuilder("Tanh", op_registrations);
  }

  {  // ArgMax/ArgMin
    CreateArgMaxMinOpBuilder("ArgMax", op_registrations);
    CreateArgMaxMinOpBuilder("ArgMin", op_registrations);
  }

  {  // Cast
    CreateCastOpBuilder("Cast", op_registrations);
  }

  {  // Clip
    CreateClipOpBuilder("Clip", op_registrations);
  }

  {  // Conv
    CreateConvOpBuilder("Conv", op_registrations);
    CreateConvOpBuilder("ConvTranspose", op_registrations);
  }

  {  // Concat
    CreateConcatOpBuilder("Concat", op_registrations);
  }

  {  // Expand
    CreateExpandOpBuilder("Expand", op_registrations);
  }

  {  // Gather
    CreateGatherOpBuilder("Gather", op_registrations);
  }

  {  // Flatten
    CreateFlattenOpBuilder("Flatten", op_registrations);
  }

  {  // Gemm/MatMul
    CreateGemmOpBuilder("Gemm", op_registrations);
    CreateGemmOpBuilder("MatMul", op_registrations);
  }

  {  // Logical
    CreateLogicalOpBuilder("Equal", op_registrations);
    CreateLogicalOpBuilder("Greater", op_registrations);
    CreateLogicalOpBuilder("GreaterOrEqual", op_registrations);
    CreateLogicalOpBuilder("Less", op_registrations);
    CreateLogicalOpBuilder("LessOrEqual", op_registrations);
  }

  {  // Max/Min
    CreateMaxMinOpBuilder("Max", op_registrations);
    CreateMaxMinOpBuilder("Min", op_registrations);
  }

  {  // Normalization
    CreateNormalizationOpBuilder("BatchNormalization", op_registrations);
    CreateNormalizationOpBuilder("InstanceNormalization", op_registrations);
    CreateNormalizationOpBuilder("LayerNormalization", op_registrations);
  }

  {  // Pad
    CreatePadOpBuilder("Pad", op_registrations);
  }

  {  // Pool
    CreatePoolOpBuilder("AveragePool", op_registrations);
    CreatePoolOpBuilder("GlobalAveragePool", op_registrations);
    CreatePoolOpBuilder("GlobalMaxPool", op_registrations);
    CreatePoolOpBuilder("GlobalLpPool", op_registrations);
    CreatePoolOpBuilder("LpPool", op_registrations);
    CreatePoolOpBuilder("MaxPool", op_registrations);
  }

  {  // Reduction
    CreateReductionOpBuilder("ReduceL1", op_registrations);
    CreateReductionOpBuilder("ReduceL2", op_registrations);
    CreateReductionOpBuilder("ReduceLogSum", op_registrations);
    CreateReductionOpBuilder("ReduceLogSumExp", op_registrations);
    CreateReductionOpBuilder("ReduceMax", op_registrations);
    CreateReductionOpBuilder("ReduceMean", op_registrations);
    CreateReductionOpBuilder("ReduceMin", op_registrations);
    CreateReductionOpBuilder("ReduceProd", op_registrations);
    CreateReductionOpBuilder("ReduceSum", op_registrations);
    CreateReductionOpBuilder("ReduceSumSquare", op_registrations);
  }

  {  // Reshape
    CreateReshapeOpBuilder("Reshape", op_registrations);
  }

  {  // Resize
    CreateResizeOpBuilder("Resize", op_registrations);
  }

  {  // Shape
    CreateShapeOpBuilder("Shape", op_registrations);
  }

  {  // Slice
    CreateSliceOpBuilder("Slice", op_registrations);
  }

  {  // Softmax
    CreateSoftmaxOpBuilder("Softmax", op_registrations);
  }

  {  // Split
    CreateSplitOpBuilder("Split", op_registrations);
  }

  {  // Squeeze/Unsqueeze
    CreateSqueezeUnsqueezeOpBuilder("Squeeze", op_registrations);
    CreateSqueezeUnsqueezeOpBuilder("Unsqueeze", op_registrations);
  }

  {  // Transpose
    CreateTransposeOpBuilder("Transpose", op_registrations);
  }

  return op_registrations;
}

const InlinedHashMap<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

}  // namespace webnn
}  // namespace onnxruntime
