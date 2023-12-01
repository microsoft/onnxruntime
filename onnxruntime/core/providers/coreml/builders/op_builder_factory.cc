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

  {  // Add/Mul/Pow/Sub/Div
    CreateBinaryOpBuilder("Add", op_registrations);
    CreateBinaryOpBuilder("Mul", op_registrations);
    CreateBinaryOpBuilder("Pow", op_registrations);
    CreateBinaryOpBuilder("Sub", op_registrations);
    CreateBinaryOpBuilder("Div", op_registrations);
  }

  {  // Activations
    CreateActivationOpBuilder("Sigmoid", op_registrations);
    CreateActivationOpBuilder("Tanh", op_registrations);
    CreateActivationOpBuilder("Relu", op_registrations);
    CreateActivationOpBuilder("PRelu", op_registrations);
    CreateActivationOpBuilder("LeakyRelu", op_registrations);
  }

  {  // Transpose
    CreateTransposeOpBuilder("Transpose", op_registrations);
  }

  {  // Conv
    CreateConvOpBuilder("Conv", op_registrations);
  }

  {  // Batch Normalization
    CreateBatchNormalizationOpBuilder("BatchNormalization", op_registrations);
  }

  {  // Reshape
    CreateReshapeOpBuilder("Reshape", op_registrations);
  }

  {  // DepthToSpace
    CreateDepthToSpaceOpBuilder("DepthToSpace", op_registrations);
  }

  {  // Pool
    CreatePoolOpBuilder("GlobalAveragePool", op_registrations);
    CreatePoolOpBuilder("GlobalMaxPool", op_registrations);
    CreatePoolOpBuilder("AveragePool", op_registrations);
    CreatePoolOpBuilder("MaxPool", op_registrations);
  }

  {  // Concat
    CreateConcatOpBuilder("Concat", op_registrations);
  }

  {  // Resize
    CreateResizeOpBuilder("Resize", op_registrations);
  }

  {  // Gemm/MatMul
    CreateGemmOpBuilder("Gemm", op_registrations);
    CreateGemmOpBuilder("MatMul", op_registrations);
  }

  {  // Clip
    CreateClipOpBuilder("Clip", op_registrations);
  }

  {  // Squeeze
    CreateSqueezeOpBuilder("Squeeze", op_registrations);
  }

  {  // ArgMax
    CreateArgMaxOpBuilder("ArgMax", op_registrations);
  }

  {  // Cast
    CreateCastOpBuilder("Cast", op_registrations);
  }

  {  // Flatten
    CreateFlattenOpBuilder("Flatten", op_registrations);
  }

  {  // LRN
    CreateLRNOpBuilder("LRN", op_registrations);
  }

  {  // Pad
    CreatePadOpBuilder("Pad", op_registrations);
  }

  {  // Unary
    CreateUnaryOpBuilder("Sqrt", op_registrations);
    CreateUnaryOpBuilder("Reciprocal", op_registrations);
  }

  {  // Reduction
     // ReduceMean is used in layer normalization which seems to be problematic in Python tests.
    CreateReductionOpBuilder("ReduceMean", op_registrations);
    CreateReductionOpBuilder("ReduceSum", op_registrations);
  }

  {  // Shape
    CreateShapeOpBuilder("Shape", op_registrations);
  }

  {  // Gather
    CreateGatherOpBuilder("Gather", op_registrations);
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

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

}  // namespace coreml
}  // namespace onnxruntime
