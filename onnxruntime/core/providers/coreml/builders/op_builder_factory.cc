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

  {  // Add
    CreateBinaryOpBuilder("Add", op_registrations);
  }

  {  // Activations
    CreateActivationOpBuilder("Sigmoid", op_registrations);
    CreateActivationOpBuilder("Tanh", op_registrations);
    CreateActivationOpBuilder("Relu", op_registrations);
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

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

}  // namespace coreml
}  // namespace onnxruntime