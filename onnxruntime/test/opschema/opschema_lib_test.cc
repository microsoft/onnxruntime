// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <array>
#include <iostream>

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/ms_opset.h"

int main() {
  // Full registration also pulls legacy MatMulFpQ4's pre-existing MLAS dependency.
  const std::array schemas{
      onnxruntime::contrib::GetOpSchema<onnxruntime::contrib::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          Microsoft, 1, ScaledSiLU)>(),
      onnxruntime::contrib::GetOpSchema<onnxruntime::contrib::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          Microsoft, 1, CausalConvWithState)>(),
      onnxruntime::contrib::GetOpSchema<onnxruntime::contrib::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          Microsoft, 1, MoEGroupRouter)>()};
  for (const auto& schema : schemas) {
    if (schema.Name().empty() || schema.domain() != onnxruntime::kMSDomain || schema.SinceVersion() != 1) {
      std::cerr << "Invalid standalone contrib schema: " << schema.Name() << '\n';
      return 1;
    }
  }
  return 0;
}
