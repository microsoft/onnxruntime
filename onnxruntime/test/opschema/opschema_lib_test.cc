// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <array>
#include <iostream>

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/ms_opset.h"
#include "core/graph/contrib_ops/onnx_deprecated_opset.h"

int main() {
  size_t enumerated = 0;
  bool valid = true;
  onnxruntime::contrib::OpSet_ONNX_Deprecated::ForEachSchema([&](ONNX_NAMESPACE::OpSchema&& schema) {
    ++enumerated;
    valid &= !schema.Name().empty() && schema.domain() == onnxruntime::kOnnxDomain && schema.SinceVersion() > 0;
  });
  if (!valid || enumerated == 0) {
    std::cerr << "Invalid standalone deprecated ONNX schema enumeration\n";
    return 1;
  }
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
  auto pending = ONNX_NAMESPACE::OpSchema()
                     .Input(0, "X", "", "T")
                     .Output(0, "Y", "", "T")
                     .TypeConstraint("T", {"tensor(float)"}, "");
  onnxruntime::contrib::RecordPendingSourcePolicy(pending, "SchemaOnlyPendingSourcePolicy", 1, "Pending");
  for (const auto* schema : std::array<const ONNX_NAMESPACE::OpSchema*, 2>{&pending, &schemas.back()}) {
    if (schema->max_input() != static_cast<int>(schema->inputs().size()) ||
        schema->max_output() != static_cast<int>(schema->outputs().size()) ||
        schema->inputs().empty() || schema->inputs()[0].GetTypes().empty()) {
      std::cerr << "Standalone policy-authored schema was not finalized: " << schema->Name() << '\n';
      return 1;
    }
  }
  return 0;
}
