// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iostream>

#include "core/graph/contrib_ops/contrib_defs.h"

int main() {
  onnxruntime::contrib::RegisterContribSchemas();
  for (const char* name : {"ScaledSiLU", "CausalConvWithState", "MoEGroupRouter",
                           "GatherPacked", "MatMulPacked", "MoEPacked"}) {
    const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema(name, 1, onnxruntime::kMSDomain);
    if (!schema || schema->SinceVersion() != 1) {
      std::cerr << "Missing standalone contrib schema: " << name << '\n';
      return 1;
    }
  }
  return 0;
}
