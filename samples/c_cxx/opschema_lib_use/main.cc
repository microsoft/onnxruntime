// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// A sample/test application using the opschema library of ORT's custom ops.

#include <iostream>

#pragma warning(push)
#pragma warning(disable : 4100)

#include "onnx/defs/schema.h"

#pragma warning(pop)

// 
// #include "orttraining/core/graph/training_op_defs.h"

namespace onnxruntime {
extern void RegisterOrtOpSchemas();
} 

void Check(const char* domain, int version, const char* opname) {
  const onnx::OpSchema* schema = onnx::OpSchemaRegistry::Schema(opname, version, domain);
  std::cout << opname << ": " << ((schema != nullptr) ? "Found schema.\n" : "Error: Schema not found.\n");
}

int main() {
  onnxruntime::RegisterOrtOpSchemas();

  constexpr const char* kMSDomain = "com.microsoft";

  Check(kMSDomain, 1, "ReluGrad");
}