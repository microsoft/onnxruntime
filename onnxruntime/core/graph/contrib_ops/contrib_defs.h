// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace contrib {
#define ONNX_CONTRIB_OPERATOR_SCHEMA(name) \
  ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ(Counter, name)         \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

#define ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(name, schema_func) \
  ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(__COUNTER__, name, schema_func)
#define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(Counter, name, schema_func) \
  ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)
#define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func) \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(                \
      op_schema_register_once##name##Counter) ONNX_UNUSED =                     \
      schema_func(ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__))

void RegisterContribSchemas();

constexpr const float kDefaultSkipLayerNormEpsilon = 1e-12f;
constexpr const float kDefaultEmbedLayerNormEpsilon = 1e-12f;
}  // namespace contrib
}  // namespace onnxruntime
