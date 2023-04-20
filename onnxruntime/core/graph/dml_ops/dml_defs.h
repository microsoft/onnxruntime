// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif
#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"

namespace onnxruntime {
namespace dml {
#define MS_DML_OPERATOR_SCHEMA(name) \
  MS_DML_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define MS_DML_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  MS_DML_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define MS_DML_OPERATOR_SCHEMA_UNIQ(Counter, name)               \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

#define MS_DML_OPERATOR_SCHEMA_ELSEWHERE(name, schema_func) \
  MS_DML_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(__COUNTER__, name, schema_func)
#define MS_DML_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(Counter, name, schema_func) \
  MS_DML_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)
#define MS_DML_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func) \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(          \
      op_schema_register_once##name##Counter) ONNX_UNUSED =               \
      schema_func(ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__))

void RegisterDmlSchemas();
}  // namespace dml
}  // namespace onnxruntime
