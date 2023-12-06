// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/onnx_protobuf.h"
#include "core/graph/contrib_ops/ms_schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif

#define ONNX_MS_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, Microsoft, ::onnxruntime::kMSDomain, ver, true, impl)

// They are in ONNX domain but they are in our source code
#define ONNX_CONTRIB_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, Onnx, ::ONNX_NAMESPACE::ONNX_DOMAIN, ver, true, impl)

namespace onnxruntime {
namespace contrib {
namespace utils {
inline bool HasDimValue(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  return dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue;
}
inline bool HasRawData(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  // Can not be UNDEFINED and can not be STRING but test for STRING is usually performed separately
  // to return an error
  return ten_proto.data_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED &&
         ten_proto.has_raw_data();  // XXX: Figure out how to do in proto3
}
}  // namespace utils

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
void RegisterNchwcSchemas();
void RegisterQuantizationSchemas();

#if defined(ORT_USE_NCCL)
void RegisterCollectiveOps();
#endif

constexpr const float kDefaultSkipLayerNormEpsilon = 1e-12f;
constexpr const float kDefaultEmbedLayerNormEpsilon = 1e-12f;
}  // namespace contrib
}  // namespace onnxruntime
