// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace neutron {

#define NEUTRON_OPERATOR_SCHEMA(name) \
  NEUTRON_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define NEUTRON_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  NEUTRON_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define NEUTRON_OPERATOR_SCHEMA_UNIQ(Counter, name)              \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

void RegisterNeutronSchemas();
}  // namespace neutron
}  // namespace onnxruntime
