// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace optimizer_utils {

// Check if TensorProto contains a floating point type.
static bool IsFloatingPointDataType(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return !(tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
           tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
           tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
}
}  // namespace optimizer_utils
}  // namespace onnxruntime
