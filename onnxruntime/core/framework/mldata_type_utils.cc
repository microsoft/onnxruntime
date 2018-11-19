// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mldata_type_utils.h"

namespace onnxruntime {
namespace utils {
MLDataType GetMLDataType(const onnxruntime::NodeArg& arg) {
  const ONNX_NAMESPACE::DataType ptype = arg.Type();
  const ONNX_NAMESPACE::TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(ptype);
  return DataTypeImpl::TypeFromProto(type_proto);
}
}  // namespace utils
}  // namespace onnxruntime
