// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mldata_type_utils.h"

namespace onnxruntime {
namespace utils {
MLDataType GetMLDataType(const onnxruntime::NodeArg& arg) {
  auto type_proto = arg.TypeAsProto();
  ORT_ENFORCE(nullptr != type_proto);
  return DataTypeImpl::TypeFromProto(*type_proto);
}
}  // namespace utils
}  // namespace onnxruntime
