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

bool IsOptionalTensor(MLDataType type) {
  return type->IsOptionalType() &&
         type->AsOptionalType()->GetElementType()->IsTensorType();
}

bool IsOptionalSeqTensor(MLDataType type) {
  return type->IsOptionalType() &&
         type->AsOptionalType()->GetElementType()->IsTensorSequenceType();
}

}  // namespace utils
}  // namespace onnxruntime
