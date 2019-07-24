// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tvm_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

#define RETURN_DLDATATYPE_IF_MATCH(type, type_code) \
  if (ml_type == DataTypeImpl::GetType<type>()) {   \
    return {type_code, sizeof(type) * 8, 1};        \
  }

// DLDataType: {DLDataTypeCode, bits, lanes}
DLDataType ToTvmDLDataType(MLDataType ml_type) {
  RETURN_DLDATATYPE_IF_MATCH(int8_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(uint8_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(int16_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(uint16_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(int32_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(uint32_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(int64_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(uint64_t, kDLInt);

  RETURN_DLDATATYPE_IF_MATCH(float, kDLFloat);
  RETURN_DLDATATYPE_IF_MATCH(double, kDLFloat);

  ORT_NOT_IMPLEMENTED("converting MLDataType ", ml_type, " to tvm DLDataType is not implemented");
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
