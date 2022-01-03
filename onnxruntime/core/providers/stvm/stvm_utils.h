// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_UTILS_H
#define STVM_UTILS_H

#include "stvm_common.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/ortdevice.h"
#include "core/common/common.h"

namespace onnxruntime {

inline DLDataType GetDataType(ONNXTensorElementDataType type) {
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return {kDLFloat, 64, 1};
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return {kDLFloat, 16, 1};
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return {kDLFloat, 32, 1};
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return {kDLInt, 64, 1};
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return {kDLInt, 32, 1};
  } else {
    ORT_NOT_IMPLEMENTED("Unsupported data type");
  }
}

inline DLDataType GetDataTypeFromProto() {
  return {kDLFloat, 32, 1};
}

inline DLDevice GetDLDevice(const OrtDevice& device) {
  DLDevice context;
  switch (device.Type()) {
    case OrtDevice::CPU:
      context = {kDLCPU, 0};
      break;
    case OrtDevice::GPU:
      context = {kDLVulkan, 0};
      break;
    default:
      ORT_NOT_IMPLEMENTED("Unsupported device");
      break;
  }
  return context;
}

}  // namespace onnxruntime

#endif // STVM_UTILS_H
