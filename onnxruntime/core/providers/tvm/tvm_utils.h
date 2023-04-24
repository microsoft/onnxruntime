// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_UTILS_H
#define TVM_UTILS_H

#include <string>

#include "tvm_common.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/ortdevice.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm {

inline DLDataType GetDataType(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return {kDLUInt, 8, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return {kDLInt, 8, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return {kDLUInt, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return {kDLInt, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return {kDLUInt, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return {kDLInt, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return {kDLUInt, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return {kDLInt, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return {kDLFloat, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return {kDLFloat, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return {kDLFloat, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return {kDLUInt, 1, 1};
    default:
      ORT_NOT_IMPLEMENTED("Unsupported data type");
  }
}

inline DLDevice GetDLDevice(OrtMemoryInfoDeviceType device_type) {
  DLDevice context;
  switch (device_type) {
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

std::string readFromFile(const std::string& file_path);

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_UTILS_H
