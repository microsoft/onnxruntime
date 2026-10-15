// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdint>

#include "custom_op_library.h"

namespace {

struct TensorMetadata {
  size_t length;
  ONNXTensorElementDataType type;
};

thread_local TensorMetadata metadata{};

OrtStatus* ORT_API_CALL GetValue(const OrtValue*, int, OrtAllocator*, OrtValue** value) noexcept {
  *value = reinterpret_cast<OrtValue*>(&metadata);
  return nullptr;
}

OrtStatus* ORT_API_CALL GetTensorTypeAndShape(const OrtValue*, OrtTensorTypeAndShapeInfo** info) noexcept {
  *info = reinterpret_cast<OrtTensorTypeAndShapeInfo*>(&metadata);
  return nullptr;
}

OrtStatus* ORT_API_CALL GetTensorElementType(const OrtTensorTypeAndShapeInfo*,
                                             ONNXTensorElementDataType* type) noexcept {
  *type = metadata.type;
  return nullptr;
}

OrtStatus* ORT_API_CALL GetTensorShapeElementCount(const OrtTensorTypeAndShapeInfo*, size_t* length) noexcept {
  *length = metadata.length;
  return nullptr;
}

// An incorrect capacity check must fail the test before it can read nonexistent tensor data.
OrtStatus* ORT_API_CALL GetTensorMutableData(OrtValue*, void**) noexcept {
  return reinterpret_cast<OrtStatus*>(&metadata);
}

OrtStatus* ORT_API_CALL GetStringTensorDataLength(const OrtValue*, size_t*) noexcept {
  return reinterpret_cast<OrtStatus*>(&metadata);
}

OrtErrorCode ORT_API_CALL GetErrorCode(const OrtStatus*) noexcept {
  return ORT_FAIL;
}

const char* ORT_API_CALL GetErrorMessage(const OrtStatus*) noexcept {
  return "Unexpected tensor data access in the array capacity test";
}

void ORT_API_CALL ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo*) {}
void ORT_API_CALL ReleaseValue(OrtValue*) {}
void ORT_API_CALL ReleaseStatus(OrtStatus*) {}

const OrtApi api = [] {
  OrtApi result{};
  result.GetValue = GetValue;
  result.GetTensorTypeAndShape = GetTensorTypeAndShape;
  result.GetTensorElementType = GetTensorElementType;
  result.GetTensorShapeElementCount = GetTensorShapeElementCount;
  result.GetTensorMutableData = GetTensorMutableData;
  result.GetStringTensorDataLength = GetStringTensorDataLength;
  result.GetErrorCode = GetErrorCode;
  result.GetErrorMessage = GetErrorMessage;
  result.ReleaseTensorTypeAndShapeInfo = ReleaseTensorTypeAndShapeInfo;
  result.ReleaseValue = ReleaseValue;
  result.ReleaseStatus = ReleaseStatus;
  return result;
}();

}  // namespace

// The unused JNI environment and class arguments are opaque: this test library needs no JNI headers.
extern "C" ORT_EXPORT int64_t ORT_API_CALL Java_ai_onnxruntime_OnnxTensorTest_getCapacityTestApi(
    void*, void*, int64_t length, int32_t type) {
  constexpr std::array types{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING};
  if (type < 0 || static_cast<size_t>(type) >= types.size()) {
    return 0;
  }
  metadata = {static_cast<size_t>(length), types[static_cast<size_t>(type)]};
  return reinterpret_cast<int64_t>(&api);
}
