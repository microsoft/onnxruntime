// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "common.h"
#include "ort_instance_data.h"
#include "ort_singleton_data.h"
#include "tensor_helper.h"
#include "inference_session_wrap.h"

// napi_float16_array was added in Node.js 23 (N-API version 10).
// Define it for older Node.js versions to support Float16Array input tensors.
#ifndef napi_float16_array
#define napi_float16_array static_cast<napi_typedarray_type>(11)
#endif

// make sure consistent with origin definition
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == 0, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == 1, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 == 2, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 == 3, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 == 4, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 == 5, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 == 6, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 == 7, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING == 8, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL == 9, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 == 10, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE == 11, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 == 12, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 == 13, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 == 14, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 == 15, "definition not consistent with OnnxRuntime");
static_assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 == 16, "definition not consistent with OnnxRuntime");
constexpr size_t ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT = 17;

// size of element in bytes for each data type. 0 indicates not supported.
constexpr size_t DATA_TYPE_ELEMENT_SIZE_MAP[] = {
    0,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED     not supported
    4,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    1,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    1,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    2,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    2,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    4,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    8,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    0,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING        N/A
    1,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    2,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    8,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    4,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    8,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    0,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64     not supported
    0,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128    not supported
    0   // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16      not supported
};
static_assert(sizeof(DATA_TYPE_ELEMENT_SIZE_MAP) == sizeof(size_t) * ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT,
              "definition not matching");

constexpr std::underlying_type_t<napi_typedarray_type> DATA_TYPE_TYPEDARRAY_MAP[] = {
    std::underlying_type_t<napi_typedarray_type>(-1),  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED     not supported
    napi_float32_array,                                // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    napi_uint8_array,                                  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    napi_int8_array,                                   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    napi_uint16_array,                                 // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    napi_int16_array,                                  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    napi_int32_array,                                  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    napi_bigint64_array,                               // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    std::underlying_type_t<napi_typedarray_type>(-1),  // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING        not supported
    napi_uint8_array,                                  // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    napi_uint16_array,                                 // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16       FLOAT16 uses Uint16Array
    napi_float64_array,                                // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    napi_uint32_array,                                 // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    napi_biguint64_array,                              // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    std::underlying_type_t<napi_typedarray_type>(-1),  // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64     not supported
    std::underlying_type_t<napi_typedarray_type>(-1),  // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128    not supported
    std::underlying_type_t<napi_typedarray_type>(-1)   // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16      not supported
};
static_assert(sizeof(DATA_TYPE_TYPEDARRAY_MAP) == sizeof(napi_typedarray_type) * ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT,
              "definition not matching");

constexpr const char* DATA_TYPE_ID_TO_NAME_MAP[] = {
    nullptr,    // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
    "float32",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    "uint8",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    "int8",     // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    "uint16",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    "int16",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    "int32",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    "int64",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    "string",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    "bool",     // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    "float16",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    "float64",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    "uint32",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    "uint64",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    nullptr,    // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
    nullptr,    // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
    nullptr     // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
};
static_assert(sizeof(DATA_TYPE_ID_TO_NAME_MAP) == sizeof(const char*) * ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT,
              "definition not matching");

const std::unordered_map<std::string, ONNXTensorElementDataType> DATA_TYPE_NAME_TO_ID_MAP = {
    {"float32", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
    {"uint8", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8},
    {"int8", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8},
    {"uint16", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16},
    {"int16", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16},
    {"int32", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
    {"int64", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
    {"string", ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
    {"bool", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL},
    {"float16", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16},
    {"float64", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE},
    {"uint32", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32},
    {"uint64", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64},
};

// currently only support tensor
Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value, OrtMemoryInfo* cpu_memory_info,
                               OrtMemoryInfo* webgpu_memory_info, NapiValueUsage usage,
                               std::vector<OrtValueOwner>* value_owners,
                               NapiTensorConversion* conversion) {
  ORT_NAPI_THROW_TYPEERROR_IF(!value.IsObject(), env, "Tensor must be an object.");

  // check 'dims'
  auto tensorObject = value.As<Napi::Object>();
  auto dimsValue = tensorObject.Get("dims");
  ORT_NAPI_THROW_TYPEERROR_IF(!dimsValue.IsArray(), env, "Tensor.dims must be an array.");

  auto dimsArray = dimsValue.As<Napi::Array>();
  auto len = dimsArray.Length();
  size_t elementSize = 1;
  std::vector<int64_t> dims;
  if (len > 0) {
    dims.reserve(len);
    for (uint32_t i = 0; i < len; i++) {
      Napi::Value dimValue = dimsArray[i];
      ORT_NAPI_THROW_TYPEERROR_IF(!dimValue.IsNumber(), env, "Tensor.dims[", i, "] is not a number.");
      auto dimNumber = dimValue.As<Napi::Number>();
      double dimDouble = dimNumber.DoubleValue();
      ORT_NAPI_THROW_RANGEERROR_IF(std::floor(dimDouble) != dimDouble || dimDouble < 0 || dimDouble > 4294967295, env,
                                   "Tensor.dims[", i, "] is invalid: ", dimDouble);
      int64_t dim = static_cast<int64_t>(dimDouble);
      dims.push_back(dim);
      elementSize *= dim;
    }
  }

  // check 'location'
  auto tensorLocationValue = tensorObject.Get("location");
  ORT_NAPI_THROW_TYPEERROR_IF(!tensorLocationValue.IsString(), env, "Tensor.location must be a string.");
  DataLocation tensorLocation = ParseDataLocation(tensorLocationValue.As<Napi::String>().Utf8Value());
  ORT_NAPI_THROW_RANGEERROR_IF(tensorLocation == DATA_LOCATION_NONE, env, "Tensor.location is not supported.");

  // check 'data' and 'type'
  auto tensorTypeValue = tensorObject.Get("type");
  ORT_NAPI_THROW_TYPEERROR_IF(!tensorTypeValue.IsString(), env, "Tensor.type must be a string.");

  auto tensorTypeString = tensorTypeValue.As<Napi::String>().Utf8Value();

  if (tensorTypeString == "string") {
    ORT_NAPI_THROW_TYPEERROR_IF(usage == NapiValueUsage::kPreallocatedOutput, env,
                                "Preallocated string output tensors are not supported.");

    auto tensorDataValue = tensorObject.Get("data");
    if (conversion != nullptr) {
      conversion->data = tensorDataValue;
    }

    ORT_NAPI_THROW_TYPEERROR_IF(tensorLocation != DATA_LOCATION_CPU, env, "Tensor.location must be 'cpu' for string tensors.");
    ORT_NAPI_THROW_TYPEERROR_IF(!tensorDataValue.IsArray(), env, "Tensor.data must be an array for string tensors.");

    auto tensorDataArray = tensorDataValue.As<Napi::Array>();
    auto tensorDataSize = tensorDataArray.Length();
    std::vector<std::string> stringData;
    std::vector<const char*> stringDataCStr;
    stringData.reserve(tensorDataSize);
    stringDataCStr.reserve(tensorDataSize);
    for (uint32_t i = 0; i < tensorDataSize; i++) {
      auto currentData = tensorDataArray.Get(i);
      ORT_NAPI_THROW_TYPEERROR_IF(!currentData.IsString(), env, "Tensor.data[", i, "] must be a string.");

      auto currentString = currentData.As<Napi::String>();
      stringData.emplace_back(currentString.Utf8Value());
      stringDataCStr.emplace_back(stringData[i].c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto tensor = Ort::Value::CreateTensor(allocator, dims.empty() ? nullptr : &dims[0], dims.size(),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    if (stringDataCStr.size() > 0) {
      Ort::ThrowOnError(Ort::GetApi().FillStringTensor(tensor, &stringDataCStr[0], stringDataCStr.size()));
    }
    return tensor;
  } else {
    // lookup numeric tensor types
    auto v = DATA_TYPE_NAME_TO_ID_MAP.find(tensorTypeString);
    ORT_NAPI_THROW_TYPEERROR_IF(v == DATA_TYPE_NAME_TO_ID_MAP.end(), env,
                                "Tensor.type is not supported: ", tensorTypeString);
    ONNXTensorElementDataType elemType = v->second;

    if (tensorLocation == DATA_LOCATION_CPU) {
      auto tensorDataValue = tensorObject.Get("data");
      ORT_NAPI_THROW_TYPEERROR_IF(!tensorDataValue.IsTypedArray(), env,
                                  "Tensor.data must be a typed array for numeric tensor.");

      auto tensorDataTypedArray = tensorDataValue.As<Napi::TypedArray>();
      std::underlying_type_t<napi_typedarray_type> typedArrayType = tensorDataValue.As<Napi::TypedArray>().TypedArrayType();

      // For float16 tensors, accept both Uint16Array and Float16Array.
      // Float16Array is a newer JavaScript type (ES2024) that may be passed by users.
      // Both use 16-bit storage, so they are compatible at the binary level.
      bool isValidTypedArray = (DATA_TYPE_TYPEDARRAY_MAP[elemType] == typedArrayType);
      if (!isValidTypedArray && elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // Accept Float16Array (napi_float16_array = 11) for float16 tensors
        isValidTypedArray = (typedArrayType == napi_float16_array);
      }

      ORT_NAPI_THROW_TYPEERROR_IF(!isValidTypedArray, env,
                                  "Tensor.data must be a typed array (", DATA_TYPE_TYPEDARRAY_MAP[elemType],
                                  elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? " or Float16Array" : "",
                                  ") for ", tensorTypeString, " tensors, but got typed array (", typedArrayType, ").");

      auto tensorDataArrayBuffer = tensorDataTypedArray.ArrayBuffer();
      if (conversion != nullptr) {
        conversion->data = tensorDataValue;
        conversion->dataArrayBuffer = tensorDataArrayBuffer;
        conversion->dataByteOffset = tensorDataTypedArray.ByteOffset();
        conversion->dataByteLength = tensorDataTypedArray.ByteLength();
      }

      char* buffer = reinterpret_cast<char*>(tensorDataArrayBuffer.Data());
      size_t bufferByteOffset = tensorDataTypedArray.ByteOffset();
      size_t bufferByteLength = tensorDataTypedArray.ByteLength();
      // Wrapping the JS buffer validates that it is large enough for 'dims'. The wrapper itself is
      // never handed to ORT: the buffer may be detached, resized or rewritten from JS while an
      // asynchronous run is in flight.
      auto sourceValue = Ort::Value::CreateTensor(cpu_memory_info, buffer + bufferByteOffset, bufferByteLength,
                                                  dims.empty() ? nullptr : &dims[0], dims.size(), elemType);

      if (usage == NapiValueUsage::kPreallocatedOutput) {
        // Hand the declared type and shape back so that the result can be validated against them
        // before it is copied into the caller's buffer. ORT performs those checks itself for a
        // preallocated fetch (InferenceSession::ValidateInputsOutputs and
        // IExecutionFrame::GetOrCreateNodeOutputMLValue) but skips them entirely for an
        // unallocated one, so they have to happen here instead.
        if (conversion != nullptr) {
          conversion->declared.elementType = elemType;
          conversion->declared.dims = dims;
        }

        // Return an empty OrtValue so that ORT allocates the output itself. Handing ORT a
        // preallocated fetch is not safe: it is free to replace the fetch instead of filling it
        // (see utils::BatchOrCopyMLValue, which assigns over the target when the producing device
        // already satisfies it), which would leave our tensor unwritten and silently copy
        // uninitialized memory back to the caller. The caller copies ORT's own output into the JS
        // buffer once the run completes.
        return Ort::Value{nullptr};
      }

      Ort::AllocatorWithDefaultOptions allocator;
      auto copiedValue = Ort::Value::CreateTensor(allocator, dims.empty() ? nullptr : &dims[0], dims.size(), elemType);
      const size_t tensorByteLength = sourceValue.GetTensorSizeInBytes();
      if (tensorByteLength > 0) {
        memcpy(copiedValue.GetTensorMutableRawData(), sourceValue.GetTensorRawData(), tensorByteLength);
      }
      return copiedValue;
    } else {
      ORT_NAPI_THROW_TYPEERROR_IF(tensorLocation != DATA_LOCATION_GPU_BUFFER, env, "Tensor.location must be 'gpu-buffer' for IO binding.");

      auto gpuBufferValue = tensorObject.Get("gpuBuffer");
      // nodejs: tensor.gpuBuffer is no longer a GPUBuffer in nodejs. It is an External holding the OrtValue owner.
      ORT_NAPI_THROW_TYPEERROR_IF(!gpuBufferValue.IsExternal(), env, "Tensor.gpuBuffer must be an external object.");
      if (conversion != nullptr) {
        conversion->gpuBuffer = gpuBufferValue;
      }
      auto* valueOwner = gpuBufferValue.As<Napi::External<OrtValueOwner>>().Data();
      ORT_NAPI_THROW_ERROR_IF(valueOwner == nullptr || !*valueOwner, env, "Tensor.gpuBuffer has been disposed.");
      Ort::Value dataValue(valueOwner->get());
      void* gpuBuffer = dataValue.GetTensorMutableRawData();
      dataValue.release();
      if (value_owners != nullptr) {
        value_owners->push_back(*valueOwner);
      }

      // The OrtValue behind the External is authoritative for type, shape and allocation size.
      // Tensor.type and Tensor.dims are ordinary mutable Javascript properties, so a caller can
      // declare a larger shape over a smaller device allocation; check the declaration against the
      // owned value before handing ORT a view over it, which for a preallocated output would
      // otherwise be an out-of-bounds device write.
      OrtTensorTypeAndShapeInfo* ownedTypeAndShape = nullptr;
      Ort::ThrowOnError(Ort::GetApi().GetTensorTypeAndShape(valueOwner->get(), &ownedTypeAndShape));
      Ort::TensorTypeAndShapeInfo ownedInfo{ownedTypeAndShape};
      ORT_NAPI_THROW_TYPEERROR_IF(ownedInfo.GetElementType() != elemType, env,
                                  "Tensor.type does not match the type of the GPU buffer it wraps.");
      ORT_NAPI_THROW_ERROR_IF(ownedInfo.GetShape() != dims, env,
                              "Tensor.dims does not match the shape of the GPU buffer it wraps.");

      if (conversion != nullptr) {
        conversion->declared.elementType = elemType;
        conversion->declared.dims = dims;
      }

      size_t dataByteLength = DATA_TYPE_ELEMENT_SIZE_MAP[elemType] * elementSize;
      return Ort::Value::CreateTensor(webgpu_memory_info, gpuBuffer, dataByteLength, dims.empty() ? nullptr : &dims[0], dims.size(), elemType);
    }
  }
}

namespace {
std::string DescribeElementType(ONNXTensorElementDataType type) {
  const auto index = static_cast<size_t>(type);
  if (index < ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT && DATA_TYPE_ID_TO_NAME_MAP[index] != nullptr) {
    return DATA_TYPE_ID_TO_NAME_MAP[index];
  }
  return "element type " + std::to_string(index);
}

std::string DescribeShape(const std::vector<int64_t>& dims) {
  std::string description = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      description += ",";
    }
    description += std::to_string(dims[i]);
  }
  return description + "]";
}
}  // namespace

void ValidateOrtValueMatchesDeclared(Napi::Env env, const Ort::Value& value,
                                     const PreallocatedOutputInfo& expected) {
  // ORT allocated this output itself and so never saw the type and shape the caller declared.
  // Reject a mismatch rather than reinterpreting the model's bytes as the declared type or
  // leaving part of the caller's buffer holding data from a previous run.
  auto typeAndShapeInfo = value.GetTensorTypeAndShapeInfo();
  const auto elementType = typeAndShapeInfo.GetElementType();
  ORT_NAPI_THROW_TYPEERROR_IF(elementType != expected.elementType, env,
                              "Preallocated output tensor has type ", DescribeElementType(expected.elementType),
                              ", but the model produced ", DescribeElementType(elementType), ".");

  const auto shape = typeAndShapeInfo.GetShape();
  ORT_NAPI_THROW_ERROR_IF(shape != expected.dims, env, "Preallocated output tensor has shape ",
                          DescribeShape(expected.dims), ", but the model produced ", DescribeShape(shape), ".");
}

void ValidateOrtValueForNapiTypedArray(Napi::Env env, const Ort::Value& value, Napi::Value destination,
                                       const PreallocatedOutputInfo& expected) {
  ValidateOrtValueMatchesDeclared(env, value, expected);

  ORT_NAPI_THROW_TYPEERROR_IF(!destination.IsTypedArray(), env,
                              "Preallocated output Tensor.data must remain a typed array.");

  auto typedArray = destination.As<Napi::TypedArray>();
  const size_t sourceByteLength = value.GetTensorSizeInBytes();
  ORT_NAPI_THROW_ERROR_IF(typedArray.ByteLength() < sourceByteLength, env,
                          "Preallocated output tensor buffer was detached or is too small.");
  if (sourceByteLength == 0) {
    return;
  }

  auto arrayBuffer = typedArray.ArrayBuffer();
  const size_t byteOffset = typedArray.ByteOffset();
  const size_t arrayBufferByteLength = arrayBuffer.ByteLength();
  ORT_NAPI_THROW_ERROR_IF(byteOffset > arrayBufferByteLength ||
                              sourceByteLength > arrayBufferByteLength - byteOffset,
                          env, "Preallocated output tensor buffer was detached or is too small.");
  auto* data = static_cast<char*>(arrayBuffer.Data());
  ORT_NAPI_THROW_ERROR_IF(data == nullptr, env, "Preallocated output tensor buffer was detached.");
}

void CopyOrtValueToNapiTypedArray(Napi::Env env, const Ort::Value& value, Napi::Value destination,
                                  const PreallocatedOutputInfo& expected) {
  // Precondition: ValidateOrtValueForNapiTypedArray() has already accepted this pair. Callers with
  // several preallocated outputs must validate all of them before copying any, so that a rejection
  // cannot leave some caller buffers already overwritten.
  (void)expected;

  const size_t sourceByteLength = value.GetTensorSizeInBytes();
  if (sourceByteLength == 0) {
    return;
  }

  auto typedArray = destination.As<Napi::TypedArray>();
  auto* data = static_cast<char*>(typedArray.ArrayBuffer().Data());
  // Unreachable while the precondition holds, but a dead pointer here would be a silent memcpy into
  // freed memory rather than an exception.
  ORT_NAPI_THROW_ERROR_IF(data == nullptr, env, "Preallocated output tensor buffer was detached.");
  memcpy(data + typedArray.ByteOffset(), value.GetTensorRawData(), sourceByteLength);
}

Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value&& value, std::shared_ptr<Ort::Session> session) {
  Napi::EscapableHandleScope scope(env);

  auto typeInfo = value.GetTypeInfo();
  auto onnxType = typeInfo.GetONNXType();

  ORT_NAPI_THROW_ERROR_IF(onnxType != ONNX_TYPE_TENSOR, env, "Non tensor type is temporarily not supported.");

  auto tensorTypeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
  auto elemType = tensorTypeAndShapeInfo.GetElementType();

  // type
  auto typeCstr = DATA_TYPE_ID_TO_NAME_MAP[elemType];
  ORT_NAPI_THROW_ERROR_IF(typeCstr == nullptr, env, "Tensor type (", elemType, ") is not supported.");
  auto type = Napi::String::New(env, typeCstr);

  // dims
  const size_t dimsCount = tensorTypeAndShapeInfo.GetDimensionsCount();
  std::vector<int64_t> dimsVector;
  if (dimsCount > 0) {
    dimsVector = tensorTypeAndShapeInfo.GetShape();
  }
  auto dims = Napi::Array::New(env, dimsCount);
  for (uint32_t i = 0; i < dimsCount; i++) {
    dims[i] = dimsVector[i];
  }

  // location
  auto memoryInfo = value.GetTensorMemoryInfo();
  bool isGpuBuffer = memoryInfo.GetDeviceType() == OrtMemoryInfoDeviceType_GPU &&
                     memoryInfo.GetAllocatorName() == "WebGPU_Buf";

  // size
  auto size = tensorTypeAndShapeInfo.GetElementCount();

  // data
  if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    // string data
    auto stringArray = Napi::Array::New(env, size);
    if (size > 0) {
      auto tempBufferLength = value.GetStringTensorDataLength();
      // create buffer of length (tempBufferLength + 1) to make sure `&tempBuffer[0]` is always valid
      std::vector<char> tempBuffer(tempBufferLength + 1);
      std::vector<size_t> tempOffsets;
      tempOffsets.resize(size);
      value.GetStringTensorContent(&tempBuffer[0], tempBufferLength, &tempOffsets[0], size);

      for (uint32_t i = 0; i < size; i++) {
        stringArray[i] =
            Napi::String::New(env, &tempBuffer[0] + tempOffsets[i],
                              i == size - 1 ? tempBufferLength - tempOffsets[i] : tempOffsets[i + 1] - tempOffsets[i]);
      }
    }

    // new Tensor("string", stringArray /* string[] */, dims /* number[] */)
    return scope.Escape(OrtInstanceData::TensorConstructor(env)
                            .New({Napi::String::New(env, "string"),
                                  stringArray,
                                  dims}));
  } else {
    // number data
    if (isGpuBuffer) {
      // Tensor.fromGpuBuffer(buffer, options)
      Napi::Function tensorFromGpuBuffer = OrtInstanceData::TensorConstructor(env)
                                               .Value()
                                               .Get("fromGpuBuffer")
                                               .As<Napi::Function>();
      // Capturing 'session' keeps the execution provider that owns this buffer alive for exactly as
      // long as the value is: the capture outlives the release below and is dropped right after it.
      auto releaseOrtValue = [session](OrtValue* value) mutable {
        if (OrtSingletonData::GetOrtObjects()) {
          Ort::GetApi().ReleaseValue(value);
          return;
        }
        // The ORT singleton is already gone. Releasing the value would call into an unloaded
        // library, and so would dropping the last reference to the session, so leak both.
        new std::shared_ptr<Ort::Session>(std::move(session));
      };
      // Build the shared owner out of the unique_ptr rather than from its raw pointer: if allocating
      // the control block throws, ownership stays with the unique_ptr instead of the deleter running
      // once for the failed shared_ptr and again while unwinding.
      auto underlyingOrtValue = std::unique_ptr<OrtValue, decltype(releaseOrtValue)>(value.release(), releaseOrtValue);
      auto valueOwner = std::make_unique<OrtValueOwner>(std::move(underlyingOrtValue));

      auto options = Napi::Object::New(env);
      options.Set("dataType", type);
      options.Set("dims", dims);
      options.Set("dispose", Napi::Function::New(env, [](const Napi::CallbackInfo& info) {
                    auto tensor = info.This().As<Napi::Object>();
                    auto gpuBuffer = tensor.Get("gpuBuffer");
                    if (gpuBuffer.IsExternal()) {
                      auto* valueOwner = gpuBuffer.As<Napi::External<OrtValueOwner>>().Data();
                      if (valueOwner != nullptr) {
                        // Returning a device buffer to the provider is device work, so it has to
                        // happen under the device lock -- but never by blocking this thread, which
                        // would stall the event loop for the length of an in-flight inference.
                        OrtInstanceData::ReleaseDeviceObject(
                            std::make_shared<OrtValueOwner>(std::move(*valueOwner)));
                        valueOwner->reset();
                      }
                    }
                  }));
      options.Set("download", Napi::Function::New(
                                  env, [](const Napi::CallbackInfo& info) {
                                    NAPI_THROW("not implemented");
                                  },
                                  "download"));

      auto external = Napi::External<OrtValueOwner>::New(env, valueOwner.get(), [](Napi::Env, OrtValueOwner* value) {
        // Collected rather than disposed: the same device work, and the same reason not to block.
        OrtInstanceData::ReleaseDeviceObject(std::shared_ptr<OrtValueOwner>(value));
      });
      valueOwner.release();
      return scope.Escape(tensorFromGpuBuffer.Call({external, options}));
    } else {
      const size_t byteLength = value.GetTensorSizeInBytes();
      Napi::ArrayBuffer arrayBuffer;
      // Hand the OrtValue's memory straight to Javascript so the output is not copied. Electron's
      // V8 Memory Cage rejects external ArrayBuffers, as does any build with the V8 sandbox
      // enabled, so attempt the allocation and fall back to a copy rather than testing for a
      // particular runtime by name. A refusal is cached per environment: it cannot change for the
      // life of the process, and re-probing would cost a failed call per output.
      bool usingExternalBuffer = false;
      if (byteLength > 0 && !OrtInstanceData::ExternalArrayBuffersRefused(env)) {
        napi_value externalArrayBuffer = nullptr;
        const napi_status status = napi_create_external_arraybuffer(
            env, value.GetTensorMutableRawData(), byteLength,
            [](napi_env, void*, void* hint) {
              // The environment cleanup hook may run before N-API finalizers during shutdown.
              if (OrtSingletonData::GetOrtObjects()) {
                Ort::GetApi().ReleaseValue(static_cast<OrtValue*>(hint));
              }
            },
            static_cast<OrtValue*>(value), &externalArrayBuffer);
        if (status == napi_ok) {
          // Ownership of the OrtValue moves to the ArrayBuffer: the finalizer above releases it, so
          // the buffer stays valid for as long as Javascript can reach it rather than dying with the
          // vector this value was returned in.
          value.release();
          arrayBuffer = Napi::ArrayBuffer(env, externalArrayBuffer);
          usingExternalBuffer = true;
        } else if (status == napi_no_external_buffers_allowed) {
          // Only this status means the runtime will never accept them. Caching any other failure
          // would downgrade every later output to a copy over something transient.
          OrtInstanceData::MarkExternalArrayBuffersRefused(env);
        }
      }

      if (!usingExternalBuffer) {
        arrayBuffer = Napi::ArrayBuffer::New(env, byteLength);
        if (byteLength > 0) {
          memcpy(arrayBuffer.Data(), value.GetTensorRawData(), byteLength);
        }
      }
      napi_value typedArrayData;
      napi_status status =
          napi_create_typedarray(env, (napi_typedarray_type)DATA_TYPE_TYPEDARRAY_MAP[elemType], size, arrayBuffer, 0, &typedArrayData);
      NAPI_THROW_IF_FAILED(env, status, Napi::Value);

      // new Tensor(type, typedArrayData, dims)
      return scope.Escape(OrtInstanceData::TensorConstructor(env)
                              .New({type,
                                    Napi::Value(env, typedArrayData),
                                    dims}));
    }
  }
}
