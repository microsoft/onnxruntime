// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <cmath>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "common.h"
#include "device_lock.h"
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

namespace {
std::string DescribeShape(const std::vector<int64_t>& dims) {
  std::ostringstream description;
  description << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    description << (i > 0 ? "," : "") << dims[i];
  }
  description << "]";
  return description.str();
}

const char* DescribeType(ONNXTensorElementDataType type) {
  const char* name = static_cast<size_t>(type) < ONNX_TENSOR_ELEMENT_DATA_TYPE_COUNT ? DATA_TYPE_ID_TO_NAME_MAP[type]
                                                                                     : nullptr;
  return name != nullptr ? name : "unsupported";
}

DeviceValuePtr MakeDeviceValue(Ort::Value&& value, std::shared_ptr<SessionCore> session) {
  return DeviceValuePtr(new DeviceValue{std::move(session), std::move(value)}, [](DeviceValue* deviceValue) {
    ReleaseUnderDeviceLock([deviceValue] { delete deviceValue; });
  });
}

// An output handed to Javascript as an external ArrayBuffer over ORT's memory. The singleton reference keeps ORT
// alive until the ArrayBuffer is collected, which can be after the environment cleanup hook has run.
struct ExternalOutput {
  Ort::Value value;
  OrtSingletonData::Reference ort;
};

// Electron's V8 memory cage, and any build with the V8 sandbox, refuse external ArrayBuffers for the life of the
// process, so a refusal is remembered rather than retried for every output.
std::atomic<bool> external_array_buffers_refused{false};

Napi::ArrayBuffer CreateOutputArrayBuffer(Napi::Env env, Ort::Value&& value) {
  const size_t byteLength = value.GetTensorSizeInBytes();
  if (byteLength > 0 && !external_array_buffers_refused.load(std::memory_order_relaxed)) {
    std::unique_ptr<ExternalOutput> output{new ExternalOutput{std::move(value), {}}};
    napi_value arrayBuffer = nullptr;
    napi_status status = napi_create_external_arraybuffer(
        env, output->value.GetTensorMutableRawData(), byteLength,
        [](napi_env, void*, void* hint) { delete static_cast<ExternalOutput*>(hint); }, output.get(), &arrayBuffer);
    if (status == napi_ok) {
      output.release();
      return Napi::ArrayBuffer(env, arrayBuffer);
    }
    if (status == napi_no_external_buffers_allowed) {
      external_array_buffers_refused.store(true, std::memory_order_relaxed);
    }
    value = std::move(output->value);
  }

  auto arrayBuffer = Napi::ArrayBuffer::New(env, byteLength);
  if (byteLength > 0) {
    memcpy(arrayBuffer.Data(), value.GetTensorRawData(), byteLength);
  }
  return arrayBuffer;
}
}  // namespace

// currently only support tensor
NapiTensor NapiValueToOrtValue(Napi::Env env, Napi::Value value, bool is_preallocated_output,
                               OrtMemoryInfo* cpu_memory_info, OrtAllocator* cpu_allocator,
                               OrtMemoryInfo* webgpu_memory_info) {
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

  NapiTensor result;
  if (tensorTypeString == "string") {
    ORT_NAPI_THROW_TYPEERROR_IF(is_preallocated_output, env, "Preallocated string output tensors are not supported.");

    auto tensorDataValue = tensorObject.Get("data");

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
    result.value = Ort::Value::CreateTensor(allocator, dims.empty() ? nullptr : &dims[0], dims.size(),
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    if (stringDataCStr.size() > 0) {
      Ort::ThrowOnError(Ort::GetApi().FillStringTensor(result.value, &stringDataCStr[0], stringDataCStr.size()));
    }
    return result;
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

      char* buffer = reinterpret_cast<char*>(tensorDataTypedArray.ArrayBuffer().Data());
      size_t bufferByteOffset = tensorDataTypedArray.ByteOffset();
      size_t bufferByteLength = tensorDataTypedArray.ByteLength();
      // A view over the Javascript buffer, created only to check that it is large enough for 'dims'.
      auto source = Ort::Value::CreateTensor(cpu_memory_info, buffer + bufferByteOffset, bufferByteLength,
                                             dims.empty() ? nullptr : &dims[0], dims.size(), elemType);

      if (is_preallocated_output) {
        // ORT may replace a preallocated fetch instead of writing into it, so it is not given one. ORT also skips
        // its type and shape checks for an output it allocates itself; ValidatePreallocatedOutput() does them.
        result.data = tensorDataValue;
        result.type = elemType;
        result.dims = std::move(dims);
        return result;
      }

      // The session's allocator follows its enableCpuMemArena setting, so with the arena on, repeated runs reuse
      // memory that is already mapped.
      result.value = Ort::Value::CreateTensor(cpu_allocator, dims.empty() ? nullptr : &dims[0], dims.size(), elemType);
      const size_t byteLength = source.GetTensorSizeInBytes();
      if (byteLength > 0) {
        memcpy(result.value.GetTensorMutableRawData(), source.GetTensorRawData(), byteLength);
      }
      return result;
    } else {
      ORT_NAPI_THROW_TYPEERROR_IF(tensorLocation != DATA_LOCATION_GPU_BUFFER, env, "Tensor.location must be 'gpu-buffer' for IO binding.");

      auto gpuBufferValue = tensorObject.Get("gpuBuffer");
      // nodejs: tensor.gpuBuffer is no longer a GPUBuffer in nodejs. It is an external object holding a DeviceValuePtr.
      ORT_NAPI_THROW_TYPEERROR_IF(!gpuBufferValue.IsExternal(), env, "Tensor.gpuBuffer must be an external object.");
      auto* deviceValue = gpuBufferValue.As<Napi::External<DeviceValuePtr>>().Data();
      ORT_NAPI_THROW_ERROR_IF(deviceValue == nullptr || *deviceValue == nullptr, env,
                              "Tensor.gpuBuffer has been disposed.");
      result.device_value = *deviceValue;
      Ort::Value& owned = result.device_value->value;

      // Tensor.type and Tensor.dims are plain Javascript properties. A view declaring a larger shape than the
      // allocation would let ORT read or write past its end.
      auto ownedInfo = owned.GetTensorTypeAndShapeInfo();
      ORT_NAPI_THROW_TYPEERROR_IF(ownedInfo.GetElementType() != elemType, env,
                                  "Tensor.type does not match the type of the GPU buffer it wraps.");
      ORT_NAPI_THROW_ERROR_IF(ownedInfo.GetShape() != dims, env,
                              "Tensor.dims does not match the shape of the GPU buffer it wraps.");

      result.value = Ort::Value::CreateTensor(webgpu_memory_info, owned.GetTensorMutableRawData(),
                                              owned.GetTensorSizeInBytes(), dims.empty() ? nullptr : &dims[0],
                                              dims.size(), elemType);
      return result;
    }
  }
}

void ValidatePreallocatedOutput(Napi::Env env, const Ort::Value& value, Napi::Value typed_array,
                                ONNXTensorElementDataType type, const std::vector<int64_t>& dims) {
  auto typeAndShapeInfo = value.GetTensorTypeAndShapeInfo();
  ORT_NAPI_THROW_TYPEERROR_IF(typeAndShapeInfo.GetElementType() != type, env, "Preallocated output tensor has type ",
                              DescribeType(type), ", but the model produced ",
                              DescribeType(typeAndShapeInfo.GetElementType()), ".");
  ORT_NAPI_THROW_ERROR_IF(typeAndShapeInfo.GetShape() != dims, env, "Preallocated output tensor has shape ",
                          DescribeShape(dims), ", but the model produced ",
                          DescribeShape(typeAndShapeInfo.GetShape()), ".");

  // The buffer was checked when the run started, but Javascript may have detached it since.
  ORT_NAPI_THROW_TYPEERROR_IF(!typed_array.IsTypedArray(), env, "Preallocated output Tensor.data must be a typed array.");
  const size_t byteLength = value.GetTensorSizeInBytes();
  auto typedArray = typed_array.As<Napi::TypedArray>();
  ORT_NAPI_THROW_ERROR_IF(typedArray.ByteLength() < byteLength ||
                              (byteLength > 0 && typedArray.ArrayBuffer().Data() == nullptr),
                          env, "Preallocated output tensor buffer was detached or is too small.");
}

void CopyOrtValueToNapiTypedArray(const Ort::Value& value, Napi::Value typed_array) {
  const size_t byteLength = value.GetTensorSizeInBytes();
  if (byteLength > 0) {
    auto typedArray = typed_array.As<Napi::TypedArray>();
    memcpy(static_cast<char*>(typedArray.ArrayBuffer().Data()) + typedArray.ByteOffset(), value.GetTensorRawData(),
           byteLength);
  }
}

Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value&& value, const std::shared_ptr<SessionCore>& session) {
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

      auto deviceValue = std::make_unique<DeviceValuePtr>(MakeDeviceValue(std::move(value), session));
      auto gpuBuffer = Napi::External<DeviceValuePtr>::New(env, deviceValue.get(),
                                                           [](Napi::Env, DeviceValuePtr* p) { delete p; });
      deviceValue.release();

      auto options = Napi::Object::New(env);
      options.Set("dataType", type);
      options.Set("dims", dims);
      // Drops the Tensor's reference. A run that is using the buffer holds its own and keeps it alive until it ends.
      options.Set("dispose", Napi::Function::New(
                                 env, [](const Napi::CallbackInfo& info) {
                                   if (info.This().IsObject()) {
                                     auto gpuBuffer = info.This().As<Napi::Object>().Get("gpuBuffer");
                                     if (gpuBuffer.IsExternal()) {
                                       gpuBuffer.As<Napi::External<DeviceValuePtr>>().Data()->reset();
                                     }
                                   }
                                   return info.Env().Undefined();
                                 },
                                 "dispose"));
      options.Set("download", Napi::Function::New(
                                  env, [](const Napi::CallbackInfo& info) {
                                    NAPI_THROW("not implemented");
                                  },
                                  "download"));

      return scope.Escape(tensorFromGpuBuffer.Call({gpuBuffer, options}));
    } else {
      auto arrayBuffer = CreateOutputArrayBuffer(env, std::move(value));
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
