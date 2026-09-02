// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <napi.h>
#include <vector>

#include "onnxruntime_cxx_api.h"

using OrtValueOwner = std::shared_ptr<OrtValue>;

enum class NapiValueUsage {
  kInput,
  kPreallocatedOutput,
};

// the type and shape a caller declared for a preallocated output, captured while the tensor is
// validated so that the model's actual output can be checked against it before being copied back
struct PreallocatedOutputInfo {
  ONNXTensorElementDataType elementType{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> dims;
};

// what a tensor conversion read out of a Javascript OnnxValue
//
// Tensor.location, Tensor.data and Tensor.gpuBuffer are ordinary Javascript properties, so they may
// be accessors that return a different object on each read. The values that were actually
// validated and used are handed back here; callers must pin, lease and copy back into exactly
// these objects rather than reading the tensor a second time.
struct NapiTensorConversion {
  // Tensor.data, and its backing ArrayBuffer for a numeric tensor. 'dataArrayBuffer' stays empty
  // for a string tensor, whose data is a plain array.
  Napi::Value data;
  Napi::Value dataArrayBuffer;
  // Region of 'dataArrayBuffer' the tensor occupies, used to lease only what is actually written.
  size_t dataByteOffset{0};
  size_t dataByteLength{0};
  // Tensor.gpuBuffer, for a gpu-buffer tensor.
  Napi::Value gpuBuffer;
  // Only filled in when 'usage' is kPreallocatedOutput.
  PreallocatedOutputInfo declared;
};

// convert a Javascript OnnxValue object to an OrtValue object
//
// Returns an empty Ort::Value for a preallocated CPU output: the caller's buffer is never handed
// to ORT, so ORT must allocate the output and the result copied back with
// CopyOrtValueToNapiTypedArray(). 'conversion' receives the JS values that were read, and must be
// supplied whenever 'usage' is kPreallocatedOutput so that the declared type and shape survive.
Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value, OrtMemoryInfo* cpu_memory_info,
                               OrtMemoryInfo* webgpu_memory_info, NapiValueUsage usage,
                               std::vector<OrtValueOwner>* value_owners = nullptr,
                               NapiTensorConversion* conversion = nullptr);

// check that an OrtValue tensor matches the type and shape the caller declared for a preallocated
// output; applies to device outputs too, which have no Javascript buffer to copy into
void ValidateOrtValueMatchesDeclared(Napi::Env env, const Ort::Value& value, const PreallocatedOutputInfo& expected);

// check that an OrtValue tensor matches the caller's declared type and shape and fits in its
// Javascript TypedArray, without writing anything
//
// Callers holding several preallocated outputs should validate all of them before copying any, so
// that a rejection cannot leave some of the caller's buffers already overwritten.
void ValidateOrtValueForNapiTypedArray(Napi::Env env, const Ort::Value& value, Napi::Value destination,
                                       const PreallocatedOutputInfo& expected);

// copy an OrtValue tensor into an existing Javascript TypedArray that
// ValidateOrtValueForNapiTypedArray() has already accepted
void CopyOrtValueToNapiTypedArray(Napi::Env env, const Ort::Value& value, Napi::Value destination);

// convert an OrtValue object to a Javascript OnnxValue object
//
// 'session' is the session that produced the value. A device-backed output keeps a reference to it,
// because its buffer belongs to an allocator owned by that session's execution provider and cannot
// be released once the session is gone.
Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value&& value, std::shared_ptr<Ort::Session> session = nullptr);

enum DataLocation {
  DATA_LOCATION_NONE = 0,
  DATA_LOCATION_CPU = 1,
  DATA_LOCATION_CPU_PINNED = 2,
  DATA_LOCATION_TEXTURE = 3,
  DATA_LOCATION_GPU_BUFFER = 4,
  DATA_LOCATION_ML_TENSOR = 5
};

inline DataLocation ParseDataLocation(const std::string& location) {
  if (location == "cpu") {
    return DATA_LOCATION_CPU;
  } else if (location == "cpu-pinned") {
    return DATA_LOCATION_CPU_PINNED;
  } else if (location == "texture") {
    return DATA_LOCATION_TEXTURE;
  } else if (location == "gpu-buffer") {
    return DATA_LOCATION_GPU_BUFFER;
  } else if (location == "ml-tensor") {
    return DATA_LOCATION_ML_TENSOR;
  } else {
    return DATA_LOCATION_NONE;
  }
}
