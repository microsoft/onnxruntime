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

// convert a Javascript OnnxValue object to an OrtValue object
//
// Returns an empty Ort::Value for a preallocated CPU output: the caller's buffer is never handed
// to ORT, so ORT must allocate the output and the result copied back with
// CopyOrtValueToNapiTypedArray(). 'preallocated_info' receives the declared type and shape in that
// case, and must be supplied whenever 'usage' is kPreallocatedOutput.
Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value, OrtMemoryInfo* cpu_memory_info,
                               OrtMemoryInfo* webgpu_memory_info, NapiValueUsage usage,
                               std::vector<OrtValueOwner>* value_owners = nullptr,
                               PreallocatedOutputInfo* preallocated_info = nullptr);

// copy an OrtValue tensor into an existing Javascript TypedArray, after checking that it matches
// the type and shape the caller declared for the preallocated output
void CopyOrtValueToNapiTypedArray(Napi::Env env, const Ort::Value& value, Napi::Value destination,
                                  const PreallocatedOutputInfo& expected);

// convert an OrtValue object to a Javascript OnnxValue object
Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value&& value);

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
