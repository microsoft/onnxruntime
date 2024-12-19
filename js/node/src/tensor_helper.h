// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>
#include <vector>

#include "onnxruntime_cxx_api.h"

// convert a Javascript OnnxValue object to an OrtValue object
Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value, OrtMemoryInfo* cpu_memory_info, OrtMemoryInfo* webgpu_memory_info);

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
