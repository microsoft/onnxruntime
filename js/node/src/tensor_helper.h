// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <napi.h>
#include <vector>

#include "onnxruntime_cxx_api.h"

struct SessionCore;

// A device allocation owned by a gpu-buffer Tensor. It keeps the session that produced it alive, because the buffer
// belongs to an allocator of that session's execution provider. Destruction is posted with ReleaseUnderDeviceLock(),
// so the last reference may be dropped on any thread.
struct DeviceValue {
  // Declared first so that it is released last, after the value.
  std::shared_ptr<SessionCore> session;
  Ort::Value value;
};
using DeviceValuePtr = std::shared_ptr<DeviceValue>;

// a Javascript Tensor converted for a run
struct NapiTensor {
  // what ORT is given; empty for a preallocated CPU output (see NapiValueToOrtValue)
  Ort::Value value{nullptr};
  // for a gpu-buffer tensor: the allocation `value` is a view over, held for as long as ORT may use the view
  DeviceValuePtr device_value;
  // for a preallocated CPU output: the typed array to copy the result into, and the type and shape it was declared
  // with, which the result has to match
  Napi::Value data;
  ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> dims;
};

// convert a Javascript OnnxValue object to an OrtValue object
//
// The run is asynchronous and Javascript can detach or rewrite a buffer while it is in flight, so ORT is never handed
// Javascript memory: CPU inputs are copied into memory from 'cpu_allocator', and a preallocated CPU output is left
// for ORT to allocate and copied back with CopyOrtValueToNapiTypedArray() once the run has completed.
NapiTensor NapiValueToOrtValue(Napi::Env env, Napi::Value value, bool is_preallocated_output,
                               OrtMemoryInfo* cpu_memory_info, OrtAllocator* cpu_allocator,
                               OrtMemoryInfo* webgpu_memory_info);

// check that an OrtValue matches the type and shape declared for a preallocated output and fits in its typed array
void ValidatePreallocatedOutput(Napi::Env env, const Ort::Value& value, Napi::Value typed_array,
                                ONNXTensorElementDataType type, const std::vector<int64_t>& dims);

// copy an OrtValue into a typed array that ValidatePreallocatedOutput() has accepted
void CopyOrtValueToNapiTypedArray(const Ort::Value& value, Napi::Value typed_array);

// convert an OrtValue object to a Javascript OnnxValue object
//
// 'session' is the session that produced the value; a gpu-buffer Tensor keeps it alive (see DeviceValue).
Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value&& value, const std::shared_ptr<SessionCore>& session);

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
