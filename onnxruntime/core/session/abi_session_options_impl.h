// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include "core/framework/onnx_object_cxx.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"

struct ONNXRuntimeSessionOptions : public onnxruntime::ObjectBase<ONNXRuntimeSessionOptions> {
  onnxruntime::SessionOptions value;
  std::vector<std::string> custom_op_paths;
  std::vector<ONNXRuntimeProviderFactoryInterface**> provider_factories;
  ONNXRuntimeSessionOptions() = default;
  ~ONNXRuntimeSessionOptions();
  ONNXRuntimeSessionOptions(const ONNXRuntimeSessionOptions& other);
  ONNXRuntimeSessionOptions& operator=(const ONNXRuntimeSessionOptions& other);
};
