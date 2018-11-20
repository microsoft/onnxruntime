// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"

struct ONNXRuntimeSessionOptions {
  const ONNXObject* const cls;
  std::atomic_int ref_count;
  onnxruntime::SessionOptions value;
  std::vector<std::string> custom_op_paths;
  std::vector<ONNXRuntimeProviderFactoryPtr*> provider_factories;
  ONNXRuntimeSessionOptions();
  ~ONNXRuntimeSessionOptions();
  ONNXRuntimeSessionOptions(const ONNXRuntimeSessionOptions& other);
  ONNXRuntimeSessionOptions& operator=(const ONNXRuntimeSessionOptions& other);
};