// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include "core/framework/onnx_object_cxx.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"

struct OrtSessionOptions : public onnxruntime::ObjectBase<OrtSessionOptions> {
  onnxruntime::SessionOptions value;
  std::vector<std::string> custom_op_paths;
  std::vector<OrtProviderFactoryInterface**> provider_factories;
  OrtSessionOptions() = default;
  ~OrtSessionOptions();
  OrtSessionOptions(const OrtSessionOptions& other);
  OrtSessionOptions& operator=(const OrtSessionOptions& other);
};
