// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/providers.h"

struct OrtSessionOptions {
  onnxruntime::SessionOptions value;
  std::vector<std::string> custom_op_paths;
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
  std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> provider_factories;
  OrtSessionOptions() = default;
  ~OrtSessionOptions();
  OrtSessionOptions(const OrtSessionOptions& other);
  OrtSessionOptions& operator=(const OrtSessionOptions& other);
};
