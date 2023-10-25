// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include "core/common/status.h"
#include "core/common/path_string.h"
#include "core/framework/session_options.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/providers.h"

struct OrtSessionOptions {
  onnxruntime::SessionOptions value;
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
  std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> provider_factories;
  std::unordered_map<std::string, std::string> provider_options_;
  std::string external_shared_lib_path_;
  OrtSessionOptions() = default;
  ~OrtSessionOptions();
  OrtSessionOptions(const OrtSessionOptions& other);
  OrtSessionOptions& operator=(const OrtSessionOptions& other);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  onnxruntime::Status RegisterCustomOpsLibrary(onnxruntime::PathString library_name);
#endif
};
