// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
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
  OrtSessionOptions() = default;
  ~OrtSessionOptions();
  OrtSessionOptions(const OrtSessionOptions& other);
  OrtSessionOptions& operator=(const OrtSessionOptions& other);

  const onnxruntime::ConfigOptions& GetConfigOptions() const noexcept;

  // Adds the given provider options to the session config options using a key with the format:
  // "ep.<lowercase_provider_name>.<PROVIDER_OPTION_KEY>"
  onnxruntime::Status AddProviderOptionsToConfigOptions(
      const std::unordered_map<std::string, std::string>& provider_options, const char* provider_name);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  onnxruntime::Status RegisterCustomOpsLibrary(onnxruntime::PathString library_name);
#endif

  // get the EP prefix to used when an EP specific option is added to config_options.
  // e.g. for EP called 'MyEP' an options 'device_id' would be added as 'ep.myep.device_id'
  //      with GetProviderOptionPrefix returning 'ep.myep.'
  static std::string GetProviderOptionPrefix(const char* provider_name);
};
