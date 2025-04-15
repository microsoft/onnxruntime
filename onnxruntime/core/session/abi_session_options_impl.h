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

  // This is only set when we provide an OrtSessionOptions to OrtEpFactory::CreateEp.
  // In that scenario the InferenceSession has been created with the user provided OrtSessionOptions,
  // so InferenceSession::GetSessionOptions() is the source of truth and existing_value is set to that.
  // When set, `value` should be ignored.
  // `const SessionOptions` as the expected usage in CreateEp does not involve modifying the value.
  std::optional<const onnxruntime::SessionOptions*> existing_value;

  std::vector<OrtCustomOpDomain*> custom_op_domains_;
  std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> provider_factories;
  OrtSessionOptions() = default;
  ~OrtSessionOptions();
  OrtSessionOptions(const OrtSessionOptions& other);
  OrtSessionOptions& operator=(const OrtSessionOptions& other);

  const onnxruntime::ConfigOptions& GetConfigOptions() const noexcept;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  onnxruntime::Status RegisterCustomOpsLibrary(onnxruntime::PathString library_name);
#endif
};
