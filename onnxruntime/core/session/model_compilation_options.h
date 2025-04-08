// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include "core/common/status.h"
#include "core/common/path_string.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

// TODO(adrianlizarraga): Clean up: add constructors, member public/private access, etc.
struct ModelCompilationOptions {
  const OrtEnv* env = nullptr;
  std::unique_ptr<OrtSessionOptions> session_options_ = nullptr;
  OrtSessionOptions* session_options_override_ = nullptr;
  std::string input_model_path;
  const void* input_model_data = nullptr;
  size_t input_model_data_size;

  OrtSessionOptions* GetSessionOptions() const;
  void ResetInputModelSettings();
  Status ResetOutputModelSettings();
  Status CheckInputModelSettings() const;
  Status CheckOutputModelSettings() const;
  Status Check() const;
};
}  // namespace onnxruntime
