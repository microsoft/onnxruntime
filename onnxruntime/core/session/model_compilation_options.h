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

struct ModelCompilationOptions {
  const OrtEnv* env = nullptr;
  std::unique_ptr<OrtSessionOptions> session_options = nullptr;
  std::string input_model_path;
  const void* input_model_data = nullptr;
  size_t input_model_data_size;

  void ResetInputModelSettings();
  Status ResetOutputModelSettings();
  Status CheckInputModelSettings() const;
  Status CheckOutputModelSettings() const;
  Status Check() const;
};
}  // namespace onnxruntime
