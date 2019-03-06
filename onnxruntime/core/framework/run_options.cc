
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/run_options.h"
#include "core/session/onnxruntime_c_api.h"
#include <stdexcept>
#include <memory>

ORT_API(OrtRunOptions*, OrtCreateRunOptions) {
  std::unique_ptr<OrtRunOptions> options = std::make_unique<OrtRunOptions>();
  return options.release();
}

ORT_API_STATUS_IMPL(OrtRunOptionsSetRunLogVerbosityLevel, _In_ OrtRunOptions* options, unsigned int value) {
  options->run_log_verbosity_level = value;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsSetRunTag, _In_ OrtRunOptions* options, _In_ const char* run_tag) {
  if (run_tag)
    options->run_tag = run_tag;
  return nullptr;
}

ORT_API(unsigned int, OrtRunOptionsGetRunLogVerbosityLevel, _In_ OrtRunOptions* options) {
  return options->run_log_verbosity_level;
}

ORT_API(const char*, OrtRunOptionsGetRunTag, _In_ OrtRunOptions* options) {
  return options->run_tag.c_str();
}

ORT_API(void, OrtRunOptionsSetTerminate, _In_ OrtRunOptions* options, bool value) {
  options->terminate = value;
}
