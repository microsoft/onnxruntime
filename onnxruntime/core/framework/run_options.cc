
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/run_options.h"
#include "core/session/onnxruntime_c_api.h"
#include <stdexcept>
#include <memory>

ONNXRUNTIME_API(ONNXRuntimeRunOptions*, ONNXRuntimeCreateRunOptions) {
  std::unique_ptr<ONNXRuntimeRunOptions> options = std::make_unique<ONNXRuntimeRunOptions>();
  return options.release();
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeRunOptionsSetRunLogVerbosityLevel, _In_ ONNXRuntimeRunOptions* options, unsigned int value) {
  options->run_log_verbosity_level = value;
  return nullptr;
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeRunOptionsSetRunTag, _In_ ONNXRuntimeRunOptions* options, _In_ const char* run_tag) {
  if (run_tag)
    options->run_tag = run_tag;
  return nullptr;
}

ONNXRUNTIME_API(unsigned int, ONNXRuntimeRunOptionsGetRunLogVerbosityLevel, _In_ ONNXRuntimeRunOptions* options) {
  return options->run_log_verbosity_level;
}
ONNXRUNTIME_API(const char*, ONNXRuntimeRunOptionsGetRunTag, _In_ ONNXRuntimeRunOptions* options) {
  return options->run_tag.c_str();
}

ONNXRUNTIME_API(void, ONNXRuntimeRunOptionsSetTerminate, _In_ ONNXRuntimeRunOptions* options, bool value) {
  options->terminate = value;
}
