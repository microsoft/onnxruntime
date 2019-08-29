
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/run_options.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/error_code_helper.h"

ORT_API_STATUS_IMPL(OrtCreateRunOptions, OrtRunOptions** out) {
  API_IMPL_BEGIN
  *out = new OrtRunOptions();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtRunOptionsSetRunLogVerbosityLevel, _In_ OrtRunOptions* options, int value) {
  options->run_log_verbosity_level = value;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsSetRunLogSeverityLevel, _In_ OrtRunOptions* options, int value) {
  options->run_log_severity_level = value;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsSetRunTag, _In_ OrtRunOptions* options, _In_ const char* run_tag) {
  if (run_tag)
    options->run_tag = run_tag;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, int* out) {
  *out = options->run_log_verbosity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, int* out) {
  *out = options->run_log_severity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsGetRunTag, _In_ const OrtRunOptions* options, const char** out) {
  *out = options->run_tag.c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsEnableTerminate, _Inout_ OrtRunOptions* options) {
  options->terminate = true;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtRunOptionsDisableTerminate, _Inout_ OrtRunOptions* options) {
  options->terminate = false;
  return nullptr;
}
