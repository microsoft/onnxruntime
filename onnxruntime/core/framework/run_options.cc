
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/run_options.h"
#include "core/framework/run_options_c_api.h"
#include <stdexcept>
#include <memory>

uint32_t ONNXRUNTIME_API_STATUSCALL ReleaseRunOptions(void* this_) {
  ONNXRuntimeRunOptions* this_ptr = static_cast<ONNXRuntimeRunOptions*>(this_);
  if (--this_ptr->ref_count == 0)
    delete this_ptr;
  return 0;
}

uint32_t ONNXRUNTIME_API_STATUSCALL AddRefRunOptions(void* this_) {
  ONNXRuntimeRunOptions* this_ptr = static_cast<ONNXRuntimeRunOptions*>(this_);
  ++this_ptr->ref_count;
  return 0;
}

constexpr ONNXObject mkl_cls = {
    AddRefRunOptions,
    ReleaseRunOptions,
};

ONNXRuntimeRunOptions::ONNXRuntimeRunOptions() : cls(&mkl_cls), ref_count(1) {
}

ONNXRuntimeRunOptions& ONNXRuntimeRunOptions::operator=(const ONNXRuntimeRunOptions&) {
  throw std::runtime_error("not implemented");
}
ONNXRuntimeRunOptions::ONNXRuntimeRunOptions(const ONNXRuntimeRunOptions& other)
    : cls(&mkl_cls), ref_count(1), run_log_verbosity_level(other.run_log_verbosity_level), run_tag(other.run_tag) {
}

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