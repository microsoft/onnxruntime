// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/onnx_attention_cpu_impl.h"

#include <string>

#include "core/platform/env_var.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace {

constexpr const char* kCpuAttentionImplEnvVar = "ORT_ONNX_ATTENTION_CPU_IMPL";

Status ParseCpuAttentionImpl(std::string_view value, CpuAttentionImpl& impl) {
  if (value == "unfused") {
    impl = CpuAttentionImpl::kUnfused;
    return Status::OK();
  }
  if (value == "flash_specialized") {
    impl = CpuAttentionImpl::kFlashSpecialized;
    return Status::OK();
  }
  if (value == "flash_flex") {
    impl = CpuAttentionImpl::kFlashFlex;
    return Status::OK();
  }
  if (value == "auto") {
    impl = CpuAttentionImpl::kAuto;
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Invalid CPU ONNX Attention implementation '", value,
                         "'. Valid values are: unfused, flash_specialized, flash_flex, auto.");
}

Status ParseStrictFlag(std::string_view value, bool& strict) {
  if (value == "0") {
    strict = false;
    return Status::OK();
  }
  if (value == "1") {
    strict = true;
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Invalid CPU ONNX Attention strict flag '", value,
                         "'. Valid values are: 0, 1.");
}

}  // namespace

Status ResolveCpuAttentionSelection(const ConfigOptions& config_options,
                                    CpuAttentionSelection& selection) {
  std::string impl_value =
      config_options.GetConfigOrDefault(kOrtSessionOptionsOnnxAttentionCpuImpl, "auto");
  const std::string env_value = detail::GetEnvironmentVar(kCpuAttentionImplEnvVar);
  if (!env_value.empty()) {
    impl_value = env_value;
  }

  ORT_RETURN_IF_ERROR(ParseCpuAttentionImpl(impl_value, selection.impl));
  ORT_RETURN_IF_ERROR(ParseStrictFlag(
      config_options.GetConfigOrDefault(kOrtSessionOptionsOnnxAttentionCpuImplStrict, "0"),
      selection.strict));

  return Status::OK();
}

std::string_view CpuAttentionImplToString(CpuAttentionImpl impl) {
  switch (impl) {
    case CpuAttentionImpl::kUnfused:
      return "unfused";
    case CpuAttentionImpl::kFlashSpecialized:
      return "flash_specialized";
    case CpuAttentionImpl::kFlashFlex:
      return "flash_flex";
    case CpuAttentionImpl::kAuto:
      return "auto";
  }

  ORT_THROW("Unknown CPU ONNX Attention implementation.");
}

}  // namespace onnxruntime
