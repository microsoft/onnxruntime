
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/run_options.h"

#include <mutex>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif

class OrtRunOptions::TerminationState {
 public:
  explicit TerminationState(bool stop_requested = false) {
    if (stop_requested) {
      stop_source_.request_stop();
    }
  }

  std::stop_token GetToken() const {
    std::lock_guard lock(mutex_);
    return stop_source_.get_token();
  }

  void RequestStop() {
    std::stop_source stop_source;
    {
      std::lock_guard lock(mutex_);
      stop_source = stop_source_;
    }

    stop_source.request_stop();
  }

  void Reset() {
    std::lock_guard lock(mutex_);
    stop_source_ = std::stop_source{};
  }

 private:
  mutable std::mutex mutex_;
  std::stop_source stop_source_;
};

OrtRunOptions::OrtRunOptions() : termination_state_{std::make_shared<TerminationState>()} {
}

OrtRunOptions::OrtRunOptions(const OrtRunOptions& other)
    : run_log_severity_level{other.run_log_severity_level},
      run_log_verbosity_level{other.run_log_verbosity_level},
      run_tag{other.run_tag},
      only_execute_path_to_fetches{other.only_execute_path_to_fetches},
      enable_profiling{other.enable_profiling},
      profile_file_prefix{other.profile_file_prefix},
#ifdef ENABLE_TRAINING
      training_mode{other.training_mode},
#endif
      config_options{other.config_options},
      active_adapters{other.active_adapters},
      sync_stream{other.sync_stream},
      termination_state_{
          std::make_shared<TerminationState>(other.GetTerminateToken().stop_requested())} {
}

OrtRunOptions& OrtRunOptions::operator=(const OrtRunOptions& other) {
  if (this != &other) {
    OrtRunOptions copy{other};
    *this = std::move(copy);
  }

  return *this;
}

OrtRunOptions::~OrtRunOptions() = default;

std::stop_token OrtRunOptions::GetTerminateToken() const {
  return termination_state_->GetToken();
}

void OrtRunOptions::RequestTerminate() {
  termination_state_->RequestStop();
}

void OrtRunOptions::ResetTerminate() {
  termination_state_->Reset();
}

ORT_API_STATUS_IMPL(OrtApis::CreateRunOptions, _Outptr_ OrtRunOptions** out) {
  API_IMPL_BEGIN
  *out = new OrtRunOptions();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value) {
  options->run_log_verbosity_level = value;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value) {
  options->run_log_severity_level = value;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetRunTag, _Inout_ OrtRunOptions* options, _In_ const char* run_tag) {
  if (run_tag)
    options->run_tag = run_tag;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out) {
  *out = options->run_log_verbosity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out) {
  *out = options->run_log_severity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsGetRunTag, _In_ const OrtRunOptions* options, _Out_ const char** out) {
  *out = options->run_tag.c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetTerminate, _Inout_ OrtRunOptions* options) {
  API_IMPL_BEGIN
  options->RequestTerminate();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options) {
  API_IMPL_BEGIN
  options->ResetTerminate();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::RunOptionsSetSyncStream, _Inout_ OrtRunOptions* options, _In_ OrtSyncStream* sync_stream) {
  options->sync_stream = sync_stream;
}

ORT_API_STATUS_IMPL(OrtApis::AddRunConfigEntry, _Inout_ OrtRunOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value) {
  return onnxruntime::ToOrtStatus(options->config_options.AddConfigEntry(config_key, config_value));
}

ORT_API(const char*, OrtApis::GetRunConfigEntry, _In_ const OrtRunOptions* options, _In_z_ const char* config_key) {
  const auto& config_options = options->config_options.GetConfigOptionsMap();
  if (auto it = config_options.find(config_key); it != config_options.end()) {
    return it->second.c_str();
  } else {
    return nullptr;
  }
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsAddActiveLoraAdapter, _Inout_ OrtRunOptions* options,
                    const _In_ OrtLoraAdapter* adapter) {
  API_IMPL_BEGIN
  auto* lora_adapter = reinterpret_cast<const onnxruntime::lora::LoraAdapter*>(adapter);
  options->active_adapters.push_back(lora_adapter);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsEnableProfiling, _Inout_ OrtRunOptions* options,
                    _In_ const ORTCHAR_T* profile_file_prefix) {
  options->enable_profiling = true;
  options->profile_file_prefix = profile_file_prefix;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsDisableProfiling, _Inout_ OrtRunOptions* options) {
  options->enable_profiling = false;
  options->profile_file_prefix.clear();
  return nullptr;
}
