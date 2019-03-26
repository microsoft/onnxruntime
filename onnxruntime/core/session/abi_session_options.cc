// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include <cstring>
#include <cassert>
#include "core/session/inference_session.h"
#include "abi_session_options_impl.h"

OrtSessionOptions::~OrtSessionOptions() {
}

OrtSessionOptions& OrtSessionOptions::operator=(const OrtSessionOptions&) {
  throw std::runtime_error("not implemented");
}
OrtSessionOptions::OrtSessionOptions(const OrtSessionOptions& other)
    : value(other.value), custom_op_paths(other.custom_op_paths), provider_factories(other.provider_factories) {
}

ORT_API(OrtSessionOptions*, OrtCreateSessionOptions) {
  std::unique_ptr<OrtSessionOptions> options = std::make_unique<OrtSessionOptions>();
  return options.release();
}

ORT_API(void, OrtReleaseSessionOptions, OrtSessionOptions* ptr) {
  delete ptr;
}

ORT_API(OrtSessionOptions*, OrtCloneSessionOptions, OrtSessionOptions* input) {
  try {
    return new OrtSessionOptions(*input);
  } catch (std::exception&) {
    return nullptr;
  }
}

ORT_API(void, OrtEnableSequentialExecution, _In_ OrtSessionOptions* options) {
  options->value.enable_sequential_execution = true;
}
ORT_API(void, OrtDisableSequentialExecution, _In_ OrtSessionOptions* options) {
  options->value.enable_sequential_execution = false;
}

// enable profiling for this session.
ORT_API(void, OrtEnableProfiling, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix) {
  options->value.enable_profiling = true;
  options->value.profile_file_prefix = profile_file_prefix;
}
ORT_API(void, OrtDisableProfiling, _In_ OrtSessionOptions* options) {
  options->value.enable_profiling = false;
  options->value.profile_file_prefix.clear();
}

ORT_API(void, OrtEnableMemPattern, _In_ OrtSessionOptions*) {}
ORT_API(void, OrtDisableMemPattern, _In_ OrtSessionOptions*) {}

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ORT_API(void, OrtEnableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = true;
}

ORT_API(void, OrtDisableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = false;
}

///< logger id to use for session output
ORT_API(void, OrtSetSessionLogId, _In_ OrtSessionOptions* options, const char* logid) {
  options->value.session_logid = logid;
}

///< applies to session load, initialization, etc
ORT_API(void, OrtSetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, uint32_t session_log_verbosity_level) {
  options->value.session_log_verbosity_level = session_log_verbosity_level;
}

///How many threads in the session thread pool.
ORT_API(int, OrtSetSessionThreadPoolSize, _In_ OrtSessionOptions* options, int session_thread_pool_size) {
  if (session_thread_pool_size <= 0) return -1;
  options->value.session_thread_pool_size = session_thread_pool_size;
  return 0;
}

ORT_API(void, OrtAppendCustomOpLibPath, _In_ OrtSessionOptions* options, const char* lib_path) {
  options->custom_op_paths.emplace_back(lib_path);
}
