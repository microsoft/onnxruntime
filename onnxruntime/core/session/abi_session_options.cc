// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/framework/error_code_helper.h"
#include <cstring>
#include <cassert>
#include "core/session/inference_session.h"
#include "abi_session_options_impl.h"

OrtSessionOptions::~OrtSessionOptions() = default;

OrtSessionOptions& OrtSessionOptions::operator=(const OrtSessionOptions&) {
  throw std::runtime_error("not implemented");
}
OrtSessionOptions::OrtSessionOptions(const OrtSessionOptions& other)
    : value(other.value), provider_factories(other.provider_factories) {
}

ORT_API_STATUS_IMPL(OrtCreateSessionOptions, OrtSessionOptions** out) {
  API_IMPL_BEGIN
  *out = new OrtSessionOptions();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtReleaseSessionOptions, OrtSessionOptions* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtCloneSessionOptions, const OrtSessionOptions* input, OrtSessionOptions** out) {
  API_IMPL_BEGIN
  *out = new OrtSessionOptions(*input);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtEnableSequentialExecution, _In_ OrtSessionOptions* options) {
  options->value.enable_sequential_execution = true;
  return nullptr;
}
ORT_API_STATUS_IMPL(OrtDisableSequentialExecution, _In_ OrtSessionOptions* options) {
  options->value.enable_sequential_execution = false;
  return nullptr;
}

// enable profiling for this session.
ORT_API_STATUS_IMPL(OrtEnableProfiling, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix) {
  options->value.enable_profiling = true;
  options->value.profile_file_prefix = profile_file_prefix;
  return nullptr;
}
ORT_API_STATUS_IMPL(OrtDisableProfiling, _In_ OrtSessionOptions* options) {
  options->value.enable_profiling = false;
  options->value.profile_file_prefix.clear();
  return nullptr;
}

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ORT_API_STATUS_IMPL(OrtEnableMemPattern, _In_ OrtSessionOptions* options) {
  options->value.enable_mem_pattern = true;
  return nullptr;
}
ORT_API_STATUS_IMPL(OrtDisableMemPattern, _In_ OrtSessionOptions* options) {
  options->value.enable_mem_pattern = false;
  return nullptr;
}

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ORT_API_STATUS_IMPL(OrtEnableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = true;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtDisableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = false;
  return nullptr;
}

///< logger id to use for session output
ORT_API_STATUS_IMPL(OrtSetSessionLogId, _In_ OrtSessionOptions* options, const char* logid) {
  options->value.session_logid = logid;
  return nullptr;
}

///< applies to session load, initialization, etc
ORT_API_STATUS_IMPL(OrtSetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, int session_log_verbosity_level) {
  options->value.session_log_verbosity_level = session_log_verbosity_level;
  return nullptr;
}

// Set Graph optimization level.
// Available options are : 0, 1, 2.
ORT_API_STATUS_IMPL(OrtSetSessionGraphOptimizationLevel, _In_ OrtSessionOptions* options, int graph_optimization_level) {
  if (graph_optimization_level < 0) {
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, "graph_optimization_level is not valid");
  }
  if (graph_optimization_level >= static_cast<int>(onnxruntime::TransformerLevel::MaxTransformerLevel))
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, "graph_optimization_level is not valid");
  options->value.graph_optimization_level = static_cast<onnxruntime::TransformerLevel>(graph_optimization_level);
  return nullptr;
}

///How many threads in the session thread pool.
ORT_API_STATUS_IMPL(OrtSetSessionThreadPoolSize, _In_ OrtSessionOptions* options, int session_thread_pool_size) {
  if (session_thread_pool_size <= 0)
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, "session_thread_pool_size is not valid");
  options->value.session_thread_pool_size = session_thread_pool_size;
  return nullptr;
}
