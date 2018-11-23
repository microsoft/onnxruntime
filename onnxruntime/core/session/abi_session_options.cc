// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include <cstring>
#include <cassert>
#include "core/session/inference_session.h"
#include "abi_session_options_impl.h"



ONNXRuntimeSessionOptions::~ONNXRuntimeSessionOptions() {
  assert(ref_count == 0);
  for (ONNXRuntimeProviderFactoryPtr* p : provider_factories) {
    ONNXRuntimeReleaseObject(p);
  }
}

ONNXRuntimeSessionOptions& ONNXRuntimeSessionOptions::operator=(const ONNXRuntimeSessionOptions&) {
  throw std::runtime_error("not implemented");
}
ONNXRuntimeSessionOptions::ONNXRuntimeSessionOptions(const ONNXRuntimeSessionOptions& other)
    : value(other.value), custom_op_paths(other.custom_op_paths), provider_factories(other.provider_factories) {
  for (ONNXRuntimeProviderFactoryPtr* p : other.provider_factories) {
    ONNXRuntimeAddRefToObject(p);
  }
}
ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCreateSessionOptions) {
  std::unique_ptr<ONNXRuntimeSessionOptions> options = std::make_unique<ONNXRuntimeSessionOptions>();
  return options.release();
}

ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCloneSessionOptions, ONNXRuntimeSessionOptions* input) {
  try {
    return new ONNXRuntimeSessionOptions(*input);
  } catch (std::exception&) {
    return nullptr;
  }
}

ONNXRUNTIME_API(void, ONNXRuntimeSessionOptionsAppendExecutionProvider, _In_ ONNXRuntimeSessionOptions* options, _In_ ONNXRuntimeProviderFactoryPtr* f) {
  ONNXRuntimeAddRefToObject(f);
  options->provider_factories.push_back(f);
}

ONNXRUNTIME_API(void, ONNXRuntimeEnableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options) {
  options->value.enable_sequential_execution = true;
}
ONNXRUNTIME_API(void, ONNXRuntimeDisableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options) {
  options->value.enable_sequential_execution = false;
}

// enable profiling for this session.
ONNXRUNTIME_API(void, ONNXRuntimeEnableProfiling, _In_ ONNXRuntimeSessionOptions* options, _In_ const char* profile_file_prefix) {
  options->value.enable_profiling = true;
  options->value.profile_file_prefix = profile_file_prefix;
}
ONNXRUNTIME_API(void, ONNXRuntimeDisableProfiling, _In_ ONNXRuntimeSessionOptions* options) {
  options->value.enable_profiling = false;
  options->value.profile_file_prefix.clear();
}

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ONNXRUNTIME_API(void, ONNXRuntimeEnableMemPattern, _In_ ONNXRuntimeSessionOptions* options) {
  options->value.enable_mem_pattern = true;
}
ONNXRUNTIME_API(void, ONNXRuntimeDisableMemPattern, _In_ ONNXRuntimeSessionOptions* options) {
  options->value.enable_mem_pattern = false;
}

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ONNXRUNTIME_API(void, ONNXRuntimeEnableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options) {
  options->value.enable_cpu_mem_arena = true;
}

ONNXRUNTIME_API(void, ONNXRuntimeDisableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options) {
  options->value.enable_cpu_mem_arena = false;
}

///< logger id to use for session output
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogId, _In_ ONNXRuntimeSessionOptions* options, const char* logid) {
  options->value.session_logid = logid;
}

///< applies to session load, initialization, etc
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogVerbosityLevel, _In_ ONNXRuntimeSessionOptions* options, uint32_t session_log_verbosity_level) {
  options->value.session_log_verbosity_level = session_log_verbosity_level;
}

///How many threads in the session thread pool.
ONNXRUNTIME_API(int, ONNXRuntimeSetSessionThreadPoolSize, _In_ ONNXRuntimeSessionOptions* options, int session_thread_pool_size) {
  if (session_thread_pool_size <= 0) return -1;
  options->value.session_thread_pool_size = session_thread_pool_size;
  return 0;
}


ONNXRUNTIME_API(void, ONNXRuntimeAddCustomOp, _In_ ONNXRuntimeSessionOptions* options, const char* custom_op_path) {
  options->custom_op_paths.emplace_back(custom_op_path);
}