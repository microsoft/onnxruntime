// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_runtime_options.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

constexpr const char* OPT_RUNTIME = "runtime";
constexpr const char* OPT_PRIORITY = "priority";
constexpr const char* BUFFER_TYPE = "buffer_type";
constexpr const char* ENABLE_INIT_CACHE = "enable_init_cache";

void SnpeRuntimeOptions::ParseOptions() {
  if (const auto runtime_opt_it = runtime_options_.find(OPT_RUNTIME); runtime_opt_it != runtime_options_.end()) {
    runtime_target_ = SnpeRuntimeWrapper(runtime_opt_it->second);
    LOGS_DEFAULT(VERBOSE) << "Located user specified runtime target: " << runtime_opt_it->second;
  }
  LOGS_DEFAULT(VERBOSE) << "Runtime target: " << runtime_target_.ToString();

  // Option Priority
  if (const auto priority_opt_it = runtime_options_.find(OPT_PRIORITY); priority_opt_it != runtime_options_.end()) {
    if (priority_opt_it->second == "low") {
      execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::LOW;
    } else if (priority_opt_it->second == "normal") {
      execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::NORMAL;
    } else {
      LOGS_DEFAULT(INFO) << "Invalid execution priority, defaulting to LOW";
      execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::LOW;
    }

    LOGS_DEFAULT(VERBOSE) << "Located user specified execution priority " << priority_opt_it->second;
  }

  // buffer type
  if (const auto buffer_type_it = runtime_options_.find(BUFFER_TYPE); buffer_type_it != runtime_options_.end()) {
    if (buffer_type_it->second == "TF8") {
      buffer_type_ = BufferType::TF8;
    } else if (buffer_type_it->second == "TF16") {
      buffer_type_ = BufferType::TF16;
    } else if (buffer_type_it->second == "ITENSOR") {
      buffer_type_ = BufferType::ITENSOR;
    } else if (buffer_type_it->second == "UINT8") {
      buffer_type_ = BufferType::UINT8;
    } else if (buffer_type_it->second == "FLOAT") {
      buffer_type_ = BufferType::FLOAT;
    } else {
      LOGS_DEFAULT(ERROR) << "Invalid buffer type: " << buffer_type_it->second;
      buffer_type_ = BufferType::UNKNOWN;
    }
    LOGS_DEFAULT(VERBOSE) << "Buffer type: " << buffer_type_it->second;
  }

  if (const auto enable_init_cache_pos = runtime_options_.find(ENABLE_INIT_CACHE); enable_init_cache_pos != runtime_options_.end()) {
    if (enable_init_cache_pos->second == "1") {
      enable_init_cache_ = true;
      LOGS_DEFAULT(VERBOSE) << "enable_init_cache enabled.";
    }
  }
}

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
