// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_runtime_options.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

void SnpeRuntimeOptions::ParseOptions() {
  static const std::string OPT_RUNTIME = "runtime";
  static const std::string OPT_PRIORITY = "priority";
  static const std::string BUFFER_TYPE = "buffer_type";

  // Option - Runtime
  if (runtime_options_.find(OPT_RUNTIME) != runtime_options_.end()) {
    runtime_target_ = SnpeRuntimeWrapper(runtime_options_[OPT_RUNTIME]);
    LOGS_DEFAULT(INFO) << "Located user specified runtime target: " << runtime_options_[OPT_RUNTIME];
  }
  LOGS_DEFAULT(INFO) << "Runtime target: " << runtime_target_.ToString();

  // Option Priority
  if (runtime_options_.find(OPT_PRIORITY) != runtime_options_.end()) {
    if (runtime_options_[OPT_PRIORITY] == "low") {
      execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::LOW;
    } else if (runtime_options_[OPT_PRIORITY] == "normal") {
      execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::NORMAL;
    } else {
      LOGS_DEFAULT(INFO) << "Invalid execution priority, defaulting to LOW";
      execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::LOW;
    }

    LOGS_DEFAULT(INFO) << "Located user specified execution priority " << runtime_options_[OPT_PRIORITY];
  }

  // buffer type
  if (runtime_options_.find(BUFFER_TYPE) != runtime_options_.end()) {
    if (runtime_options_[BUFFER_TYPE] == "TF8") {
      buffer_type_ = BufferType::TF8;
    } else if (runtime_options_[BUFFER_TYPE] == "TF16") {
      buffer_type_ = BufferType::TF16;
    } else if (runtime_options_[BUFFER_TYPE] == "ITENSOR") {
      buffer_type_ = BufferType::ITENSOR;
    } else if (runtime_options_[BUFFER_TYPE] == "UINT8") {
      buffer_type_ = BufferType::UINT8;
    } else if (runtime_options_[BUFFER_TYPE] == "FLOAT") {
      buffer_type_ = BufferType::FLOAT;
    } else {
      LOGS_DEFAULT(ERROR) << "Invalid buffer type: " << runtime_options_[BUFFER_TYPE];
      buffer_type_ = BufferType::UNKNOWN;
    }
  }
}

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
