// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "DlSystem/DlEnums.hpp"
#include "core/providers/snpe/snpe_runtime_wrapper.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

enum class BufferType : int {
  UNKNOWN = -1,
  ITENSOR,
  TF8,
  TF16,
  UINT8,
  FLOAT
};

class SnpeRuntimeOptions {
 public:
  SnpeRuntimeOptions() : runtime_target_(),
                         execution_priority_(zdl::DlSystem::ExecutionPriorityHint_t::NORMAL),
                         runtime_options_(),
                         buffer_type_(BufferType::ITENSOR) {
  }

  explicit SnpeRuntimeOptions(const std::unordered_map<std::string, std::string>& options)
      : runtime_target_(),
        execution_priority_(zdl::DlSystem::ExecutionPriorityHint_t::NORMAL),
        runtime_options_(options),
        buffer_type_(BufferType::ITENSOR) {
    ParseOptions();
  }

  const SnpeRuntimeWrapper& GetRuntimeTarget() const {
    return runtime_target_;
  }

  zdl::DlSystem::ExecutionPriorityHint_t GetExecutionPriority() const {
    return execution_priority_;
  }

  BufferType GetBufferType() const {
    return buffer_type_;
  }

  bool GetInitCacheMode() const {
    return enable_init_cache_;
  }

 private:
  void ParseOptions();

 private:
  SnpeRuntimeWrapper runtime_target_;
  zdl::DlSystem::ExecutionPriorityHint_t execution_priority_;
  std::unordered_map<std::string, std::string> runtime_options_;
  BufferType buffer_type_;
  bool enable_init_cache_ = false;
};

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
