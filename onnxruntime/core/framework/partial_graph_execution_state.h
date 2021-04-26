//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ENABLE_TRAINING
#include "core/common/common.h"
#include "core/framework/execution_frame.h"

namespace onnxruntime {
struct PartialGraphExecutionState {
 public:
  PartialGraphExecutionState() {
    execution_frame_ = nullptr;
  }

  ~PartialGraphExecutionState() = default;

  void SetProgramCounterStart(size_t start) { program_counter_start_ = start; }
  void SetProgramCounterEnd(size_t end) { program_counter_end_ = end; }

  size_t GetProgramCounterStart() { return program_counter_start_; }
  size_t GetProgramCounterEnd() { return program_counter_end_; }

  void SetExecutionFrame(std::unique_ptr<ExecutionFrame> frame) {
    execution_frame_ = std::move(frame);
  }

  const std::unique_ptr<ExecutionFrame>& GetExecutionFrame() const { return execution_frame_; }

 private:
  std::unique_ptr<ExecutionFrame> execution_frame_;
  size_t program_counter_start_;
  size_t program_counter_end_;
};
}  // namespace onnxruntime
#endif
