//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ENABLE_TRAINING
#include "core/common/common.h"
#include "core/framework/ort_value.h"
#include "core/framework/execution_frame.h"
#include "core/common/inlined_containers.h"
#include "core/framework/program_region.h"

namespace onnxruntime {

struct PartialGraphExecutionState {
 public:
  PartialGraphExecutionState() : execution_frame_(nullptr) {
  }

  ~PartialGraphExecutionState() = default;

  void SetProgramCounterStart(size_t start) { program_counter_start_ = start; }
  void SetProgramCounterEnd(size_t end) { program_counter_end_ = end; }

  size_t GetProgramCounterStart() { return program_counter_start_; }
  size_t GetProgramCounterEnd() { return program_counter_end_; }

  ProgramRegion& GetProgramRegions(const SessionState& session_state);

  std::shared_ptr<ExecutionFrame> GetExecutionFrame(gsl::span<const int> feed_mlvalue_idxs,
                                                    gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                                    gsl::span<const OrtValue> fetches,
                                                    const InlinedHashMap<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                                    const SessionState& session_state,
                                                    const std::vector<Stream*>* device_streams);

 private:
  // Temporary use shared_ptr to make it transfer between mutliple execution context
  // TODO: use a better way to transfer ownership.
  std::shared_ptr<ExecutionFrame> execution_frame_;
  size_t program_counter_start_{0};
  size_t program_counter_end_{0};

  std::vector<ProgramRegion> program_regions_;
};
}  // namespace onnxruntime
#endif
