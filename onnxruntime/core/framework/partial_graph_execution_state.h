//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ENABLE_TRAINING
#include "core/common/common.h"
#include "core/framework/ort_value.h"
#include "core/framework/iexecutor.h"
#include "core/common/inlined_containers.h"
#include "core/framework/program_region.h"

namespace onnxruntime {

class StreamExecutionContext;
class DeviceStreamCollection;

struct PartialGraphExecutionState {
 public:
  PartialGraphExecutionState() : execution_context_(nullptr), device_stream_collection_(nullptr) {
  }

  ~PartialGraphExecutionState();

  void SetProgramCounterStart(size_t start) { program_counter_start_ = start; }
  void SetProgramCounterEnd(size_t end) { program_counter_end_ = end; }

  size_t GetProgramCounterStart() { return program_counter_start_; }
  size_t GetProgramCounterEnd() { return program_counter_end_; }

  ProgramRegion& GetProgramRegions(const SessionState& session_state);

  StreamExecutionContext& GetExecutionContext(gsl::span<const int>& feed_mlvalue_idxs, gsl::span<const OrtValue>& feeds,
                                              gsl::span<const int>& fetch_mlvalue_idxs, std::vector<OrtValue>& fetches,
                                              const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                              const SessionState& session_state,
                                              const logging::Logger& sess_logger,
                                              const DeviceStreamCollection* device_streams);
  DeviceStreamCollection* GetDeviceStreamCollection(const SessionState& session_state);

 private:
  std::unique_ptr<StreamExecutionContext> execution_context_;
  size_t program_counter_start_{0};
  size_t program_counter_end_{0};

  std::vector<ProgramRegion> program_regions_;
  std::unique_ptr<DeviceStreamCollection> device_stream_collection_;
};
}  // namespace onnxruntime
#endif
