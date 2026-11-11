// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/common/profiler_common.h"

namespace onnxruntime::webgpu::ep {

class Profiler final : public OrtEpProfilerImpl {
 public:
  Profiler(std::unique_ptr<profiling::EpProfiler> impl, bool& profiling_active);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Profiler);

 private:
  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL StartProfilingImpl(
      OrtEpProfilerImpl* this_ptr, int64_t profiling_start_offset_ns) noexcept;
  static OrtStatus* ORT_API_CALL EndProfilingImpl(
      OrtEpProfilerImpl* this_ptr, OrtProfilingEventsContainer* events_container) noexcept;

  std::unique_ptr<profiling::EpProfiler> impl_;
  bool& profiling_active_;
  bool profiling_started_{false};
  TimePoint profiling_start_time_;
};

}  // namespace onnxruntime::webgpu::ep
