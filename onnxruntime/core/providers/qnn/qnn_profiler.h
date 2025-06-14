// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/common/profiler_common.h"

namespace onnxruntime::qnn {

class ProfilingEventStore;

/**
 * The QNN EP `profiling::EpProfiler` implementation.
 * It interacts with the QNN EP via its reference to a shared `ProfilingEventStore`.
 */
class QnnProfiler : public profiling::EpProfiler {
 public:
  QnnProfiler(std::shared_ptr<ProfilingEventStore> event_store);

  bool StartProfiling(TimePoint start_time) override;
  void EndProfiling(TimePoint start_time, profiling::Events& events) override;

 private:
  std::shared_ptr<ProfilingEventStore> event_store_;
};

}  // namespace onnxruntime::qnn
