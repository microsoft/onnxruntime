// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "core/common/common.h"
#include "core/framework/kernel_pilot_moe_deferred_record.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {

// Provider-facing view of the per-Run MoE logging state.
class IKernelPilotMoeLoggingContext {
 public:
  virtual ~IKernelPilotMoeLoggingContext() = default;

  virtual const std::string& RequestId() const noexcept = 0;
  virtual TimePoint StartProfiling() const = 0;
  virtual uint64_t ProfilerStartTimeNs() const noexcept = 0;
  virtual void RecordMoeRoutingEvent(const TimePoint& start_time,
                                     const TimePoint& end_time,
                                     std::string_view node_name,
                                     NodeIndex node_index,
                                     std::string_view node_type,
                                     std::string expert_ids_json,
                                     std::string router_weights_json,
                                     int64_t num_rows,
                                     int64_t top_k,
                                     int execution_device_id,
                                     int64_t completion_ns,
                                     std::string_view completion_timestamp_source) const = 0;
  virtual void AddDeferredRecord(std::unique_ptr<KernelPilotMoeDeferredRecord> record) const = 0;
  virtual bool TryReserveMoeRoutingRecord(size_t element_count) const = 0;
};

}  // namespace onnxruntime
