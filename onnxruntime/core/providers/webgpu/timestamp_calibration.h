// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>

#include "core/common/common.h"
#include "core/providers/webgpu/webgpu_external_header.h"

namespace onnxruntime::webgpu {

// Correlated timestamps, expressed in Dawn's query units and relative to the ORT profiler's origin.
struct TimestampCalibration {
  uint64_t gpu_timestamp_ns;
  int64_t cpu_timestamp_ns;
  double gpu_to_cpu_scale = 1.0;

  int64_t ToCpuMicroseconds(uint64_t timestamp_ns) const;
  int64_t DurationMicroseconds(uint64_t duration_ns) const;
};

// Unsupported backends/builds and failed native calls return no calibration.
std::optional<TimestampCalibration> CalibrateTimestamps(const wgpu::Device& device,
                                                        wgpu::BackendType backend_type,
                                                        TimePoint profiling_start_time);

namespace detail {

// Match Dawn's D3D12 query normalization, including its fixed-point timestamp period.
uint64_t NormalizeD3D12Timestamp(uint64_t timestamp, uint64_t frequency);

}  // namespace detail
}  // namespace onnxruntime::webgpu
