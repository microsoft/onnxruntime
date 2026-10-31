// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/timestamp_calibration.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string_view>

#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12) && !defined(USE_EXTERNAL_DAWN)
#include "dawn/native/D3D12Backend.h"
#include "dawn/native/DawnNative.h"
#endif

namespace onnxruntime::webgpu {
namespace {

constexpr uint64_t kMaxTimestampFrequency = 1000000000;
// Keep the timestamp period between 1 and 2^16 nanoseconds, as required by Dawn's conversion.
constexpr uint64_t kMinTimestampFrequency = (kMaxTimestampFrequency + 65535) / 65536;

struct D3D12TimestampPeriod {
  uint32_t multiplier;
  uint32_t shift;
};

D3D12TimestampPeriod GetD3D12TimestampPeriod(uint64_t frequency) {
  ORT_ENFORCE(frequency >= kMinTimestampFrequency && frequency <= kMaxTimestampFrequency);
  // Dawn's DeviceD3D12 and QueryHelper::TimestampParams use a float period and a
  // truncated fixed-point multiplier. Using the exact frequency here instead can
  // shift the absolute timestamp by seconds on a long-running device clock.
  const float period = static_cast<float>(1e9 / static_cast<double>(frequency));
  const uint32_t upper_log2 = static_cast<uint32_t>(std::ceil(std::log2(period)));
  const uint32_t shift = 16u - std::min(upper_log2, 16u);
  return {static_cast<uint32_t>(period * static_cast<float>(1u << shift)), shift};
}

}  // namespace

int64_t TimestampCalibration::ToCpuMicroseconds(uint64_t timestamp_ns) const {
  // Subtract before converting to double to retain precision for large device clock values.
  const double delta_ns = timestamp_ns >= gpu_timestamp_ns
                              ? static_cast<double>(timestamp_ns - gpu_timestamp_ns)
                              : -static_cast<double>(gpu_timestamp_ns - timestamp_ns);
  return static_cast<int64_t>(std::round((static_cast<double>(cpu_timestamp_ns) + delta_ns * gpu_to_cpu_scale) / 1000.0));
}

int64_t TimestampCalibration::DurationMicroseconds(uint64_t duration_ns) const {
  return static_cast<int64_t>(std::round(static_cast<double>(duration_ns) * gpu_to_cpu_scale / 1000.0));
}

namespace detail {

uint64_t NormalizeD3D12Timestamp(uint64_t timestamp, uint64_t frequency) {
  const auto [multiplier, shift] = GetD3D12TimestampPeriod(frequency);
  // Split the multiplication so the intermediate product does not overflow.
  return (timestamp >> shift) * multiplier +
         (((timestamp & ((uint64_t{1} << shift) - 1)) * multiplier) >> shift);
}

}  // namespace detail

std::optional<TimestampCalibration> CalibrateTimestamps(const wgpu::Device& device,
                                                        wgpu::BackendType backend_type,
                                                        TimePoint profiling_start_time) {
#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12) && !defined(USE_EXTERNAL_DAWN)
  if (!device || backend_type != wgpu::BackendType::D3D12) {
    return std::nullopt;
  }

  // These toggles change the query clock domain. Do not mix it with the calibrated domain.
  const auto toggles = dawn::native::GetTogglesUsed(device.Get());
  if (std::find(toggles.begin(), toggles.end(), std::string_view{"disable_timestamp_query_conversion"}) != toggles.end() ||
      std::find(toggles.begin(), toggles.end(), std::string_view{"timestamp_quantization"}) != toggles.end()) {
    return std::nullopt;
  }

  const auto queue = dawn::native::d3d12::GetD3D12CommandQueue(device.Get());
  UINT64 frequency = 0;
  UINT64 gpu_ticks = 0;
  UINT64 cpu_ticks = 0;
  LARGE_INTEGER qpc_frequency{};
  if (!queue || FAILED(queue->GetTimestampFrequency(&frequency)) ||
      frequency < kMinTimestampFrequency || frequency > kMaxTimestampFrequency ||
      FAILED(queue->GetClockCalibration(&gpu_ticks, &cpu_ticks)) ||
      !QueryPerformanceFrequency(&qpc_frequency) || qpc_frequency.QuadPart <= 0) {
    return std::nullopt;
  }

  // Bridge QPC to TimePoint without assuming their epochs match (including on non-MSVC toolchains).
  const auto before = std::chrono::high_resolution_clock::now();
  LARGE_INTEGER qpc_now{};
  if (!QueryPerformanceCounter(&qpc_now)) {
    return std::nullopt;
  }
  const auto after = std::chrono::high_resolution_clock::now();
  const auto midpoint = before + (after - before) / 2;
  const double cpu_delta_ticks = cpu_ticks >= static_cast<uint64_t>(qpc_now.QuadPart)
                                     ? static_cast<double>(cpu_ticks - static_cast<uint64_t>(qpc_now.QuadPart))
                                     : -static_cast<double>(static_cast<uint64_t>(qpc_now.QuadPart) - cpu_ticks);
  const auto cpu_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(midpoint - profiling_start_time).count() +
                                static_cast<int64_t>(std::round(cpu_delta_ticks * 1e9 / static_cast<double>(qpc_frequency.QuadPart)));
  // Correct the rounded query clock's rate as well as its epoch, avoiding drift during long runs.
  const auto [multiplier, shift] = GetD3D12TimestampPeriod(frequency);
  const double gpu_to_cpu_scale = 1e9 * static_cast<double>(uint64_t{1} << shift) /
                                  (static_cast<double>(frequency) * multiplier);
  return TimestampCalibration{detail::NormalizeD3D12Timestamp(gpu_ticks, frequency), cpu_timestamp_ns, gpu_to_cpu_scale};
#else
  ORT_UNUSED_PARAMETER(device);
  ORT_UNUSED_PARAMETER(backend_type);
  ORT_UNUSED_PARAMETER(profiling_start_time);
  return std::nullopt;
#endif
}

}  // namespace onnxruntime::webgpu
