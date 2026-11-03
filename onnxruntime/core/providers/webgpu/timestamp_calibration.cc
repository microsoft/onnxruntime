// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/timestamp_calibration.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#endif

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

#if defined(_WIN32)
std::optional<TimestampCalibration> FromQpcSample(const OrtWebGpuClockCalibration& sample,
                                                  TimePoint profiling_start_time) {
  if (!std::isfinite(sample.gpu_to_cpu_scale) || sample.gpu_to_cpu_scale <= 0 ||
      sample.cpu_qpc_ticks > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  LARGE_INTEGER qpc_frequency{};
  if (!QueryPerformanceFrequency(&qpc_frequency) || qpc_frequency.QuadPart <= 0) {
    return std::nullopt;
  }
  // Bridge QPC to TimePoint without assuming their epochs match.
  const auto before = std::chrono::high_resolution_clock::now();
  LARGE_INTEGER qpc_now{};
  if (!QueryPerformanceCounter(&qpc_now) || qpc_now.QuadPart < 0) {
    return std::nullopt;
  }
  const auto after = std::chrono::high_resolution_clock::now();
  const auto midpoint = before + (after - before) / 2;
  const auto cpu_delta_ticks = static_cast<int64_t>(sample.cpu_qpc_ticks) - qpc_now.QuadPart;
  const long double cpu_timestamp_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(midpoint - profiling_start_time).count() +
      std::round(static_cast<long double>(cpu_delta_ticks) * 1e9L / qpc_frequency.QuadPart);
  if (cpu_timestamp_ns <= static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
      cpu_timestamp_ns >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  return TimestampCalibration{sample.gpu_timestamp_ns, static_cast<int64_t>(cpu_timestamp_ns), sample.gpu_to_cpu_scale};
}
#endif

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

std::optional<TimestampCalibration> CalibrateHostTimestamps(
    const wgpu::Device& device, wgpu::BackendType backend_type, TimePoint profiling_start_time,
    const OrtWebGpuClockCalibrationCallbackInfo& callback_info) {
#if defined(_WIN32)
  if (!device || backend_type != wgpu::BackendType::D3D12 ||
      callback_info.version != 1 || callback_info.callback == nullptr) {
    return std::nullopt;
  }
  ORT_TRY {
    OrtWebGpuClockCalibration sample{};
    if (!callback_info.callback(device.Get(), &sample, callback_info.user_data)) {
      return std::nullopt;
    }
    return FromQpcSample(sample, profiling_start_time);
  }
  ORT_CATCH(...) {
    return std::nullopt;
  }
#else
  ORT_UNUSED_PARAMETER(device);
  ORT_UNUSED_PARAMETER(backend_type);
  ORT_UNUSED_PARAMETER(profiling_start_time);
  ORT_UNUSED_PARAMETER(callback_info);
  return std::nullopt;
#endif
}

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
  if (!queue || FAILED(queue->GetTimestampFrequency(&frequency)) ||
      frequency < kMinTimestampFrequency || frequency > kMaxTimestampFrequency ||
      FAILED(queue->GetClockCalibration(&gpu_ticks, &cpu_ticks))) {
    return std::nullopt;
  }
  // Correct the rounded query clock's rate as well as its epoch, avoiding drift during long runs.
  const auto [multiplier, shift] = GetD3D12TimestampPeriod(frequency);
  const double gpu_to_cpu_scale = 1e9 * static_cast<double>(uint64_t{1} << shift) /
                                  (static_cast<double>(frequency) * multiplier);
  return FromQpcSample({detail::NormalizeD3D12Timestamp(gpu_ticks, frequency), cpu_ticks, gpu_to_cpu_scale},
                       profiling_start_time);
#else
  ORT_UNUSED_PARAMETER(device);
  ORT_UNUSED_PARAMETER(backend_type);
  ORT_UNUSED_PARAMETER(profiling_start_time);
  return std::nullopt;
#endif
}

}  // namespace onnxruntime::webgpu
