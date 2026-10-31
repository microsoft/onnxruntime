// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdint>

#include "gtest/gtest.h"
#include "core/providers/webgpu/timestamp_calibration.h"

namespace onnxruntime::test {

TEST(WebGpuTimestampCalibrationTest, PreservesQueueingDelayAndProfilingOrigin) {
  // The device clock has a different epoch. The first dispatch starts 5 ms after calibration.
  const webgpu::TimestampCalibration calibration{9000000000000000000ULL, 20000000};
  EXPECT_EQ(calibration.ToCpuMicroseconds(9000000000005000000ULL), 25000);
  EXPECT_EQ(calibration.ToCpuMicroseconds(9000000000010000000ULL), 30000);

  // A later run is calibrated again against the same session origin.
  const webgpu::TimestampCalibration next_run{9000000001000000000ULL, 1020000000};
  EXPECT_EQ(next_run.ToCpuMicroseconds(9000000001005000000ULL), 1025000);
}

TEST(WebGpuTimestampCalibrationTest, HandlesTimestampsBeforeCalibration) {
  const webgpu::TimestampCalibration calibration{10000000000000000ULL, 2000000};
  EXPECT_EQ(calibration.ToCpuMicroseconds(9999999999000000ULL), 1000);
  EXPECT_EQ(calibration.ToCpuMicroseconds(9999999997000000ULL), -1000);
  EXPECT_EQ(calibration.ToCpuMicroseconds(10000000000000499ULL), 2000);
  EXPECT_EQ(calibration.ToCpuMicroseconds(10000000000000500ULL), 2001);
}

TEST(WebGpuTimestampCalibrationTest, MatchesDawnFixedPointNormalization) {
  EXPECT_EQ(webgpu::detail::NormalizeD3D12Timestamp(123456789, 10000000), 12345678900ULL);
  // 27 MHz uses multiplier 37925 and shift 10, rather than the exact 1000 / 27 period.
  // At this clock value, the difference exceeds a second.
  EXPECT_EQ(webgpu::detail::NormalizeD3D12Timestamp(100000000000000ULL, 27000000),
            3703613281250000ULL);
  EXPECT_EQ(webgpu::detail::NormalizeD3D12Timestamp(123456789, 27000000), 4572362034ULL);
  // The full timestamp * multiplier would overflow uint64_t.
  EXPECT_EQ(webgpu::detail::NormalizeD3D12Timestamp(1000000000000000ULL, 10000000),
            100000000000000000ULL);
}

TEST(WebGpuTimestampCalibrationTest, UnsupportedBackendHasNoCalibration) {
  EXPECT_FALSE(webgpu::CalibrateTimestamps({}, wgpu::BackendType::Vulkan,
                                           std::chrono::high_resolution_clock::now()));
}

TEST(WebGpuTimestampCalibrationTest, CompensatesForRoundedQueryClockRate) {
  constexpr uint64_t frequency = 27000000;
  constexpr uint64_t gpu_ticks = 100000000000000;
  const auto start = webgpu::detail::NormalizeD3D12Timestamp(gpu_ticks, frequency);
  const auto end = webgpu::detail::NormalizeD3D12Timestamp(gpu_ticks + frequency, frequency);
  const webgpu::TimestampCalibration calibration{start, 2000000000, 1e9 * 1024 / (27000000.0 * 37925)};
  EXPECT_EQ(calibration.ToCpuMicroseconds(end), 3000000);
  EXPECT_EQ(calibration.DurationMicroseconds(end - start), 1000000);
}

}  // namespace onnxruntime::test
