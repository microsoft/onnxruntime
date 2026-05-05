/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax_rvv_compare.cpp

Abstract:

    This module implements a standalone RVV versus scalar validation and
    timing tool for the Softmax critical path on riscv64.

--*/

#include "mlas.h"

#include <iostream>

#if !defined(MLAS_TARGET_RISCV64)

int main() {
  std::cout << "softmax_rvv_compare is only supported on riscv64." << std::endl;
  return 0;
}

#elif !defined(MLAS_USE_RVV)

int main() {
  std::cout << "softmax_rvv_compare requires an RVV-enabled MLAS build." << std::endl;
  return 0;
}

#else

#include "core/mlas/lib/mlasi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

struct CompareStats {
  float max_abs_diff = 0.0f;
  float max_rel_diff = 0.0f;
  double checksum_scalar = 0.0;
  double checksum_rvv = 0.0;
};

struct TimingStats {
  double scalar_ms = 0.0;
  double rvv_ms = 0.0;
};

void ScalarSoftmaxRow(const float* input, float* output, size_t d, bool log_softmax, bool smooth_softmax) {
  float maximum = MlasReduceMaximumF32Kernel(input, d);
  if (smooth_softmax && maximum < 0.0f) {
    maximum = 0.0f;
  }

  const float negative_maximum = -maximum;

  if (log_softmax) {
    float accumulation = MlasComputeSumExpF32Kernel(input, nullptr, d, &negative_maximum);
    if (smooth_softmax) {
      accumulation += std::exp(-maximum);
    }

    const float parameters[2] = {negative_maximum, std::log(accumulation)};
    MlasComputeLogSoftmaxOutputF32Kernel(input, output, d, parameters);
    return;
  }

  float accumulation = MlasComputeSumExpF32Kernel(input, output, d, &negative_maximum);
  if (smooth_softmax) {
    accumulation += std::exp(-maximum);
  }

  const float parameters[1] = {1.0f / accumulation};
  MlasComputeSoftmaxOutputF32Kernel(output, d, parameters);
}

void RvvSoftmaxRow(const float* input, float* output, size_t d, bool log_softmax, bool smooth_softmax) {
  auto& platform = GetMlasPlatform();

  float maximum = platform.ReduceMaximumF32Kernel(input, d);
  if (smooth_softmax && maximum < 0.0f) {
    maximum = 0.0f;
  }

  const float negative_maximum = -maximum;

  if (log_softmax) {
    float accumulation = platform.ComputeSumExpF32Kernel(input, nullptr, d, &negative_maximum);
    if (smooth_softmax) {
      accumulation += std::exp(-maximum);
    }

    const float parameters[2] = {negative_maximum, std::log(accumulation)};
    platform.ComputeLogSoftmaxOutputF32Kernel(input, output, d, parameters);
    return;
  }

  float accumulation = platform.ComputeSumExpF32Kernel(input, output, d, &negative_maximum);
  if (smooth_softmax) {
    accumulation += std::exp(-maximum);
  }

  const float parameters[1] = {1.0f / accumulation};
  platform.ComputeSoftmaxOutputF32Kernel(output, d, parameters);
}

CompareStats CompareCase(size_t rows, size_t d, bool log_softmax, bool smooth_softmax) {
  std::vector<float> input(rows * d);
  std::vector<float> scalar_output(rows * d);
  std::vector<float> rvv_output(rows * d);

  std::mt19937 rng(
      static_cast<uint32_t>(rows * 131 + d * 17 + (log_softmax ? 7 : 0) + (smooth_softmax ? 19 : 0)));
  std::uniform_real_distribution<float> dist(-150.0f, 190.0f);

  for (float& value : input) {
    value = dist(rng);
  }

  for (size_t row = 0; row < rows; ++row) {
    const float* row_input = input.data() + row * d;
    ScalarSoftmaxRow(row_input, scalar_output.data() + row * d, d, log_softmax, smooth_softmax);
    RvvSoftmaxRow(row_input, rvv_output.data() + row * d, d, log_softmax, smooth_softmax);
  }

  CompareStats stats;
  for (size_t i = 0; i < rows * d; ++i) {
    const float scalar = scalar_output[i];
    const float rvv = rvv_output[i];
    const float abs_diff = std::fabs(scalar - rvv);
    const float rel_diff = abs_diff / std::max(std::fabs(scalar), 1.0e-12f);
    stats.max_abs_diff = std::max(stats.max_abs_diff, abs_diff);
    stats.max_rel_diff = std::max(stats.max_rel_diff, rel_diff);
    stats.checksum_scalar += scalar;
    stats.checksum_rvv += rvv;
  }

  return stats;
}

TimingStats TimeCase(size_t rows, size_t d, size_t repeats, bool log_softmax, bool smooth_softmax) {
  std::vector<float> input(rows * d);
  std::vector<float> scalar_output(rows * d);
  std::vector<float> rvv_output(rows * d);

  std::mt19937 rng(static_cast<uint32_t>(rows * 97 + d * 29 + repeats));
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  for (float& value : input) {
    value = dist(rng);
  }

  const auto scalar_begin = std::chrono::steady_clock::now();
  for (size_t repeat = 0; repeat < repeats; ++repeat) {
    for (size_t row = 0; row < rows; ++row) {
      ScalarSoftmaxRow(input.data() + row * d, scalar_output.data() + row * d, d, log_softmax, smooth_softmax);
    }
  }
  const auto scalar_end = std::chrono::steady_clock::now();

  const auto rvv_begin = std::chrono::steady_clock::now();
  for (size_t repeat = 0; repeat < repeats; ++repeat) {
    for (size_t row = 0; row < rows; ++row) {
      RvvSoftmaxRow(input.data() + row * d, rvv_output.data() + row * d, d, log_softmax, smooth_softmax);
    }
  }
  const auto rvv_end = std::chrono::steady_clock::now();

  TimingStats stats;
  stats.scalar_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(scalar_end - scalar_begin).count();
  stats.rvv_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(rvv_end - rvv_begin).count();
  return stats;
}

void PrintCompareCase(const std::string& name, size_t rows, size_t d, bool log_softmax, bool smooth_softmax) {
  const auto stats = CompareCase(rows, d, log_softmax, smooth_softmax);
  std::cout << name << " rows=" << rows << " d=" << d << " log_softmax=" << log_softmax
            << " smooth=" << smooth_softmax << '\n';
  std::cout << "  max_abs_diff=" << std::setprecision(9) << stats.max_abs_diff
            << " max_rel_diff=" << stats.max_rel_diff << '\n';
  std::cout << "  checksum_scalar=" << std::setprecision(12) << stats.checksum_scalar
            << " checksum_rvv=" << stats.checksum_rvv << '\n';
}

void PrintTimingCase(
    const std::string& name, size_t rows, size_t d, size_t repeats, bool log_softmax, bool smooth_softmax) {
  const auto stats = TimeCase(rows, d, repeats, log_softmax, smooth_softmax);
  const double speedup = stats.rvv_ms > 0.0 ? stats.scalar_ms / stats.rvv_ms : 0.0;
  std::cout << name << " rows=" << rows << " d=" << d << " repeats=" << repeats
            << " log_softmax=" << log_softmax << " smooth=" << smooth_softmax << '\n';
  std::cout << "  scalar_ms=" << std::fixed << std::setprecision(3) << stats.scalar_ms
            << " rvv_ms=" << stats.rvv_ms << " speedup=" << speedup << "x\n";
}

}  // namespace

int main() {
  auto& platform = GetMlasPlatform();

  std::cout << std::boolalpha;
  std::cout << "dispatch_is_rvv_reduce="
            << (platform.ReduceMaximumF32Kernel == MlasReduceMaximumF32KernelRvv) << '\n';
  std::cout << "dispatch_is_rvv_sumexp="
            << (platform.ComputeSumExpF32Kernel == MlasComputeSumExpF32KernelRvv) << '\n';
  std::cout << "dispatch_is_rvv_softmax="
            << (platform.ComputeSoftmaxOutputF32Kernel == MlasComputeSoftmaxOutputF32KernelRvv) << '\n';
  std::cout << "dispatch_is_rvv_logsoftmax="
            << (platform.ComputeLogSoftmaxOutputF32Kernel == MlasComputeLogSoftmaxOutputF32KernelRvv) << '\n';
  std::cout << '\n';

  PrintCompareCase("regression_case_3x128_softmax", 3, 128, false, true);
  PrintCompareCase("regression_case_3x128_logsoftmax", 3, 128, true, true);
  PrintCompareCase("regression_case_63x95_softmax", 63, 95, false, true);
  PrintCompareCase("regression_case_16x211_softmax", 16, 211, false, true);
  std::cout << '\n';

  PrintTimingCase("perf_case_attention_like", 4096, 128, 100, false, true);
  PrintTimingCase("perf_case_long_seq", 1024, 1024, 20, false, true);

  return 0;
}

#endif
