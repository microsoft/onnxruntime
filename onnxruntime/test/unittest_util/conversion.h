// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>

#include "core/common/float16.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace test {
// These converters use std::transform rather than an Eigen Map cast-assignment. The Eigen path
// vectorizes to a 16-wide _mm512_loadu_ps under -march=native, which GCC 15 mis-flags with a
// -Warray-bounds false positive when the helper is inlined into callers that pass small fixed-size
// buffers (it assumes the guarded packet load runs at input_size < 16). The element-wise transforms
// keep identical conversion semantics (Eigen::half round-to-nearest for float<->half; static_cast
// for the integer casts) without the vectorized load.
inline void ConvertFloatToMLFloat16(const float* f_datat, MLFloat16* h_data, size_t input_size) {
  auto* h = static_cast<Eigen::half*>(static_cast<void*>(h_data));
  std::transform(f_datat, f_datat + input_size, h, [](float f) { return static_cast<Eigen::half>(f); });
}

inline void ConvertFloatToUint8_t(const float* f_datat, uint8_t* u8_data, size_t input_size) {
  std::transform(f_datat, f_datat + input_size, u8_data, [](float f) { return static_cast<uint8_t>(f); });
}

inline void ConvertFloatToInt8_t(const float* f_datat, int8_t* i8_data, size_t input_size) {
  std::transform(f_datat, f_datat + input_size, i8_data, [](float f) { return static_cast<int8_t>(f); });
}

inline void ConvertMLFloat16ToFloat(const MLFloat16* h_data, float* f_data, size_t input_size) {
  const auto* h = static_cast<const Eigen::half*>(static_cast<const void*>(h_data));
  std::transform(h, h + input_size, f_data, [](Eigen::half h) { return static_cast<float>(h); });
}

inline std::vector<MLFloat16> FloatsToMLFloat16s(const std::vector<float>& f) {
  std::vector<MLFloat16> m(f.size());
  ConvertFloatToMLFloat16(f.data(), m.data(), f.size());
  return m;
}

inline std::vector<BFloat16> MakeBFloat16(const std::initializer_list<float>& input) {
  std::vector<BFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output), [](float f) { return BFloat16(f); });
  return output;
}

inline std::vector<BFloat16> FloatsToBFloat16s(const std::vector<float>& input) {
  std::vector<BFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output), [](float f) { return BFloat16(f); });
  return output;
}
}  // namespace test
}  // namespace onnxruntime
