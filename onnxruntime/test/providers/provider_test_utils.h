// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test/providers/checkers.h"
#include "test/providers/op_tester.h"
#include "test/providers/model_tester.h"

namespace onnxruntime {
namespace test {
inline void ConvertFloatToMLFloat16(const float* f_datat, MLFloat16* h_data, size_t input_size) {
  auto in_vector = ConstEigenVectorMap<float>(f_datat, input_size);
  auto output_vector = EigenVectorMap<Eigen::half>(static_cast<Eigen::half*>(static_cast<void*>(h_data)), input_size);
  output_vector = in_vector.template cast<Eigen::half>();
}

inline void ConvertMLFloat16ToFloat(const MLFloat16* h_data, float* f_data, size_t input_size) {
  auto in_vector =
      ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(h_data)), input_size);
  auto output_vector = EigenVectorMap<float>(f_data, input_size);
  output_vector = in_vector.template cast<float>();
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
