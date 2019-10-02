#pragma once

#include<random>

#include "core/util/math.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <class T>
inline void FillRandom(std::vector<T>& val, T min, T max) {
  static std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(min, max);
  for (size_t i = 0; i < val.size(); ++i) {
    val[i] = T(distribution(generator));
  }
}

inline std::pair<float, float> MeanStdev(std::vector<float>& v) {
  float sum = std::accumulate(v.begin(), v.end(), 0.0f);
  float mean = sum / v.size();

  std::vector<float> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(),
                 std::bind(std::minus<float>(), std::placeholders::_1, mean));
  float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0f);
  float stdev = std::sqrt(sq_sum / v.size());

  return std::make_pair(mean, stdev);
}

inline void Normalize(std::vector<float>& v,
                      std::pair<float, float>& mean_stdev, bool normalize_variance) {
  float mean = mean_stdev.first;
  float stdev = mean_stdev.second;

  std::transform(v.begin(), v.end(), v.begin(),
                 std::bind(std::minus<float>(), std::placeholders::_1, mean));

  if (normalize_variance) {
    std::transform(v.begin(), v.end(), v.begin(),
                   std::bind(std::divides<float>(), std::placeholders::_1, stdev));
  }
}
}  // namespace test
}  // namespace onnxruntime
