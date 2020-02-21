// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>

#include "core/util/math.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

class RandomValueGenerator {
 public:
  enum class RandomSeedType {
    kStatic,      // static value
    kPerProcess,  // value that is fixed per process (generated or static)
    kDynamic,     // dynamic value
  };

  static constexpr auto k_default_random_seed_type = RandomSeedType::kPerProcess;

  explicit RandomValueGenerator(RandomSeedType random_seed_type = k_default_random_seed_type);

  template <class T>
  inline std::vector<T> Uniform(const std::vector<int64_t>& dims, float min, float max) {
    int64_t size = std::accumulate(dims.cbegin(), dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    std::vector<T> val(size);
    std::uniform_real_distribution<float> distribution(min, max);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = T(distribution(generator_));
    }
    return val;
  }

  template <class T>
  inline std::vector<T> OneHot(const std::vector<int64_t>& dims, int64_t stride) {
    int64_t size = std::accumulate(dims.cbegin(), dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    std::vector<T> val(size, T(0));
    std::uniform_int_distribution<int64_t> distribution(0, stride - 1);
    for (size_t offset = 0; offset < val.size(); offset += stride) {
      size_t rand_index = static_cast<size_t>(distribution(generator_));
      val[offset + rand_index] = T(1);
    }
    return val;
  }

 private:
  std::default_random_engine generator_;
};

template <class T>
inline std::vector<T> FillZeros(const std::vector<int64_t>& dims) {
  int64_t size = std::accumulate(dims.cbegin(), dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  std::vector<T> val(size, T(0));
  return val;
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

inline std::vector<MLFloat16> ToFloat16(const std::vector<float>& data) {
  std::vector<MLFloat16> result;
  result.reserve(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    result.push_back(MLFloat16(math::floatToHalf(data[i])));
  }
  return result;
}

}  // namespace test
}  // namespace onnxruntime
