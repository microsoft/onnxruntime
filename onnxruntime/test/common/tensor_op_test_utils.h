// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/util/math.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

namespace detail {
inline int64_t SizeFromDims(const std::vector<int64_t>& dims) {
  const int64_t size = std::accumulate(
      dims.cbegin(), dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  ORT_ENFORCE(size >= 0);
  return size;
}
}  // namespace detail

class RandomValueGenerator {
 public:
  RandomValueGenerator();

  // Random values generated are in the range [min, max).
  template <typename TFloat>
  typename std::enable_if<
      std::is_floating_point<TFloat>::value,
      std::vector<TFloat>>::type
  Uniform(const std::vector<int64_t>& dims, TFloat min, TFloat max) {
    std::vector<TFloat> val(detail::SizeFromDims(dims));
    std::uniform_real_distribution<TFloat> distribution(min, max);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  // Random values generated are in the range [min, max).
  template <typename TInt>
  typename std::enable_if<
      std::is_integral<TInt>::value,
      std::vector<TInt>>::type
  Uniform(const std::vector<int64_t>& dims, TInt min, TInt max) {
    std::vector<TInt> val(detail::SizeFromDims(dims));
    std::uniform_int_distribution<TInt> distribution(min, max - 1);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  template <class T>
  inline std::vector<T> OneHot(const std::vector<int64_t>& dims, int64_t stride) {
    std::vector<T> val(detail::SizeFromDims(dims), T(0));
    std::uniform_int_distribution<int64_t> distribution(0, stride - 1);
    for (size_t offset = 0; offset < val.size(); offset += stride) {
      size_t rand_index = static_cast<size_t>(distribution(generator_));
      val[offset + rand_index] = T(1);
    }
    return val;
  }

 private:
  const RandomSeedType random_seed_;
  std::default_random_engine generator_;
  // while this instance is in scope, output some context information on test failure like the random seed value
  const ::testing::ScopedTrace output_trace_;
};

template <class T>
inline std::vector<T> FillZeros(const std::vector<int64_t>& dims) {
  std::vector<T> val(detail::SizeFromDims(dims), T(0));
  return val;
}

// Returns a vector of `count` values which start at `start` and change by increments of `step`.
template <typename T>
inline std::vector<T> ValueRange(
    size_t count, T start = static_cast<T>(0), T step = static_cast<T>(1)) {
  std::vector<T> result;
  result.reserve(count);
  T curr = start;
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(curr);
    curr += step;
  }
  return result;
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
