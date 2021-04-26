// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/common/type_utils.h"
#include "core/util/math.h"

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
  using RandomEngine = std::default_random_engine;
  using RandomSeedType = RandomEngine::result_type;

  explicit RandomValueGenerator(optional<RandomSeedType> seed = {});

  RandomSeedType GetRandomSeed() const {
    return random_seed_;
  }

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
      std::is_integral<TInt>::value && !utils::IsByteType<TInt>::value,
      std::vector<TInt>>::type
  Uniform(const std::vector<int64_t>& dims, TInt min, TInt max) {
    std::vector<TInt> val(detail::SizeFromDims(dims));
    std::uniform_int_distribution<TInt> distribution(min, max - 1);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  template <typename TByte>
  typename std::enable_if<
      utils::IsByteType<TByte>::value,
      std::vector<TByte>>::type
  Uniform(const std::vector<int64_t>& dims, TByte min, TByte max) {
    std::vector<TByte> val(detail::SizeFromDims(dims));
    std::uniform_int_distribution<int32_t> distribution(min, max - 1);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = static_cast<TByte>(distribution(generator_));
    }
    return val;
  }

  // Gaussian distribution for float
  template <typename TFloat>
  typename std::enable_if<
      std::is_floating_point<TFloat>::value,
      std::vector<TFloat>>::type
  Gaussian(const std::vector<int64_t>& dims, TFloat mean, TFloat stddev) {
    std::vector<TFloat> val(detail::SizeFromDims(dims));
    std::normal_distribution<TFloat> distribution(mean, stddev);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  // Gaussian distribution for Integer
  template <typename TInt>
  typename std::enable_if<
      std::is_integral<TInt>::value,
      std::vector<TInt>>::type
  Gaussian(const std::vector<int64_t>& dims, TInt mean, TInt stddev) {
    std::vector<TInt> val(detail::SizeFromDims(dims));
    std::normal_distribution<float> distribution(static_cast<float>(mean), static_cast<float>(stddev));
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = static_cast<TInt>(std::round(distribution(generator_)));
    }
    return val;
  }

  // Gaussian distribution for Integer and Clamp to [min, max]
  template <typename TInt>
  typename std::enable_if<
      std::is_integral<TInt>::value,
      std::vector<TInt>>::type
  Gaussian(const std::vector<int64_t>& dims, TInt mean, TInt stddev, TInt min, TInt max) {
    std::vector<TInt> val(detail::SizeFromDims(dims));
    std::normal_distribution<float> distribution(static_cast<float>(mean), static_cast<float>(stddev));
    for (size_t i = 0; i < val.size(); ++i) {
      int64_t round_val = static_cast<int64_t>(std::round(distribution(generator_)));
      val[i] = static_cast<TInt>(std::min<int64_t>(std::max<int64_t>(round_val, min), max));
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
  RandomEngine generator_;
  // while this instance is in scope, output some context information on test failure like the random seed value
  const ::testing::ScopedTrace output_trace_;
};

template <class T>
inline std::vector<T> FillZeros(const std::vector<int64_t>& dims) {
  std::vector<T> val(detail::SizeFromDims(dims), T{});
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

inline void CheckTensor(const Tensor& expected_tensor, const Tensor& output_tensor, double rtol, double atol) {
  ORT_ENFORCE(expected_tensor.Shape() == output_tensor.Shape(),
              "Expected output shape [" + expected_tensor.Shape().ToString() +
                  "] did not match run output shape [" +
                  output_tensor.Shape().ToString() + "]");

  ASSERT_TRUE(expected_tensor.DataType() == DataTypeImpl::GetType<float>()) << "Compare with non float number is not supported yet. ";
  auto expected = expected_tensor.Data<float>();
  auto output = output_tensor.Data<float>();
  for (auto i = 0; i < expected_tensor.Shape().Size(); ++i) {
    const auto expected_value = expected[i], actual_value = output[i];
    if (std::isnan(expected_value)) {
      ASSERT_TRUE(std::isnan(actual_value)) << "value mismatch at index " << i << "; expected is NaN, actual is not NaN";
    } else if (std::isinf(expected_value)) {
      ASSERT_EQ(expected_value, actual_value) << "value mismatch at index " << i;
    } else {
      double diff = fabs(expected_value - actual_value);
      ASSERT_TRUE(diff <= (atol + rtol * fabs(expected_value))) << "value mismatch at index " << i << "; expected: " << expected_value << ", actual: " << actual_value;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
