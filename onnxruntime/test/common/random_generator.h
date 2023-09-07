// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/common/gsl.h"
#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/common/type_utils.h"
#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

namespace detail {
inline int64_t SizeFromDims(gsl::span<const int64_t> dims, gsl::span<const int64_t> strides = {}) {
  int64_t size = 1;
  if (strides.empty()) {
    size = std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  } else {
    ORT_ENFORCE(dims.size() == strides.size());
    for (size_t dim = 0; dim < dims.size(); ++dim) {
      if (dims[dim] == 0) {
        size = 0;
        break;
      }
      size += strides[dim] * (dims[dim] - 1);
    }
  }

  ORT_ENFORCE(size >= 0);
  return size;
}
}  // namespace detail

class RandomValueGenerator {
 public:
  using RandomEngine = std::default_random_engine;
  using RandomSeedType = RandomEngine::result_type;

  explicit RandomValueGenerator(optional<RandomSeedType> seed = {})
      : random_seed_{seed.has_value() ? *seed : static_cast<RandomSeedType>(GetTestRandomSeed())},
        generator_{random_seed_},
        output_trace_{__FILE__, __LINE__, "ORT test random seed: " + std::to_string(random_seed_)} {
  }

  RandomSeedType GetRandomSeed() const {
    return random_seed_;
  }

  // Random values generated are in the range [min, max).
  template <typename TFloat>
  typename std::enable_if<
      std::is_floating_point<TFloat>::value,
      std::vector<TFloat>>::type
  Uniform(gsl::span<const int64_t> dims, TFloat min, TFloat max) {
    std::vector<TFloat> val(detail::SizeFromDims(dims));
    std::uniform_real_distribution<TFloat> distribution(min, max);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  // Random values generated are in the range [min, max).
  template <typename TFloat16>
  typename std::enable_if<
      std::is_same_v<TFloat16, MLFloat16>,
      std::vector<TFloat16>>::type
  Uniform(gsl::span<const int64_t> dims, float min, float max) {
    std::vector<TFloat16> val(detail::SizeFromDims(dims));
    std::uniform_real_distribution<float> distribution(min, max);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = TFloat16(static_cast<float>(distribution(generator_)));
    }
    return val;
  }

  // Random values generated are in the range [min, max).
  template <typename TInt>
  typename std::enable_if<
      std::is_integral<TInt>::value && !utils::IsByteType<TInt>::value,
      std::vector<TInt>>::type
  Uniform(gsl::span<const int64_t> dims, TInt min, TInt max) {
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
  Uniform(gsl::span<const int64_t> dims, TByte min, TByte max) {
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
  Gaussian(gsl::span<const int64_t> dims, TFloat mean, TFloat stddev) {
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

}  // namespace test
}  // namespace onnxruntime
