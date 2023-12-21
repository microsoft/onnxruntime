// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>
#include <type_traits>

#include "core/common/gsl.h"

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/common/type_utils.h"
#include "core/framework/tensor.h"
#include "core/util/math.h"
#include "core/platform/threadpool.h"
#include "core/util/thread_utils.h"
#include "test/common/random_generator.h"

namespace onnxruntime {
namespace test {

// This class provides similar functionality as `RandomValueGenerator` but generates `fixed` patterns
// for given tensor element type and shape. It should be used in unstable tests because
// `purely random` patterns can easily trigger numerical errors.
class FixedPatternValueGenerator {
 public:
  explicit FixedPatternValueGenerator();

  template <typename TValue>
  std::vector<TValue>
  Discrete(gsl::span<const int64_t> dims, gsl::span<const TValue> value_candidates) {
    std::vector<TValue> values(detail::SizeFromDims(dims));
    std::uniform_int_distribution<size_t> distribution(0, value_candidates.size() - 1);
    // Tier 2 RNG. Use it if `RandomValueGenerator::Uniform` method causes large numerical errors
    // (e.g., when elementwise relative error > 1e-3).
    //
    // The generated tensor is more numerically stable than other `RandomValueGenerator::Uniform` methods.
    // If a test constantly fails, it's better to use `Discrete` method. For example,
    // we call `Discrete` method to generate a tensor with shape [2, 3] and value candidates
    // [-1, 0, 1] to eliminate the error propagation caused by * and / in matrix multiplication.
    // To trigger mild error propagation in * and /, try value candidates [-1.5, -1, 0, 1, 1.5].
    //
    // Suggested value_candidates to alleviate numerical error (listed
    // from smallest error to largest error):
    //  [0]
    //  [1]
    //  [0, 1]
    //  [-1, 0, 1]
    //  [-2, -1, 0, 1, 2]
    //  [-1.5, -1, 0, 1, 1.5]
    //  [-2, -1.5, -1, 0, 1, 1.5, 2]
    for (size_t i = 0; i < values.size(); ++i) {
      // To maximize stability, use constant_generator_ since
      // it's always initialized with 0.
      auto index = distribution(generator_);
      values[i] = value_candidates[index];
    }
    return values;
  }

  template <typename TValue>
  std::vector<TValue>
  Circular(gsl::span<const int64_t> dims, gsl::span<const TValue> value_candidates) {
    // Tier 3 RNG. Use it if `Discrete` method causes large numerical errors
    // (e.g., when elementwise relative error > 1e-3).
    // Suggested value_candidates to alleviate numerical error (listed
    // from smallest error to largest error):
    //  [0]
    //  [1]
    //  [0, 1]
    //  [-1, 0, 1]
    //  [-2, -1, 0, 1, 2]
    //  [-1.5, -1, 0, 1, 1.5]
    //  [-2, -1.5, -1, 0, 1, 1.5, 2]
    std::vector<TValue> values(detail::SizeFromDims(dims));
    for (size_t i = 0; i < values.size(); ++i) {
      auto index = i % value_candidates.size();
      values[i] = value_candidates[index];
    }
    return values;
  }

 private:
  std::default_random_engine generator_;
  const ::testing::ScopedTrace output_trace_;
};

template <class T>
inline std::vector<T> FillZeros(const std::vector<int64_t>& dims) {
  std::vector<T> val(detail::SizeFromDims(dims), T{});
  return val;
}

// Returns a vector of `count` values which start at `start` and change by increments of `step`.
template <typename T>
inline std::vector<T> ValueRange(size_t count, T start = static_cast<T>(0.f), T step = static_cast<T>(1.f)) {
  std::vector<T> result;
  result.reserve(count);
  T curr = start;
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(curr);
    curr += step;
  }
  return result;
}

template <>
inline std::vector<MLFloat16> ValueRange<MLFloat16>(size_t count, MLFloat16 start, MLFloat16 step) {
  std::vector<MLFloat16> result;
  result.reserve(count);
  float curr = start.ToFloat();
  float f_step = step.ToFloat();
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(MLFloat16(curr));
    curr += f_step;
  }
  return result;
}

inline std::pair<float, float> MeanStdev(gsl::span<const float> v) {
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
                      const std::pair<float, float>& mean_stdev, bool normalize_variance) {
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
    result.push_back(MLFloat16(data[i]));
  }
  return result;
}

inline std::vector<BFloat16> ToBFloat16(const std::vector<float>& data) {
  std::vector<BFloat16> result;
  result.reserve(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    result.push_back(BFloat16(data[i]));
  }
  return result;
}

inline void CheckTensor(const Tensor& expected_tensor, const Tensor& output_tensor, double rtol, double atol) {
  ORT_ENFORCE(expected_tensor.Shape() == output_tensor.Shape(),
              "Expected output shape [" + expected_tensor.Shape().ToString() +
                  "] did not match run output shape [" +
                  output_tensor.Shape().ToString() + "]");

  ASSERT_TRUE(expected_tensor.DataType() == DataTypeImpl::GetType<float>())
      << "Compare with non float number is not supported yet. ";
  auto expected = expected_tensor.Data<float>();
  auto output = output_tensor.Data<float>();
  for (auto i = 0; i < expected_tensor.Shape().Size(); ++i) {
    const auto expected_value = expected[i], actual_value = output[i];
    if (std::isnan(expected_value)) {
      ASSERT_TRUE(std::isnan(actual_value)) << "value mismatch at index " << i
                                            << "; expected is NaN, actual is not NaN";
    } else if (std::isinf(expected_value)) {
      ASSERT_EQ(expected_value, actual_value) << "value mismatch at index " << i;
    } else {
      double diff = fabs(expected_value - actual_value);
      ASSERT_TRUE(diff <= (atol + rtol * fabs(expected_value))) << "value mismatch at index " << i << "; expected: "
                                                                << expected_value << ", actual: " << actual_value;
    }
  }
}

class ParallelRandomValueGenerator {
 public:
  using RandomEngine = std::default_random_engine;
  using RandomSeedType = RandomEngine::result_type;

  ParallelRandomValueGenerator(RandomSeedType base_seed)
      : base_seed_{base_seed} {
  }

  // Random values generated are in the range [min, max).
  template <typename TFloat>
  typename std::enable_if<
      std::is_floating_point<TFloat>::value,
      std::vector<TFloat>>::type
  Uniform(gsl::span<const int64_t> dims, TFloat min, TFloat max) {
    OrtThreadPoolParams to;
    to.thread_pool_size = 16;
    auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                            concurrency::ThreadPoolType::INTRA_OP);
    static double cost = 1;
    RandomSeedType base_seed = base_seed_;
    std::vector<TFloat> val(detail::SizeFromDims(dims));
    concurrency::ThreadPool::TryParallelFor(
        tp.get(), val.size(), cost,
        [&min, &max, &base_seed, &val](
            std::ptrdiff_t begin, std::ptrdiff_t end) {
          RandomSeedType seed = base_seed;
          auto new_seed = static_cast<std::ptrdiff_t>(base_seed) + begin;
          if (new_seed < static_cast<std::ptrdiff_t>(std::numeric_limits<RandomSeedType>::max()))
            seed = static_cast<RandomSeedType>(new_seed);
          RandomEngine generator{seed};
          std::uniform_real_distribution<TFloat> distribution(min, max);
          for (std::ptrdiff_t di = begin; di != end; ++di) {
            val[di] = distribution(generator);
          }
        });

    return val;
  }

  // Random values generated are in the range [min, max).
  template <typename TFloat16>
  typename std::enable_if<
      std::is_same_v<TFloat16, MLFloat16>,
      std::vector<TFloat16>>::type
  Uniform(gsl::span<const int64_t> dims, float min, float max) {
    OrtThreadPoolParams to;
    to.thread_pool_size = 16;
    auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                            concurrency::ThreadPoolType::INTRA_OP);
    static double cost = 1;
    RandomSeedType base_seed = base_seed_;
    std::vector<TFloat16> val(detail::SizeFromDims(dims));
    concurrency::ThreadPool::TryParallelFor(
        tp.get(), val.size(), cost,
        [&min, &max, &base_seed, &val](
            std::ptrdiff_t begin, std::ptrdiff_t end) {
          RandomSeedType seed = base_seed;
          auto new_seed = static_cast<std::ptrdiff_t>(base_seed) + begin;
          if (new_seed < static_cast<std::ptrdiff_t>(std::numeric_limits<RandomSeedType>::max()))
            seed = static_cast<RandomSeedType>(new_seed);
          RandomEngine generator{seed};
          std::uniform_real_distribution<float> distribution(min, max);
          for (std::ptrdiff_t di = begin; di != end; ++di) {
            val[di] = TFloat16(static_cast<float>(distribution(generator)));
            ;
          }
        });

    return val;
  }

 private:
  const RandomSeedType base_seed_;
};

}  // namespace test
}  // namespace onnxruntime
