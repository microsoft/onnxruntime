// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define ORT_SCATTER_HOST_DEVICE __host__ __device__
#else
#define ORT_SCATTER_HOST_DEVICE
#endif

namespace onnxruntime::scatter_internal {

// Divide before summing, retaining remainders exactly. Neither a widened integer
// sum nor floating-point conversion can represent all int64/uint64 means.
template <typename T>
class IntegerMean {
 public:
  ORT_SCATTER_HOST_DEVICE explicit IntegerMean(uint64_t count) : count_(count) {}

  ORT_SCATTER_HOST_DEVICE void Add(T value) {
    bool negative = false;
    if constexpr (std::is_signed_v<T>) {
      negative = value < 0;
    }
    const uint64_t magnitude = negative ? uint64_t{0} - static_cast<uint64_t>(value)
                                        : static_cast<uint64_t>(value);
    auto& quotient = negative ? negative_quotient_ : positive_quotient_;
    auto& remainder = negative ? negative_remainder_ : positive_remainder_;
    quotient += magnitude / count_;
    const uint64_t r = magnitude % count_;
    if (remainder >= count_ - r) {
      ++quotient;
      remainder -= count_ - r;
    } else {
      remainder += r;
    }
  }

  ORT_SCATTER_HOST_DEVICE T Get() const {
    if constexpr (std::is_signed_v<T>) {
      if (positive_quotient_ < negative_quotient_ ||
          (positive_quotient_ == negative_quotient_ && positive_remainder_ < negative_remainder_)) {
        const uint64_t magnitude = negative_quotient_ - positive_quotient_ +
                                   (positive_remainder_ < negative_remainder_);
        // Avoid negating INT64_MIN or converting its magnitude to a signed type.
        return static_cast<T>(-static_cast<int64_t>(magnitude - 1) - 1);
      }
    }
    return static_cast<T>(positive_quotient_ - negative_quotient_ -
                          (positive_remainder_ < negative_remainder_));
  }

 private:
  uint64_t count_;
  uint64_t positive_quotient_{0};
  uint64_t negative_quotient_{0};
  uint64_t positive_remainder_{0};
  uint64_t negative_remainder_{0};
};

template <typename T>
ORT_SCATTER_HOST_DEVICE T MeanCast(double value) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(value);
  } else {
    return T(static_cast<float>(value));
  }
}

template <typename T>
ORT_SCATTER_HOST_DEVICE T MeanOfLinkedUpdates(T original, const T* updates, const int64_t* next,
                                              int64_t head, int64_t stride, int64_t offset) {
  uint64_t count = 1;
  for (int64_t i = head; i >= 0; i = next[i]) {
    ++count;
  }
  if constexpr (std::is_integral_v<T>) {
    if constexpr (sizeof(T) < sizeof(int64_t)) {
      // Avoid per-update division when a widened sum is provably safe.
      constexpr uint64_t magnitude_limit = uint64_t{1} << std::numeric_limits<T>::digits;
      if (count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / magnitude_limit) {
        int64_t sum = static_cast<int64_t>(original);
        for (int64_t i = head; i >= 0; i = next[i]) {
          sum += static_cast<int64_t>(updates[i * stride + offset]);
        }
        const int64_t divisor = static_cast<int64_t>(count);
        return static_cast<T>(sum / divisor - (sum % divisor < 0));
      }
    }
    IntegerMean<T> mean(count);
    mean.Add(original);
    for (int64_t i = head; i >= 0; i = next[i]) {
      mean.Add(updates[i * stride + offset]);
    }
    return mean.Get();
  } else {
    using Accumulator = std::conditional_t<std::is_same_v<T, double>, double, float>;
    Accumulator sum = static_cast<Accumulator>(original);
    Accumulator scale = 0;
    bool positive_infinity = false;
    bool negative_infinity = false;
    bool nan = false;
    auto add = [&](Accumulator value) {
      nan |= value != value;
      positive_infinity |= value == std::numeric_limits<Accumulator>::infinity();
      negative_infinity |= value == -std::numeric_limits<Accumulator>::infinity();
      const Accumulator magnitude = value < 0 ? -value : value;
      if (magnitude > scale) scale = magnitude;
    };
    add(sum);
    for (int64_t i = head; i >= 0; i = next[i]) {
      const Accumulator value = static_cast<Accumulator>(updates[i * stride + offset]);
      add(value);
      sum += value;
    }
    if (nan || (positive_infinity && negative_infinity)) {
      return MeanCast<T>(std::numeric_limits<double>::quiet_NaN());
    }
    if (positive_infinity || negative_infinity) {
      return MeanCast<T>(positive_infinity ? std::numeric_limits<double>::infinity()
                                           : -std::numeric_limits<double>::infinity());
    }
    if (sum <= std::numeric_limits<Accumulator>::max() && sum >= -std::numeric_limits<Accumulator>::max()) {
      return MeanCast<T>(sum / static_cast<Accumulator>(count));
    }
    // Replay only finite-sum overflow in double, rescaled for double inputs too.
    // The usual half/bfloat16/float path needs no per-update FP64 arithmetic.
    double scaled_sum = static_cast<double>(original) / static_cast<double>(scale);
    for (int64_t i = head; i >= 0; i = next[i]) {
      scaled_sum += static_cast<double>(updates[i * stride + offset]) / static_cast<double>(scale);
    }
    double normalized = scaled_sum / static_cast<double>(count);
    // Rounding must not turn a finite mean into infinity at DBL_MAX.
    if (normalized > 1) normalized = 1;
    if (normalized < -1) normalized = -1;
    return MeanCast<T>(normalized * scale);
  }
}

}  // namespace onnxruntime::scatter_internal

#undef ORT_SCATTER_HOST_DEVICE
