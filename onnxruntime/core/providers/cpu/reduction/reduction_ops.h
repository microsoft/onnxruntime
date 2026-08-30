// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CORE_PROVIDERS_CPU_REDUCTION_OPS_H
#define CORE_PROVIDERS_CPU_REDUCTION_OPS_H

#include <cmath>
#include <limits>
#include <type_traits>

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/containers.h"
#include "core/util/math.h"
#endif
#include "core/framework/math.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/reduction/reduction_kernel_base.h"
#include "core/common/safeint.h"

namespace onnxruntime {

enum FastReduceKind {
  kNone = 0,   // no fast implementation
  kK = 1,      // kept dim = no reduce
  kR = 2,      // reduced dim = all reduced
  kKR = 4,     // kept dim, reduced dim
  kRK = 8,     // reduced dim, kept dim
  kKRK = 16,   // kept dim, reduced dim, kept dim
  kRKR = 32,   // reduced dim, kept dim, reduced dim
  kEmpty = 64  // empty reduce
};

FastReduceKind operator|(FastReduceKind a, FastReduceKind b);

bool operator==(FastReduceKind a, FastReduceKind b);

bool operator!=(FastReduceKind a, FastReduceKind b);

constexpr bool IsFastReduceKindAvailable(FastReduceKind scenario, FastReduceKind available) {
  return (static_cast<uint8_t>(scenario) & static_cast<uint8_t>(available)) > 0;
}
/* Evaluate the cost of parallelized FastReduce implementations. */
constexpr TensorOpCost ParallelReduceFastCost(int64_t n_row, int64_t n_col, int64_t element_size, int n_ops) {
  return TensorOpCost{static_cast<double>(n_row * n_col * element_size),
                      static_cast<double>(n_row * element_size),
                      static_cast<double>(n_row * n_col * element_size * n_ops)};
}

/**
  This only improves reduce function when reduced axes are contiguous:
  if len(shape) == 4, any single axis is ok, axes=(0, 1) or (1, 2) or (2, 3) is ok,
  axes=(0, 2) is not covered by this change, former implementation prevails.
  In that case, the shape can be compressed into three cases:
  (K = axis not reduced, R = reduced axis):

  *  KR - reduction on the last dimensions
  *  RK - reduction on the first dimensions
  *  KRK - reduction on the middle dimensions.
  *  RKR - reduction on all dimensions but the middle ones

  For these three configuration, the reduction may be optimized
  with vectors operations. Method WhichFastReduce() returns which case
  case be optimized for which aggregator.
*/
FastReduceKind OptimizeShapeForFastReduce(gsl::span<const int64_t> input_shape,
                                          gsl::span<const int64_t> reduced_axes,
                                          TensorShapeVector& fast_shape,
                                          TensorShapeVector& fast_output_shape,
                                          TensorShapeVector& fast_axes,
                                          bool keep_dims, bool noop_with_empty_axes = false);

class ResultsNoTransposePrepareForReduce {
 public:
  TensorShapeVector input_shape;
  TensorShapeVector reduced_axes;
  TensorShapeVector projected_index;
  int64_t last_loop_red_size;
  int64_t last_loop_red_inc;
  TensorShapeVector unprojected_index;
  int64_t last_loop_size;
  int64_t last_loop_inc;

  ResultsNoTransposePrepareForReduce() : input_shape(), reduced_axes(), projected_index(), unprojected_index() {
    last_loop_red_size = 0;
    last_loop_red_inc = 0;
    last_loop_size = 0;
    last_loop_inc = 0;
  }

  bool equal(gsl::span<const int64_t> local_input_shape, gsl::span<const int64_t> local_reduced_axes);
  void ValidateNotEmpty();
};

template <typename T>
inline T reduce_sqrt(T value) { return std::sqrt(value); }

template <>
inline int64_t reduce_sqrt<int64_t>(int64_t value) {
  return static_cast<int64_t>(std::sqrt(static_cast<double>(value)));
}

template <>
inline int32_t reduce_sqrt<int32_t>(int32_t value) {
  return static_cast<int32_t>(std::sqrt(static_cast<double>(value)));
}

template <typename T>
inline T reduce_log(T value) { return static_cast<T>(std::log(value)); }

template <>
inline int64_t reduce_log<int64_t>(int64_t value) {
  return static_cast<int64_t>(std::log(static_cast<double>(value)));
}

template <>
inline int32_t reduce_log<int32_t>(int32_t value) {
  return static_cast<int32_t>(std::log(static_cast<double>(value)));
}

template <typename T>
inline T reduce_exp(T value) { return static_cast<T>(std::exp(value)); }

template <typename T>
inline bool reduce_isinf(T value) { return std::isinf(value); }

template <>
inline bool reduce_isinf<int8_t>(int8_t) { return false; }

template <>
inline bool reduce_isinf<uint8_t>(uint8_t) { return false; }

template <>
inline bool reduce_isinf<int32_t>(int32_t) { return false; }

template <>
inline bool reduce_isinf<int64_t>(int64_t) { return false; }

template <typename T>
inline bool reduce_isnan(T value) { return std::isnan(value); }

template <>
inline bool reduce_isnan<int8_t>(int8_t) { return false; }

template <>
inline bool reduce_isnan<uint8_t>(uint8_t) { return false; }

template <>
inline bool reduce_isnan<int32_t>(int32_t) { return false; }

template <>
inline bool reduce_isnan<int64_t>(int64_t) { return false; }

// Integer types whose precision exceeds double's 53-bit mantissa, where summing in double
// loses precision for large values. For these, reduction aggregators use Kahan compensated
// summation. int32_t/uint32_t (and narrower) fit exactly in double, so plain double
// accumulation is sufficient.
template <typename T>
inline constexpr bool kReduceUseKahan = std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;

class ReduceAggregatorBase {
 public:
  // Fast reduction: see OptimizeShapeForFastReduce's comment.
  static inline FastReduceKind WhichFastReduce() { return FastReduceKind::kNone; }
  static void FastReduceKR(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*);
  static void FastReduceRK(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*);
  static void FastReduceKRK(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*);
  static void FastReduceRKR(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*);
};

template <typename T, typename TVAL = T>
class ReduceAggregator : public ReduceAggregatorBase {
 public:
  typedef T input_type;
  typedef TVAL value_type;

 protected:
  int64_t N_;
  T accumulator_;

 public:
  inline ReduceAggregator(int64_t N, const T& init) {
    N_ = N;
    accumulator_ = init;
  }
  inline void update(const T&) {}
  inline void update0(const T&) {}
  inline TVAL aggall(const T*) {}
  inline TVAL get_value() { return accumulator_; }
  static void fill_for_empty_set(Tensor&) { ORT_NOT_IMPLEMENTED(); }

 protected:
  static void CommonFastReduceRKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                                  Tensor& output, concurrency::ThreadPool* tp,
                                  std::function<TVAL(const T*)> f_init,
                                  std::function<void(TVAL&, const T*, int64_t)> f_update) {
    const T* data = input.Data<T>();
    TVAL* out = output.MutableData<TVAL>();
    int64_t d0 = fast_shape[0];
    int64_t d2 = fast_shape[2];
    int64_t inc = d2 * fast_shape[1];

    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<ptrdiff_t>(fast_shape[1]), ParallelReduceFastCost(fast_shape[1], fast_shape[0] * fast_shape[2], sizeof(T), 6),
        [data, out, d0, d2, inc, f_init, f_update](ptrdiff_t begin, ptrdiff_t last) {
          const T* p;
          for (ptrdiff_t d = begin; d < last; ++d) {
            p = data + d * d2;
            out[d] = f_init(p);
            for (int64_t i = 0; i < d0; ++i, p += inc) {
              f_update(out[d], p, d2);
            }
          }
        });
  }
};

template <typename T>
class ReduceAggregatorSum : public ReduceAggregator<T, T> {
 protected:
  // For integer types, accumulate in double to avoid signed overflow UB.
  // When overflow occurs, results saturate to T::max()/T::lowest() rather than wrapping.
  // Note: NumPy/PyTorch use modular (wrapping) arithmetic for integers, but wrapping
  // is non-deterministic in ORT due to parallelized reduction order. Saturation provides
  // well-defined behavior. Other EPs (CUDA/DML) may still wrap.
  // Double has range ~1.8e308, sufficient for any practical sum of int32/int64 values.
  // For int64 values > 2^53, double cannot represent every integer exactly;
  // Kahan summation minimizes but does not eliminate rounding in that regime.
  double double_accumulator_ = 0.0;
  double kahan_compensation_ = 0.0;

 public:
  inline ReduceAggregatorSum(int64_t N, const T&) : ReduceAggregator<T, T>(N, 0) {}
  inline void update(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (kReduceUseKahan<T>) {
        double y = static_cast<double>(v) - kahan_compensation_;
        double t = double_accumulator_ + y;
        kahan_compensation_ = (t - double_accumulator_) - y;
        double_accumulator_ = t;
      } else {
        double_accumulator_ += static_cast<double>(v);
      }
    } else {
      this->accumulator_ += v;
    }
  }
  static T aggall(const T* from_data, int64_t size) {
    if constexpr (std::is_integral_v<T>) {
      double sum = 0.0;
      if constexpr (kReduceUseKahan<T>) {
        double comp = 0.0;
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(size); i < n; ++i) {
          double y = static_cast<double>(from_data[i]) - comp;
          double t = sum + y;
          comp = (t - sum) - y;
          sum = t;
        }
      } else {
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(size); i < n; ++i) {
          sum += static_cast<double>(from_data[i]);
        }
      }
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (sum >= t_max) return std::numeric_limits<T>::max();
      if (sum <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(sum);
    } else {
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
                 from_data, onnxruntime::narrow<size_t>(size))
          .sum();
    }
  }
  inline T aggall(const T* from_data) {
    return aggall(from_data, this->N_);
  }
  inline T get_value() {
    if constexpr (std::is_integral_v<T>) {
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (double_accumulator_ >= t_max) return std::numeric_limits<T>::max();
      if (double_accumulator_ <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(double_accumulator_);
    } else {
      return this->accumulator_;
    }
  }

  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(0);
  }

  // Fast reduction paths use parallelized vectorized operations.
  // For integer types, FastReduceRK and FastReduceKRK accumulate through double
  // intermediates to avoid signed overflow UB, then saturating-cast back to T.
  static inline FastReduceKind WhichFastReduce() {
    return FastReduceKind::kKR | FastReduceKind::kRK | FastReduceKind::kKRK | FastReduceKind::kRKR;
  }

  static void FastReduceKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1];
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(fast_shape[0]), ParallelReduceFastCost(1, stridei, sizeof(T), 6),
        [data, stridei, out](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t d = first; d < last; ++d) {
            out[d] = aggall(data + d * stridei, stridei);
          }
        });
  }

  static void FastReduceRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    int64_t N = fast_shape[1];
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t n_rows = fast_shape[0];

    if constexpr (std::is_integral_v<T>) {
      // Accumulate rows in double to avoid signed overflow UB, then saturate.
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<std::ptrdiff_t>(N), ParallelReduceFastCost(1, n_rows, sizeof(T), 6),
          [data, out, N, n_rows](ptrdiff_t begin, ptrdiff_t end) {
            for (ptrdiff_t col = begin; col < end; ++col) {
              double sum = 0.0;
              for (int64_t row = 0; row < n_rows; ++row) {
                sum += static_cast<double>(data[row * N + col]);
              }
              constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
              constexpr double t_min = static_cast<double>(std::numeric_limits<T>::lowest());
              if (sum >= t_max) {
                out[col] = std::numeric_limits<T>::max();
              } else if (sum <= t_min) {
                out[col] = std::numeric_limits<T>::lowest();
              } else {
                out[col] = static_cast<T>(sum);
              }
            }
          });
    } else {
      memcpy(out, data, SafeInt<size_t>(N) * sizeof(T));
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<std::ptrdiff_t>(N), ParallelReduceFastCost(1, n_rows, sizeof(T), 6),
          [data, out, N, n_rows](ptrdiff_t begin, ptrdiff_t end) {
            for (int64_t row = 1; row < n_rows; ++row) {
              EigenVectorArrayMap<T>(out + begin, end - begin) += ConstEigenVectorArrayMap<T>(
                  data + row * N + begin, end - begin);
            }
          });
    }
  }

  static void FastReduceKRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    int64_t N = fast_shape[2];
    const T* data = input.Data<T>();
    int64_t stridei = fast_shape[1] * fast_shape[2];
    int64_t strideo = fast_shape[2];
    T* out = output.MutableData<T>();

    if constexpr (std::is_integral_v<T>) {
      // Accumulate the middle dimension in double to avoid signed overflow UB.
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<ptrdiff_t>(fast_shape[0]),
          ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T), 6),
          [data, fast_shape, stridei, strideo, out, N](ptrdiff_t begin, ptrdiff_t last) {
            for (ptrdiff_t d = begin; d < last; ++d) {
              for (int64_t col = 0; col < N; ++col) {
                double sum = 0.0;
                for (int64_t row = 0; row < fast_shape[1]; ++row) {
                  sum += static_cast<double>(data[stridei * d + row * N + col]);
                }
                constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
                constexpr double t_min = static_cast<double>(std::numeric_limits<T>::lowest());
                if (sum >= t_max) {
                  out[strideo * d + col] = std::numeric_limits<T>::max();
                } else if (sum <= t_min) {
                  out[strideo * d + col] = std::numeric_limits<T>::lowest();
                } else {
                  out[strideo * d + col] = static_cast<T>(sum);
                }
              }
            }
          });
    } else {
      std::vector<T> one(onnxruntime::narrow<size_t>(fast_shape[1]), 1);
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<ptrdiff_t>(fast_shape[0]),
          ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T), 6),
          [one, data, fast_shape, stridei, strideo, out, N](ptrdiff_t begin, ptrdiff_t last) {
            for (ptrdiff_t d = begin; d < last; ++d) {
              math::MatMul<T>(1, onnxruntime::narrow<ptrdiff_t>(N),
                              onnxruntime::narrow<ptrdiff_t>(fast_shape[1]),
                              one.data(), data + stridei * d,
                              out + strideo * d, nullptr, nullptr);
            }
          });
    }
  }

  static void FastReduceRKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    if constexpr (std::is_integral_v<T>) {
      // Use double accumulation across outer-dimension partial sums.
      const T* data = input.Data<T>();
      T* out = output.MutableData<T>();
      int64_t d0 = fast_shape[0];
      int64_t d2 = fast_shape[2];
      int64_t inc = d2 * fast_shape[1];

      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<ptrdiff_t>(fast_shape[1]),
          ParallelReduceFastCost(fast_shape[1], fast_shape[0] * fast_shape[2], sizeof(T), 6),
          [data, out, d0, d2, inc](ptrdiff_t begin, ptrdiff_t last) {
            for (ptrdiff_t d = begin; d < last; ++d) {
              const T* p = data + d * d2;
              double sum = 0.0;
              for (int64_t i = 0; i < d0; ++i, p += inc) {
                for (int64_t j = 0; j < d2; ++j) {
                  sum += static_cast<double>(p[j]);
                }
              }
              constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
              constexpr double t_min = static_cast<double>(std::numeric_limits<T>::lowest());
              if (sum >= t_max) {
                out[d] = std::numeric_limits<T>::max();
              } else if (sum <= t_min) {
                out[d] = std::numeric_limits<T>::lowest();
              } else {
                out[d] = static_cast<T>(sum);
              }
            }
          });
    } else {
      ReduceAggregator<T, T>::CommonFastReduceRKR(
          input, fast_shape, output, tp,
          [=](const T*) -> T { return 0; },
          [=](T& value, const T* p, int64_t size) {
            value += aggall(p, size);
          });
    }
  }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorSumSquare : public ReduceAggregator<T, TVAL> {
  // For integer types, accumulate sum-of-squares in double to avoid signed overflow UB.
  // Same rationale as ReduceAggregatorL2: squaring int32 values > 46340 overflows,
  // and summing squared values compounds the problem.
  double double_accumulator_ = 0.0;
  double kahan_compensation_ = 0.0;  // Kahan compensation for int64+

 public:
  inline ReduceAggregatorSumSquare(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 0) {}
  inline TVAL aggall(const T* from_data) {
    if constexpr (std::is_integral_v<T>) {
      double sum = 0.0;
      if constexpr (kReduceUseKahan<T>) {
        double comp = 0.0;
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          double dv = static_cast<double>(from_data[i]);
          double y = dv * dv - comp;
          double t = sum + y;
          comp = (t - sum) - y;
          sum = t;
        }
      } else {
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          double dv = static_cast<double>(from_data[i]);
          sum += dv * dv;
        }
      }
      constexpr double t_max = static_cast<double>(std::numeric_limits<TVAL>::max());
      if (sum >= t_max) return std::numeric_limits<TVAL>::max();
      return static_cast<TVAL>(sum);
    } else {
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
                 from_data, onnxruntime::narrow<size_t>(this->N_))
          .squaredNorm();
    }
  }
  inline void update(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      double dv = static_cast<double>(v);
      if constexpr (kReduceUseKahan<T>) {
        double y = dv * dv - kahan_compensation_;
        double t = double_accumulator_ + y;
        kahan_compensation_ = (t - double_accumulator_) - y;
        double_accumulator_ = t;
      } else {
        double_accumulator_ += dv * dv;
      }
    } else {
      this->accumulator_ += v * v;
    }
  }
  inline TVAL get_value() {
    if constexpr (std::is_integral_v<T>) {
      constexpr double t_max = static_cast<double>(std::numeric_limits<TVAL>::max());
      if (double_accumulator_ >= t_max) return std::numeric_limits<TVAL>::max();
      return static_cast<TVAL>(double_accumulator_);
    } else {
      return this->accumulator_;
    }
  }
  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(0);
  }
};

template <typename T>
class ReduceAggregatorMean : public ReduceAggregatorSum<T> {
 public:
  inline ReduceAggregatorMean(int64_t N, const T&) : ReduceAggregatorSum<T>(N, 0) {}
  static T aggall(const T* from_data, int64_t size) {
    if constexpr (std::is_integral_v<T>) {
      // Accumulate in double, divide, then saturate back to T.
      double sum = 0.0;
      if constexpr (kReduceUseKahan<T>) {
        double comp = 0.0;
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(size); i < n; ++i) {
          double y = static_cast<double>(from_data[i]) - comp;
          double t = sum + y;
          comp = (t - sum) - y;
          sum = t;
        }
      } else {
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(size); i < n; ++i) {
          sum += static_cast<double>(from_data[i]);
        }
      }
      double result = sum / static_cast<double>(size);
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (result >= t_max) return std::numeric_limits<T>::max();
      if (result <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(result);
    } else {
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
                 from_data, onnxruntime::narrow<size_t>(size))
          .mean();
    }
  }
  inline T aggall(const T* from_data) {
    return aggall(from_data, this->N_);
  }
  inline T get_value() {
    if constexpr (std::is_integral_v<T>) {
      double result = this->double_accumulator_ / static_cast<double>(this->N_);
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (result >= t_max) return std::numeric_limits<T>::max();
      if (result <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(result);
    } else {
      return this->accumulator_ / static_cast<T>(this->N_);
    }
  }

  // Fast reduction
  // WhichFastReduce() already defined in ReduceAggregatorSum

  static void FastReduceKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    if constexpr (std::is_integral_v<T>) {
      // For integers: compute sum in double and divide before saturating to T.
      const T* data = input.Data<T>();
      T* out = output.MutableData<T>();
      int64_t stridei = fast_shape[1];
      double divisor = static_cast<double>(stridei);
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<std::ptrdiff_t>(fast_shape[0]),
          ParallelReduceFastCost(1, stridei, sizeof(T), 6),
          [data, stridei, out, divisor](ptrdiff_t first, ptrdiff_t last) {
            for (ptrdiff_t d = first; d < last; ++d) {
              double sum = 0.0;
              for (int64_t i = 0; i < stridei; ++i) {
                sum += static_cast<double>(data[d * stridei + i]);
              }
              double result = sum / divisor;
              constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
              constexpr double t_min = static_cast<double>(std::numeric_limits<T>::lowest());
              if (result >= t_max)
                out[d] = std::numeric_limits<T>::max();
              else if (result <= t_min)
                out[d] = std::numeric_limits<T>::lowest();
              else
                out[d] = static_cast<T>(result);
            }
          });
    } else {
      ReduceAggregatorSum<T>::FastReduceKR(input, fast_shape, output, tp);
      T* out = output.MutableData<T>();
      T* end = out + fast_shape[0];
      for (; out != end; ++out) {
        *out /= static_cast<T>(fast_shape[1]);
      }
    }
  }

  static void FastReduceRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    if constexpr (std::is_integral_v<T>) {
      // For integers: compute sum in double and divide before saturating to T.
      int64_t N = fast_shape[1];
      const T* data = input.Data<T>();
      T* out = output.MutableData<T>();
      int64_t n_rows = fast_shape[0];
      double divisor = static_cast<double>(n_rows);
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<std::ptrdiff_t>(N),
          ParallelReduceFastCost(1, n_rows, sizeof(T), 6),
          [data, out, N, n_rows, divisor](ptrdiff_t begin, ptrdiff_t end) {
            for (ptrdiff_t col = begin; col < end; ++col) {
              double sum = 0.0;
              for (int64_t row = 0; row < n_rows; ++row) {
                sum += static_cast<double>(data[row * N + col]);
              }
              double result = sum / divisor;
              constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
              constexpr double t_min = static_cast<double>(std::numeric_limits<T>::lowest());
              if (result >= t_max)
                out[col] = std::numeric_limits<T>::max();
              else if (result <= t_min)
                out[col] = std::numeric_limits<T>::lowest();
              else
                out[col] = static_cast<T>(result);
            }
          });
    } else {
      ReduceAggregatorSum<T>::FastReduceRK(input, fast_shape, output, tp);
      T* out = output.MutableData<T>();
      T* end = out + fast_shape[1];
      for (; out != end; ++out) {
        *out /= static_cast<T>(fast_shape[0]);
      }
    }
  }

  static void FastReduceKRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    if constexpr (std::is_integral_v<T>) {
      // For integers: compute sum in double and divide before saturating to T.
      int64_t N = fast_shape[2];
      const T* data = input.Data<T>();
      int64_t stridei = fast_shape[1] * fast_shape[2];
      int64_t strideo = fast_shape[2];
      T* out = output.MutableData<T>();
      double divisor = static_cast<double>(fast_shape[1]);
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<ptrdiff_t>(fast_shape[0]),
          ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T), 6),
          [data, fast_shape, stridei, strideo, out, N, divisor](ptrdiff_t begin, ptrdiff_t last) {
            for (ptrdiff_t d = begin; d < last; ++d) {
              for (int64_t col = 0; col < N; ++col) {
                double sum = 0.0;
                for (int64_t row = 0; row < fast_shape[1]; ++row) {
                  sum += static_cast<double>(data[stridei * d + row * N + col]);
                }
                double result = sum / divisor;
                constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
                constexpr double t_min = static_cast<double>(std::numeric_limits<T>::lowest());
                if (result >= t_max)
                  out[strideo * d + col] = std::numeric_limits<T>::max();
                else if (result <= t_min)
                  out[strideo * d + col] = std::numeric_limits<T>::lowest();
                else
                  out[strideo * d + col] = static_cast<T>(result);
              }
            }
          });
    } else {
      ReduceAggregatorSum<T>::FastReduceKRK(input, fast_shape, output, tp);
      int64_t strideo = fast_shape[2];
      T* out = output.MutableData<T>();
      T* begin;
      T* end;
      T div = static_cast<T>(fast_shape[1]);
      for (int64_t d = 0; d < fast_shape[0]; ++d) {
        begin = out + strideo * d;
        end = begin + strideo;
        for (; begin != end; ++begin) {
          *begin /= div;
        }
      }
    }
  }

  static void FastReduceRKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    if constexpr (std::is_integral_v<T>) {
      // For integers: compute sum in double and divide before saturating to T.
      const T* data = input.Data<T>();
      T* out = output.MutableData<T>();
      int64_t d0 = fast_shape[0];
      int64_t d2 = fast_shape[2];
      int64_t inc = d2 * fast_shape[1];
      double divisor = static_cast<double>(d0 * d2);
      concurrency::ThreadPool::TryParallelFor(
          tp, onnxruntime::narrow<ptrdiff_t>(fast_shape[1]),
          ParallelReduceFastCost(fast_shape[1], fast_shape[0] * fast_shape[2], sizeof(T), 6),
          [data, out, d0, d2, inc, divisor](ptrdiff_t begin, ptrdiff_t last) {
            for (ptrdiff_t d = begin; d < last; ++d) {
              const T* p = data + d * d2;
              double sum = 0.0;
              for (int64_t i = 0; i < d0; ++i, p += inc) {
                for (int64_t j = 0; j < d2; ++j) {
                  sum += static_cast<double>(p[j]);
                }
              }
              double result = sum / divisor;
              constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
              constexpr double t_min = static_cast<double>(std::numeric_limits<T>::lowest());
              if (result >= t_max)
                out[d] = std::numeric_limits<T>::max();
              else if (result <= t_min)
                out[d] = std::numeric_limits<T>::lowest();
              else
                out[d] = static_cast<T>(result);
            }
          });
    } else {
      ReduceAggregatorSum<T>::FastReduceRKR(input, fast_shape, output, tp);
      T* out = output.MutableData<T>();
      T div = static_cast<T>(fast_shape[0] * fast_shape[2]);
      T* end = out + fast_shape[1];
      for (; out != end; ++out) {
        *out /= div;
      }
    }
  }
};

template <typename T>
class ReduceAggregatorMax : public ReduceAggregator<T> {
 public:
  inline ReduceAggregatorMax(int64_t N, const T& init) : ReduceAggregator<T, T>(N, init) {}
  static T aggall(const T* from_data, int64_t size) {
    if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
      return Eigen::Map<const Eigen::Matrix<bool, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(size)).cast<int>().maxCoeff();
    } else { /* generic impl */
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(size)).maxCoeff();
    }
  }
  inline T aggall(const T* from_data) {
    return aggall(from_data, this->N_);
  }
  inline void update(const T& v) { this->accumulator_ = v > this->accumulator_ ? v : this->accumulator_; }

  // fill_for_empty_set: ReduceMax on empty set is semantically undefined for bool.
  static void fill_for_empty_set(Tensor& output) {
    if constexpr (std::is_same_v<T, bool>) {
      ORT_NOT_IMPLEMENTED("ReduceMax is not defined for empty set with bool type");
    } else if constexpr (std::is_integral_v<T>) {
      // For integers, infinity() returns 0. Use lowest() (most negative value).
      EigenMap<T>(output).array() = std::numeric_limits<T>::lowest();
    } else {
      EigenMap<T>(output).array() = -std::numeric_limits<T>::infinity();
    }
  }

  // Fast reduction
  static inline FastReduceKind WhichFastReduce() {
    return FastReduceKind::kKR | FastReduceKind::kRK | FastReduceKind::kKRK | FastReduceKind::kRKR;
  }
  
  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(0);
    if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
      ORT_NOT_IMPLEMENTED();
    } else {
      EigenMap<T>(output).array() = -std::numeric_limits<T>::infinity();
    }
  }

  static void FastReduceKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1];
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(fast_shape[0]), ParallelReduceFastCost(1, stridei, sizeof(T), 6),
        [data, stridei, out](std::ptrdiff_t first, std::ptrdiff_t last) {
          if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
            EigenVectorMap<bool>(out + first, last - first) = ConstEigenMatrixMap<bool>(
                                                                  data + first * stridei, onnxruntime::narrow<size_t>(stridei), last - first)
                                                                  .cast<unsigned char>()
                                                                  .colwise()
                                                                  .maxCoeff()
                                                                  .cast<bool>();
          } else {
            EigenVectorMap<T>(out + first, last - first) = ConstEigenMatrixMap<T>(
                                                               data + first * stridei, onnxruntime::narrow<size_t>(stridei), last - first)
                                                               .colwise()
                                                               .maxCoeff();
          }
        });
  }

  static void FastReduceRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    int64_t n_rows = fast_shape[0];
    int64_t N = fast_shape[1];
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    memcpy(out, data, SafeInt<size_t>(N) * sizeof(T));

    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(N), ParallelReduceFastCost(1, n_rows, sizeof(T), 6),
        [data, out, N, n_rows](ptrdiff_t begin, ptrdiff_t end) {
          const T* p;
          for (int64_t row = 1; row < n_rows; ++row) {
            p = data + row * N;
            for (int64_t j = begin; j < end; ++j) {
              if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
                out[j] = out[j] || p[j];
              } else {
                if (out[j] < p[j])
                  out[j] = p[j];
              }
            }
          }
        });
  }

  static void FastReduceKRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1] * fast_shape[2];
    int64_t strideo = fast_shape[2];
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(fast_shape[0]), ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T), 6),
        [data, fast_shape, stridei, strideo, out](ptrdiff_t begin, ptrdiff_t end) {
          for (ptrdiff_t j = begin; j < end; ++j) {
            if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
              EigenVectorMap<bool>(out + j * strideo, onnxruntime::narrow<size_t>(strideo)) =
                  ConstEigenMatrixMap<bool>(
                      data + j * stridei, onnxruntime::narrow<size_t>(fast_shape[2]), onnxruntime::narrow<size_t>(fast_shape[1]))
                      .cast<unsigned char>()
                      .rowwise()
                      .maxCoeff()
                      .cast<bool>();
            } else {
              EigenVectorMap<T>(out + j * strideo, onnxruntime::narrow<size_t>(strideo)) =
                  ConstEigenMatrixMap<T>(
                      data + j * stridei, onnxruntime::narrow<size_t>(fast_shape[2]), onnxruntime::narrow<size_t>(fast_shape[1]))
                      .rowwise()
                      .maxCoeff();
            }
          }
        });
  }

  static void FastReduceRKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregator<T, T>::CommonFastReduceRKR(
        input, fast_shape, output, tp,
        [=](const T* p) -> T { return p[0]; },
        [=](T& value, const T* p, int64_t size) {
          T v = aggall(p, size);
          if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
            value = value || v;
          } else {
            if (v > value)
              value = v;
          }
        });
  }
};

template <typename T, typename TVAL = int64_t>
class ReduceAggregatorArgMinMax : public ReduceAggregator<T, TVAL> {
 protected:
  int64_t arg_;
  int64_t index_;

 public:
  inline ReduceAggregatorArgMinMax(int64_t N, const T& init) : ReduceAggregator<T, TVAL>(N, init) {
    arg_ = 0;
    index_ = 0;
  }
  inline TVAL get_value() { return arg_; }
  inline void enforce(const ResultsNoTransposePrepareForReduce& res) {
    ORT_ENFORCE(res.projected_index.size() == 0, "Only one axis is allowed for reduction.");
  }
};

template <typename T, typename TVAL = int64_t>
class ReduceAggregatorArgMax : public ReduceAggregatorArgMinMax<T, TVAL> {
 public:
  inline ReduceAggregatorArgMax(int64_t N, const T& init) : ReduceAggregatorArgMinMax<T, TVAL>(N, init) {}
  inline TVAL aggall(const T* from_data) {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).maxCoeff(&this->arg_);
    return this->get_value();
  }
  inline void update(const T& v) {
    if (v > this->accumulator_) {
      this->accumulator_ = v;
      this->arg_ = this->index_;
    }
    ++this->index_;
  }
};

template <typename T, typename TVAL = int64_t>
class ReduceAggregatorArgMaxLastIndex : public ReduceAggregatorArgMax<T, TVAL> {
 public:
  inline ReduceAggregatorArgMaxLastIndex(int64_t N, const T& init) : ReduceAggregatorArgMax<T, TVAL>(N, init) {}
  inline TVAL aggall(const T* from_data) {
    for (int64_t i = 0; i < this->N_; ++i) {
      update(from_data[i]);
    }
    return this->get_value();
  }
  inline void update(const T& v) {
    if (v >= this->accumulator_) {
      this->accumulator_ = v;
      this->arg_ = this->index_;
    }
    ++this->index_;
  }
};

template <typename T, typename TVAL = int64_t>
class ReduceAggregatorArgMin : public ReduceAggregatorArgMinMax<T, TVAL> {
 public:
  inline ReduceAggregatorArgMin(int64_t N, const T& init) : ReduceAggregatorArgMinMax<T, TVAL>(N, init) {}
  inline TVAL aggall(const T* from_data) {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).minCoeff(&this->arg_);
    return this->get_value();
  }
  inline void update(const T& v) {
    if (v < this->accumulator_) {
      this->accumulator_ = v;
      this->arg_ = this->index_;
    }
    ++this->index_;
  }
};

template <typename T, typename TVAL = int64_t>
class ReduceAggregatorArgMinLastIndex : public ReduceAggregatorArgMin<T, TVAL> {
 public:
  inline ReduceAggregatorArgMinLastIndex(int64_t N, const T& init) : ReduceAggregatorArgMin<T, TVAL>(N, init) {}
  inline TVAL aggall(const T* from_data) {
    for (int64_t i = 0; i < this->N_; ++i) {
      update(from_data[i]);
    }
    return this->get_value();
  }
  inline void update(const T& v) {
    if (v <= this->accumulator_) {
      this->accumulator_ = v;
      this->arg_ = this->index_;
    }
    ++this->index_;
  }
};

template <typename T>
class ReduceAggregatorMin : public ReduceAggregator<T, T> {
 public:
  inline ReduceAggregatorMin(int64_t N, const T& init) : ReduceAggregator<T, T>(N, init) {}
  static T aggall(const T* from_data, int64_t size) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(size)).minCoeff();
  }
  inline T aggall(const T* from_data) {
    return aggall(from_data, this->N_);
  }
  inline void update(const T& v) { this->accumulator_ = v < this->accumulator_ ? v : this->accumulator_; }

  // fill_for_empty_set: ReduceMin on empty set is semantically undefined for bool.
  static void fill_for_empty_set(Tensor& output) {
    if constexpr (std::is_same_v<T, bool>) {
      ORT_NOT_IMPLEMENTED("ReduceMin is not defined for empty set with bool type");
    } else if constexpr (std::is_integral_v<T>) {
      // For integers, infinity() returns 0. Use max() (largest value).
      EigenMap<T>(output).array() = std::numeric_limits<T>::max();
    } else {
      EigenMap<T>(output).array() = std::numeric_limits<T>::infinity();
    }
  }

  // Fast reduction
  static inline FastReduceKind WhichFastReduce() {
    return FastReduceKind::kKR | FastReduceKind::kRK | FastReduceKind::kKRK | FastReduceKind::kRKR;
  }

  static void FastReduceKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1];
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(fast_shape[0]), ParallelReduceFastCost(1, stridei, sizeof(T), 6),
        [data, stridei, out](std::ptrdiff_t first, std::ptrdiff_t last) {
          if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
            EigenVectorMap<bool>(out + first, last - first) = ConstEigenMatrixMap<bool>(
                                                                  data + first * stridei, onnxruntime::narrow<size_t>(stridei), last - first)
                                                                  .cast<unsigned char>()
                                                                  .colwise()
                                                                  .minCoeff()
                                                                  .cast<bool>();
          } else {
            EigenVectorMap<T>(out + first, last - first) = ConstEigenMatrixMap<T>(
                                                               data + first * stridei, onnxruntime::narrow<size_t>(stridei), last - first)
                                                               .colwise()
                                                               .minCoeff();
          }
        });
  }

  static void FastReduceRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    int64_t n_rows = fast_shape[0];
    int64_t N = fast_shape[1];
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    memcpy(out, data, SafeInt<size_t>(N) * sizeof(T));

    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(N), ParallelReduceFastCost(1, n_rows, sizeof(T), 6),
        [data, out, N, n_rows](ptrdiff_t begin, ptrdiff_t end) {
          const T* p;
          for (int64_t row = 1; row < n_rows; ++row) {
            p = data + row * N;
            for (int64_t j = begin; j < end; ++j) {
              if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
                out[j] = out[j] && p[j];
              } else {
                if (out[j] > p[j])
                  out[j] = p[j];
              }
            }
          }
        });
  }

  static void FastReduceKRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1] * fast_shape[2];
    int64_t strideo = fast_shape[2];
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(fast_shape[0]), ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T), 6),
        [data, fast_shape, stridei, strideo, out](ptrdiff_t begin, ptrdiff_t end) {
          for (ptrdiff_t j = begin; j < end; ++j) {
            if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
              EigenVectorMap<bool>(out + j * strideo, onnxruntime::narrow<size_t>(strideo)) =
                  ConstEigenMatrixMap<bool>(
                      data + j * stridei, onnxruntime::narrow<size_t>(fast_shape[2]), onnxruntime::narrow<size_t>(fast_shape[1]))
                      .cast<unsigned char>()
                      .rowwise()
                      .minCoeff()
                      .cast<bool>();
            } else {
              EigenVectorMap<T>(out + j * strideo, onnxruntime::narrow<size_t>(strideo)) =
                  ConstEigenMatrixMap<T>(
                      data + j * stridei, onnxruntime::narrow<size_t>(fast_shape[2]), onnxruntime::narrow<size_t>(fast_shape[1]))
                      .rowwise()
                      .minCoeff();
            }
          }
        });
  }

  static void FastReduceRKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregator<T, T>::CommonFastReduceRKR(
        input, fast_shape, output, tp,
        [=](const T* p) -> T { return p[0]; },
        [=](T& value, const T* p, int64_t size) {
          T v = aggall(p, size);
          if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
            value = value && v;
          } else {
            if (v < value)
              value = v;
          }
        });
  }
};

template <typename T>
class ReduceAggregatorProd : public ReduceAggregator<T, T> {
  // For integer types, accumulate product in double to avoid signed overflow UB.
  // Double range (~1.8e308) is far larger than int64 range (~9.2e18).
  // Precision loss is possible for int64 products > 2^53, but this is far better
  // than undefined behavior from signed overflow.
  double double_accumulator_ = 1.0;

 public:
  inline ReduceAggregatorProd(int64_t N, const T&) : ReduceAggregator<T, T>(N, 1) {}
  inline T aggall(const T* from_data) {
    if constexpr (std::is_integral_v<T>) {
      double prod = 1.0;
      for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
        prod *= static_cast<double>(from_data[i]);
      }
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (prod >= t_max) return std::numeric_limits<T>::max();
      if (prod <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(prod);
    } else {
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
                 from_data, onnxruntime::narrow<size_t>(this->N_))
          .prod();
    }
  }
  inline void update(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      double_accumulator_ *= static_cast<double>(v);
    } else {
      this->accumulator_ *= v;
    }
  }
  inline T get_value() {
    if constexpr (std::is_integral_v<T>) {
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (double_accumulator_ >= t_max) return std::numeric_limits<T>::max();
      if (double_accumulator_ <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(double_accumulator_);
    } else {
      return this->accumulator_;
    }
  }
  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(1);
  }
};

// Saturating absolute value for norm reductions.
// For signed integer types, abs(MIN_VALUE) overflows because -MIN_VALUE > MAX_VALUE.
// This returns MAX_VALUE (the closest representable value) instead of invoking UB.
template <typename T>
inline T saturating_abs(const T& v) {
  if constexpr (std::is_signed_v<T> && std::is_integral_v<T>) {
    if (v == std::numeric_limits<T>::min()) {
      return std::numeric_limits<T>::max();
    }
  }
  return v > T(0) ? v : -v;
}

template <typename T>
class ReduceAggregatorL1 : public ReduceAggregator<T, T> {
  // For integer types, accumulate sum-of-absolute-values in double to avoid overflow UB.
  //
  // Problem: ReduceL1 computes sum(|x_i|). For integer types, the sum can overflow:
  //   - int32: just 3 elements of value 1,000,000,000 sum to 3×10^9 > INT32_MAX
  //   - The abs step itself overflows for INT_MIN (handled by saturating_abs)
  //
  // Solution: Accumulate in double (range ~1.8e308), then clamp when casting back to T.
  // The double accumulator cannot overflow to infinity: even summing INT64_MAX for every
  // element, overflow requires > 1.9e289 elements — physically impossible.
  // For int32 inputs, double accumulation is exact: |x_i| fits in 31 bits, and double's
  // 53-bit mantissa can represent sums exactly up to 2^53 (~4.5 million max-magnitude
  // elements). Kahan summation is unnecessary and would only add overhead.
  // For int64 inputs, values > 2^53 and large reductions may lose precision when
  // accumulated in double. Kahan compensated summation is used for types >= 64 bits
  // to reduce accumulation error from O(N) to O(1) ULPs.
  double double_accumulator_ = 0.0;
  double kahan_compensation_ = 0.0;  // Kahan compensation term for int64+

 public:
  inline ReduceAggregatorL1(int64_t N, const T&) : ReduceAggregator<T, T>(N, 0) {}
  inline T aggall(const T* from_data) {
    if constexpr (std::is_integral_v<T>) {
      double sum = 0.0;
      if constexpr (kReduceUseKahan<T>) {
        // Kahan compensated summation for int64+ to minimize precision loss.
        double comp = 0.0;
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          double y = std::abs(static_cast<double>(from_data[i])) - comp;
          double t = sum + y;
          comp = (t - sum) - y;
          sum = t;
        }
      } else {
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          sum += std::abs(static_cast<double>(from_data[i]));
        }
      }
      // Saturate to max representable value if the L1-norm exceeds T's range.
      constexpr double max_val = static_cast<double>(std::numeric_limits<T>::max());
      if (sum >= max_val) return std::numeric_limits<T>::max();
      return static_cast<T>(sum);
    } else {
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).cwiseAbs().sum();
    }
  }
  inline void update(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (kReduceUseKahan<T>) {
        // Kahan compensated summation for int64+.
        double y = std::abs(static_cast<double>(v)) - kahan_compensation_;
        double t = double_accumulator_ + y;
        kahan_compensation_ = (t - double_accumulator_) - y;
        double_accumulator_ = t;
      } else {
        double_accumulator_ += std::abs(static_cast<double>(v));
      }
    } else {
      this->accumulator_ += saturating_abs(v);
    }
  }
  inline T get_value() {
    if constexpr (std::is_integral_v<T>) {
      constexpr double max_val = static_cast<double>(std::numeric_limits<T>::max());
      if (double_accumulator_ >= max_val) return std::numeric_limits<T>::max();
      return static_cast<T>(double_accumulator_);
    } else {
      return this->accumulator_;
    }
  }

  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(0);
  }
};

template <typename T>
class ReduceAggregatorL2 : public ReduceAggregator<T, T> {
  // For integer types, accumulate sum-of-squares in double to avoid signed overflow UB.
  //
  // Problem: ReduceL2 computes sqrt(sum(x_i^2)). For integer types, squaring can overflow:
  //   - int32: any |v| > 46340 causes v*v > INT32_MAX (signed overflow = UB)
  //   - int64: any |v| > 3,037,000,499 causes v*v > INT64_MAX
  //   - INT_MIN: (-2^31)^2 = 2^62 which wraps to 0 in int32, giving sqrt(0) = 0 (wrong)
  //
  // Solution: Promote each element to double before squaring. Double has:
  //   - Range up to ~1.8e308 (sufficient for any sum of int64 squares). The accumulator
  //     cannot overflow to infinity: even summing INT64_MAX^2 for every element would
  //     require > 2.1e270 elements — physically impossible.
  //   - 53-bit mantissa: int32 values are exactly representable in double, but their
  //     squares can require up to 62 bits (e.g., (2^31-1)^2 ≈ 2^62). For |v| > 2^26,
  //     v*v exceeds 2^53 and loses precision. However, this is at most 1 ULP of error
  //     per element — far better than integer overflow. Kahan summation is omitted for
  //     int32 since the per-element squaring dominates error, not accumulation.
  //   - For int64, values > 2^53 lose precision in both squaring and accumulation.
  //     Kahan compensated summation is used for types >= 64 bits to reduce accumulation
  //     error from O(N) to O(1) ULPs (squaring precision loss remains inherent).
  //
  // The final cast back to T is clamped to numeric_limits<T>::max() to avoid UB when the
  // L2-norm exceeds the representable range (e.g., ReduceL2([INT_MIN]) ≈ 2^31 > INT32_MAX).
  double double_accumulator_ = 0.0;
  double kahan_compensation_ = 0.0;  // Kahan compensation term for int64+

 public:
  inline ReduceAggregatorL2(int64_t N, const T&) : ReduceAggregator<T, T>(N, 0) {}
  inline T aggall(const T* from_data) {
    if constexpr (std::is_integral_v<T>) {
      double sum = 0.0;
      if constexpr (kReduceUseKahan<T>) {
        // Kahan compensated summation for int64+ to minimize precision loss.
        double comp = 0.0;
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          double dv = static_cast<double>(from_data[i]);
          double y = dv * dv - comp;
          double t = sum + y;
          comp = (t - sum) - y;
          sum = t;
        }
      } else {
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          double dv = static_cast<double>(from_data[i]);
          sum += dv * dv;
        }
      }
      double result = std::sqrt(sum);
      // Saturate to max representable value if the norm exceeds T's range.
      constexpr double max_val = static_cast<double>(std::numeric_limits<T>::max());
      if (result >= max_val) return std::numeric_limits<T>::max();
      return static_cast<T>(result);
    } else {
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).norm();
    }
  }
  inline void update(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      double dv = static_cast<double>(v);
      if constexpr (kReduceUseKahan<T>) {
        // Kahan compensated summation for int64+.
        double y = dv * dv - kahan_compensation_;
        double t = double_accumulator_ + y;
        kahan_compensation_ = (t - double_accumulator_) - y;
        double_accumulator_ = t;
      } else {
        double_accumulator_ += dv * dv;
      }
    } else {
      this->accumulator_ += v * v;
    }
  }
  inline T get_value() {
    if constexpr (std::is_integral_v<T>) {
      double result = std::sqrt(double_accumulator_);
      // Saturate to max representable value to avoid UB when casting back to integer.
      constexpr double max_val = static_cast<double>(std::numeric_limits<T>::max());
      if (result >= max_val) return std::numeric_limits<T>::max();
      return static_cast<T>(result);
    } else {
      return reduce_sqrt<T>(this->accumulator_);
    }
  }
  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(0);
  }
};

template <typename T>
class ReduceAggregatorLogSum : public ReduceAggregator<T, T> {
  // For integer types, accumulate in double to avoid signed overflow UB before log.
  double double_accumulator_ = 0.0;
  double kahan_compensation_ = 0.0;

 public:
  inline ReduceAggregatorLogSum(int64_t N, const T&) : ReduceAggregator<T, T>(N, 0) {}
  inline T aggall(const T* from_data) {
    if constexpr (std::is_integral_v<T>) {
      double sum = 0.0;
      if constexpr (kReduceUseKahan<T>) {
        double comp = 0.0;
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          double y = static_cast<double>(from_data[i]) - comp;
          double t = sum + y;
          comp = (t - sum) - y;
          sum = t;
        }
      } else {
        for (size_t i = 0, n = onnxruntime::narrow<size_t>(this->N_); i < n; ++i) {
          sum += static_cast<double>(from_data[i]);
        }
      }
      if (sum <= 0.0) {
        // log is undefined for non-positive values; map to min (closest integer analogue of -inf).
        return std::numeric_limits<T>::min();
      }
      double result = std::log(sum);
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (result >= t_max) return std::numeric_limits<T>::max();
      if (result <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(result);
    } else {
      return reduce_log<T>(Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
                               from_data, onnxruntime::narrow<size_t>(this->N_))
                               .sum());
    }
  }
  inline void update(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (kReduceUseKahan<T>) {
        double y = static_cast<double>(v) - kahan_compensation_;
        double t = double_accumulator_ + y;
        kahan_compensation_ = (t - double_accumulator_) - y;
        double_accumulator_ = t;
      } else {
        double_accumulator_ += static_cast<double>(v);
      }
    } else {
      this->accumulator_ += v;
    }
  }
  inline T get_value() {
    if constexpr (std::is_integral_v<T>) {
      if (double_accumulator_ <= 0.0) {
        // log is undefined for non-positive values; map to min (closest integer analogue of -inf).
        return std::numeric_limits<T>::min();
      }
      double result = std::log(double_accumulator_);
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (result >= t_max) return std::numeric_limits<T>::max();
      if (result <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(result);
    } else {
      return reduce_log<T>(this->accumulator_);
    }
  }
  static void fill_for_empty_set(Tensor& output) {
    if constexpr (std::is_integral_v<T>) {
      // log(0) for integers: use min value (closest to -inf semantics).
      EigenMap<T>(output).array() = std::numeric_limits<T>::min();
    } else {
      EigenMap<T>(output).array() = -std::numeric_limits<T>::infinity();
    }
  }
};

template <typename T>
class ReduceAggregatorLogSumExp : public ReduceAggregator<T, T> {
 protected:
  T max_;
  int64_t max_count_ = 0;  // Counter for integer types (avoids overflow in T accumulator)

 public:
  inline ReduceAggregatorLogSumExp(int64_t N, const T& init) : ReduceAggregator<T, T>(N, 0) {
    max_ = reduce_isinf(init) ? this->accumulator_ : init;
  }
  inline T aggall(const T* from_data) {
    max_ = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
               from_data, onnxruntime::narrow<size_t>(this->N_))
               .maxCoeff();
    if constexpr (std::is_integral_v<T>) {
      // For integer types: exp(v - max_) is 1 when v == max_, else 0 (truncation
      // of exp(negative) < 1.0 to integer). No negative-value elements contribute
      // because v <= max_ always, so v - max_ <= 0, and exp(<=0) truncates to 0 or 1.
      // Avoid signed overflow in (v - max_) by counting equal-to-max elements directly.
      int64_t num_maxval_elements = 0;
      for (int64_t i = 0; i < this->N_; ++i) {
        if (from_data[i] == max_) ++num_maxval_elements;
      }
      // Result: log(num_maxval_elements) + max_. Use double to detect overflow before casting.
      double log_count = std::log(static_cast<double>(num_maxval_elements));
      double result = log_count + static_cast<double>(max_);
      // Saturate to avoid signed overflow when casting back to T.
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (result >= t_max) return std::numeric_limits<T>::max();
      if (result <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(result);
    } else {
      // For floating-point types: if max_ is infinite, return max_ early.
      // When max_ = -inf: all inputs are -inf, exp(x_i)=0, log(0) = -inf.
      // When max_ = +inf: result is +inf (avoids +inf - (+inf) = NaN).
      if (reduce_isinf(max_)) {
        return max_;
      }
      for (int64_t i = 0; i < this->N_; ++i) {
        update(from_data[i]);
      }
      return get_value();
    }
  }
  inline void update0(const T& v) {
    max_ = (reduce_isinf(v) || reduce_isnan(v) || v < max_) ? max_ : v;
  }
  inline void update(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      // For integer types: exp(v - max_) truncates to 1 if v == max_, else 0.
      // Use direct comparison to avoid signed overflow in (v - max_).
      if (v == max_) ++max_count_;
    } else {
      // For floating-point types: if v is -inf, exp(-inf) = 0 regardless of max_.
      // Skip to avoid -inf - (-inf) = NaN when max_ is also -inf.
      if (v == -std::numeric_limits<T>::infinity()) return;
      this->accumulator_ += reduce_exp(v - max_);
    }
  }
  inline T get_value() {
    if constexpr (std::is_integral_v<T>) {
      // log(count) + max_. Use double to detect overflow.
      double log_count = std::log(static_cast<double>(max_count_));
      double result = log_count + static_cast<double>(max_);
      constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
      constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
      if (result >= t_max) return std::numeric_limits<T>::max();
      if (result <= t_min) return std::numeric_limits<T>::min();
      return static_cast<T>(result);
    } else {
      // For floating-point types: if max_ is infinite, return max_ directly.
      if (reduce_isinf(max_)) {
        return max_;
      }
      return reduce_log<T>(this->accumulator_) + max_;
    }
  }
  static void fill_for_empty_set(Tensor& output) {
    if constexpr (std::is_integral_v<T>) {
      // For integers, -infinity() returns 0. Use min() (closest to -inf semantics).
      EigenMap<T>(output).array() = std::numeric_limits<T>::min();
    } else {
      EigenMap<T>(output).array() = -std::numeric_limits<T>::infinity();
    }
  }
};

// Traits indicating whether a reduction aggregator applies element-wise transforms
// in addition to reduction.
// For axes=[] with noop_with_empty_axes=1, no reduction is performed; we either
// apply the aggregator's element-wise PreOp/PostOp to each element, or
// return the input unchanged for identity aggregators.
//
// PreOp: per-element transform during accumulation (AGG::update).
// PostOp: transform applied after accumulation (AGG::get_value).
// Add a specialization for aggregators that define a PreOp and/or PostOp.
template <typename AGG>
struct ReduceAggTraits {
  static constexpr bool kHasPreOp = false;
  static constexpr bool kHasPostOp = false;
};

template <typename T>
struct ReduceAggTraits<ReduceAggregatorL1<T>> {
  static constexpr bool kHasPreOp = true;
  static constexpr bool kHasPostOp = false;
};

template <typename T>
struct ReduceAggTraits<ReduceAggregatorL2<T>> {
  static constexpr bool kHasPreOp = true;
  static constexpr bool kHasPostOp = true;
};

template <typename T, typename TVAL>
struct ReduceAggTraits<ReduceAggregatorSumSquare<T, TVAL>> {
  static constexpr bool kHasPreOp = true;
  static constexpr bool kHasPostOp = false;
};

template <typename T>
struct ReduceAggTraits<ReduceAggregatorLogSum<T>> {
  static constexpr bool kHasPreOp = false;
  static constexpr bool kHasPostOp = true;
};

template <typename T>
struct ReduceAggTraits<ReduceAggregatorLogSumExp<T>> {
  static constexpr bool kHasPreOp = true;
  static constexpr bool kHasPostOp = true;
};

void NoTransposePrepareForReduce(const TensorShape& new_input_shape,
                                 gsl::span<const int64_t> reduced_axes,
                                 ResultsNoTransposePrepareForReduce& results);

template <typename AGG>
void NoTransposeReduce1Loop(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                            gsl::span<const int64_t> reduced_axes, concurrency::ThreadPool* tp,
                            ResultsNoTransposePrepareForReduce& last_results);

// Specific case for ReduceLogSumExp.
template <typename AGG>
void NoTransposeReduce2Loops(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                             gsl::span<const int64_t> reduced_axes, concurrency::ThreadPool* tp,
                             ResultsNoTransposePrepareForReduce& last_results);

template <typename AGG>
void CommonReduce1Loop(OpKernelContext* ctx,
                       const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                       bool noop_with_empty_axes = false);

// Specific case for ReduceLogSumExp.
template <typename AGG>
void CommonReduce2Loops(OpKernelContext* ctx,
                        const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                        bool noop_with_empty_axes = false);

template <bool allow_multi_axes>
class ReduceKernel : public OpKernel, public ReduceKernelBase<allow_multi_axes> {
 protected:
  ReduceKernel(const OpKernelInfo& info) : OpKernel(info), ReduceKernelBase<allow_multi_axes>(info) {}
};

template <typename T>
class ReduceL1 final : public ReduceKernel<true> {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceL2 final : public ReduceKernel<true> {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceLogSum final : public ReduceKernel<true> {
 public:
  ReduceLogSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceLogSumExp final : public ReduceKernel<true> {
 public:
  ReduceLogSumExp(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceMax final : public ReduceKernel<true> {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceMean final : public ReduceKernel<true> {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceMin final : public ReduceKernel<true> {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceProd final : public ReduceKernel<true> {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceSum final : public ReduceKernel<true> {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  // For external calls requiring ReduceSum implementation - will return the reduced output.
  //`input_shape_override` overrides the shape of `input` for compute purposes.
  static std::unique_ptr<Tensor> Impl(const Tensor& input, gsl::span<const int64_t> reduce_axes,
                                      AllocatorPtr allocator, concurrency::ThreadPool* tp, bool keep_dims,
                                      const TensorShape* input_shape_override = nullptr);
};

template <typename T>
class ReduceSumSquare final : public ReduceKernel<true> {
 public:
  ReduceSumSquare(const OpKernelInfo& info) : ReduceKernel<true>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ArgMax final : public ReduceKernel<false> {
 public:
  ArgMax(const OpKernelInfo& info) : ReduceKernel<false>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ArgMin final : public ReduceKernel<false> {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel<false>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime

#endif  // !CORE_PROVIDERS_CPU_REDUCTION_OPS_H
