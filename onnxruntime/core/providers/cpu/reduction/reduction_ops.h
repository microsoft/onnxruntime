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
 public:
  inline ReduceAggregatorSum(int64_t N, const T&) : ReduceAggregator<T, T>(N, 0) {}
  inline void update(const T& v) { this->accumulator_ += v; }
  static T aggall(const T* from_data, int64_t size) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(size)).sum();
  }
  inline T aggall(const T* from_data) {
    return aggall(from_data, this->N_);
  }

  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(0);
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

  static void FastReduceKRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    int64_t N = fast_shape[2];
    const T* data = input.Data<T>();
    int64_t stridei = fast_shape[1] * fast_shape[2];
    int64_t strideo = fast_shape[2];
    T* out = output.MutableData<T>();
    std::vector<T> one(onnxruntime::narrow<size_t>(fast_shape[1]), 1);
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<ptrdiff_t>(fast_shape[0]), ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T), 6),
        [one, data, fast_shape, stridei, strideo, out, N](ptrdiff_t begin, ptrdiff_t last) {
          for (ptrdiff_t d = begin; d < last; ++d) {
            // TODO(hasesh): Plumb through the mlas backend kernel selector config here
            math::MatMul<T>(1, onnxruntime::narrow<ptrdiff_t>(N), onnxruntime::narrow<ptrdiff_t>(fast_shape[1]), one.data(), data + stridei * d, out + strideo * d, nullptr, nullptr);
          }
        });
  }

  static void FastReduceRKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregator<T, T>::CommonFastReduceRKR(
        input, fast_shape, output, tp,
        [=](const T*) -> T { return 0; },
        [=](T& value, const T* p, int64_t size) {
          value += aggall(p, size);
        });
  }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorSumSquare : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorSumSquare(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 0) {}
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).squaredNorm();
  }
  inline void update(const T& v) { this->accumulator_ += v * v; }
  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = static_cast<T>(0);
  }
};

template <typename T>
class ReduceAggregatorMean : public ReduceAggregatorSum<T> {
 public:
  inline ReduceAggregatorMean(int64_t N, const T&) : ReduceAggregatorSum<T>(N, 0) {}
  static T aggall(const T* from_data, int64_t size) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(size)).mean();
  }
  inline T aggall(const T* from_data) {
    return aggall(from_data, this->N_);
  }
  inline T get_value() { return this->accumulator_ / static_cast<T>(this->N_); }

  // Fast reduction
  // WhichFastReduce() already defined in ReduceAggregatorSum

  static void FastReduceKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregatorSum<T>::FastReduceKR(input, fast_shape, output, tp);
    // TODO: use MLAS or BLAS
    T* out = output.MutableData<T>();
    T* end = out + fast_shape[0];
    for (; out != end; ++out) {
      *out /= static_cast<T>(fast_shape[1]);
    }
  }

  static void FastReduceRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregatorSum<T>::FastReduceRK(input, fast_shape, output, tp);
    // TODO: use MLAS or BLAS
    T* out = output.MutableData<T>();
    T* end = out + fast_shape[1];
    for (; out != end; ++out) {
      *out /= static_cast<T>(fast_shape[0]);
    }
  }

  static void FastReduceKRK(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
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

  static void FastReduceRKR(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregatorSum<T>::FastReduceRKR(input, fast_shape, output, tp);
    T* out = output.MutableData<T>();
    T div = static_cast<T>(fast_shape[0] * fast_shape[2]);
    T* end = out + fast_shape[1];
    for (; out != end; ++out) {
      *out /= div;
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

  static void fill_for_empty_set(Tensor& output) {
    if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
      ORT_NOT_IMPLEMENTED();
    } else {
      EigenMap<T>(output).array() = -std::numeric_limits<T>::infinity();
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

  static void fill_for_empty_set(Tensor& output) {
    if constexpr (std::is_same_v<bool, T>) { /* bool specific impl */
      ORT_NOT_IMPLEMENTED();
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
 public:
  inline ReduceAggregatorProd(int64_t N, const T&) : ReduceAggregator<T, T>(N, 1) {}
  inline T aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).prod();
  }
  inline void update(const T& v) { this->accumulator_ *= v; }
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
      if constexpr (sizeof(T) >= sizeof(int64_t)) {
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
      if constexpr (sizeof(T) >= sizeof(int64_t)) {
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
      if constexpr (sizeof(T) >= sizeof(int64_t)) {
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
      if constexpr (sizeof(T) >= sizeof(int64_t)) {
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
 public:
  inline ReduceAggregatorLogSum(int64_t N, const T&) : ReduceAggregator<T, T>(N, 0) {}
  inline T aggall(const T* from_data) {
    return reduce_log<T>(Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).sum());
  }
  inline void update(const T& v) { this->accumulator_ += v; }
  inline T get_value() { return reduce_log<T>(this->accumulator_); }
  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = -std::numeric_limits<T>::infinity();
  }
};

template <typename T>
class ReduceAggregatorLogSumExp : public ReduceAggregator<T, T> {
 protected:
  T max_;

 public:
  inline ReduceAggregatorLogSumExp(int64_t N, const T& init) : ReduceAggregator<T, T>(N, 0) {
    max_ = reduce_isinf(init) ? this->accumulator_ : init;
  }
  inline T aggall(const T* from_data) {
    max_ = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, onnxruntime::narrow<size_t>(this->N_)).maxCoeff();
    for (int64_t i = 0; i < this->N_; ++i) {
      update(from_data[i]);
    }
    return get_value();
  }
  inline void update0(const T& v) {
    max_ = (reduce_isinf(v) || reduce_isnan(v) || v < max_) ? max_ : v;
  }
  inline void update(const T& v) { this->accumulator_ += reduce_exp(v - max_); }
  inline T get_value() { return reduce_log<T>(this->accumulator_) + max_; }
  static void fill_for_empty_set(Tensor& output) {
    EigenMap<T>(output).array() = -std::numeric_limits<T>::infinity();
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
