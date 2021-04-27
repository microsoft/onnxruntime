// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CORE_PROVIDERS_CPU_REDUCTION_OPS_H
#define CORE_PROVIDERS_CPU_REDUCTION_OPS_H

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/containers.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"
#include "core/common/safeint.h"
#include <cmath>

namespace onnxruntime {

/* Common verficiations. */
void OrtEnforce_ReduceAggregatorKR(const std::vector<int64_t>& fast_shape, const Tensor& output);
void OrtEnforce_ReduceAggregatorRK(const std::vector<int64_t>& fast_shape, const Tensor& output);
void OrtEnforce_ReduceAggregatorKRK(const std::vector<int64_t>& fast_shape, const Tensor& output);

typedef enum class FastReduceKind : uint8_t {
  kNone = 0,   // no fast implementation
  kK = 1,      // kept dim = no reduce
  kR = 2,      // reduced dim = all reduced
  kKR = 4,     // kept dim, reduced dim
  kRK = 8,     // reduced dim, kept dim
  kKRK = 16,   // kept dim, reduced dim, kept dim
  kEmpty = 32  // empty reduce
} FastReduceKind;

FastReduceKind operator|(FastReduceKind a, FastReduceKind b);

bool operator==(FastReduceKind a, FastReduceKind b);

bool operator!=(FastReduceKind a, FastReduceKind b);

bool IsFastReduceKindAvailable(FastReduceKind scenario, FastReduceKind available);

/**
  This only improves reduce function when reduced axes are contiguous:
  if len(shape) == 4, any single axis is ok, axes=(0, 1) or (1, 2) or (2, 3) is ok,
  axes=(0, 2) is not covered by this change, former implementation prevails.
  In that case, the shape can be compressed into three cases: 
  (K = axis not reduced, R = reduced axis):

  *  KR - reduction on the last dimensions
  *  RK - reduction on the first dimensions
  *  KRK - reduction on the middle dimensions.
   
  For these three configuration, the reduction may be optimized
  with vectors operations. Method WhichFastReduce() returns which case
  case be optimized for which aggregator.
*/
FastReduceKind OptimizeShapeForFastReduce(const std::vector<int64_t>& input_shape,
                                          const std::vector<int64_t>& reduced_axes,
                                          std::vector<int64_t>& fast_shape,
                                          std::vector<int64_t>& fast_output_shape,
                                          std::vector<int64_t>& fast_axes,
                                          bool keep_dims, bool noop_with_empty_axes = false);

class ResultsNoTransposePrepareForReduce {
 public:
  std::vector<int64_t> input_shape;
  std::vector<int64_t> reduced_axes;
  std::vector<int64_t> projected_index;
  int64_t last_loop_red_size;
  int64_t last_loop_red_inc;
  std::vector<int64_t> unprojected_index;
  int64_t last_loop_size;
  int64_t last_loop_inc;

  ResultsNoTransposePrepareForReduce() : input_shape(), reduced_axes(), projected_index(), unprojected_index() {
    last_loop_red_size = 0;
    last_loop_red_inc = 0;
    last_loop_size = 0;
    last_loop_inc = 0;
  }

  bool equal(const std::vector<int64_t>& local_input_shape, const std::vector<int64_t>& local_reduced_axes);
  void OrtEnforceNotEmpty();
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

TensorOpCost ParallelReduceFastCost(int64_t n_row, int64_t n_col, int64_t element_size);

class _ReduceAggregator {
 public:
  // Fast reduction: see OptimizeShapeForFastReduce's comment.
  static inline FastReduceKind WhichFastReduce() { return FastReduceKind::kNone; }
  static void FastReduceKR(const Tensor&, const std::vector<int64_t>&, Tensor&, concurrency::ThreadPool*);
  static void FastReduceRK(const Tensor&, const std::vector<int64_t>&, Tensor&, concurrency::ThreadPool*);
  static void FastReduceKRK(const Tensor&, const std::vector<int64_t>&, Tensor&, concurrency::ThreadPool*);
};

template <typename T, typename TVAL = T>
class ReduceAggregator : public _ReduceAggregator {
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
  inline void update(const T&) { ORT_ENFORCE(false, "must be overloaded."); }
  inline void update0(const T&) { ORT_ENFORCE(false, "must be overloaded."); }
  inline TVAL aggall(const T*) { ORT_ENFORCE(false, "must be overloaded."); }
  inline TVAL get_value() { return accumulator_; }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorSum : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorSum(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 0) {}
  inline void update(const T& v) { this->accumulator_ += v; }
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).sum();
  }

  // Fast reduction
  static inline FastReduceKind WhichFastReduce() {
    return FastReduceKind::kKR | FastReduceKind::kRK | FastReduceKind::kKRK;
  }

  static void FastReduceKR(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorKR(fast_shape, output);
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1];
    concurrency::ThreadPool::TryParallelFor(
        tp, fast_shape[0], ParallelReduceFastCost(1, stridei, sizeof(T)),
        [data, stridei, out](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t d = first; d < last; ++d) {
            out[d] = ConstEigenVectorArrayMap<T>(data + d * stridei, stridei).sum();
          }
        });
  }

  static void FastReduceRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorRK(fast_shape, output);
    int64_t N = fast_shape[1];
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();

    int64_t n_rows = fast_shape[0];
    memcpy(out, data, N * sizeof(T));
    concurrency::ThreadPool::TryParallelFor(
        tp, N, ParallelReduceFastCost(1, n_rows, sizeof(T)),
        [data, out, N, n_rows](ptrdiff_t begin, ptrdiff_t end) {
          for (int64_t row = 1; row < n_rows; ++row) {
            EigenVectorArrayMap<T>(out + begin, end - begin) += ConstEigenVectorArrayMap<T>(
                data + row * N + begin, end - begin);
          }
        });
  }

  static void FastReduceKRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorKRK(fast_shape, output);
    int64_t N = fast_shape[2];
    const T* data = input.Data<T>();
    int64_t stridei = fast_shape[1] * fast_shape[2];
    int64_t strideo = fast_shape[2];
    T* out = output.MutableData<T>();
    std::vector<T> one(fast_shape[1], 1);
    concurrency::ThreadPool::TryParallelFor(
        tp, fast_shape[0], ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T)),
        [one, data, fast_shape, stridei, strideo, out, N](ptrdiff_t begin, ptrdiff_t last) {
          for (ptrdiff_t d = begin; d < last; ++d) {
            math::MatMul<T>(1, N, fast_shape[1], one.data(), data + stridei * d, out + strideo * d, nullptr);
          }
        });
  }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorSumSquare : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorSumSquare(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 0) {}
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).squaredNorm();
  }
  inline void update(const T& v) { this->accumulator_ += v * v; }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorMean : public ReduceAggregatorSum<T, TVAL> {
 public:
  inline ReduceAggregatorMean(int64_t N, const T&) : ReduceAggregatorSum<T, TVAL>(N, 0) {}
  inline T aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).mean();
  }
  inline T get_value() { return this->accumulator_ / static_cast<T>(this->N_); }

  // Fast reduction
  // WhichFastReduce() already defined in ReduceAggregatorSum

  static void FastReduceKR(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregatorSum<T, TVAL>::FastReduceKR(input, fast_shape, output, tp);
    // TODO: use MLAS or BLAS
    T* out = output.MutableData<T>();
    T* end = out + fast_shape[0];
    for (; out != end; ++out) {
      *out /= static_cast<T>(fast_shape[1]);
    }
  }

  static void FastReduceRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregatorSum<T, TVAL>::FastReduceRK(input, fast_shape, output, tp);
    // TODO: use MLAS or BLAS
    T* out = output.MutableData<T>();
    T* end = out + fast_shape[1];
    for (; out != end; ++out) {
      *out /= static_cast<T>(fast_shape[0]);
    }
  }

  static void FastReduceKRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    ReduceAggregatorSum<T, TVAL>::FastReduceKRK(input, fast_shape, output, tp);
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
};

template <typename T, typename TVAL = T>
class ReduceAggregatorMax : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorMax(int64_t N, const T& init) : ReduceAggregator<T, TVAL>(N, init) {}
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).maxCoeff();
  }
  inline void update(const T& v) { this->accumulator_ = v > this->accumulator_ ? v : this->accumulator_; }

  // Fast reduction
  static inline FastReduceKind WhichFastReduce() {
    return FastReduceKind::kKR | FastReduceKind::kRK | FastReduceKind::kKRK;
  }

  static void FastReduceKR(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorKR(fast_shape, output);
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1];
    concurrency::ThreadPool::TryParallelFor(
        tp, fast_shape[0], ParallelReduceFastCost(1, stridei, sizeof(T)),
        [data, stridei, out](std::ptrdiff_t first, std::ptrdiff_t last) {
          EigenVectorMap<T>(out + first, last - first) = ConstEigenMatrixMap<T>(
                                                             data + first * stridei, stridei, last - first)
                                                             .colwise()
                                                             .maxCoeff();
        });
  }

  static void FastReduceRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorRK(fast_shape, output);
    int64_t n_rows = fast_shape[0];
    int64_t N = fast_shape[1];
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    memcpy(out, data, N * sizeof(T));

    concurrency::ThreadPool::TryParallelFor(
        tp, N, ParallelReduceFastCost(1, n_rows, sizeof(T)),
        [data, out, N, n_rows](ptrdiff_t begin, ptrdiff_t end) {
          const T* p;
          for (int64_t row = 1; row < n_rows; ++row) {
            p = data + row * N;
            for (int64_t j = begin; j < end; ++j) {
              out[j] = out[j] > p[j] ? out[j] : p[j];
            }
          }
        });
  }

  static void FastReduceKRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorKRK(fast_shape, output);
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1] * fast_shape[2];
    int64_t strideo = fast_shape[2];
    concurrency::ThreadPool::TryParallelFor(
        tp, fast_shape[0], ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T)),
        [data, fast_shape, stridei, strideo, out](ptrdiff_t begin, ptrdiff_t end) {
          for (ptrdiff_t j = begin; j < end; ++j) {
            EigenVectorMap<T>(out + j * strideo, strideo) =
                ConstEigenMatrixMap<T>(
                    data + j * stridei, fast_shape[2], fast_shape[1])
                    .rowwise()
                    .maxCoeff();
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
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).maxCoeff(&this->arg_);
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
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).minCoeff(&this->arg_);
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

template <typename T, typename TVAL = T>
class ReduceAggregatorMin : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorMin(int64_t N, const T& init) : ReduceAggregator<T, TVAL>(N, init) {}
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).minCoeff();
  }
  inline void update(const T& v) { this->accumulator_ = v < this->accumulator_ ? v : this->accumulator_; }

  // Fast reduction
  static inline FastReduceKind WhichFastReduce() {
    return FastReduceKind::kKR | FastReduceKind::kRK | FastReduceKind::kKRK;
  }

  static void FastReduceKR(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorKR(fast_shape, output);
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1];
    concurrency::ThreadPool::TryParallelFor(
        tp, fast_shape[0], ParallelReduceFastCost(1, stridei, sizeof(T)),
        [data, stridei, out](std::ptrdiff_t first, std::ptrdiff_t last) {
          EigenVectorMap<T>(out + first, last - first) = ConstEigenMatrixMap<T>(
                                                             data + first * stridei, stridei, last - first)
                                                             .colwise()
                                                             .minCoeff();
        });
  }

  static void FastReduceRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                           Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorRK(fast_shape, output);
    int64_t n_rows = fast_shape[0];
    int64_t N = fast_shape[1];
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    memcpy(out, data, N * sizeof(T));

    concurrency::ThreadPool::TryParallelFor(
        tp, N, ParallelReduceFastCost(1, n_rows, sizeof(T)),
        [data, out, N, n_rows](ptrdiff_t begin, ptrdiff_t end) {
          const T* p;
          for (int64_t row = 1; row < n_rows; ++row) {
            p = data + row * N;
            for (int64_t j = begin; j < end; ++j) {
              out[j] = out[j] < p[j] ? out[j] : p[j];
            }
          }
        });
  }

  static void FastReduceKRK(const Tensor& input, const std::vector<int64_t>& fast_shape,
                            Tensor& output, concurrency::ThreadPool* tp) {
    OrtEnforce_ReduceAggregatorKRK(fast_shape, output);
    const T* data = input.Data<T>();
    T* out = output.MutableData<T>();
    int64_t stridei = fast_shape[1] * fast_shape[2];
    int64_t strideo = fast_shape[2];
    concurrency::ThreadPool::TryParallelFor(
        tp, fast_shape[0], ParallelReduceFastCost(fast_shape[1], fast_shape[2], sizeof(T)),
        [data, fast_shape, stridei, strideo, out](ptrdiff_t begin, ptrdiff_t end) {
          for (ptrdiff_t j = begin; j < end; ++j) {
            EigenVectorMap<T>(out + j * strideo, strideo) =
                ConstEigenMatrixMap<T>(
                    data + j * stridei, fast_shape[2], fast_shape[1])
                    .rowwise()
                    .minCoeff();
          }
        });
  }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorProd : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorProd(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 1) {}
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).prod();
  }
  inline void update(const T& v) { this->accumulator_ *= v; }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorL1 : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorL1(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 0) {}
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).cwiseAbs().sum();
  }
  inline void update(const T& v) { this->accumulator_ += v > 0 ? v : -v; }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorL2 : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorL2(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 0) {}
  inline TVAL aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).norm();
  }
  inline void update(const T& v) { this->accumulator_ += v * v; }
  inline TVAL get_value() { return reduce_sqrt<T>(this->accumulator_); }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorLogSum : public ReduceAggregator<T, TVAL> {
 public:
  inline ReduceAggregatorLogSum(int64_t N, const T&) : ReduceAggregator<T, TVAL>(N, 0) {}
  inline T aggall(const T* from_data) {
    return reduce_log<T>(Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).sum());
  }
  inline void update(const T& v) { this->accumulator_ += v; }
  inline TVAL get_value() { return reduce_log<T>(this->accumulator_); }
};

template <typename T, typename TVAL = T>
class ReduceAggregatorLogSumExp : public ReduceAggregator<T, TVAL> {
 protected:
  T max_;

 public:
  inline ReduceAggregatorLogSumExp(int64_t N, const T& init) : ReduceAggregator<T, TVAL>(N, 0) {
    max_ = reduce_isinf(init) ? this->accumulator_ : init;
  }
  inline TVAL aggall(const T* from_data) {
    max_ = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, this->N_).maxCoeff();
    for (int64_t i = 0; i < this->N_; ++i) {
      update(from_data[i]);
    }
    return get_value();
  }
  inline void update0(const T& v) {
    max_ = (reduce_isinf(v) || reduce_isnan(v) || v < max_) ? max_ : v;
  }
  inline void update(const T& v) { this->accumulator_ += reduce_exp(v - max_); }
  inline TVAL get_value() { return reduce_log<T>(this->accumulator_) + max_; }
};

void NoTransposePrepareForReduce(const TensorShape& new_input_shape,
                                 const std::vector<int64_t>& reduced_axes,
                                 ResultsNoTransposePrepareForReduce& results);

template <typename AGG>
void NoTransposeReduce1Loop(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                            const std::vector<int64_t>& reduced_axes, concurrency::ThreadPool* tp,
                            ResultsNoTransposePrepareForReduce& last_results);

// Specific case for ReduceLogSumExp.
template <typename AGG>
void NoTransposeReduce2Loops(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                             const std::vector<int64_t>& reduced_axes, concurrency::ThreadPool* tp,
                             ResultsNoTransposePrepareForReduce& last_results);

template <typename AGG>
void CommonReduce1Loop(OpKernelContext* ctx,
                       const std::vector<int64_t>& axes_, int64_t keepdims_,
                       bool noop_with_empty_axes = false);

// Specific case for ReduceLogSumExp.
template <typename AGG>
void CommonReduce2Loops(OpKernelContext* ctx,
                        const std::vector<int64_t>& axes_, int64_t keepdims_,
                        bool noop_with_empty_axes = false);

template <bool allow_multi_axes>
class ReduceKernelBase {
 protected:
  ReduceKernelBase(const OpKernelInfo& info, optional<int64_t> keepdims_override = {}) {
    if (allow_multi_axes) {
      axes_ = info.GetAttrsOrDefault<int64_t>("axes");
    } else {
      auto v = info.GetAttrOrDefault<int64_t>("axis", 0);
      axes_.push_back(v);
    }
    int64_t keepdims = 1;
    if (keepdims_override.has_value()) {
      keepdims = keepdims_override.value();
    } else {
      ORT_ENFORCE(info.GetAttr("keepdims", &keepdims).IsOK());
    }
    keepdims_ = (keepdims == 1);
    int64_t noop_with_empty_axes = info.GetAttrOrDefault<int64_t>("noop_with_empty_axes", 0);
    noop_with_empty_axes_ = (noop_with_empty_axes == 1);
    int64_t select_last_index = info.GetAttrOrDefault<int64_t>("select_last_index", 0);
    select_last_index_ = (select_last_index != 0);
  }

  std::vector<int64_t> axes_;
  bool keepdims_;
  bool noop_with_empty_axes_;
  bool select_last_index_;
};

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
  static Tensor Impl(const Tensor& input, const std::vector<int64_t>& reduce_axes,
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
