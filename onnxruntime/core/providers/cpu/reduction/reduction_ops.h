// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CORE_PROVIDERS_CPU_REDUCTION_OPS_H
#define CORE_PROVIDERS_CPU_REDUCTION_OPS_H

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/containers.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

template <typename T>
class ReduceAggregator {
 protected:
  int64_t N_;
  T accumulator_;

 public:
  inline ReduceAggregator(int64_t N) {
    N_ = N;
    accumulator_ = 0;
  }
  inline void update(const T& v) { accumulator_ += v; }
  inline T aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, N_).sum();
  }
  inline T get_value() { return accumulator_; }
};

template <typename T>
class ReduceAggregatorSum : public ReduceAggregator<T> {
 public:
  inline ReduceAggregatorSum(int64_t N) : ReduceAggregator(N) {}
};

template <typename T>
class ReduceAggregatorMean : public ReduceAggregator<T> {
 public:
  inline ReduceAggregatorMean(int64_t N) : ReduceAggregator(N) {}
  inline T aggall(const T* from_data) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(from_data, N_).mean();
  }
  inline T get_value() { return accumulator_ / static_cast<T>(N_); }
};

bool NeedsTransposeForReduce(const Tensor* input_tensor_ptr,
                             const std::vector<int64_t>& axes_,
                             std::vector<int64_t>& axes,
                             TensorShape& new_input_shape,
                             std::vector<int64_t>& output_shape,
                             bool& empty_reduce,
                             const TensorShape* input_shape_override);

class ResultsExperimentalPrepareForReduce {
 public:
  std::vector<int64_t> input_shape;
  std::vector<int64_t> reduced_axes;
  std::vector<int64_t> projected_index;
  int64_t last_loop_red_size;
  int64_t last_loop_red_inc;
  std::vector<int64_t> unprojected_index;
  int64_t last_loop_size;
  int64_t last_loop_inc;
  bool equal(const std::vector<int64_t>& local_input_shape, const std::vector<int64_t>& local_reduced_axes) {
    if (input_shape.size() != local_input_shape.size())
      return false;
    if (reduced_axes.size() != local_reduced_axes.size())
      return false;
    for (std::vector<int64_t>::const_iterator it1 = input_shape.begin(), it2 = local_input_shape.begin();
         it1 != input_shape.end(); ++it1, ++it2) {
      if (*it1 != *it2)
        return false;
    }
    for (std::vector<int64_t>::const_iterator it1 = reduced_axes.begin(), it2 = local_reduced_axes.begin();
         it1 != reduced_axes.end(); ++it1, ++it2) {
      if (*it1 != *it2)
        return false;
    }
    return true;
  }
};

void ExperimentalPrepareForReduce(const TensorShape& new_input_shape,
                                  const std::vector<int64_t>& reduced_axes,
                                  ResultsExperimentalPrepareForReduce& results);

template <typename T, typename AGG>
void ExperimentalReduce(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                        const std::vector<int64_t>& reduced_axes, concurrency::ThreadPool* tp,
                        ResultsExperimentalPrepareForReduce& last_results);

template <typename T, typename AGG>
void CommonReduce(OpKernelContext* ctx,
                  const std::vector<int64_t> axes_, int64_t keepdims_,
                  ResultsExperimentalPrepareForReduce& last_results);

template <typename T>
bool PrepareForReduce(const Tensor* input_tensor_ptr,
                      FastAllocVector<T>& transposed_input_data,
                      int64_t& block_size,
                      int64_t& blocks,
                      const std::vector<int64_t>& axes_,
                      bool keepdims_,
                      /*out*/ std::vector<int64_t>& reduced_dims,
                      bool check_no_transpose = false,
                      const TensorShape* input_shape_override = nullptr);

template <typename T>
void ReduceSumCore(const T* input_data, T* output_data, bool no_transpose,
                   int64_t blocks, int64_t block_size, FastAllocVector<T>& transposed_input_data,
                   concurrency::ThreadPool* tp);

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

  // Caches the configuration of the last execution.
  mutable ResultsExperimentalPrepareForReduce last_results_;
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
