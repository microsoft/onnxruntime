// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/providers/common.h"

using namespace std;
namespace onnxruntime {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion)                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      float,                                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),   \
      x<float>);                                                                      \
                                                                                      \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      int32_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()), \
      x<int32_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(x, startVer, endVer)              \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      float,                                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),   \
      x<float>);                                                                      \
                                                                                      \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      int32_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()), \
      x<int32_t>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(x, sinceVersion)               \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                    \
      x,                                                                             \
      sinceVersion,                                                                  \
      double,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), \
      x<double>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(x, startVer, endVer) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                          \
      x,                                                                             \
      startVer,                                                                      \
      endVer,                                                                        \
      double,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), \
      x<double>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(x, sinceVersion)                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      int64_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()), \
      x<int64_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(x, startVer, endVer)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      int64_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()), \
      x<int64_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(x, startVer, endVer)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                          \
      x,                                                                             \
      startVer,                                                                      \
      endVer,                                                                        \
      int8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()), \
      x<int8_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(x, startVer, endVer)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      uint8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()), \
      x<uint8_t>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(x, sinceVersion)                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                    \
      x,                                                                             \
      sinceVersion,                                                                  \
      int8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()), \
      x<int8_t>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(x, sinceVersion)                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      uint8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()), \
      x<uint8_t>);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceL1, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceL2, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceLogSum, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceLogSumExp, 13);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSumExp, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSumExp, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceLogSumExp, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 11, 11);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMax, 12, 12);

REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(ReduceMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(ReduceMax, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMean, 13);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMean, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMean, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceMean, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 11, 11);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMin, 12, 12);

REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(ReduceMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(ReduceMin, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceProd, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceProd, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceProd, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceProd, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceSum, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceSum, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceSum, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSumSquare, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSumSquare, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceSumSquare, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceSumSquare, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMax, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ArgMax, 11, 12)
REGISTER_UNARY_ELEMENTWISE_KERNEL(ArgMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ArgMax, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMin, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ArgMin, 11, 12)
REGISTER_UNARY_ELEMENTWISE_KERNEL(ArgMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ArgMin, 13);

FastReduceKind operator|(FastReduceKind a, FastReduceKind b) {
  return static_cast<FastReduceKind>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

bool operator==(FastReduceKind a, FastReduceKind b) {
  return static_cast<uint8_t>(a) == static_cast<uint8_t>(b);
}

bool operator!=(FastReduceKind a, FastReduceKind b) {
  return static_cast<uint8_t>(a) != static_cast<uint8_t>(b);
}

bool IsFastReduceKindAvailable(FastReduceKind scenario, FastReduceKind available) {
  return (static_cast<uint8_t>(scenario) & static_cast<uint8_t>(available)) > 0;
}

bool ResultsNoTransposePrepareForReduce::equal(const std::vector<int64_t>& local_input_shape,
                                               const std::vector<int64_t>& local_reduced_axes) {
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

void ResultsNoTransposePrepareForReduce::OrtEnforceNotEmpty() {
  ORT_ENFORCE(last_loop_red_size > 0);
  ORT_ENFORCE(last_loop_size > 0);
  ORT_ENFORCE(projected_index.size() > 0);
}

static void OrtEnforceMustBeOverloaded() {
  ORT_ENFORCE(false, "must be overloaded.");
}

void OrtEnforce_ReduceAggregatorKR(const std::vector<int64_t>& fast_shape, const Tensor& output) {
  ORT_ENFORCE(fast_shape.size() == 2, "Only works on matrices with two dimensions.");
  ORT_ENFORCE(fast_shape[0] == output.Shape().Size(), "Output size mismatch.");
}

void OrtEnforce_ReduceAggregatorRK(const std::vector<int64_t>& fast_shape, const Tensor& output) {
  ORT_ENFORCE(fast_shape.size() == 2, "Only works on matrices with two dimensions.");
  ORT_ENFORCE(fast_shape[1] == output.Shape().Size(), "Output size mismatch.");
}

void OrtEnforce_ReduceAggregatorKRK(const std::vector<int64_t>& fast_shape, const Tensor& output) {
  ORT_ENFORCE(fast_shape.size() == 3, "Only works on matrices with two dimensions.");
  ORT_ENFORCE(fast_shape[0] * fast_shape[2] == output.Shape().Size(), "Output size mismatch.");
}

void _ReduceAggregator::FastReduceKR(const Tensor&, const std::vector<int64_t>&, Tensor&, concurrency::ThreadPool*) {
  OrtEnforceMustBeOverloaded();
}
void _ReduceAggregator::FastReduceRK(const Tensor&, const std::vector<int64_t>&, Tensor&, concurrency::ThreadPool*) {
  OrtEnforceMustBeOverloaded();
}
void _ReduceAggregator::FastReduceKRK(const Tensor&, const std::vector<int64_t>&, Tensor&, concurrency::ThreadPool*) {
  OrtEnforceMustBeOverloaded();
}

void NoTransposePrepareForReduce(const TensorShape& new_input_shape,
                                 const std::vector<int64_t>& reduced_axes,
                                 ResultsNoTransposePrepareForReduce& results) {
  // Common initialisation for the indices.
  std::vector<int64_t> cumulative_shape = new_input_shape.GetDims();
  cumulative_shape[cumulative_shape.size() - 1] = 1;
  for (int i = static_cast<int>(cumulative_shape.size()) - 2; i >= 0; --i) {
    cumulative_shape[i] = cumulative_shape[i + 1] * new_input_shape[i + 1];
  }
  int64_t projection_size = 1;
  for (auto a : reduced_axes) {
    projection_size *= new_input_shape[a];
  }

  int last_reduced_axis = static_cast<int>(reduced_axes.size()) - 1;
  int loop_reduced_axis = 1;
  results.last_loop_red_size = new_input_shape[reduced_axes[last_reduced_axis]];
  results.last_loop_red_inc = cumulative_shape[reduced_axes[last_reduced_axis]];
  projection_size /= new_input_shape[reduced_axes[last_reduced_axis]];
  --last_reduced_axis;
  while (last_reduced_axis >= 0) {
    if (reduced_axes[last_reduced_axis] != reduced_axes[last_reduced_axis + 1] - 1)
      break;
    results.last_loop_red_size *= new_input_shape[reduced_axes[last_reduced_axis]];
    projection_size /= new_input_shape[reduced_axes[last_reduced_axis]];
    --last_reduced_axis;
    ++loop_reduced_axis;
  }

  // Builds the list of indices projected into the same sum.
  int reduced_axes_size = static_cast<int>(reduced_axes.size()) - loop_reduced_axis;
  if (reduced_axes_size == 0) {
    results.projected_index.resize(1, 0);
  } else {
    results.projected_index.resize(projection_size);
    std::vector<int64_t> projected_indices(reduced_axes_size, 0);
    int64_t current_index = 0;
    size_t current_pos = 0;
    int j;
    for (current_pos = 0; current_pos < results.projected_index.size(); ++current_pos) {
      results.projected_index[current_pos] = current_index;
      ++projected_indices[projected_indices.size() - 1];
      current_index += cumulative_shape[reduced_axes[reduced_axes_size - 1]];
      for (j = reduced_axes_size - 1; j > 0; --j) {
        if (projected_indices[j] < new_input_shape[reduced_axes[j]])
          break;
        projected_indices[j] -= new_input_shape[reduced_axes[j]];
        current_index -= new_input_shape[reduced_axes[j]] * cumulative_shape[reduced_axes[j]];
        ++projected_indices[j - 1];
        current_index += cumulative_shape[reduced_axes[j - 1]];
      }
    }
  }

  // Builds the list of indices for the unprojected sum.
  std::vector<int64_t> unreduced_axes;
  for (int64_t i = 0; i < static_cast<int64_t>(cumulative_shape.size()); ++i) {
    if (std::find(reduced_axes.begin(), reduced_axes.end(), i) != reduced_axes.end())
      continue;
    unreduced_axes.push_back(i);
  }
  int64_t unprojection_size = 1;
  for (auto a : unreduced_axes) {
    unprojection_size *= new_input_shape[a];
  }
  if (unprojection_size == 0) {
    return;
  }
  std::vector<int64_t> unprojected_indices(unreduced_axes.size(), 0);

  // The last index is usually an image size.
  // We differently process the last unprojected dimension.
  results.last_loop_size = new_input_shape[unreduced_axes[unreduced_axes.size() - 1]];
  int64_t unprojection_size_before_last = unprojection_size / results.last_loop_size;
  results.unprojected_index.reserve(unprojection_size_before_last);
  results.last_loop_inc = cumulative_shape[unreduced_axes[unreduced_axes.size() - 1]];
  if (unprojected_indices.size() <= 1) {
    results.unprojected_index.push_back(0);
  } else {
    int64_t current_index = 0;
    int j;
    for (int64_t pos = 0; pos < unprojection_size_before_last; ++pos) {
      results.unprojected_index.push_back(current_index);
      ++unprojected_indices[unprojected_indices.size() - 2];
      current_index += cumulative_shape[unreduced_axes[unreduced_axes.size() - 2]];
      for (j = static_cast<int>(unreduced_axes.size()) - 2; j > 0; --j) {
        if (unprojected_indices[j] < new_input_shape[unreduced_axes[j]])
          break;
        unprojected_indices[j] -= new_input_shape[unreduced_axes[j]];
        current_index -= new_input_shape[unreduced_axes[j]] * cumulative_shape[unreduced_axes[j]];
        ++unprojected_indices[j - 1];
        current_index += cumulative_shape[unreduced_axes[j - 1]];
      }
    }
  }
}

void OrtEnforceNoTransposeReduce(int64_t count) {
  ORT_ENFORCE(count == 1, "Reduction on all axes, output size should be 1.");
}

template <typename AGG>
void NoTransposeReduce1Loop(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                            const std::vector<int64_t>& reduced_axes, concurrency::ThreadPool* tp,
                            ResultsNoTransposePrepareForReduce& last_results) {
  auto output_shape = output->Shape();
  const typename AGG::input_type* from_data = input.template Data<typename AGG::input_type>();
  typename AGG::value_type* to_data = output->template MutableData<typename AGG::value_type>();
  int64_t count = output_shape.Size();

  if (reduced_axes.size() == 0 || reduced_axes.size() == new_input_shape.NumDimensions()) {
    OrtEnforceNoTransposeReduce(count);
    int64_t input_size = new_input_shape.Size();
    to_data[0] = AGG(input_size, from_data[0]).aggall(from_data);
    return;
  }

  if (!last_results.equal(new_input_shape.GetDims(), reduced_axes)) {
    NoTransposePrepareForReduce(new_input_shape, reduced_axes, last_results);
    if (last_results.last_loop_red_size == 0 || last_results.last_loop_size == 0)
      return;
  }
  last_results.OrtEnforceNotEmpty();
  int64_t denominator = last_results.last_loop_red_size * last_results.projected_index.size();

  auto fn = [&](std::ptrdiff_t first, std::ptrdiff_t end) {
    int64_t loop;
    const typename AGG::input_type* loop_red_ptr;
    const typename AGG::input_type* loop_red_ptr_end;
    int64_t current_index = first * last_results.last_loop_size;
    for (int64_t main_index = first; main_index < end; ++main_index) {
      for (loop = 0; loop < last_results.last_loop_size; ++loop, ++current_index) {
        int64_t origin = last_results.unprojected_index[main_index] + loop * last_results.last_loop_inc;
        AGG accumulator(denominator, from_data[origin + last_results.projected_index[0]]);
        for (auto it = last_results.projected_index.begin(); it != last_results.projected_index.end(); ++it) {
          loop_red_ptr = from_data + (origin + *it);
          loop_red_ptr_end = loop_red_ptr + last_results.last_loop_red_size * last_results.last_loop_red_inc;
          for (; loop_red_ptr != loop_red_ptr_end; loop_red_ptr += last_results.last_loop_red_inc) {
            accumulator.update(*loop_red_ptr);
          }
        }
        to_data[current_index] = accumulator.get_value();
      }
    }
  };

  auto cost = TensorOpCost{(double)(last_results.projected_index.size() * sizeof(typename AGG::input_type) *
                                    last_results.last_loop_size * last_results.last_loop_red_size),
                           (double)last_results.last_loop_size * last_results.last_loop_red_size,
                           (double)last_results.projected_index.size() * last_results.last_loop_size *
                               last_results.last_loop_red_size};
  concurrency::ThreadPool::TryParallelFor(tp, count / last_results.last_loop_size, cost, fn);
}

template <typename AGG>
void NoTransposeReduce2Loops(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                             const std::vector<int64_t>& reduced_axes, concurrency::ThreadPool* tp,
                             ResultsNoTransposePrepareForReduce& last_results) {
  auto output_shape = output->Shape();
  const typename AGG::input_type* from_data = input.template Data<typename AGG::input_type>();
  typename AGG::value_type* to_data = output->template MutableData<typename AGG::value_type>();
  int64_t count = output_shape.Size();

  if (reduced_axes.size() == 0 || reduced_axes.size() == new_input_shape.NumDimensions()) {
    OrtEnforceNoTransposeReduce(count);
    int64_t input_size = new_input_shape.Size();
    to_data[0] = AGG(input_size, from_data[0]).aggall(from_data);
    return;
  }

  if (!last_results.equal(new_input_shape.GetDims(), reduced_axes)) {
    NoTransposePrepareForReduce(new_input_shape, reduced_axes, last_results);
    if (last_results.last_loop_red_size == 0 || last_results.last_loop_size == 0)
      return;
  }
  last_results.OrtEnforceNotEmpty();
  int64_t denominator = last_results.last_loop_red_size * last_results.projected_index.size();

  auto fn = [&](std::ptrdiff_t first, std::ptrdiff_t end) {
    int64_t loop;
    const typename AGG::input_type* loop_red_ptr;
    const typename AGG::input_type* loop_red_ptr_end;
    int64_t current_index = first * last_results.last_loop_size;
    for (int64_t main_index = first; main_index < end; ++main_index) {
      for (loop = 0; loop < last_results.last_loop_size; ++loop, ++current_index) {
        int64_t origin = last_results.unprojected_index[main_index] + loop * last_results.last_loop_inc;
        AGG accumulator(denominator, from_data[origin + last_results.projected_index[0]]);
        for (auto it = last_results.projected_index.begin(); it != last_results.projected_index.end(); ++it) {
          loop_red_ptr = from_data + (origin + *it);
          loop_red_ptr_end = loop_red_ptr + last_results.last_loop_red_size * last_results.last_loop_red_inc;
          for (; loop_red_ptr != loop_red_ptr_end; loop_red_ptr += last_results.last_loop_red_inc) {
            accumulator.update0(*loop_red_ptr);
          }
        }
        for (auto it = last_results.projected_index.begin(); it != last_results.projected_index.end(); ++it) {
          loop_red_ptr = from_data + (origin + *it);
          loop_red_ptr_end = loop_red_ptr + last_results.last_loop_red_size * last_results.last_loop_red_inc;
          for (; loop_red_ptr != loop_red_ptr_end; loop_red_ptr += last_results.last_loop_red_inc) {
            accumulator.update(*loop_red_ptr);
          }
        }
        to_data[current_index] = accumulator.get_value();
      }
    }
  };

  auto cost = TensorOpCost{(double)(last_results.projected_index.size() * sizeof(typename AGG::input_type) *
                                    last_results.last_loop_size * last_results.last_loop_red_size),
                           (double)last_results.last_loop_size * last_results.last_loop_red_size,
                           (double)last_results.projected_index.size() * last_results.last_loop_size *
                               last_results.last_loop_red_size * 2};
  concurrency::ThreadPool::TryParallelFor(tp, count / last_results.last_loop_size, cost, fn);
}

void DropDimensions(const std::vector<int64_t>& input_shape,
                    const std::vector<int64_t>& axes,
                    std::vector<int64_t>& dropped_axes) {
  auto dropped_dims = input_shape;
  for (auto i : axes) {
    dropped_dims[i] = -1;
  }
  for (auto it = dropped_dims.begin(); it != dropped_dims.end(); ++it) {
    if (*it != -1) {
      dropped_axes.push_back(*it);
    }
  }
}

FastReduceKind OptimizeShapeForFastReduce(const std::vector<int64_t>& input_shape,
                                          const std::vector<int64_t>& reduced_axes,
                                          std::vector<int64_t>& fast_shape,
                                          std::vector<int64_t>& fast_output_shape,
                                          std::vector<int64_t>& fast_axes,
                                          bool keep_dims, bool noop_with_empty_axes) {
  if (input_shape.empty()) {
    fast_shape = input_shape;
    fast_output_shape = input_shape;
    fast_axes = reduced_axes;
    return FastReduceKind::kNone;
  }

  std::set<int64_t> axes;
  if (reduced_axes.size() == 0 && !noop_with_empty_axes) {
    for (int64_t i = 0; i < (int64_t)input_shape.size(); ++i) {
      axes.insert(i);
    }
  } else {
    for (auto it = reduced_axes.begin(); it != reduced_axes.end(); ++it) {
      axes.insert(HandleNegativeAxis(*it, static_cast<int64_t>(input_shape.size())));
    }
  }

  fast_output_shape.clear();
  fast_output_shape.reserve(input_shape.size());
  bool empty_reduce = false;
  std::vector<bool> reduce(input_shape.size());
  for (int64_t i = 0; i < (int64_t)input_shape.size(); ++i) {
    reduce[i] = axes.find(i) != axes.end();
    if (reduce[i]) {
      empty_reduce |= input_shape[i] == 0;
      if (keep_dims)
        fast_output_shape.push_back(input_shape[i] > 0 ? 1 : 0);
    } else {
      fast_output_shape.push_back(input_shape[i]);
    }
  }

  if (empty_reduce) {
    return FastReduceKind::kEmpty;
  }

  if (reduced_axes.empty()) {
    fast_shape.resize(1);
    fast_shape[0] = 1;
    for (auto a : input_shape) {
      fast_shape[0] *= a;
    }
    if (noop_with_empty_axes) {
      fast_axes.clear();
      fast_output_shape = input_shape;
      return FastReduceKind::kK;
    } else {
      if (keep_dims) {
        fast_output_shape.resize(input_shape.size(), 1);
      } else {
        fast_output_shape.clear();
      }
      fast_axes.resize(1);
      fast_axes[0] = 0;
      return FastReduceKind::kR;
    }
  }

  fast_shape.clear();
  fast_axes.clear();
  fast_shape.reserve(input_shape.size());
  fast_axes.reserve(reduced_axes.size());

  fast_shape.push_back(input_shape[0]);
  if (reduce[0])
    fast_axes.push_back(0);
  for (size_t i = 1; i < input_shape.size(); ++i) {
    if (reduce[i] == reduce[i - 1]) {
      fast_shape[fast_shape.size() - 1] *= input_shape[i];
    } else {
      if (reduce[i]) {
        fast_axes.push_back(fast_shape.size());
      }
      fast_shape.push_back(input_shape[i]);
    }
  }
  if (fast_shape.size() == 1) {
    return reduce[0] ? FastReduceKind::kR : FastReduceKind::kK;
  }
  if (fast_shape.size() == 2) {
    return reduce[0] ? FastReduceKind::kRK : FastReduceKind::kKR;
  }
  if (fast_shape.size() == 3 && !reduce[0]) {
    return FastReduceKind::kKRK;
  }
  return FastReduceKind::kNone;
}

void OrtEnforceCommonFastReduce(const Tensor* axes_tensor) {
  ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
  ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
              "An axes tensor must be a vector tensor.");
}

//template <typename T, typename TVAL>
bool CommonFastReduceCopy(OpKernelContext* ctx, std::vector<int64_t>& input_axes, bool noop_with_empty_axes) {
  if (ctx->InputCount() == 2) {
    // second input holds the axes.
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    OrtEnforceCommonFastReduce(axes_tensor);
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->template Data<int64_t>();
    input_axes.insert(input_axes.begin(), data, data + nDims);
    if (input_axes.empty() && noop_with_empty_axes) {
      const Tensor* input = ctx->Input<Tensor>(0);
      auto* output = ctx->Output(0, input->Shape());
      memcpy(reinterpret_cast<void*>(output->template MutableData<float>()),
             reinterpret_cast<const void*>(input->template Data<float>()),
             input->SizeInBytes());
      return true;
    }
  }
  return false;
}

typedef void fast_reduce_fct(const Tensor& input, const std::vector<int64_t>& fast_shape,
                             Tensor& output, concurrency::ThreadPool* tp);

bool CommonFastReduceSwitch(OpKernelContext* ctx,
                            const std::vector<int64_t>& axes_,
                            int64_t keepdims_,
                            bool noop_with_empty_axes,
                            FastReduceKind& fast_kind,
                            std::vector<int64_t>& fast_shape,
                            std::vector<int64_t>& output_shape,
                            std::vector<int64_t>& fast_axes,
                            FastReduceKind which_fast_reduce,
                            fast_reduce_fct* case_kr,
                            fast_reduce_fct* case_rk,
                            fast_reduce_fct* case_krk) {
  std::vector<int64_t> axes;
  const Tensor* input = ctx->Input<Tensor>(0);
  auto reduced_dims = input->Shape().GetDims();
  std::vector<int64_t> input_axes;

  if (CommonFastReduceCopy(ctx, input_axes, noop_with_empty_axes)) {
    return true;
  }

  fast_kind = OptimizeShapeForFastReduce(
      reduced_dims, input_axes.empty() ? axes_ : input_axes,
      fast_shape, output_shape, fast_axes, keepdims_, noop_with_empty_axes);
  if (which_fast_reduce != FastReduceKind::kNone) {
    if (IsFastReduceKindAvailable(fast_kind, which_fast_reduce)) {
      Tensor* output = ctx->Output(0, output_shape);
      switch (fast_kind) {
        case FastReduceKind::kKR: {
          case_kr(*input, fast_shape, *output, ctx->GetOperatorThreadPool());
          return true;
        }
        case FastReduceKind::kRK: {
          case_rk(*input, fast_shape, *output, ctx->GetOperatorThreadPool());
          return true;
        }
        case FastReduceKind::kKRK: {
          case_krk(*input, fast_shape, *output, ctx->GetOperatorThreadPool());
          return true;
        }
        case FastReduceKind::kR:
        case FastReduceKind::kK:
        case FastReduceKind::kNone:
        default:
          // Former implementation prevails in this case.
          break;
      }
    }
  }
  return false;
}

template <typename AGG>
bool CommonFastReduce(OpKernelContext* ctx,
                      const std::vector<int64_t>& axes_,
                      int64_t keepdims_,
                      bool noop_with_empty_axes,
                      FastReduceKind& fast_kind,
                      std::vector<int64_t>& fast_shape,
                      std::vector<int64_t>& output_shape,
                      std::vector<int64_t>& fast_axes) {
  return CommonFastReduceSwitch(ctx, axes_, keepdims_, noop_with_empty_axes, fast_kind, fast_shape, output_shape, fast_axes,
                                AGG::WhichFastReduce(), &AGG::FastReduceKR, &AGG::FastReduceRK, &AGG::FastReduceKRK);
}

static void OrtEnforceKeepDims(const TensorShape& shape, int64_t keepdims) {
  ORT_ENFORCE(keepdims,
              "Can't reduce on dim with value of 0 if 'keepdims' is false. "
              "Invalid output shape would be produced. input_shape:",
              shape);
}

static void OrtEnforceKeepDims(const Tensor* input, int64_t keepdims) {
  OrtEnforceKeepDims(input->Shape(), keepdims);
}

template <typename AGG>
void CommonReduce1Loop(OpKernelContext* ctx,
                       const std::vector<int64_t>& axes_, int64_t keepdims_,
                       bool noop_with_empty_axes) {
  FastReduceKind fast_kind;
  std::vector<int64_t> fast_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> fast_axes;
  if (CommonFastReduce<AGG>(ctx, axes_, keepdims_, noop_with_empty_axes,
                            fast_kind, fast_shape, output_shape, fast_axes)) {
    return;
  }

  const Tensor* input = ctx->Input<Tensor>(0);
  Tensor* output = ctx->Output(0, output_shape);
  if (fast_kind == FastReduceKind::kEmpty) {
    const TensorShape& new_input_shape = input->Shape();
    if (new_input_shape.Size() == 1) {
      const typename AGG::input_type* from_data = input->template Data<typename AGG::input_type>();
      typename AGG::value_type* to_data = output->template MutableData<typename AGG::value_type>();
      AGG agg(1, *from_data);
      agg.update(*from_data);
      *to_data = agg.get_value();
    } else {
      OrtEnforceKeepDims(input, keepdims_);
    }
    return;
  }

  ResultsNoTransposePrepareForReduce last_results;
  NoTransposeReduce1Loop<AGG>(output, fast_shape, *input, fast_axes, ctx->GetOperatorThreadPool(), last_results);
}

template <typename AGG>
void CommonReduce2Loops(OpKernelContext* ctx,
                        const std::vector<int64_t>& axes_, int64_t keepdims_,
                        bool noop_with_empty_axes) {
  FastReduceKind fast_kind;
  std::vector<int64_t> fast_shape, output_shape, fast_axes;
  if (CommonFastReduce<AGG>(ctx, axes_, keepdims_, noop_with_empty_axes,
                            fast_kind, fast_shape, output_shape, fast_axes)) {
    return;
  }

  const Tensor* input = ctx->Input<Tensor>(0);
  Tensor* output = ctx->Output(0, output_shape);
  if (fast_kind == FastReduceKind::kEmpty) {
    const TensorShape& new_input_shape = input->Shape();
    if (new_input_shape.Size() == 1) {
      const typename AGG::input_type* from_data = input->template Data<typename AGG::input_type>();
      typename AGG::value_type* to_data = output->template MutableData<typename AGG::value_type>();
      AGG agg(1, *from_data);
      agg.update0(*from_data);
      agg.update(*from_data);
      *to_data = agg.get_value();
    } else {
      OrtEnforceKeepDims(input, keepdims_);
    }
    return;
  }

  ResultsNoTransposePrepareForReduce last_results;
  NoTransposeReduce2Loops<AGG>(output, fast_shape, *input, fast_axes, ctx->GetOperatorThreadPool(), last_results);
}

template <typename T>
Status ReduceL1<T>::Compute(OpKernelContext* ctx) const {
  // The following variable does not change if the input tensor and the
  // axes do not either. It could be either cached in ctx or precomputed
  // in the constructor if shape and axes are known at this stage.
  CommonReduce1Loop<ReduceAggregatorL1<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceL2<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorL2<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceLogSum<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorLogSum<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceLogSumExp<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce2Loops<ReduceAggregatorLogSumExp<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceMax<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorMax<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceMean<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorMean<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceMin<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorMin<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceProd<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorProd<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ReduceSum<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorSum<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Tensor ReduceSum<T>::Impl(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                          AllocatorPtr allocator, concurrency::ThreadPool* tp, bool keep_dims,
                          const TensorShape* input_shape_override) {
  std::vector<int64_t> axes;
  std::vector<int64_t> output_shape, fast_shape, fast_axes;
  TensorShape new_input_shape = input_shape_override == nullptr ? input.Shape() : *input_shape_override;
  auto reduced_dims = new_input_shape.GetDims();

  FastReduceKind fast_kind = OptimizeShapeForFastReduce(
      reduced_dims, reduce_axes, fast_shape, output_shape, fast_axes, keep_dims, false);

  Tensor output(input.DataType(), keep_dims ? output_shape : std::vector<int64_t>(), allocator);

  if (fast_kind == FastReduceKind::kEmpty) {
    if (new_input_shape.Size() == 1) {
      const T* from_data = input.template Data<T>();
      T* to_data = output.template MutableData<T>();
      *to_data = *from_data;
    } else {
      OrtEnforceKeepDims(new_input_shape, keep_dims);
    }
    return output;
  }

  if (IsFastReduceKindAvailable(fast_kind, ReduceAggregatorSum<T>::WhichFastReduce())) {
    switch (fast_kind) {
      case FastReduceKind::kKR: {
        ReduceAggregatorSum<T>::FastReduceKR(input, fast_shape, output, tp);
        return output;
      }
      case FastReduceKind::kRK: {
        ReduceAggregatorSum<T>::FastReduceRK(input, fast_shape, output, tp);
        return output;
      }
      case FastReduceKind::kKRK: {
        ReduceAggregatorSum<T>::FastReduceKRK(input, fast_shape, output, tp);
        return output;
      }
      case FastReduceKind::kR:
      case FastReduceKind::kK:
      case FastReduceKind::kNone:
      default:
        // Former implementation prevails in this case.
        break;
    }
  }

  ResultsNoTransposePrepareForReduce last_results;
  NoTransposeReduce1Loop<ReduceAggregatorSum<T>>(&output, fast_shape, input, fast_axes, tp, last_results);
  return output;
}

template <typename T>
Status ReduceSumSquare<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorSumSquare<T>>(ctx, axes_, keepdims_);
  return Status::OK();
}

template <typename T>
Status ArgMax<T>::Compute(OpKernelContext* ctx) const {
  if (select_last_index_) {
    CommonReduce1Loop<ReduceAggregatorArgMaxLastIndex<T>>(ctx, axes_, keepdims_);
  } else {
    CommonReduce1Loop<ReduceAggregatorArgMax<T>>(ctx, axes_, keepdims_);
  }
  return Status::OK();
}

template <typename T>
Status ArgMin<T>::Compute(OpKernelContext* ctx) const {
  if (select_last_index_) {
    CommonReduce1Loop<ReduceAggregatorArgMinLastIndex<T>>(ctx, axes_, keepdims_);
  } else {
    CommonReduce1Loop<ReduceAggregatorArgMin<T>>(ctx, axes_, keepdims_);
  }
  return Status::OK();
}

// Explicit template instantiation -
// Even though there are kernels registered for ReduceSum op for these types,
// these are needed because we seem to get linker errors without these when the linker
// tries to resolve symbols in the einsum_auxiliary_ops obj file
template class ReduceSum<float>;
template class ReduceSum<int32_t>;
template class ReduceSum<double>;
template class ReduceSum<int64_t>;

template void CommonReduce1Loop<ReduceAggregatorSum<float>>(OpKernelContext* ctx,
                                                            const std::vector<int64_t>& axes_, int64_t keepdims_,
                                                            bool noop_with_empty_axes);
template void CommonReduce1Loop<ReduceAggregatorSum<int32_t>>(OpKernelContext* ctx,
                                                              const std::vector<int64_t>& axes_, int64_t keepdims_,
                                                              bool noop_with_empty_axes);
template void CommonReduce1Loop<ReduceAggregatorSum<double>>(OpKernelContext* ctx,
                                                             const std::vector<int64_t>& axes_, int64_t keepdims_,
                                                             bool noop_with_empty_axes);
template void CommonReduce1Loop<ReduceAggregatorSum<int64_t>>(OpKernelContext* ctx,
                                                              const std::vector<int64_t>& axes_, int64_t keepdims_,
                                                              bool noop_with_empty_axes);

}  // namespace onnxruntime
