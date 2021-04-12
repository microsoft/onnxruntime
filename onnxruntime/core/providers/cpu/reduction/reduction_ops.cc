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

bool SetupForReduce(const Tensor* input_tensor_ptr,
                    const std::vector<int64_t>& axes_,
                    std::vector<int64_t>& axes,
                    TensorShape& new_input_shape,
                    std::vector<int64_t>& output_shape,
                    bool& empty_reduce,
                    const TensorShape* input_shape_override) {
  ORT_ENFORCE(input_tensor_ptr != nullptr, "Input to be reduced is null");

  if (input_shape_override) {
    ORT_ENFORCE(input_tensor_ptr->Shape().Size() == input_shape_override->Size(),
                "The input shape override's size does not match the input tensor's shape size");
  }

  new_input_shape = input_shape_override ? *input_shape_override : input_tensor_ptr->Shape();
  size_t ndim = new_input_shape.NumDimensions();
  if (ndim == 0) {
    empty_reduce = true;
    return false;
  }

  axes.reserve(axes_.size());
  for (int64_t axis : axes_) {
    axes.push_back(HandleNegativeAxis(axis, static_cast<int64_t>(ndim)));
  }

  if (axes.empty()) {
    // This is the default case for non-arg kind reductions. Reduce on all dimensions.
    for (size_t i = 0; i < ndim; i++) {
      axes.push_back(i);
    }
  }

  std::sort(axes.begin(), axes.end());

  // If all reduced axes are located at the tail of the input shape, then copy could be skipped is required
  bool need_copy = true;
  if (axes.size() <= ndim && ndim > 0 &&
      axes.front() == static_cast<int64_t>(ndim - axes.size()) &&
      axes.back() == static_cast<int64_t>(ndim) - 1) {
    need_copy = false;
  }

  empty_reduce = false;
  output_shape = new_input_shape.GetDims();
  for (auto a : axes) {
    output_shape[a] = new_input_shape[a] > 0 ? 1 : 0;
    empty_reduce |= output_shape[a] == 0;
  }
  return need_copy;
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

template <typename T, typename AGG>
void NoTransposeReduce(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                       const std::vector<int64_t>& reduced_axes, concurrency::ThreadPool* tp,
                       ResultsNoTransposePrepareForReduce& last_results) {
  auto output_shape = output->Shape();
  const T* from_data = input.template Data<T>();
  typename AGG::value_type* to_data = output->template MutableData<typename AGG::value_type>();
  int64_t count = output_shape.Size();

  if (reduced_axes.size() == 0 || reduced_axes.size() == new_input_shape.NumDimensions()) {
    ORT_ENFORCE(count == 1, "Reduction on all axes, output size should be 1.");
    int64_t input_size = new_input_shape.Size();
    to_data[0] = AGG(input_size, from_data[0]).aggall(from_data);
    return;
  }

  if (!last_results.equal(new_input_shape.GetDims(), reduced_axes)) {
    NoTransposePrepareForReduce(new_input_shape, reduced_axes, last_results);
    if (last_results.last_loop_red_size == 0 || last_results.last_loop_size == 0)
      return;
  }
  ORT_ENFORCE(last_results.last_loop_red_size > 0);
  ORT_ENFORCE(last_results.last_loop_size > 0);
  ORT_ENFORCE(last_results.projected_index.size() > 0);
  int64_t denominator = last_results.last_loop_red_size * last_results.projected_index.size();

  if (AGG::two_loops()) {
    auto fn = [&](std::ptrdiff_t first, std::ptrdiff_t end) {
      int64_t loop;
      const T* loop_red_ptr;
      const T* loop_red_ptr_end;
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

    auto cost = TensorOpCost{(double)(last_results.projected_index.size() * sizeof(T) * last_results.last_loop_size * last_results.last_loop_red_size),
                             (double)last_results.last_loop_size * last_results.last_loop_red_size,
                             (double)last_results.projected_index.size() * last_results.last_loop_size * last_results.last_loop_red_size * 2};
    concurrency::ThreadPool::TryParallelFor(tp, count / last_results.last_loop_size, cost, fn);
  } else {
    auto fn = [&](std::ptrdiff_t first, std::ptrdiff_t end) {
      int64_t loop;
      const T* loop_red_ptr;
      const T* loop_red_ptr_end;
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

    auto cost = TensorOpCost{(double)(last_results.projected_index.size() * sizeof(T) * last_results.last_loop_size * last_results.last_loop_red_size),
                             (double)last_results.last_loop_size * last_results.last_loop_red_size,
                             (double)last_results.projected_index.size() * last_results.last_loop_size * last_results.last_loop_red_size};
    concurrency::ThreadPool::TryParallelFor(tp, count / last_results.last_loop_size, cost, fn);
  }
}

void DropDimensions(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& axes, std::vector<int64_t>& dropped_axes) {
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

template <typename T, typename AGG>
void CommonReduce(OpKernelContext* ctx,
                  const std::vector<int64_t> axes_, int64_t keepdims_,
                  ResultsNoTransposePrepareForReduce& last_results,
                  bool noop_with_empty_axes) {
  std::vector<int64_t> axes;
  const Tensor* input = ctx->Input<Tensor>(0);
  auto reduced_dims = input->Shape().GetDims();
  std::vector<int64_t> output_shape;
  bool empty_reduce;
  TensorShape new_input_shape;

  if (ctx->InputCount() == 2) {
    // second input holds the axes.
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->template Data<int64_t>();
    std::vector<int64_t> input_axes(data, data + nDims);
    if (input_axes.empty() && noop_with_empty_axes) {
      auto* output = ctx->Output(0, input->Shape());
      memcpy(output->template MutableData<typename AGG::value_type>(), input->template Data<T>(), input->SizeInBytes());
      return;
    }
    SetupForReduce(input, input_axes, axes, new_input_shape, output_shape, empty_reduce, nullptr);
  } else {
    SetupForReduce(input, axes_, axes, new_input_shape, output_shape, empty_reduce, nullptr);
  }

  if (empty_reduce) {
    Tensor* output = ctx->Output(0, keepdims_ ? output_shape : std::vector<int64_t>());
    if (new_input_shape.Size() == 1) {
      const T* from_data = input->template Data<T>();
      typename AGG::value_type* to_data = output->template MutableData<typename AGG::value_type>();
      AGG agg(1, *from_data);
      if (agg.two_loops()) {
        agg.update0(*from_data);
        agg.update(*from_data);
      } else {
        agg.update(*from_data);
      }
      *to_data = agg.get_value();
    } else {
      ORT_ENFORCE(keepdims_,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input->Shape());
    }
    return;
  }

  Tensor* output;
  if (keepdims_) {
    output = ctx->Output(0, output_shape);
  } else {
    std::vector<int64_t> dropped_axes;
    DropDimensions(output_shape, axes, dropped_axes);
    output = ctx->Output(0, dropped_axes);
  }
  NoTransposeReduce<T, AGG>(output, new_input_shape, *input, axes, ctx->GetOperatorThreadPool(), last_results);
}

template <typename T>
Status ReduceL1<T>::Compute(OpKernelContext* ctx) const {
  // The following variable does not change if the input tensor and the
  // axes do not either. It could be either cached in ctx or precomputed
  // in the constructor if shape and axes are known at this stage.
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorL1<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceL2<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorL2<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceLogSum<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorLogSum<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceLogSumExp<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorLogSumExp<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceMax<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorMax<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceMean<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorMean<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceMin<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorMin<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceProd<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorProd<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ReduceSum<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorSum<T>>(ctx, axes_, keepdims_, last_results, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
std::unique_ptr<Tensor> ReduceSum<T>::Impl(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                                           AllocatorPtr allocator, concurrency::ThreadPool* tp, bool keep_dims,
                                           const TensorShape* input_shape_override) {
  std::vector<int64_t> axes;
  auto reduced_dims = input.Shape().GetDims();
  std::vector<int64_t> output_shape;
  TensorShape new_input_shape;
  bool empty_reduce;
  SetupForReduce(&input, reduce_axes, axes, new_input_shape, output_shape, empty_reduce, input_shape_override);

  if (empty_reduce) {
    auto output = make_unique<Tensor>(input.DataType(), keep_dims ? output_shape : std::vector<int64_t>(), allocator);
    if (new_input_shape.Size() == 1) {
      const T* from_data = input.template Data<T>();
      T* to_data = output->template MutableData<T>();
      *to_data = *from_data;
    } else {
      ORT_ENFORCE(keep_dims,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  new_input_shape);
    }
    return output;
  }

  if (keep_dims) {
    ResultsNoTransposePrepareForReduce last_results;
    auto output = make_unique<Tensor>(input.DataType(), output_shape, allocator);
    NoTransposeReduce<T, ReduceAggregatorSum<T>>(output.get(), new_input_shape, input, axes, tp, last_results);
    return output;
  } else {
    ResultsNoTransposePrepareForReduce last_results;
    std::vector<int64_t> dropped_axes;
    DropDimensions(output_shape, axes, dropped_axes);
    auto output = make_unique<Tensor>(input.DataType(), dropped_axes, allocator);
    NoTransposeReduce<T, ReduceAggregatorSum<T>>(output.get(), new_input_shape, input, axes, tp, last_results);
    return output;
  }
}

template <typename T>
Status ReduceSumSquare<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  CommonReduce<T, ReduceAggregatorSumSquare<T>>(ctx, axes_, keepdims_, last_results);
  return Status::OK();
}

template <typename T>
Status ArgMax<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  if (select_last_index_) {
    CommonReduce<T, ReduceAggregatorArgMaxLastIndex<T>>(ctx, axes_, keepdims_, last_results);
  } else {
    CommonReduce<T, ReduceAggregatorArgMax<T>>(ctx, axes_, keepdims_, last_results);
  }
  return Status::OK();
}

template <typename T>
Status ArgMin<T>::Compute(OpKernelContext* ctx) const {
  ResultsNoTransposePrepareForReduce last_results;
  if (select_last_index_) {
    CommonReduce<T, ReduceAggregatorArgMinLastIndex<T>>(ctx, axes_, keepdims_, last_results);
  } else {
    CommonReduce<T, ReduceAggregatorArgMin<T>>(ctx, axes_, keepdims_, last_results);
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

}  // namespace onnxruntime
