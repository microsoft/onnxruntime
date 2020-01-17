/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "core/providers/cpu/math/top_k.h"
#include "core/providers/common.h"
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;
namespace onnxruntime {

template <typename T>
struct GreaterValueCmp {
  using DataType = T;
  bool operator()(const pair<T, int64_t>& lhs, const pair<T, int64_t>& rhs) {
    return (lhs.first > rhs.first ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

template <typename T>
struct LesserValueCmp {
  using DataType = T;
  bool operator()(const pair<T, int64_t>& lhs, const pair<T, int64_t>& rhs) {
    return (lhs.first < rhs.first ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

// Static helpers that implement the core logic for each of the 'TopK' operator flavor

// Selects the top k elements (largest or smallest based on template parameter)
template <class Comparator>
static vector<pair<typename Comparator::DataType, int64_t>> select_top_k(
    const ConstEigenMatrixMapRowMajor<typename Comparator::DataType>& raw_data, int64_t row_num, int64_t num_blocks,
    int64_t block_slice, int64_t inter_block_offset, const unsigned k,
    bool sort_top_k) {
  // create a data holder and insert elements
  vector<pair<typename Comparator::DataType, int64_t>> data_holder;
  data_holder.reserve(num_blocks);
  for (int64_t l = 0; l < num_blocks; ++l) {
    data_holder.push_back({raw_data(row_num, l * block_slice + inter_block_offset), l});
  }

  // find the top k (largest or smallest) elements in the data holder - O(n)
  nth_element(data_holder.begin(), data_holder.begin() + (k - 1), data_holder.end(), Comparator());

  // sort the top k elements if needed - O (k log k)
  if (sort_top_k) {
    std::sort(data_holder.begin(), data_holder.begin() + k, Comparator());
  }

  // the data_holder now contains the top k elements in the first k indices
  return data_holder;
}

// Given an input tensor 'input' and metadata values - 'k' and 'axis_parsed',
// this method will extract the sorted top k largest/smallest elements and place them in the output tensor 'values'
// along with the metadata output 'indices'
template <bool largest, bool sorted, class Comparator>
static void extract_top_k_elements(const Tensor* input, const TensorShape& input_shape, Tensor* values,
                                   Tensor* indices, const TensorShape& output_shape, const unsigned k,
                                   const unsigned axis_parsed) {
  // Cache some values that will be used in the implementation below
  const int64_t rows = input_shape.SizeToDimension(static_cast<size_t>(axis_parsed));
  const int64_t cols = input->Shape().Size() / rows;
  auto input_map =
      ConstEigenMatrixMapRowMajor<typename Comparator::DataType>(
          static_cast<const typename Comparator::DataType*>(input->template Data<typename Comparator::DataType>()), rows, cols);

  // Use Eigen maps to allow indexing into the 2d tensors like Values_map(i,j)
  const int64_t reduced_cols = output_shape.SizeFromDimension(static_cast<size_t>(axis_parsed));
  auto values_map = EigenMatrixMapRowMajor<typename Comparator::DataType>(
      values->template MutableData<typename Comparator::DataType>(), rows, reduced_cols);
  auto indices_map = EigenMatrixMapRowMajor<int64_t>(indices->template MutableData<int64_t>(), rows, reduced_cols);

  // This is basically the number of elements within each of the "k" rows
  const int64_t block_slice = reduced_cols / k;
  const int64_t num_blocks = input_shape[axis_parsed];

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < block_slice; ++j) {
      // Since sorted == true, we will use a Heap to hold the top K values in sorted fashion
      if (sorted) {  // The optimizer will clean-up the redundant condition based on the template parameter 'sorted'
        auto n_casted = static_cast<double>(num_blocks);
        auto k_casted = static_cast<double>(k);
        if ((n_casted + k_casted * log(k_casted)) < (n_casted * log(k_casted))) {
          // Select first  - O(n), then sort O(k * ln(k))
          // Overall complexity =  O (n + k * ln(k))
          const auto& data_holder = select_top_k<Comparator>(input_map, i, num_blocks, block_slice, j, k, true);
          for (int64_t l = 0; l < k; ++l) {
            const auto& elem = data_holder[l];
            auto col_index = l * block_slice + j;
            values_map(i, col_index) = elem.first;
            indices_map(i, col_index) = elem.second;
          }
        } else {
          // Perform sorted selection by passing 'n' elements over a heap of size 'k'
          // overall complexity =  O (n * ln(k))

          // Build a min-heap/max-heap, the heap element is pair of (value, idx)
          // The top of the heap is the smallest/largest value depending on whether it is a min-heap/max-heap
          // This is a min-heap if largest == true, this is a max-heap if largest == false
          priority_queue<pair<typename Comparator::DataType, int64_t>, vector<pair<typename Comparator::DataType, int64_t>>, Comparator> heap;

          // Maintain the size of heap to be less or equal to k, so the
          // heap will hold the k largest/smallest values
          for (int64_t l = 0; l < num_blocks; ++l) {
            const auto value = input_map(i, l * block_slice + j);
            // largest == true: insert into the min-heap if the size is < k or if the new
            // element is greater than the min element in the min-heap

            // largest == false: insert into the min-heap if the size is < k or if the new
            // element is lesser than the max element in the max-heap
            if ((heap.size() < k) || (largest && value > heap.top().first) ||
                (!largest && value < heap.top().first)) {  // the optimizer will clean-up the redundant condition based
                                                           // on the template parameter 'largest'
              heap.push({value, l});
            }
            if (heap.size() > k) {
              heap.pop();
            }
          }
          // Extract these k elements and place them in the results placeholder
          for (int64_t l = 0; l < k; ++l) {
            const auto& elem = heap.top();
            auto col_index = (k - l - 1) * block_slice + j;
            values_map(i, col_index) = elem.first;
            indices_map(i, col_index) = elem.second;
            heap.pop();
          }
        }
      } else {  // sorted == false
        // The optimizer will clean-up the redundant condition based on the template parameter 'sorted'

        // If the top K values are not required to be sorted, we use a more optimal selection algorithm
        // Average - O(n). Worst - O(n * ln(n)) or O(n^2) depending on the implementation, where 'n' is the number of input

        const auto& data_holder = select_top_k<Comparator>(input_map, i, num_blocks, block_slice, j, k, false);

        // Insert the top 'k' (largest or smallest) elements into the final output buffers
        for (int64_t l = 0; l < k; ++l) {
          const auto& elem = data_holder[l];
          auto col_index = l * block_slice + j;
          values_map(i, col_index) = elem.first;
          indices_map(i, col_index) = elem.second;
        }
      }
    }
  }
}

// Wrapper over core TopK implementation
template <typename T>
static Status TopKImpl(OpKernelContext* p_op_kernel_context, const Tensor* input, const int axis, const unsigned k,
                       bool largest = true, bool sorted = true) {
  const TensorShape& input_shape = input->Shape();
  // Will return axis_ as is if positive or fixes it in case it is negative
  const auto axis_parsed = HandleNegativeAxis(axis, static_cast<int64_t>(input_shape.NumDimensions()));
  // Check to ensure k is within the bounds of what is available in that specific axis
  if (input_shape[axis_parsed] < k) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "k argument [", k,
                           "] should not be greater than specified axis dim value [", input_shape[axis_parsed], "]");
  }

  // Resize output tensors to be the same shape as the input except
  // for the specified dimension ((i.e.) axis_parsed), which will be of size k. E.x. for an input tensor
  // of shape [3, 4, 5] and k=2 with axis_parsed=1, both of the outputs will be shape [3, 2, 5]
  TensorShape output_shape = input_shape;
  output_shape[axis_parsed] = k;
  auto* values = p_op_kernel_context->Output(0, output_shape);
  auto* indices = p_op_kernel_context->Output(1, output_shape);

  if (values == nullptr || indices == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "output count mismatch, expected 2 outputs to be present for TopK operator");
  }

  // no-op - no output buffers to fill - return silently
  if (k == 0) {
    return Status::OK();
  }

  if (sorted && largest) {
    // extract sorted largest TopK elements
    extract_top_k_elements<true, true, GreaterValueCmp<T>>(input, input_shape, values, indices, output_shape, k,
                                                           gsl::narrow_cast<unsigned>(axis_parsed));
  } else if (sorted && !largest) {
    // extract sorted smallest TopK elements
    extract_top_k_elements<false, true, LesserValueCmp<T>>(input, input_shape, values, indices, output_shape, k,
                                                           gsl::narrow_cast<unsigned>(axis_parsed));
  } else if (largest) {
    // extract unsorted (order undefined) largest TopK elements
    extract_top_k_elements<true, false, GreaterValueCmp<T>>(input, input_shape, values, indices, output_shape, k,
                                                            gsl::narrow_cast<unsigned>(axis_parsed));
  } else {
    // extract unsorted (order undefined) smallest TopK elements
    extract_top_k_elements<false, false, LesserValueCmp<T>>(input, input_shape, values, indices, output_shape, k,
                                                            gsl::narrow_cast<unsigned>(axis_parsed));
  }

  return Status::OK();
}

// Opset ver - 1 to 9
template <>
TopK<9, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  int64_t k_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_temp).IsOK());
  ORT_ENFORCE(k_temp > 0);
  k_ = gsl::narrow_cast<unsigned>(k_temp);

  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis_ = gsl::narrow_cast<int>(axis_temp);
}

// Opset ver - 1 to 9
template <>
Status TopK<9, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "input count mismatch, expected 1 input - the tensor to be processed");
  }

  return TopKImpl<float>(p_op_kernel_context, X, axis_, k_);
}

// Opset ver - 10
template <>
TopK<10, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis_ = gsl::narrow_cast<int>(axis_temp);
}

// Opset ver - 10
template <>
Status TopK<10, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  const auto* Y = p_op_kernel_context->Input<Tensor>(1);
  if (X == nullptr || Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "input count mismatch, expected 2 inputs - "
                           "the tensor to be processed and a tensor containing k value");
  }

  const vector<int64_t>& y_shape = Y->Shape().GetDims();
  if (y_shape.size() != 1 || y_shape[0] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "k tensor should be a 1D tensor of size 1");
  }

  auto parsed_input_k = Y->template Data<int64_t>()[0];
  if (parsed_input_k < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "value of k must not be negative");
  }

  return TopKImpl<float>(p_op_kernel_context, X, axis_, gsl::narrow_cast<unsigned>(parsed_input_k));
}

// Opset ver - 11

static void TopkOpset11ConstructorCommon(const OpKernelInfo& op_kernel_info,
                                         int& axis, bool& largest, bool& sorted) {
  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis = gsl::narrow_cast<int>(axis_temp);

  int64_t largest_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("largest", &largest_temp).IsOK());
  largest = largest_temp == 1 ? true : false;

  int64_t sorted_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("sorted", &sorted_temp).IsOK());
  sorted = sorted_temp == 1 ? true : false;
}

template <>
TopK<11, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset11ConstructorCommon(op_kernel_info, axis_, largest_, sorted_);
}

template <>
TopK<11, int64_t>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset11ConstructorCommon(op_kernel_info, axis_, largest_, sorted_);
}

template <typename T>
static Status ComputeImplOpset11(OpKernelContext* p_op_kernel_context, int axis, bool is_largest, bool is_sorted) {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  const auto* Y = p_op_kernel_context->Input<Tensor>(1);
  if (X == nullptr || Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "input count mismatch, expected 2 inputs - "
                           "the tensor to be processed and a tensor containing k value");
  }

  const vector<int64_t>& y_shape = Y->Shape().GetDims();
  if (y_shape.size() != 1 || y_shape[0] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "k tensor should be a 1D tensor of size 1");
  }

  auto parsed_input_k = Y->template Data<int64_t>()[0];
  if (parsed_input_k < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "value of k must not be negative");
  }

  return TopKImpl<T>(p_op_kernel_context, X, axis, gsl::narrow_cast<unsigned>(parsed_input_k), is_largest, is_sorted);
}

// Opset ver - 11
template <>
Status TopK<11, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset11<float>(p_op_kernel_context, axis_, largest_, sorted_);
}

template <>
Status TopK<11, int64_t>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset11<int64_t>(p_op_kernel_context, axis_, largest_, sorted_);
}

// Register necessary kernels
// spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(TopK, 1, 9,
                                   KernelDefBuilder()
                                       .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                                   TopK<9, float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(TopK, 10, 10,
                                   KernelDefBuilder()
                                       .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                                   TopK<10, float>);

#define REGISTER_TOPK_TYPED_KERNEL(OPSET, TYPE)                                                    \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(TopK,                                                             \
                                 OPSET,                                                            \
                                 TYPE,                                                             \
                                 KernelDefBuilder()                                                \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())     \
                                     .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                                 TopK<OPSET, TYPE>);

REGISTER_TOPK_TYPED_KERNEL(11, float);
REGISTER_TOPK_TYPED_KERNEL(11, int64_t);

}  // namespace onnxruntime
