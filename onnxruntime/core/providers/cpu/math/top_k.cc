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
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;
namespace onnxruntime {

template <typename T>
struct GreaterValueCmp {
  using DataType = T;
  GreaterValueCmp(const T* data = nullptr) : data_(data) {
  }

  bool operator()(const int64_t lhs_idx, const int64_t rhs_idx) const {
    return (data_[lhs_idx] > data_[rhs_idx] ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (data_[lhs_idx] == data_[rhs_idx] && lhs_idx < rhs_idx));
  }

  bool CompareValueOnly(const T& lhs, const T& rhs) const {
    return lhs > rhs;
  }

 private:
  const T* data_;
};

template <typename T>
struct LesserValueCmp {
  using DataType = T;

  LesserValueCmp(const T* data = nullptr) : data_(data) {
  }

  bool operator()(const int64_t lhs_idx, const int64_t rhs_idx) const {
    return (data_[lhs_idx] < data_[rhs_idx] ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (data_[lhs_idx] == data_[rhs_idx] && lhs_idx < rhs_idx));
  }

  bool CompareValueOnly(const T& lhs, const T& rhs) const {
    return lhs < rhs;
  }

 private:
  const T* data_;
};

/*
Maintain a binary heap where HeapComp of the parent with either child is false.
  e.g. if the comparison is 'greater than', the parent is smaller than both children.
There is no ordering within a level.

NOTE: The comparison is backwards compared to std::priority_queue as we use the same comparator for this as for
      nth_element in SelectTopK. As such for a heap selecting the largest values the comparator is 'greater than'.
*/
template <class HeapCmp>
static void HeapifyIthPosition(int64_t* heap, size_t i, size_t k, const HeapCmp& heap_cmp) {
  while (true) {
    size_t left = 2 * i + 1;
    size_t right = left + 1;
    if (right < k) {
      // need to check both left and right children as either could be replaced

      // check if we should move child up. check left node as well as whether left is preferred over right.
      // if 'i' can replace left, check whether right would replace left (if so, i replaces left as it's the weakest)
      bool i_replaces_left = heap_cmp(heap[i], heap[left]);
      if (i_replaces_left && heap_cmp(heap[right], heap[left])) {
        // left is going to be pushed up as both i and right beat it
        // NOTE: std::swap is slower as it uses std::move
        auto tmp = heap[i];
        heap[i] = heap[left];
        heap[left] = tmp;
        i = left;
      } else if (i_replaces_left || heap_cmp(heap[i], heap[right])) {
        // i_replaces_left implies left replaces right due to 'if' so replace right with i as right is the weakest.
        // also check if i only beats right
        auto tmp = heap[i];
        heap[i] = heap[right];
        heap[right] = tmp;
        i = right;
      } else
        break;
    } else if ((left < k) && heap_cmp(heap[i], heap[left])) {
      auto tmp = heap[i];
      heap[i] = heap[left];
      heap[left] = tmp;
      i = left;
    } else
      break;
  }
}

// Static helpers that implement the core logic for each of the 'TopK' operator flavor

// Selects the top k elements (largest or smallest based on template parameter)
template <class Comparator>
static void SelectTopK(const Comparator& comparer,
                       int64_t row_offset, int64_t num_blocks, int64_t block_slice, int64_t inter_block_offset,
                       const unsigned k, bool sort_top_k, vector<int64_t>& data_holder) {
  for (int64_t l = 0; l < num_blocks; ++l) {
    data_holder[l] = (row_offset + (l * block_slice + inter_block_offset));
  }

  // find the top k (largest or smallest) elements in the data holder - O(n) average. O(n*n) worst case.
  // See https://en.wikipedia.org/wiki/Quickselect
  nth_element(data_holder.begin(), data_holder.begin() + (k - 1), data_holder.end(), comparer);

  // sort the top k elements if needed - O (k log k)
  if (sort_top_k) {
    std::sort(data_holder.begin(), data_holder.begin() + k, comparer);
  }

  // the data_holder now contains the indices of the top k elements in the first k elements
}

// Given an input tensor 'input' and metadata values - 'k' and 'axis_parsed',
// this method will extract the sorted top k largest/smallest elements and place them in the output tensor 'values'
// along with the metadata output 'indices'
template <class Comparator>
static void FindTopKElements(const Tensor* input, const TensorShape& input_shape, Tensor* values,
                             Tensor* indices, const TensorShape& output_shape, const unsigned k, bool sorted,
                             const unsigned axis_parsed, concurrency::ThreadPool* threadpool) {
  // Cache some values that will be used in the implementation below
  const int64_t rows = input_shape.SizeToDimension(static_cast<size_t>(axis_parsed));
  const int64_t cols = input->Shape().Size() / rows;
  const auto* input_data = input->template Data<typename Comparator::DataType>();

  // Use Eigen maps for convenient indexing into the 2d tensors like Values_map(i,j)
  const int64_t reduced_cols = output_shape.SizeFromDimension(static_cast<size_t>(axis_parsed));

  auto* values_data = values->template MutableData<typename Comparator::DataType>();
  auto* indices_data = indices->template MutableData<int64_t>();
  auto values_map = EigenMatrixMapRowMajor<typename Comparator::DataType>(values_data, rows, reduced_cols);
  auto indices_map = EigenMatrixMapRowMajor<int64_t>(indices_data, rows, reduced_cols);

  // This is basically the number of elements within each of the "k" rows
  const int64_t num_blocks = input_shape[axis_parsed];
  const int64_t block_slice = reduced_cols / k;

  int64_t tp_threads = concurrency::ThreadPool::DegreeOfParallelism(threadpool);
  int64_t num_threads = std::min(tp_threads, rows);  // split on rows so can't have more threads than rows

  // rough attempt to make sure there's enough work for each thread. if there's insufficient work the usage of
  // too many threads degrades performance.
  // TODO: May want a different calculation for each branch below instead.
  int64_t threads_needed = static_cast<int64_t>(std::floor(input_shape.Size() * k / (128 * 1024)));
  num_threads = std::max(std::min(threads_needed, num_threads), static_cast<int64_t>(1));

  // from testing various batch sizes relative to k, the following appears to work well as a selector.
  // tested with following combinations
  //   batch_size = [ 8, 16, 32, 64, 128, 256, 512, 1024, 2048 ]
  //            k = [ 1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 128 ]
  bool use_priority_queue = k != 1 && (k < 4 || (std::log2(k) / std::log2(num_blocks)) < 0.725);

  std::function<void(std::ptrdiff_t batch)> find_top_k;

  if (k == 1) {
    // just need to compare values and not indexes as the first instance of the best value is always selected
    find_top_k =
        [num_threads, rows, block_slice, num_blocks, input_data, cols,
         &values_map, &indices_map](std::ptrdiff_t batch) {
          auto work = concurrency::ThreadPool::PartitionWork(batch, num_threads, rows);
          Comparator comparer(input_data);

          for (auto i = work.start; i < work.end; ++i) {
            auto row_offset = i * cols;
            for (int64_t j = 0; j < block_slice; ++j) {
              int64_t cur_idx = row_offset + j;

              const auto* cur_value = input_data + cur_idx;  // using pointer to data is faster than input_data[cur_idx]
              auto best = *cur_value;                        // save best value so we only have one load in the CompareValueOnly call
              int64_t top_idx = cur_idx;

              for (int64_t l = 1; l < num_blocks; ++l) {
                cur_value += block_slice;
                if (comparer.CompareValueOnly(*cur_value, best)) {
                  best = *cur_value;
                  top_idx = cur_value - input_data;
                }
              }

              values_map(i, j) = best;
              // convert overall index to result index
              // avoid '/' if possible for perf reasons
              indices_map(i, j) = block_slice == 1 ? (top_idx - row_offset - j)
                                                   : (top_idx - row_offset - j) / block_slice;
            }
          }
        };
  } else if (use_priority_queue) {
    find_top_k =
        [num_threads, rows, block_slice, num_blocks, k, sorted,
         input_data, cols, &values_map, &indices_map](std::ptrdiff_t batch) {
          auto work = concurrency::ThreadPool::PartitionWork(batch, num_threads, rows);
          Comparator comparer(input_data);

          // the heap is stored in indices_data. each iteration overwrites the old data when it adds the
          // initial k values, so we don't need to clear it.
          std::vector<int64_t> indices_data(k);
          int64_t* indices = indices_data.data();  // raw pointer is slightly faster for HeapifyIthPosition

          for (auto i = work.start; i < work.end; ++i) {
            const auto row_offset = i * cols;

            for (int64_t j = 0; j < block_slice; ++j) {
              int64_t l = 0;
              auto cur_idx = row_offset + j;

              // add first k items starting from the bottom up
              for (; l < k; ++l) {
                indices[k - l - 1] = cur_idx;
                HeapifyIthPosition(indices, k - l - 1, k, comparer);

                cur_idx += block_slice;
              }

              // insert remainder if the next value would replace the top of the heap (current worst top k value)
              // save top so we only have one load in the CompareValueOnly call
              auto top = input_data[indices[0]];
              for (; l < num_blocks; ++l) {
                // we can compare value only. if the current value is equal to the top of the heap it won't
                // replace it as the index will be higher.
                if (comparer.CompareValueOnly(input_data[cur_idx], top)) {
                  indices[0] = cur_idx;
                  HeapifyIthPosition(indices, 0, k, comparer);
                  top = input_data[indices[0]];
                }

                cur_idx += block_slice;
              }

              if (sorted) {
                // Extract these k elements and place them in the results placeholder
                for (l = 0; l < k; ++l) {
                  auto idx = indices[0];
                  auto col_index = (k - l - 1) * block_slice + j;
                  values_map(i, col_index) = input_data[idx];
                  // convert overall index to result index. avoid '/' if possible for perf reasons
                  indices_map(i, col_index) = block_slice == 1 ? (idx - row_offset - j)
                                                               : (idx - row_offset - j) / block_slice;

                  // put the last value at the top of the heap to replace the removed one, and push it into
                  // place in a heap one smaller.
                  indices[0] = indices[k - l - 1];
                  HeapifyIthPosition(indices, 0, k - l - 1, comparer);
                }
              } else {
                for (l = 0; l < k; ++l) {
                  int64_t idx = indices[l];
                  auto col_index = l * block_slice + j;
                  values_map(i, col_index) = input_data[idx];
                  // convert overall index to result index. avoid '/' if possible for perf reasons
                  indices_map(i, col_index) = block_slice == 1 ? (idx - row_offset - j)
                                                               : (idx - row_offset - j) / block_slice;
                }
              }
            }
          }
        };
  } else {
    find_top_k =
        [num_threads, rows, block_slice, num_blocks, k, sorted,
         input_data, cols,
         &values_map, &indices_map](std::ptrdiff_t batch) {
          auto work = concurrency::ThreadPool::PartitionWork(batch, num_threads, rows);
          Comparator comparer(input_data);

          // we re-use a single data_holder for performance. avoids allocating memory on each iteration.
          // the call to SelectTopK overwrites any existing data so we don't need to clear on each iteration.
          std::vector<int64_t> data_holder(num_blocks);

          for (auto i = work.start; i < work.end; ++i) {
            auto row_offset = i * cols;
            for (int64_t j = 0; j < block_slice; ++j) {
              SelectTopK<Comparator>(comparer, row_offset, num_blocks, block_slice, j, k, sorted, data_holder);

              // Insert the top 'k' (largest or smallest) elements into the final output buffers
              for (int64_t l = 0; l < k; ++l) {
                int64_t idx = data_holder[l];
                auto col_index = l * block_slice + j;
                values_map(i, col_index) = input_data[idx];
                // convert overall index to result index. avoid the cost of the '/' is possible
                indices_map(i, col_index) = block_slice == 1 ? (idx - row_offset - j)
                                                             : (idx - row_offset - j) / block_slice;
              }
            }
          }
        };
  }

  if (num_threads <= 1) {
    find_top_k(0);
  } else {
    // we want to re-use the storage variables in each lambda as much as possible to minimize allocations
    // on each iteration, so the lambda does multiple rows. e.g. the data_holder and indices_data vectors.
    // the alternative would be to use TryBatchParallelFor with the lambda doing one row.
    // Use TrySimpleParallelFor so openmp is supported correctly
    concurrency::ThreadPool::TrySimpleParallelFor(threadpool, num_threads, find_top_k);
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

  auto* threadpool = p_op_kernel_context->GetOperatorThreadPool();

  if (largest) {
    FindTopKElements<GreaterValueCmp<T>>(input, input_shape, values, indices, output_shape, k, sorted,
                                         gsl::narrow_cast<unsigned>(axis_parsed), threadpool);
  } else {
    FindTopKElements<LesserValueCmp<T>>(input, input_shape, values, indices, output_shape, k, sorted,
                                        gsl::narrow_cast<unsigned>(axis_parsed), threadpool);
  }

  return Status::OK();
}

// Opset ver - 1 to 9

static void TopkOpset9ConstructorCommon(const OpKernelInfo& op_kernel_info, int& axis, unsigned int& k) {
  int64_t k_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_temp).IsOK());
  ORT_ENFORCE(k_temp > 0);
  k = gsl::narrow_cast<unsigned>(k_temp);

  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis = gsl::narrow_cast<int>(axis_temp);
}

template <typename T>
static Status ComputeImplOpset9(OpKernelContext* p_op_kernel_context, int axis, int k) {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "input count mismatch, expected 1 input - the tensor to be processed");
  }

  return TopKImpl<T>(p_op_kernel_context, X, axis, k);
}

template <>
TopK<9, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset9ConstructorCommon(op_kernel_info, axis_, k_);
}

template <>
TopK<9, double>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset9ConstructorCommon(op_kernel_info, axis_, k_);
}

template <>
Status TopK<9, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset9<float>(p_op_kernel_context, axis_, k_);
}

template <>
Status TopK<9, double>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset9<double>(p_op_kernel_context, axis_, k_);
}

// Opset ver - 10

static void TopkOpset10ConstructorCommon(const OpKernelInfo& op_kernel_info, int& axis) {
  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis = gsl::narrow_cast<int>(axis_temp);
}

template <typename T>
static Status ComputeImplOpset1011(OpKernelContext* p_op_kernel_context, int axis, bool is_largest, bool is_sorted) {
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

template <>
TopK<10, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset10ConstructorCommon(op_kernel_info, axis_);
}

template <>
TopK<10, double>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset10ConstructorCommon(op_kernel_info, axis_);
}

template <>
Status TopK<10, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset1011<float>(p_op_kernel_context, axis_, true, true);
}

template <>
Status TopK<10, double>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset1011<double>(p_op_kernel_context, axis_, true, true);
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
TopK<11, double>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset11ConstructorCommon(op_kernel_info, axis_, largest_, sorted_);
}

template <>
TopK<11, int32_t>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset11ConstructorCommon(op_kernel_info, axis_, largest_, sorted_);
}

template <>
TopK<11, int64_t>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset11ConstructorCommon(op_kernel_info, axis_, largest_, sorted_);
}

// Opset ver - 11
template <>
Status TopK<11, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset1011<float>(p_op_kernel_context, axis_, largest_, sorted_);
}

template <>
Status TopK<11, double>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset1011<double>(p_op_kernel_context, axis_, largest_, sorted_);
}

template <>
Status TopK<11, int32_t>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset1011<int32_t>(p_op_kernel_context, axis_, largest_, sorted_);
}

template <>
Status TopK<11, int64_t>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset1011<int64_t>(p_op_kernel_context, axis_, largest_, sorted_);
}

// Register necessary kernels
// spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK

#define REGISTER_TOPK_VERSIONED_TYPED_KERNEL(OPSET1, OPSET2, TYPE)                                           \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(TopK, OPSET1, OPSET2, TYPE,                                       \
                                           KernelDefBuilder()                                                \
                                               .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())     \
                                               .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                                           TopK<OPSET2, TYPE>);

#define REGISTER_TOPK_TYPED_KERNEL(OPSET, TYPE)                                                    \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(TopK,                                                             \
                                 OPSET,                                                            \
                                 TYPE,                                                             \
                                 KernelDefBuilder()                                                \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())     \
                                     .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                                 TopK<OPSET, TYPE>);

REGISTER_TOPK_VERSIONED_TYPED_KERNEL(1, 9, float);
REGISTER_TOPK_VERSIONED_TYPED_KERNEL(1, 9, double);
REGISTER_TOPK_VERSIONED_TYPED_KERNEL(10, 10, float);
REGISTER_TOPK_VERSIONED_TYPED_KERNEL(10, 10, double);

REGISTER_TOPK_TYPED_KERNEL(11, float);
REGISTER_TOPK_TYPED_KERNEL(11, double);
REGISTER_TOPK_TYPED_KERNEL(11, int64_t);
REGISTER_TOPK_TYPED_KERNEL(11, int32_t);

}  // namespace onnxruntime
