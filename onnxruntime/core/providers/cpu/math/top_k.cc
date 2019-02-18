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
using namespace std;
namespace onnxruntime {
// spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
ONNX_CPU_OPERATOR_KERNEL(
    TopK,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    TopK<float>);

static int64_t SizeToDim(size_t k, const vector<int64_t>& dims) {
  ORT_ENFORCE(k <= dims.size());
  int64_t r = 1;
  for (size_t i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

static int64_t SizeFromDim(size_t k, const vector<int64_t>& dims) {
  ORT_ENFORCE(k <= dims.size());
  int64_t r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

template <typename T>
struct ValueCmp {
  bool operator()(
      const pair<T, int64_t>& lhs,
      const pair<T, int64_t>& rhs) {
    return (
        lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

template <>
Status TopK<float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const vector<int64_t>& in_dims = X->Shape().GetDims();
  // Will return axis_ as is if positive or fixes it in case it is negative
  auto axis_parsed = HandleNegativeAxis(axis_, in_dims.size());	
  // Check to ensure k_ is within the bounds of what is available in that specific axis 	
  if (in_dims.at(axis_parsed) < k_) {
    ostringstream err_msg;
    err_msg << "k argment [" << k_ << "] should not be greater than specified axis dim value [" << in_dims.at(axis_parsed) << "]";
    return Status(common::ONNXRUNTIME, common::FAIL, err_msg.str());
  }

  const int64_t rows = SizeToDim(axis_parsed, in_dims);
  const int64_t cols = X->Shape().Size() / rows;
  auto input_map = ConstEigenMatrixMapRowMajor<float>(
      static_cast<const float*>(X->template Data<float>()),
      rows,
      cols);

  // Resize output tensors to be the same shape as the input except
  // for the specified dimension ((i.e.) axis_parsed), which will be of size k_. E.x. for an input tensor
  // of shape [3, 4, 5] and k_=2 with axis_parsed=1, both of these will be shape [3, 2, 5]
  vector<int64_t> output_linear_shape = in_dims;
  output_linear_shape[axis_parsed] = k_;
  auto* Values = p_op_kernel_context->Output(0, output_linear_shape);
  auto* Indices = p_op_kernel_context->Output(1, output_linear_shape);

  // Use Eigen maps to allow indexing into the 2d tensors like Values_map(i,j)
  const int64_t reduced_cols = SizeFromDim(axis_parsed, output_linear_shape);
  auto Values_map = EigenMatrixMapRowMajor<float>(
      Values->template MutableData<float>(), rows, reduced_cols);
  auto Indices_map = EigenMatrixMapRowMajor<int64_t>(
      Indices->template MutableData<int64_t>(), rows, reduced_cols);

  // This is basically the number of elements within each of the "k_" rows  
  const int64_t block_slice = reduced_cols / k_;
  // Sort preserving Indices
  for (int64_t i = 0; i < rows; ++i) {
	  for (int64_t j = 0; j < block_slice; ++j) {
		// Build a min-heap, the heap element is pair of (value, idx)
		// the top of the heap is the smallest value
		priority_queue<
			pair<float, int64_t>,
			vector<pair<float, int64_t>>,
			ValueCmp<float>>
			min_heap;
		// Maintain the size of heap to be less or equal to k_, so the
		// heap will hold the k_ largest Values
		for (int64_t k = 0; k < in_dims[axis_parsed]; ++k) {
			const auto value = input_map(i, k * block_slice + j);
			if (min_heap.size() < k_ || value > min_heap.top().first) {
				min_heap.push({value, k});
			}
			if (min_heap.size() > k_) {
				min_heap.pop();
			}    
		} 
		// Extract these k_ elements and place them in the results placeholder
		for (int64_t l = 0; l < k_; ++l) {
			auto& pqElem = min_heap.top();
			auto col_index = (k_ - l -1) * block_slice + j;  
			Values_map(i, col_index) = pqElem.first;
			Indices_map(i, col_index) = pqElem.second;
			min_heap.pop();
		}
	  }
  }

  return Status::OK();
}
}  // namespace onnxruntime
